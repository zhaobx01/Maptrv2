#!/usr/bin/env bash
# 检查 MapTRv2 验证流程能否跑通（问题 1/2/3 的修复是否生效）。
#
#   bash check_val.sh                      # 1 卡，用 epoch_24.pth
#   NPROC=8 bash check_val.sh              # 8 卡
#   CKPT=/path/to/x.pth bash check_val.sh  # 指定权重
#   SKIP_RUN=1 bash check_val.sh           # 只做不需要 GPU 的前置检查
#
# 通过标准：日志里出现 NuscMap_chamfer/mAP，且没有 Traceback。
# 失败时退出码非 0，并打印原因。
set -uo pipefail

MODEL_ROOT=${MODEL_ROOT:-/workspace/model/MapTrv2}
TURBO_SRC=${TURBO_SRC:-/workspace/TurboPhysAI}
CFG_DIR=${CFG_DIR:-/workspace/TurboPhysAI/turbo_physai/optimizations/models/maptrv2/configs}
CONFIG=${CONFIG:-projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py}
CKPT=${CKPT:-/workspace/MapTRv2/work_dirs/maptrv2_nusc_r50_24ep/epoch_24.pth}
MAP_ANN=${MAP_ANN:-/workspace/MapTRv2/anno_pkls/nuscenes_map_anns_val.json}
NPROC=${NPROC:-1}
PORT=${PORT:-29503}
LOG=${LOG:-/tmp/check_val_$(date +%Y%m%d_%H%M%S).log}

export PYTHONPATH="$TURBO_SRC:${PYTHONPATH:-}"

fail() { echo "[FAIL] $*" >&2; exit 1; }
ok()   { echo "[ OK ] $*"; }

echo "== 1. 前置文件检查 =="
[[ -d $MODEL_ROOT ]]            || fail "模型目录不存在: $MODEL_ROOT"
[[ -f $MODEL_ROOT/$CONFIG ]]    || fail "config 不存在: $MODEL_ROOT/$CONFIG"
[[ -f $CKPT ]]                  || fail "checkpoint 不存在: $CKPT"
[[ -f $CFG_DIR/optimization.yaml ]] || fail "optimization.yaml 不存在: $CFG_DIR/optimization.yaml"
[[ -f $CFG_DIR/runtime.yaml ]]  || fail "runtime.yaml 不存在: $CFG_DIR/runtime.yaml"
ok "config / checkpoint / 优化配置都存在"

# 问题 3：标注文件必须存在，且不能指向只读目录
[[ -f $MAP_ANN ]] || fail "map_ann_file 不存在: $MAP_ANN"
python - "$MAP_ANN" <<'PY' || fail "map_ann_file 不是有效的 GT 标注 json"
import json, sys
d = json.load(open(sys.argv[1]))
assert "GTs" in d and len(d["GTs"]) > 0, "缺少 GTs"
print(f"[ OK ] map_ann_file 有效，GT 条数 = {len(d['GTs'])}")
PY

echo
echo "== 2. 优化配置是否包含问题 1/2 的修复 =="
# 旧的 optimization.yaml 没有这两项，必须重新 generate
grep -q "maptrv2.eval_compat" "$CFG_DIR/optimization.yaml" \
  || fail "optimization.yaml 缺少 maptrv2.eval_compat（问题 2），需要重新 generate"
grep -q "_run_ddp_forward" "$CFG_DIR/optimization.yaml" \
  || fail "optimization.yaml 缺少 _run_ddp_forward（问题 1），需要重新 generate"
ok "optimization.yaml 含 eval_compat 和 _run_ddp_forward"

echo
echo "== 3. 不需要 GPU 的单元检查 =="
python -m pytest "$TURBO_SRC/test/optimizations/test_maptrv2_eval_compat.py" -q 2>&1 | tail -3
[[ ${PIPESTATUS[0]} -eq 0 ]] || fail "test_maptrv2_eval_compat.py 未通过"
ok "单元测试通过"

if [[ ${SKIP_RUN:-0} == 1 ]]; then
  echo; echo "SKIP_RUN=1，跳过实际验证。"; exit 0
fi

echo
echo "== 4. 实际验证（${NPROC} 卡）=="
echo "日志: $LOG"
cd "$MODEL_ROOT" || fail "无法进入 $MODEL_ROOT"

turbo-physai run \
  --optimization-config "$CFG_DIR/optimization.yaml" \
  --runtime-config "$CFG_DIR/runtime.yaml" \
  --log-report \
  torchrun --nproc-per-node="$NPROC" --master-port="$PORT" tools/test.py \
    "$CONFIG" "$CKPT" \
    --launcher pytorch \
    --eval chamfer \
    --cfg-options \
      data.val.map_ann_file="$MAP_ANN" \
      data.test.map_ann_file="$MAP_ANN" \
  2>&1 | tee "$LOG"
RUN_RC=${PIPESTATUS[0]}

echo
echo "== 5. 结果判定 =="
RESULT=0

# 优化是否真的装上了
if grep -q "Summary: applied=" "$LOG"; then
  grep -m1 "Summary: applied=" "$LOG"
  grep -E "maptrv2\.(training|eval_compat): status=" "$LOG" | sort -u
  if grep -E "maptrv2\.(training|eval_compat): status=" "$LOG" | grep -vq "status=applied"; then
    echo "[WARN] training / eval_compat 不是 applied，修复可能没有生效"
    RESULT=1
  fi
else
  echo "[WARN] 日志里没有优化报告，确认是否经过 turbo-physai run"
  RESULT=1
fi

# 三类已知报错
grep -q "_use_replicated_tensor_module" "$LOG" && { echo "[FAIL] 问题 1 复现: DDP 属性缺失"; RESULT=1; }
grep -q "object has no attribute 'intersects'" "$LOG" && { echo "[FAIL] 问题 2 复现: Shapely STRtree"; RESULT=1; }
grep -qiE "Read-only file system|只读文件系统|Permission denied" "$LOG" && { echo "[FAIL] 问题 3 复现: 写只读目录"; RESULT=1; }
grep -q "Traceback" "$LOG" && { echo "[FAIL] 日志中有 Traceback:"; grep -m1 -A12 "Traceback" "$LOG"; RESULT=1; }

# 是否真的算出了指标
if grep -q "NuscMap_chamfer/mAP" "$LOG"; then
  echo "[ OK ] 已输出指标:"
  grep -E "NuscMap_chamfer/(mAP|divider_AP|ped_crossing_AP|boundary_AP)'" "$LOG" | tail -4 \
    || grep -m1 "NuscMap_chamfer/mAP" "$LOG"
else
  echo "[FAIL] 日志里没有 NuscMap_chamfer/mAP，验证没有走完"
  RESULT=1
fi

[[ $RUN_RC -ne 0 ]] && { echo "[FAIL] 验证进程退出码 = $RUN_RC"; RESULT=1; }

echo
if [[ $RESULT -eq 0 ]]; then
  echo "==> 通过：验证流程可以完整跑通。"
else
  echo "==> 未通过：见上面的 [FAIL]/[WARN]，完整日志在 $LOG"
fi
exit $RESULT
