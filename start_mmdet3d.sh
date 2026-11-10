#!/bin/bash

#######################################################################
#
# bash start_mmdet3d.sh <master_addr> <NNODES> <rank> <config_path>
# bash start_mmdet3d.sh "" "" "" ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py
#
#######################################################################
source /root/.hygon_base_env

# env var
export NCCL_TOPO_FILE=null
export NCCL_ALGO=Ring
export NCCL_RINGS="N0 0 7 6 5 4 3 2 1 N0|N1 1 2 3 4 5 6 7 0 N1|N2 2 1 0 7 6 5 4 3 N2|N3 3 4 5 6 7 0 1 2 N3|N4 4 3 2 1 0 7 6 5 N4|N5 5 6 7 0 1 2 3 4 N5|N6 6 5 4 3 2 1 0 7 N6|N7 7 0 1 2 3 4 5 6 N7"

export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32=1
export MIOPEN_FIND_MODE=1
export ROCBLAS_MATH_MODE=1

export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=${TORCHINDUCTOR_LAYOUT_OPTIMIZATION:-0}
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING:-0}

MASTER_ADDR=${1:-localhost}
MASTER_PORT=6005
NNODES=${2:-1}
NODE_RANK=${3:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}

select_socket_ifname() {
  local current="${1:-}"
  if [ -n "$current" ]; then
    IFS=',' read -ra ifaces <<< "$current"
    for iface in "${ifaces[@]}"; do
      iface=${iface#^}
      if [ -e "/sys/class/net/${iface}" ]; then
        state=$(cat "/sys/class/net/${iface}/operstate" 2>/dev/null || true)
        if [ "$state" = "up" ] || [ "$iface" = "lo" ]; then
          echo "$current"
          return
        fi
      fi
    done
  fi
  for iface_path in /sys/class/net/*; do
    iface=$(basename "$iface_path")
    state=$(cat "$iface_path/operstate" 2>/dev/null || true)
    if [ "$state" = "up" ] && [[ ! "$iface" =~ ^(lo|docker|br-) ]]; then
      echo "$iface"
      break
    fi
  done
}

SOCKET_IFNAME=$(select_socket_ifname "${NCCL_SOCKET_IFNAME:-}")
if [ -n "$SOCKET_IFNAME" ]; then
  export NCCL_SOCKET_IFNAME="$SOCKET_IFNAME"
  export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME"
fi
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
  echo "[BASH] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
fi
DISTRIBUTED_ARGS=(
  --nproc_per_node "$GPUS_PER_NODE"
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
)
CONFIG=${4:-./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py}
CFG_OPTIONS=(--cfg-options "data.workers_per_gpu=${WORKERS_PER_GPU}")
if [ -n "${SAMPLES_PER_GPU:-}" ]; then
  CFG_OPTIONS+=("data.samples_per_gpu=${SAMPLES_PER_GPU}")
fi

CONFIG_NAME=$(basename "$CONFIG")
MODEL_NAME=${CONFIG_NAME%.*}
TIME=$(date "+%Y-%m-%d_%H")

# ===== 日志文件 =====
LOG_FILE="./logs/${MODEL_NAME}_${NNODES}_${TIME}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[BASH] Logs will be written to ${LOG_FILE}"
echo "[BASH] Start time: $(date)"

# ===== 构建命令 =====
NUMA_SCRIPT='
numa_map=( $(hy-smi --showtopo | grep "Numa Node" | awk "{print \$6}") );
LOCAL_RANK=${LOCAL_RANK:-0}
NUMA_ID=${numa_map[$LOCAL_RANK]}
if [ -n "${NUMA_ID}" ]; then
  numactl --cpunodebind=${NUMA_ID} --membind=${NUMA_ID} python $(dirname "$0")/tools/train.py "$@"
else
  echo "[BASH] NUMA topology unavailable for LOCAL_RANK=${LOCAL_RANK}; running without numactl"
  python $(dirname "$0")/tools/train.py "$@"
fi
'

CMD=(
    torchrun
    "${DISTRIBUTED_ARGS[@]}"
    --no-python
    bash -c "$NUMA_SCRIPT"
    _
    "$CONFIG"
    --launcher pytorch
    "${CFG_OPTIONS[@]}"
    ${@:5}
    --deterministic
)

# ===== 打印命令 =====
echo "[BASH] Launch command:"
echo "PYTHONPATH=\"$(dirname $0)/..\":$PYTHONPATH ${CMD[*]}"
echo "----------------------------------------"

# ===== 执行命令 =====
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH "${CMD[@]}"
EXIT_CODE=$?

echo "----------------------------------------"
echo "[BASH] End time: $(date)"
echo "[BASH] Exit code: ${EXIT_CODE}"
exit $EXIT_CODE
