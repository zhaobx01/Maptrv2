# 在原配置上按环境变量追加 profiler hook；PROFILE 未设置时与原配置完全一致。
#
#   PROFILE=1 PROFILER_SKIP_FIRST=200 PROFILER_WAIT=1 PROFILER_WARMUP=5 \
#   PROFILER_ACTIVE=3 PYTHONPATH=/workspace/MapTRv2/profile_cfg:$PYTHONPATH ...
import os

_base_ = ['/workspace/model/MapTrv2/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py']

if os.environ.get('PROFILE', '0') == '1':
    custom_imports = dict(imports=['profile_hook'], allow_failed_imports=False)
    custom_hooks = [
        dict(
            type='StopAfterTraceProfilerHook',
            out_dir='profiling',
            skip_first=int(os.environ.get('PROFILER_SKIP_FIRST', 20)),
            wait=int(os.environ.get('PROFILER_WAIT', 1)),
            warmup=int(os.environ.get('PROFILER_WARMUP', 5)),
            active=int(os.environ.get('PROFILER_ACTIVE', 3)),
            repeat=1,
            stop_after_trace=os.environ.get('PROFILER_STOP_AFTER_TRACE', '1') == '1',
            with_stack=os.environ.get('PROFILER_WITH_STACK', '0') == '1',
        ),
    ]
