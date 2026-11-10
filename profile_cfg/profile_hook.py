# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""mmcv hook that captures one torch.profiler trace and then stops profiling.

Ported from a Trainer-callback implementation: ``skip_first`` skips compile
warm-up, every rank exports its own Chrome trace, rank 0 also writes an operator
table, and ``StopAfterTrace`` shuts the profiler down right after the first
successful export so the rest of the run is not slowed down.

It lives outside the model repository and is loaded through the config's
``custom_imports``, so no model source file changes.
"""

import json
import os
from pathlib import Path

import torch
from mmcv.runner import HOOKS, Hook, get_dist_info


class StopAfterTrace:
    """Stop a profiler immediately after its first successful trace export."""

    def __init__(self, profiler, on_stopped=None):
        self.profiler = profiler
        self.on_stopped = on_stopped
        self.active = False
        self.trace_exported = False

    def __enter__(self):
        self.profiler.start()
        self.active = True
        return self

    def mark_trace_exported(self):
        self.trace_exported = True

    def step(self):
        if not self.active:
            return
        self.profiler.step()
        if self.trace_exported:
            self.close()
            if self.on_stopped is not None:
                self.on_stopped()

    def close(self):
        if self.active:
            self.active = False
            self.profiler.stop()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


@HOOKS.register_module()
class StopAfterTraceProfilerHook(Hook):
    """Profile ``active`` iterations after ``skip_first + wait + warmup``.

    Args:
        out_dir: Output directory; relative paths are placed under ``work_dir``.
        skip_first / wait / warmup / active / repeat: ``torch.profiler.schedule``
            arguments. Set ``skip_first`` beyond the compile warm-up.
        stop_after_trace: Stop the profiler after the first export
            (requires ``repeat=1``).
        ranks: Ranks that export a trace; ``None`` means every rank.
        sort_by: Sort key of the rank-0 operator table.
    """

    def __init__(self,
                 out_dir='profiling',
                 skip_first=20,
                 wait=1,
                 warmup=5,
                 active=3,
                 repeat=1,
                 stop_after_trace=True,
                 with_stack=False,
                 record_shapes=False,
                 profile_memory=False,
                 ranks=None,
                 sort_by='self_cuda_time_total'):
        if stop_after_trace and repeat != 1:
            raise ValueError('stop_after_trace requires repeat=1')
        self.out_dir = out_dir
        self.schedule = dict(
            skip_first=skip_first,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat)
        self.stop_after_trace = stop_after_trace
        self.options = dict(
            with_stack=with_stack,
            record_shapes=record_shapes,
            profile_memory=profile_memory)
        self.ranks = None if ranks is None else set(ranks)
        self.sort_by = sort_by
        self._ctx = None
        self._controller = None

    def before_run(self, runner):
        self.runner = runner
        self.rank, _ = get_dist_info()
        out = Path(self.out_dir)
        if not out.is_absolute():
            out = Path(runner.work_dir) / out
        self.profile_dir = out
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        raw = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(**self.schedule),
            on_trace_ready=self._on_trace_ready,
            **self.options)
        if self.stop_after_trace:
            self._controller = StopAfterTrace(raw, on_stopped=self._record_stop)
            self._ctx = self._controller
        else:
            self._ctx = raw
        self._ctx.__enter__()
        runner.logger.info(
            f'[profiler] rank {self.rank} profiling, schedule={self.schedule}, '
            f'output={self.profile_dir}')

    def after_train_iter(self, runner):
        if self._ctx is not None:
            self._ctx.step()

    def after_run(self, runner):
        if self._ctx is not None:
            self._ctx.__exit__(None, None, None)
            self._ctx = None

    def _table(self, prof):
        try:
            return prof.key_averages().table(sort_by=self.sort_by)
        except Exception:
            return prof.key_averages().table(sort_by='self_device_time_total')

    def _on_trace_ready(self, prof):
        if self.ranks is not None and self.rank not in self.ranks:
            if self._controller is not None:
                self._controller.mark_trace_exported()
            return
        if self.rank == 0:
            table = self._table(prof)
            print(table)
            (self.profile_dir / 'rank0-key-averages.txt').write_text(table + '\n')
        path = self.profile_dir / (
            f'trace_rank_{self.rank}_iter_{self.runner.iter}.json')
        prof.export_chrome_trace(str(path))
        if self._controller is not None:
            self._controller.mark_trace_exported()
        self.runner.logger.info(f'[profiler] trace saved to {path}')

    def _record_stop(self):
        path = self.profile_dir / f'profiler-stopped-rank-{self.rank}.json'
        path.write_text(
            json.dumps({
                'rank': self.rank,
                'iter': self.runner.iter,
                'stopped_after_trace': True
            }) + '\n')
