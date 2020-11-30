import torch.autograd.profiler as prof
from torch.autograd import ProfilerActivity

from typing import Callable, Iterable, Optional

class EnablePred(object):
    """
    EnablePred describes on which steps profiler is active:
    - profiler starts in inactive state and stays in inactive state for the first 'wait' steps
    - profiler then enters a warmup state and stays in this state for the next 'warmup' steps
    - profiler then starts actively tracing/collecting stats for the next 'active' steps
    - after this, profiler returns to the inactive state and cycle repeats

    In case output_fn is specified, it is called every time the trace is ready
    """
    class Action(object):
        START_WARMUP = 0
        START_TRACE = 1
        STOP_TRACE = 2

    class State(object):
        INACTIVE = 0
        WARMUP = 1
        ACTIVE = 2

    def __init__(self, wait: int, warmup: int, active: int, output_fn: Optional[Callable[[prof.profile], None]]):
        assert wait >= 0 and warmup >= 0 and active > 0
        if warmup == 0:
            print("Warning: profiler won't be using a warmup, which can skew profiler results")
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.output_fn = output_fn

        def active_active_fn(step):
            if self._mod_step(step) == 1:
                return [EnablePred.Action.STOP_TRACE, EnablePred.Action.START_WARMUP, EnablePred.Action.START_TRACE]
            else:
                return []

        def inactive_warmup_fn(_):
            raise RuntimeError("Incorrect profiler state sequence")

        self.actions_map = {
            EnablePred.State.ACTIVE: {
                EnablePred.State.ACTIVE: active_active_fn,
                EnablePred.State.WARMUP: [EnablePred.Action.START_TRACE],
                EnablePred.State.INACTIVE: [EnablePred.Action.START_WARMUP, EnablePred.Action.START_TRACE],
            },
            EnablePred.State.WARMUP: {
                EnablePred.State.ACTIVE: [EnablePred.Action.STOP_TRACE, EnablePred.Action.START_WARMUP],
                EnablePred.State.WARMUP: [],
                EnablePred.State.INACTIVE: [EnablePred.Action.START_WARMUP],
            },
            EnablePred.State.INACTIVE: {
                EnablePred.State.ACTIVE: [EnablePred.Action.STOP_TRACE],
                EnablePred.State.WARMUP: inactive_warmup_fn,
                EnablePred.State.INACTIVE: [],
            }
        }

    def _mod_step(self, step: int):
        sum_states = self.wait + self.warmup + self.active
        r = step % sum_states
        if r == 0:
            r = sum_states
        return r

    def _num_state(self, step: int):
        mod_step = self._mod_step(step)
        if mod_step <= self.wait:
            return EnablePred.State.INACTIVE
        elif mod_step <= self.wait + self.warmup:
            return EnablePred.State.WARMUP
        else:
            return EnablePred.State.ACTIVE

    def actions(self, step: int):
        if step == 1:
            st = self._num_state(step)
            if st == EnablePred.State.ACTIVE:
                return [EnablePred.Action.START_WARMUP, EnablePred.Action.START_TRACE]
            elif st == EnablePred.State.WARMUP:
                return [EnablePred.Action.START_WARMUP]
            else:
                return []
        else:
            st = self._num_state(step)
            prev_st = self._num_state(step - 1)
            acts = self.actions_map[st][prev_st]
            if callable(acts):
                return acts(step)
            else:
                return acts


class profile(object):
    """
    PyTorch profiler context manager.

    Arguments:
        activities - list of activity groups (CPU, CUDA)
        enable_pred (optional) - iteration predicate function, used together with `next_step` call

    Notes:
     - profiler is based on the Kineto library - system profiler library, with support for CUPTI tracing
     - enable_pred is used for training loop tracing, allowing users to enable profiler on certain
       iterations and account for the warmup
     - when enable_pred is not set, profiler is always active
     - next_step uses record_function api to add information about steps in the trace
    """
    def __init__(
            self,
            activities: Iterable[ProfilerActivity],
            enable_pred: Optional[EnablePred] = None,
            record_shapes=False,
            profile_memory=False,
            with_stack=False):
        self.activities = activities
        self.enable_pred = enable_pred
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.step_num = 0
        self.profiler: Optional[prof.profile] = None
        self.step_rec_fn: Optional[prof.record_function] = None

        if not self.enable_pred:
            print("Warning: using profiler without enable predicate may result in the skewed " +
                "results, use enable_pred to control the warmup time")

    def __enter__(self):
        self.next_step()
        if not self.enable_pred:
            self._run_action(EnablePred.Action.START_WARMUP)
            self._run_action(EnablePred.Action.START_TRACE)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        if self.profiler:
            if self.enable_pred:
                if self.enable_pred._num_state(self.step_num) == EnablePred.State.WARMUP:
                    self._run_action(EnablePred.Action.START_TRACE)
            self._run_action(EnablePred.Action.STOP_TRACE, keep_profiler=True)

    def next_step(self):
        if self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self.step_num += 1
        if self.enable_pred:
            self._run_actions(self.step_num)

        self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
        self.step_rec_fn.__enter__()

    def export_chrome_trace(self, path: str):
        assert self.profiler
        return self.profiler.export_chrome_trace(path)

    def key_averages(self, group_by_input_shape: bool = False, group_by_stack_n: int = 0):
        assert self.profiler
        return self.profiler.key_averages(group_by_input_shape, group_by_stack_n)

    def _run_actions(self, step_num):
        assert self.enable_pred
        for act in self.enable_pred.actions(self.step_num):
            self._run_action(act)

    def _run_action(self, act, keep_profiler=False):
        if act == EnablePred.Action.START_WARMUP:
            self.profiler = prof.profile(
                use_cuda=(ProfilerActivity.CUDA in self.activities),
                use_cpu=(ProfilerActivity.CPU in self.activities),
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                use_kineto=True,
            )
            self.profiler._prepare_kineto_trace()
        elif act == EnablePred.Action.START_TRACE:
            assert self.profiler is not None
            self.profiler._start_kineto_trace()
        elif act == EnablePred.Action.STOP_TRACE:
            assert self.profiler is not None
            self.profiler.__exit__(None, None, None)
            if self.enable_pred and self.enable_pred.output_fn:
                self.enable_pred.output_fn(self.profiler)
            if not keep_profiler:
                self.profiler = None
