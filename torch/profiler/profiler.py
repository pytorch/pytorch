import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn

import torch
import torch.autograd.profiler as prof
from torch.autograd.profiler import profile as kineto_profile
from torch.autograd import kineto_available, ProfilerActivity


class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals
    """
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


def schedule(*, wait: int, warmup: int, active: int, repeat: int = 0, skip_first: int = 0) -> Callable:
    """
    Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip
    the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`` steps,
    then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
    The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.
    """
    def schedule_fn(step: int) -> ProfilerAction:
        assert step >= 0
        if step < skip_first:
            return ProfilerAction.NONE
        else:
            step -= skip_first
        num_steps = wait + warmup + active
        if repeat > 0 and step / num_steps >= repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps
        if mod_step < wait:
            return ProfilerAction.NONE
        elif mod_step < wait + warmup:
            return ProfilerAction.WARMUP
        else:
            return ProfilerAction.RECORD if mod_step < num_steps - 1 \
                else ProfilerAction.RECORD_AND_SAVE
    assert wait >= 0 and warmup >= 0 and active > 0 and \
           repeat >= 0 and skip_first >= 0, "Invalid profiler schedule arguments"
    if warmup == 0:
        warn("Profiler won't be using warmup, this can skew profiler results")
    return schedule_fn


def _default_schedule_fn(_: int) -> ProfilerAction:
    """
    Default profiler behavior - immediately starts recording the events,
    keeps doing it on every profiler step.
    """
    return ProfilerAction.RECORD

def tensorboard_trace_handler(dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)
        if not worker_name:
            worker_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))
        file_name = "{}.{}.pt.trace.json".format(worker_name, int(time.time() * 1000))
        if use_gzip:
            file_name = file_name + '.gz'
        prof.export_chrome_trace(os.path.join(dir_name, file_name))
    return handler_fn

def supported_activities():
    """
    Returns a set of supported profiler tracing activities.

    Note: profiler uses CUPTI library to trace on-device CUDA kernels.
    In case when CUDA is enabled but CUPTI is not available, passing
    ``ProfilerActivity.CUDA`` to profiler results in using the legacy CUDA
    profiling code (same as in the legacy ``torch.autograd.profiler``).
    This, in turn, results in including CUDA time in the profiler table output,
    but not in the JSON trace.
    """
    return torch.autograd._supported_activities()


class profile(kineto_profile):
    """Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA.
        schedule (callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (callable): callable that is called at each step when ``schedule``
            returns ``ProfilerAction.RECORD_AND_SAVE`` during the profiling.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation.
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.
        use_cuda (bool):
            .. deprecated:: 1.8.1
                use ``activities`` instead.

    .. note::
        Use :func:`~torch.profiler.schedule` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.

    .. note::
        Use :func:`~torch.profiler.tensorboard_trace_handler` to generate result files for TensorBoard:

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

        After profiling, result files can be found in the specified directory. Use the command:

        ``tensorboard --logdir dir_name``

        to see the results in TensorBoard.
        For more information, see
        `PyTorch Profiler TensorBoard Plugin <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.

    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    Using the profiler's ``schedule``, ``on_trace_ready`` and ``step`` functions:

    .. code-block:: python

        # Non-default profiler schedule allows user to turn profiler on and off
        # on different iterations of the training loop;
        # trace_handler is called every time a new trace becomes available
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # In this example with wait=1, warmup=1, active=2,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
            ) as p:
                for iter in range(N):
                    code_iteration_to_profile(iter)
                    # send a signal to the profiler that the next iteration has started
                    p.step()
    """
    def __init__(
            self,
            *,
            activities: Optional[Iterable[ProfilerActivity]] = None,
            schedule: Optional[Callable[[int], ProfilerAction]] = None,
            on_trace_ready: Optional[Callable[..., Any]] = None,
            record_shapes: bool = False,
            profile_memory: bool = False,
            with_stack: bool = False,
            with_flops: bool = False,
            with_modules: bool = False,
            # deprecated:
            use_cuda: Optional[bool] = None):

        if activities:
            activities_set = set(activities)
        else:
            activities_set = supported_activities()

        if use_cuda is not None:
            warn("use_cuda is deprecated, use activities argument instead")
            if use_cuda:
                activities_set.add(ProfilerActivity.CUDA)
            elif ProfilerActivity.CUDA in activities_set:
                activities_set.remove(ProfilerActivity.CUDA)

        assert len(activities_set) > 0, "No valid profiler activities found"

        super().__init__(
            use_cuda=(ProfilerActivity.CUDA in activities_set),
            use_cpu=(ProfilerActivity.CPU in activities_set),
            record_shapes=record_shapes,
            with_flops=with_flops,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_modules=with_modules,
            use_kineto=True
        )

        if schedule:
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            self.schedule = _default_schedule_fn
            self.record_steps = False
        self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.current_action = self.schedule(self.step_num)
        self.step_rec_fn: Optional[prof.record_function] = None

        self.action_map: Dict[Tuple[ProfilerAction, Optional[ProfilerAction]], List[Any]] = {
            # key is (prev_action, current_action), value is action list corresponding to the state pair.
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self._prepare_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [self._prepare_trace, self._start_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [self._prepare_trace, self._start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: WARMUP followed by NONE"),
                self._start_trace,
                super().stop],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self._start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self._start_trace],
            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: RECORD followed by NONE"),
                super().stop],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [
                partial(warn, "Incorrect schedule: RECORD followed by WARMUP"),
                super().stop],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [super().stop, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [super().stop, self._trace_ready, self._prepare_trace],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [
                super().stop,
                self._trace_ready,
                self._prepare_trace,
                self._start_trace],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [
                super().stop,
                self._trace_ready,
                self._prepare_trace,
                self._start_trace],
            # used for exit action
            (ProfilerAction.WARMUP, None): [self._start_trace, super().stop],
            (ProfilerAction.RECORD, None): [super().stop, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, None): [super().stop, self._trace_ready]
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self._transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()

    def stop(self):
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self._transit_action(self.current_action, None)

    def _start_trace(self):
        super()._start_trace()

        if kineto_available():
            dist_info = self._get_distributed_info()
            if dist_info:
                self.add_metadata_json("distributedInfo", json.dumps(dist_info))

    def step(self):
        """
        Signals the profiler that the next profiling step has started.
        """
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)

        self._transit_action(prev_action, self.current_action)

        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()

    def export_chrome_trace(self, path: str):
        """
        Exports the collected trace in Chrome JSON format.
        """
        if path.endswith('.gz'):
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
            fp.close()
            retvalue = super().export_chrome_trace(fp.name)
            with open(fp.name) as fin:
                with gzip.open(path, 'wt') as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
            return retvalue
        else:
            return super().export_chrome_trace(path)

    def events(self):
        """
        Returns the list of unaggregated profiler events,
        to be used in the trace callback or after the profiling is finished
        """
        return self.function_events

    def add_metadata(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a string value
        into the trace file
        """
        wrapped_value = "\"" + value.replace('"', '\\"') + "\""
        torch.autograd._add_metadata_json(key, wrapped_value)

    def add_metadata_json(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
        torch.autograd._add_metadata_json(key, value)

    def _get_distributed_info(self):
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return None

        return {
            "backend": dist.get_backend(),
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size()
        }

    def _trace_ready(self):
        if self.on_trace_ready:
            self.on_trace_ready(self)

    def _transit_action(self, prev_action, current_action):
        action_list = self.action_map.get((prev_action, current_action))
        if action_list:
            for action in action_list:
                action()
