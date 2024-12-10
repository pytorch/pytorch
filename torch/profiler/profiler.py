# mypy: allow-untyped-defs
import gzip
import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from typing_extensions import Self
from warnings import warn

import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
    _add_execution_trace_observer,
    _disable_execution_trace_observer,
    _enable_execution_trace_observer,
    _ExperimentalConfig,
    _remove_execution_trace_observer,
)
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline


__all__ = [
    "supported_activities",
    "ProfilerAction",
    "schedule",
    "tensorboard_trace_handler",
    "profile",
    "ExecutionTraceObserver",
]
PROFILER_STEP_NAME = "ProfilerStep"


class _NumpyEncoder(json.JSONEncoder):
    """
    Json encoder for numpy types (np.int, np.float, np.array etc.)
    Returns default encoder if numpy is not available
    """

    def default(self, obj):
        """Encode NumPy types to JSON"""
        try:
            import numpy as np
        except ImportError:
            return json.JSONEncoder.default(self, obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


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


class _ITraceObserver(ABC):
    """Abstract interface for a Trace observer.
    This satisfies 3 methods: start, stop and cleanup"""

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass


class _KinetoProfile:
    """Low-level profiler wrap the autograd profile

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation (see ``export_memory_timeline``
            for more details).
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPS of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles


    .. note::
        This API is experimental and subject to change in the future.

        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.
    """

    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: Optional[_ExperimentalConfig] = None,
        execution_trace_observer: Optional[_ITraceObserver] = None,
        acc_events: bool = False,
        custom_trace_id_callback: Optional[Callable[[], str]] = None,
    ):
        self.activities = set(activities) if activities else supported_activities()
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.experimental_config = experimental_config
        self.execution_trace_observer = execution_trace_observer
        self.acc_events = acc_events
        self.custom_trace_id_callback = custom_trace_id_callback
        self.profiler: Optional[prof.profile] = None
        self.mem_tl: Optional[MemoryProfileTimeline] = None
        self.use_device = None
        if ProfilerActivity.CUDA in self.activities:
            self.use_device = "cuda"
        elif ProfilerActivity.XPU in self.activities:
            self.use_device = "xpu"
        elif ProfilerActivity.MTIA in self.activities:
            self.use_device = "mtia"
        elif ProfilerActivity.PrivateUse1 in self.activities:
            self.use_device = _get_privateuse1_backend_name()

        # user-defined metadata to be amended to the trace
        self.preset_metadata: Dict[str, str] = {}

    def start(self):
        self.prepare_trace()
        self.start_trace()

    def stop(self):
        self.stop_trace()

    def prepare_trace(self):
        if (self.profiler is None) or (not self.acc_events):
            self.profiler = prof.profile(
                use_cpu=(ProfilerActivity.CPU in self.activities),
                use_device=self.use_device,
                record_shapes=self.record_shapes,
                with_flops=self.with_flops,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_modules=self.with_modules,
                use_kineto=True,
                experimental_config=self.experimental_config,
                acc_events=self.acc_events,
                custom_trace_id_callback=self.custom_trace_id_callback,
            )
        self.profiler._prepare_trace()

    def start_trace(self):
        if self.execution_trace_observer:
            self.execution_trace_observer.start()
        assert self.profiler is not None
        self.profiler._start_trace()

        if self.profile_memory:
            self.add_metadata_json("profile_memory", "1")
        if self.with_stack:
            self.add_metadata_json("with_stack", "1")
        if self.record_shapes:
            self.add_metadata_json("record_shapes", "1")
        if self.with_modules:
            self.add_metadata_json("with_modules", "1")
        if self.with_flops:
            self.add_metadata_json("with_flops", "1")

        if kineto_available():
            dist_info = self._get_distributed_info()
            if dist_info:
                self.add_metadata_json(
                    "distributedInfo", json.dumps(dist_info, cls=_NumpyEncoder)
                )

            if hasattr(torch, "_inductor"):
                import torch._inductor.config as inductor_config

                if inductor_config.triton.cudagraphs:
                    os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"
                    self.add_metadata_json("DISABLE_CUPTI_LAZY_REINIT", "1")
                    # FIXME: CUDA Graph does not work well with CUPTI teardown.
                    #   1) crashes on 1st lazy CUPTI re-init after teardown (CUDA 11)
                    #   2) crashes on 2nd non-lazy CUPTI re-init after teardown (CUDA 12)
                    # Workaround: turn off CUPTI teardown when using CUDA Graphs.
                    os.environ["TEARDOWN_CUPTI"] = "0"

            # Insert the preset user metadata to the trace
            for k, v in self.preset_metadata.items():
                self.add_metadata_json(k, v)

    def stop_trace(self):
        if self.execution_trace_observer:
            self.execution_trace_observer.stop()
        assert self.profiler is not None
        self.profiler.__exit__(None, None, None)

    def export_chrome_trace(self, path: str):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        assert self.profiler
        if path.endswith(".gz"):
            fp = tempfile.NamedTemporaryFile("w+b", suffix=".json", delete=False)
            fp.close()
            retvalue = self.profiler.export_chrome_trace(fp.name)
            with open(fp.name, "rb") as fin:
                with gzip.open(path, "wb") as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
            return retvalue
        else:
            return self.profiler.export_chrome_trace(path)

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        """Save stack traces to a file

        Args:
            path (str): save stacks file to this location;
            metric (str): metric to use: "self_cpu_time_total" or "self_cuda_time_total"
        """
        assert self.profiler
        return self.profiler.export_stacks(path, metric)

    def toggle_collection_dynamic(
        self, enable: bool, activities: Iterable[ProfilerActivity]
    ):
        """Toggle collection of activities on/off at any point of collection. Currently supports toggling Torch Ops
        (CPU) and CUDA activity supported in Kineto

        Args:
            activities (iterable): list of activity groups to use in profiling, supported values:
                ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``
        Examples:

        .. code-block:: python

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                code_to_profile_0()
                // turn off collection of all CUDA activity
                p.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                code_to_profile_1()
                // turn on collection of all CUDA activity
                p.toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.CUDA])
                code_to_profile_2()
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
        """
        if not self.profiler:
            return
        self.profiler.toggle_collection_dynamic(enable, activities)

    def key_averages(
        self, group_by_input_shape: bool = False, group_by_stack_n: int = 0
    ):
        """Averages events, grouping them by operator name and (optionally) input shapes and
        stack.

        .. note::
            To use shape/stack functionality make sure to set record_shapes/with_stack
            when creating profiler context manager.
        """
        assert self.profiler
        return self.profiler.key_averages(group_by_input_shape, group_by_stack_n)

    def events(self):
        """
        Returns the list of unaggregated profiler events,
        to be used in the trace callback or after the profiling is finished
        """
        assert self.profiler
        return self.profiler.function_events

    def add_metadata(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a string value
        into the trace file
        """
        wrapped_value = '"' + value.replace('"', '\\"') + '"'
        torch.autograd._add_metadata_json(key, wrapped_value)

    def add_metadata_json(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
        torch.autograd._add_metadata_json(key, value)

    def preset_metadata_json(self, key: str, value: str):
        """
        Preset a user defined metadata when the profiler is not started
        and added into the trace file later.
        Metadata is in the format of a string key and a valid json value
        """
        self.preset_metadata[key] = value

    def _get_distributed_info(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None

        backend = dist.get_backend()
        dist_info = {
            "backend": backend,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "pg_count": dist.get_pg_count(),
            "pg_config": dist.distributed_c10d._get_all_pg_configs(),
        }
        if backend == "nccl":
            nccl_version = torch.cuda.nccl.version()
            dist_info["nccl_version"] = ".".join(str(v) for v in nccl_version)
        return dist_info

    def _memory_profile(self) -> MemoryProfile:
        required = ("record_shapes", "profile_memory", "with_stack")
        missing = [f"{i}=True" for i in required if not getattr(self, i)]
        if missing:
            raise ValueError(f"{', '.join(missing)} required for memory profiling.")

        assert self.profiler is not None and self.profiler.kineto_results is not None
        return MemoryProfile(self.profiler.kineto_results)

    def export_memory_timeline(self, path: str, device: Optional[str] = None) -> None:
        """Export memory event information from the profiler collected
        tree for a given device, and export a timeline plot. There are 3
        exportable files using ``export_memory_timeline``, each controlled by the
        ``path``'s suffix.

        - For an HTML compatible plot, use the suffix ``.html``, and a memory timeline
          plot will be embedded as a PNG file in the HTML file.

        - For plot points consisting of ``[times, [sizes by category]]``, where
          ``times`` are timestamps and ``sizes`` are memory usage for each category.
          The memory timeline plot will be saved a JSON (``.json``) or gzipped JSON
          (``.json.gz``) depending on the suffix.

        - For raw memory points, use the suffix ``.raw.json.gz``. Each raw memory
          event will consist of ``(timestamp, action, numbytes, category)``, where
          ``action`` is one of ``[PREEXISTING, CREATE, INCREMENT_VERSION, DESTROY]``,
          and ``category`` is one of the enums from
          ``torch.profiler._memory_profiler.Category``.

        Output: Memory timeline written as gzipped JSON, JSON, or HTML.
        """
        # Default to device 0, if unset. Fallback on cpu.
        if device is None and self.use_device and self.use_device != "cuda":
            device = self.use_device + ":0"

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Construct the memory timeline plot data
        self.mem_tl = MemoryProfileTimeline(self._memory_profile())

        # Depending on the file suffix, save the data as json.gz or json.
        # For html, we can embed the image into an HTML file.
        if path.endswith(".html"):
            self.mem_tl.export_memory_timeline_html(path, device)
        elif path.endswith(".gz"):
            fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
            fp.close()
            if path.endswith("raw.json.gz"):
                self.mem_tl.export_memory_timeline_raw(fp.name, device)
            else:
                self.mem_tl.export_memory_timeline(fp.name, device)
            with open(fp.name) as fin:
                with gzip.open(path, "wt") as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
        elif path.endswith(".json"):
            self.mem_tl.export_memory_timeline_json(path, device)
        else:
            self.mem_tl.export_memory_timeline(path, device)


class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals
    """

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


def schedule(
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int = 0,
    skip_first: int = 0,
    skip_first_wait: int = 0,
) -> Callable:
    """
    Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip
    the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`` steps,
    then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
    The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.

    The ``skip_first_wait`` parameter controls whether the first ``wait`` stage should be skipped.
    This can be useful if a user wants to wait longer than ``skip_first`` between cycles, but not
    for the first profile. For example, if ``skip_first`` is 10 and ``wait`` is 20, the first cycle will
    wait 10 + 20 = 30 steps before warmup if ``skip_first_wait`` is zero, but will wait only 10
    steps if ``skip_first_wait`` is non-zero. All subsequent cycles will then wait 20 steps between the
    last active and warmup.
    """

    def schedule_fn(step: int) -> ProfilerAction:
        assert step >= 0
        if step < skip_first:
            return ProfilerAction.NONE
        else:
            step -= skip_first
        # If wait >> skip_first and we want to grab profiling early, shift left by wait if skip_first_wait is True
        if skip_first_wait != 0:
            step += wait
        num_steps = wait + warmup + active
        if repeat > 0 and step / num_steps >= repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps
        if mod_step < wait:
            return ProfilerAction.NONE
        elif mod_step < wait + warmup:
            return ProfilerAction.WARMUP
        else:
            return (
                ProfilerAction.RECORD
                if mod_step < num_steps - 1
                else ProfilerAction.RECORD_AND_SAVE
            )

    assert (
        wait >= 0 and warmup >= 0 and active > 0 and repeat >= 0 and skip_first >= 0
    ), "Invalid profiler schedule arguments"
    if warmup == 0:
        warn("Profiler won't be using warmup, this can skew profiler results")
    return schedule_fn


def _default_schedule_fn(_: int) -> ProfilerAction:
    """
    Default profiler behavior - immediately starts recording the events,
    keeps doing it on every profiler step.
    """
    return ProfilerAction.RECORD


def tensorboard_trace_handler(
    dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False
):
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
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        prof.export_chrome_trace(os.path.join(dir_name, file_name))

    return handler_fn


class profile(_KinetoProfile):
    """Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.
        schedule (Callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (Callable): callable that is called at each step when ``schedule``
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
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used for Kineto library features. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler. See the examples section below for a code sample.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles
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

            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=1),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
            ) as p:
                for iter in range(N):
                    code_iteration_to_profile(iter)
                    # send a signal to the profiler that the next iteration has started
                    p.step()

    The following sample shows how to setup up an Execution Trace Observer (`execution_trace_observer`)

    .. code-block:: python

        with torch.profiler.profile(
            ...
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback("./execution_trace.json")
            ),
        ) as p:
            for iter in range(N):
                code_iteration_to_profile(iter)
                p.step()

    You can also refer to test_execution_trace_with_kineto() in tests/profiler/test_profiler.py.
    Note: One can also pass any object satisfying the _ITraceObserver interface.
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
        experimental_config: Optional[_ExperimentalConfig] = None,
        execution_trace_observer: Optional[_ITraceObserver] = None,
        acc_events: bool = False,
        # deprecated:
        use_cuda: Optional[bool] = None,
        custom_trace_id_callback: Optional[Callable[[], str]] = None,
    ):
        activities_set = set(activities) if activities else supported_activities()
        if use_cuda is not None:
            warn(
                "`use_cuda` is deprecated, use `activities` argument instead",
                FutureWarning,
                stacklevel=2,
            )
            if use_cuda:
                activities_set.add(ProfilerActivity.CUDA)
            elif ProfilerActivity.CUDA in activities_set:
                activities_set.remove(ProfilerActivity.CUDA)
        assert len(activities_set) > 0, "No valid profiler activities found"

        super().__init__(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
            execution_trace_observer=execution_trace_observer,
            acc_events=acc_events,
            custom_trace_id_callback=custom_trace_id_callback,
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

        self.action_map: Dict[
            Tuple[ProfilerAction, Optional[ProfilerAction]], List[Any]
        ] = {
            # key is (prev_action, current_action), value is action list corresponding to the state pair.
            (ProfilerAction.NONE, ProfilerAction.NONE): [],
            (ProfilerAction.NONE, ProfilerAction.WARMUP): [self.prepare_trace],
            (ProfilerAction.NONE, ProfilerAction.RECORD): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.NONE, ProfilerAction.RECORD_AND_SAVE): [
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: WARMUP followed by NONE"),
                self.start_trace,
                self.stop_trace,
            ],
            (ProfilerAction.WARMUP, ProfilerAction.WARMUP): [],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD): [self.start_trace],
            (ProfilerAction.WARMUP, ProfilerAction.RECORD_AND_SAVE): [self.start_trace],
            (ProfilerAction.RECORD, ProfilerAction.NONE): [
                partial(warn, "Incorrect schedule: RECORD followed by NONE"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.WARMUP): [
                partial(warn, "Incorrect schedule: RECORD followed by WARMUP"),
                self.stop_trace,
            ],
            (ProfilerAction.RECORD, ProfilerAction.RECORD): [],
            (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE): [],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.NONE): [
                self.stop_trace,
                self._trace_ready,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.WARMUP): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            (ProfilerAction.RECORD_AND_SAVE, ProfilerAction.RECORD_AND_SAVE): [
                self.stop_trace,
                self._trace_ready,
                self.prepare_trace,
                self.start_trace,
            ],
            # used for exit action
            (ProfilerAction.WARMUP, None): [self.start_trace, self.stop_trace],
            (ProfilerAction.RECORD, None): [self.stop_trace, self._trace_ready],
            (ProfilerAction.RECORD_AND_SAVE, None): [
                self.stop_trace,
                self._trace_ready,
            ],
        }
        # Start tracking increments to profiler step, this will be used
        # by Kineto
        prof.KinetoStepTracker.init_step_count(PROFILER_STEP_NAME)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        prof.KinetoStepTracker.erase_step_count(PROFILER_STEP_NAME)
        if self.execution_trace_observer:
            self.execution_trace_observer.cleanup()

    def start(self):
        self._transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def stop(self):
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self._transit_action(self.current_action, None)

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
        prof.KinetoStepTracker.increment_step(PROFILER_STEP_NAME)

        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def set_custom_trace_id_callback(self, callback):
        """
        Sets a callback to be called when a new trace ID is generated.
        """
        self.custom_trace_id_callback = callback

    def get_trace_id(self):
        """
        Returns the current trace ID.
        """
        if self.profiler is None:
            return None
        return self.profiler.trace_id

    def _trace_ready(self):
        if self.on_trace_ready:
            self.on_trace_ready(self)

    def _transit_action(self, prev_action, current_action):
        action_list = self.action_map.get((prev_action, current_action))
        if action_list:
            for action in action_list:
                action()

    def _stats(self) -> Optional[prof._ProfilerStats]:
        if self.profiler is None:
            return None
        return self.profiler._stats


class ExecutionTraceObserver(_ITraceObserver):
    """Execution Trace Observer

    Each process can have a single ExecutionTraceObserver instance. The observer
    can be added to record function callbacks via calling register_callback()
    explicitly. Without calling unregister_callback(), repeated calls to
    register_callback() will not add additional observers to record function
    callbacks. Once an ExecutionTraceObserver is created, the start() and stop()
    methods control when the event data is recorded.

    Deleting or calling unregister_callback() will remove the observer from the
    record function callbacks, finalize the output file, and will stop
    incurring any overheads.
    """

    def __init__(self) -> None:
        """
        Initializes the default states.
        """
        self._registered = False
        self._execution_trace_running = False

    def __del__(self):
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    def register_callback(self, output_file_path: str) -> Self:
        """
        Adds ET observer to record function callbacks. The data will be
        written to output_file_path.
        """
        if not self._registered:
            self._output_file_path = output_file_path
            self._registered = _add_execution_trace_observer(output_file_path)
        return self

    def unregister_callback(self):
        """
        Removes ET observer from record function callbacks.
        """

        def _save_triton_kernels():
            # Save the kernel paths for the generated kernels
            from torch._inductor.codecache import PyCodeCache as PyCodeCache

            kernel_files = [
                v.__file__
                for v in PyCodeCache.modules
                if getattr(v, "__file__", None) is not None
            ]
            work_dir, file_name = os.path.split(self._output_file_path)
            resource_dir = os.path.join(
                work_dir, os.path.splitext(file_name)[0] + "_resources"
            )
            if not os.path.exists(resource_dir):
                os.mkdir(resource_dir)

            for kernel_file in kernel_files:
                if kernel_file is None:
                    continue
                name = os.path.basename(kernel_file)
                dst = os.path.join(resource_dir, name)
                shutil.copyfile(kernel_file, dst)

        if self._registered:
            self.stop()
            try:
                _save_triton_kernels()
            except Exception as e:
                warn(f"Execution trace failed to save kernels: {e}")
            _remove_execution_trace_observer()
            self._registered = False

    @property
    def is_registered(self):
        """
        Returns True if the execution trace observer is registered, otherwise False.
        """
        return self._registered

    def is_running(self):
        """
        Returns True if the observer is running, otherwise False.
        """
        return self._execution_trace_running

    def start(self):
        """
        Starts to capture.
        """
        if self._registered and not self._execution_trace_running:
            _enable_execution_trace_observer()
            self._execution_trace_running = True
            self._record_pg_config()

    def stop(self):
        """
        Stops to capture.
        """
        if self._execution_trace_running:
            _disable_execution_trace_observer()
            self._execution_trace_running = False

    def cleanup(self):
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
        self.unregister_callback()

    def get_output_file_path(self) -> str:
        """
        Returns the output file name.
        """
        if self.is_registered:
            return self._output_file_path
        else:
            raise RuntimeError(
                "A callback to the ET profiler needs to be registered "
                "first before getting the output file path"
            )

    def _record_pg_config(self) -> None:
        # Records the PG config info to the trace as node:
        #  ## process_group:init ##
        if (
            self.is_registered
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            pg_config_info = torch.distributed.distributed_c10d._world.pg_config_info
            torch.autograd._record_function_with_args_enter(
                "## process_group:init ##",
                json.dumps(pg_config_info, cls=_NumpyEncoder),
            )
