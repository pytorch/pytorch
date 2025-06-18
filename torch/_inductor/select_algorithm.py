# mypy: allow-untyped-defs
import dataclasses
import functools
import inspect
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Any, Callable, Optional, TYPE_CHECKING, Union
from typing_extensions import Self

import sympy

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters, dynamo_timed, preserve_rng_state
from torch._inductor.utils import clear_on_fresh_cache
from torch.utils._filelock import FileLock
from torch.utils._ordered_set import OrderedSet

from ..utils._sympy.functions import CeilDiv
from . import config, ir
from .autotune_process import TritonGPUBenchmarkRequest
from .codecache import code_hash, PersistentCache
from .codegen.subgraph import SubgraphChoiceCaller
from .codegen.triton_templates.caller import TritonTemplateCaller
from .exc import CUDACompileError
from .ir import ChoiceCaller, PrimitiveInfoType
from .runtime.benchmarking import benchmarker
from .utils import ceildiv, do_bench_using_profiling, is_gpu, restore_stdout_stderr
from .virtualized import V


log = logging.getLogger(__name__)

# correctness checks struggle with fp16/tf32
VERIFY: dict[str, Any] = {}
PRINT_AUTOTUNE = True
DEBUG = False


if TYPE_CHECKING:
    import concurrent


class KernelNamespace:
    pass


# these objects are imported from the generated wrapper code
extern_kernels = KernelNamespace()


@dataclasses.dataclass
class BenchmarkTensors:
    """Represents a set of inputs and outputs for autotuning with a template"""

    input_tensors: list[torch.Tensor]
    output_tensor: Optional[torch.Tensor]

    def unpack(self):
        return self.input_tensors, self.output_tensor


@dataclasses.dataclass
class AutotuneArgs:
    """During autotuning, we need to pass the same inputs to all choices.
    Note:
        Since we typically have a mix of external choices and triton choices, we create
        two lists of inputs for the same underlying buffers:
        - External inputs (for aten kernels): Include offset for sliced tensors
        - Triton inputs: Use base pointer for sliced tensors, without offset
    """

    triton: BenchmarkTensors
    extern: BenchmarkTensors
    expected: Optional[torch.Tensor] = None

    def get_benchmark_tensors(self, extern=False) -> BenchmarkTensors:
        """Returns the inputs and output tensors for a given choice."""
        bench_tensors = self.extern if extern else self.triton
        return bench_tensors

    @classmethod
    def from_choice_args(
        cls,
        example_inputs: list[torch.Tensor],
        example_inputs_extern: list[torch.Tensor],
        out: torch.Tensor,
        out_extern: torch.Tensor,
        expected: Optional[torch.Tensor] = None,
    ) -> Self:
        """Factory method to create AutotuneInputs from separate inputs/outputs"""
        return cls(
            triton=BenchmarkTensors(example_inputs, out),
            extern=BenchmarkTensors(example_inputs_extern, out_extern),
            expected=expected,
        )

    def verify(self, **kwargs):
        """Verify the correctness of the benchmarking results"""

        torch.testing.assert_close(self.extern.output_tensor, self.expected, **kwargs)


class PartialRender:
    """
    Some parts of a template need to be generated at the end, but
    inserted into the template at the start.  This allows doing a bunch
    of replacements after the initial render.
    """

    def __init__(self, code, replacement_hooks) -> None:
        super().__init__()
        self.code = code
        self.replacement_hooks = replacement_hooks

    def finalize_hook(self, hook_key: str, strict=True) -> None:
        if hook_key not in self.replacement_hooks:
            if strict:
                raise RuntimeError(
                    f"{hook_key} not registered in self.replacement_hooks"
                )
            else:
                return
        assert self.replacement_hooks[hook_key] is not None, (
            "hook_key can only be called once"
        )
        self.code = self.code.replace(hook_key, self.replacement_hooks[hook_key]())
        self.replacement_hooks[hook_key] = None

    def finalize_all(self) -> str:
        for key, fn in self.replacement_hooks.items():
            self.code = self.code.replace(key, fn())
        return self.code


@functools.cache
def _jinja2_env():
    try:
        import jinja2

        return jinja2.Environment(
            undefined=jinja2.StrictUndefined,
        )
    except ImportError:
        return None


class ExternKernelChoice:
    def __init__(
        self,
        kernel,
        cpp_kernel=None,
        *,
        name=None,
        has_out_variant=True,
        op_overload=None,
        use_fallback_kernel=False,
        kernel_creator=None,
    ) -> None:
        super().__init__()
        name = name or kernel.__name__
        assert callable(kernel)
        assert not hasattr(extern_kernels, name), f"duplicate extern kernel: {name}"
        self.name = name
        self.cpp_kernel_name = cpp_kernel
        self.has_out_variant = has_out_variant
        setattr(extern_kernels, name, kernel)
        self.op_overload = op_overload
        self.use_fallback_kernel = use_fallback_kernel
        self.kernel_creator = kernel_creator

    def to_callable(self):
        return getattr(extern_kernels, self.name)

    def call_name(self):
        return f"extern_kernels.{self.name}"

    @functools.cache  # noqa: B019
    def hash_key(self):
        fn = self.to_callable()
        parts = [
            self.name,
            getattr(fn, "__name__", ""),
            getattr(fn, "__module__", ""),
        ]
        try:
            parts.append(inspect.getsource(fn))
        except Exception:
            pass
        return code_hash("-".join(parts))

    def bind(
        self,
        input_nodes,
        layout,
        ordered_kwargs_for_cpp_kernel=(),
        **kwargs,
    ):
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        return ExternKernelCaller(
            self, input_nodes, layout, kwargs, has_out_variant=self.has_out_variant
        )


class ExternKernelCaller(ChoiceCaller):
    def __init__(
        self,
        choice: ExternKernelChoice,
        input_nodes,
        layout,
        kwargs=None,
        *,
        has_out_variant=True,
    ) -> None:
        super().__init__(choice.name, input_nodes, layout, description="")
        self.choice = choice
        self.kwargs = kwargs or {}
        self.has_out_variant = has_out_variant

    def __str__(self) -> str:
        return f"ExternKernelCaller({self.choice.call_name()})"

    def benchmark(self, *args, out):
        if out.numel() == 0:
            # no need to run the kerrnel of do benchmarking
            return 0.0
        if self.has_out_variant:
            return super().benchmark(*args, out=out)
        else:
            algo = self.to_callable()
            out_new = algo(*args)
            torch._C._dynamo.guards.assert_size_stride(
                out_new, tuple(out.size()), tuple(out.stride())
            )
            out.copy_(out_new)  # for correctness checking
            if config.profile_bandwidth_with_do_bench_using_profiling:
                return do_bench_using_profiling(lambda: algo(*args))
            return benchmarker.benchmark(algo, args, {})

    def to_callable(self):
        fn = self.choice.to_callable()
        if self.kwargs:
            return functools.partial(fn, **self.kwargs)
        return fn

    def hash_key(self):
        return "-".join(
            [
                self.choice.name,
                *[
                    f"{kwarg}={repr(self.kwargs[kwarg])}"
                    for kwarg in sorted(self.kwargs.keys())
                ],
                self.choice.hash_key(),
            ]
        )

    def output_node(self):
        if self.choice.use_fallback_kernel:
            assert self.choice.op_overload is not None, (
                "Please provide an op_overload to use ir.FallbackKernel"
            )
            inner = ir.FallbackKernel.create(
                self.choice.op_overload, *self.input_nodes, **self.kwargs
            )
        elif self.choice.kernel_creator is not None:
            inner = self.choice.kernel_creator(*self.input_nodes, **self.kwargs)
        else:
            cls = ir.ExternKernelOut if self.has_out_variant else ir.ExternKernelAlloc
            inner = cls(
                layout=self.layout,
                inputs=self.input_nodes,
                python_kernel_name=self.choice.call_name(),
                cpp_kernel_name=self.choice.cpp_kernel_name,
                ordered_kwargs_for_cpp_kernel=self.choice.ordered_kwargs_for_cpp_kernel,
                op_overload=self.choice.op_overload,
                kwargs=self.kwargs,
            )

        return ir.TensorBox.create(inner)

    def info_dict(self) -> dict[str, Union[PrimitiveInfoType, list[PrimitiveInfoType]]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
        return {
            "backend": "extern",
            "kernel_call_name": self.choice.call_name(),
        }

    def autoheuristic_id(self):
        return f"extern_{self.choice.name}"


@functools.cache
def get_mm_log_filename() -> Optional[str]:
    mm_file_name = os.environ.get("TORCHINDUCTOR_MM_LOGGING_FILE", None)
    if not mm_file_name:
        return None

    if "json" not in mm_file_name:
        mm_file_name = f"{mm_file_name}.json"

    return mm_file_name


def append_to_log(filename, data):
    lock_file = filename.replace(".json", ".lock")
    lock = FileLock(lock_file)
    with lock:
        try:
            with open(filename) as f:
                log_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []

        log_data.append(data)

        with open(filename, "w") as f:
            json.dump(log_data, f, indent=4)


class DataProcessorChoiceCallerWrapper:
    def __init__(self, wrapped, preprocessor, postprocessor) -> None:
        self._wrapped = wrapped
        if preprocessor is not None:
            self._preprocessor = preprocessor
        else:
            self._preprocessor = lambda x, y: (x, y)
        if postprocessor is not None:
            self._postprocessor = postprocessor
        else:
            self._postprocessor = lambda x: x

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def benchmark(self, *args, out) -> float:
        new_args, new_out = self._preprocessor(args, out)
        result = self._wrapped.benchmark(*new_args, out=new_out)
        new_out = self._postprocessor(new_out)
        if out is not new_out:
            out.copy_(new_out)
        return result

    def output_node(self) -> ir.TensorBox:
        result = self._wrapped.output_node()
        return self._postprocessor(result)

    def __repr__(self) -> str:
        return f"DataProcessorChoiceCallerWrapper({self._wrapped})"


class DataProcessorTemplateWrapper:
    """
    A wrapper class for a kernel template.

    This class together with `DataProcessorChoiceCallerWrapper` provides a convenient way to
    preprocess and postprocess data before and after using the wrapped template. A typical
    usage is to reorder or filter the input nodes in order to match the expected input of other
    kernel choices like a ATen kernel. A more complicated usage is to prepack the weights.
    See the example from :mod:`cpp_gemm_template` for more details.
    """

    def __init__(
        self,
        wrapped_template_cls,
        preprocessor,
        postprocessor,
        **kwargs,
    ) -> None:
        if preprocessor is not None:
            self._preprocessor = preprocessor
        else:
            self._preprocessor = lambda x, y: (x, y)
        if postprocessor is not None:
            self._postprocessor = postprocessor
        else:
            self._postprocessor = lambda x: x
        assert "input_nodes" in kwargs
        assert "layout" in kwargs
        kwargs["input_nodes"], kwargs["layout"] = preprocessor(
            kwargs["input_nodes"], kwargs["layout"]
        )
        self._wrapped = wrapped_template_cls(**kwargs)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def maybe_append_choice(self, choices, **kwargs):
        return type(self._wrapped).maybe_append_choice(self, choices, **kwargs)

    def generate(self, **kwargs):
        choice_caller = self._wrapped.generate(**kwargs)
        return DataProcessorChoiceCallerWrapper(
            choice_caller, self._preprocessor, self._postprocessor
        )

    def __repr__(self) -> str:
        return f"DataProcessorTemplateWrapper({self._wrapped})"


class ErrorFromChoice(RuntimeError):
    def __init__(self, msg, choice: ChoiceCaller, inputs_str) -> None:
        msg += f"\nFrom choice {choice}\n{inputs_str}"
        super().__init__(msg)
        self.choice = choice


class NoValidChoicesError(RuntimeError):
    pass


@functools.cache
def get_num_workers() -> int:
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])

    cpu_count = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    assert cpu_count
    return cpu_count


def create_inputs_key(input_nodes) -> str:
    return repr([AlgorithmSelectorCache.key_of(x) for x in input_nodes])


def create_precompile_key(
    name: str, inputs_key: str, choices: list[ChoiceCaller]
) -> str:
    return ":".join(
        [
            name,
            inputs_key,
            torch.get_float32_matmul_precision(),
        ]
        + [choice.kernel_hash_key() for choice in choices]
    )


# Args to FeedbackFunctions
# timings: mapping from choices to the benchmark time
# name: name of the op
# input_nodes: list of input ir.py Nodes
# choices: list of choices
# profiled time: Callable that returns a dict mapping from choices to the profiled time
FeedbackFunction = Callable[
    [
        dict[ChoiceCaller, float],
        str,
        list[Any],
        list[ChoiceCaller],
        Callable[[], dict[ChoiceCaller, float]],
    ],
    None,
]


class AlgorithmSelectorCache(PersistentCache):
    """
    A persistent cache for algorithm selection results used in autotuning of GEMMs
    and convolutions.

    This classes includes precompilation and benchmarking of the kernels.

    The cache is keyed by input characteristics (sizes, strides, dtypes, etc.) but
    doesn't depend on the output layout.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # the autotuning will get occur in the scheduler, so there is
        # no guarantee that the first lowering for a given key will also be the
        # first to benchmark it. share a single precompilation function for all lowerings
        # of a particular key
        self.precompile_cache: dict[str, Callable[[], None]] = {}
        # list of callbacks that are called after benchmarking
        self.feedback_saver_fns: list[FeedbackFunction] = []
        # cache for prescreening results to ensure deterministic candidate selection
        self.prescreening_cache: dict[str, OrderedSet[str]] = {}

        clear_on_fresh_cache(self)

    def cache_clear(self) -> None:
        self.precompile_cache.clear()
        self.prescreening_cache.clear()

    def __call__(
        self,
        name,
        choices: list[ChoiceCaller],
        input_nodes,
        layout,
        # optional dict mapping arg indices to the functions
        # generating a torch.Tensor for that input from the
        # corresponding ir.Buffer. if passed for a given
        # arg, the function will be called instead of
        # generating a random torch.Tensor for benchmarking.
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]] = None,
        precompilation_timeout_seconds: int = 60 * 60,
        return_multi_template=False,
    ):
        from .codegen.cuda.cuda_kernel import CUDATemplateCaller

        # Templates selected with input_gen_fns require specific input data to avoid IMA
        # Passing custom input gen fns to benchmark_fusion NYI, so skip deferred template selection
        # TODO(jgong5): support multi-template on CPU
        if input_gen_fns is not None or layout.device.type == "cpu":
            return_multi_template = False

        # TODO - assert that we have not mutating kernels here

        if config.test_configs.autotune_choice_name_regex is not None:
            choices = [
                c
                for c in choices
                if re.search(
                    config.test_configs.autotune_choice_name_regex,
                    c.name,
                )
            ]
        if config.test_configs.autotune_choice_desc_regex is not None:
            choices = [
                c
                for c in choices
                if re.search(
                    config.test_configs.autotune_choice_desc_regex,
                    c.description,
                )
            ]

        if mm_file_name := get_mm_log_filename():
            M, K = input_nodes[-2].get_size()[:2]
            N = input_nodes[-1].get_size()[-1]
            append_to_log(mm_file_name, {"invoke": str((M, K, N))})

        if len(choices) == 0:
            backend_config = (
                "max_autotune_gemm_backends"
                if name != "convolution"
                else "max_autotune_conv_backends"
            )
            raise NoValidChoicesError(
                f"No choices to select, please consider adding ATEN into {backend_config} "
                "config (defined in torch/_inductor/config.py) to allow at least one choice. "
            )
        log.debug("Max autotune selects from %s choices.", str(len(choices)))

        if len(choices) == 1:
            if not isinstance(choices[0], CUDATemplateCaller):
                # CUDATemplateCaller still needs to go through autotuning process to retrieve workspace size.
                return choices[0].output_node()

        @functools.cache
        def make_benchmark_fn():
            return self.make_benchmark_fn(choices, input_nodes, layout, input_gen_fns)

        inputs_key = create_inputs_key(input_nodes)

        def autotune(choices):
            log.debug("Starting autotuning")

            with dynamo_timed(
                f"{name}_template_autotuning",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="compile_time_autotune_time_us",
                metadata={
                    "autotune_strides": ", ".join(
                        [str(n.get_stride()) for n in input_nodes]
                    ),
                    "autotune_dtypes": ", ".join(
                        [str(n.get_dtype()) for n in input_nodes]
                    ),
                    "autotune_shape": ", ".join(
                        ["x".join(map(str, n.get_size())) for n in input_nodes]
                    ),
                    "autotune_offset": ", ".join(
                        [str(n.get_layout().offset) for n in input_nodes]
                    ),
                },
            ):
                return make_benchmark_fn()(choices)

        if config.autotune_in_subproc:
            # Initialize the suprocess pool so it will warmup early.
            torch._inductor.autotune_process.get_tuning_process_pool()

        def do_autotuning(choices, precompile_fn):
            precompile_start_ts = time.time()
            with dynamo_timed(
                f"{name}_template_precompiling",
                log_pt2_compile_event=True,
                dynamo_compile_column_us="compile_time_autotune_time_us",
            ):
                precompile_fn()
            precompile_elapse = time.time() - precompile_start_ts
            log.debug("Precompilation elapsed time: %.02fs", precompile_elapse)

            candidates = self.prescreen_choices(
                choices, name, inputs_key, self.prescreening_cache
            )
            prescreening_elapse: Optional[float] = None
            if candidates:
                prescreening_start_ts = time.time()
                timings = self.lookup(
                    candidates,
                    name,
                    inputs_key,
                    autotune,
                )
                choices = self.prune_choices_postscreen(
                    choices, timings, name, inputs_key, self.prescreening_cache
                )
                prescreening_elapse = time.time() - prescreening_start_ts
                log.debug("Prescreening elapsed time: %.02fs", prescreening_elapse)

            autotune_start_ts = time.time()
            timings = self.lookup(
                choices,
                name,
                inputs_key,
                autotune,
            )

            autotune_elapse = time.time() - autotune_start_ts
            log.debug("Autotuning elapsed time: %.02fs", autotune_elapse)

            if timings and all(
                not math.isfinite(timing) for timing in timings.values()
            ):
                raise NoValidChoicesError

            if make_benchmark_fn.cache_info().currsize:
                counters["inductor"]["select_algorithm_autotune"] += 1

            if (
                make_benchmark_fn.cache_info().currsize
                or log.getEffectiveLevel() == logging.DEBUG
                or config.trace.log_autotuning_results
            ):
                self.log_results(
                    name,
                    input_nodes,
                    timings,
                    autotune_elapse,
                    precompile_elapse,
                    prescreening_elapse,
                )

            def profiler_bench_function():
                # we're not running through the normal caching autotuner method here because we want to avoid returning
                # the cached value.
                # Avoid benchmarking in a separate process because it's not easy to signal to the TuningProcess that we
                # should use the profiler.
                with config.patch(
                    profile_bandwidth_with_do_bench_using_profiling=True,
                    autotune_in_subproc=False,
                ):
                    return self.make_benchmark_fn(
                        choices, input_nodes, layout, input_gen_fns
                    )(choices)

            for feedback_fn in self.feedback_saver_fns:
                # re-benchmarking the same choices with profiler is a bit expensive, so pass it in as a thunk.
                feedback_fn(
                    timings,
                    name,
                    input_nodes,
                    choices,
                    profiler_bench_function,
                )

            return timings

        precompile_fn = self.make_precompile_fn(
            choices,
            name,
            inputs_key,
            precompilation_timeout_seconds=precompilation_timeout_seconds,
        )

        if return_multi_template and (config.max_autotune or config.max_autotune_gemm):

            def get_timings():
                timings = do_autotuning(choices, precompile_fn)
                min_extern_choice = float("inf")
                for choice, timing in timings.items():
                    if isinstance(choice, ExternKernelCaller):
                        min_extern_choice = min(min_extern_choice, timing)

                timings = {
                    choice: time
                    for choice, time in timings.items()
                    if (
                        time <= min_extern_choice
                        or not isinstance(choice, ExternKernelCaller)
                    )
                }

                return timings

            # We take the union of allowed prologue inputs from all choices,
            # and, within benchmark fusion, don't allow prologue fusion for
            # choices which dont support the whole union.
            allowed_prologue_inps: OrderedSet[str] = OrderedSet()
            for c in choices:
                if isinstance(c, TritonTemplateCaller):
                    allowed_prologue_inps |= c.allowed_prologue_inps

            return torch._inductor.ir.TensorBox.create(
                torch._inductor.ir.MultiTemplateBuffer(
                    layout,
                    input_nodes,
                    get_timings,
                    choices,
                    allowed_prologue_inps,
                )
            )

        timings = do_autotuning(choices, precompile_fn)

        # if timings is empty, we really have no choice but to return a semi-random
        # choice. returning the first `ExternKernelCaller` is probably the safest bet
        # in this case, since it will generally be the ATen kernel. if there are no
        # `ExternKernelCaller`s to return, then returning the 0th kernel is our next
        # best option (ideally we'd fail whenever there is no ATen kernel to fallback
        # to, but that's not trivial to figure out)
        if timings == {}:
            for choice in choices:
                if isinstance(choice, ExternKernelCaller):
                    node = choice.output_node()
                    log.debug(
                        "Autotuning returned empty timings, falling back to first `ExternKernelCaller`: %s",
                        node,
                    )
                    return node
            node = choices[0].output_node()
            log.debug(
                "Autotuning returned empty timings, falling back to first choice: %s",
                node,
            )
            return node

        # if we got any timings at all, pick the best of those
        choice = min(timings, key=timings.__getitem__)
        node = choice.output_node()
        log.debug("Autotuning selected choice: %s", node)
        return node

    def make_precompile_fn(
        self,
        choices,
        name: str,
        inputs_key: str,
        precompilation_timeout_seconds: Optional[int] = 60 * 60,
    ) -> Callable[[], None]:
        """
        Returns a function that precompiles the given choices.
        """
        log.debug("Starting precompilation")

        def no_op(*args, **kwargs):
            return

        if (
            precompilation_timeout_seconds is None
            or precompilation_timeout_seconds <= 0
        ):
            log.debug("Precompilation timeout is None or <= 0, returning no_op")
            return no_op

        num_workers = min(get_num_workers(), len(choices))

        if num_workers <= 0:
            return no_op

        # https://github.com/python/cpython/issues/106905
        if (
            sys.version_info.major == 3
            and sys.version_info.minor == 11
            and sys.version_info.micro <= 8
        ):
            return no_op

        # check local and global cache before precompiling
        timings = self.lookup(
            choices,
            name,
            inputs_key,
            benchmark=None,
        )

        if timings and len(timings) == len(choices):
            # compilation in precompile stage is much cheaper than that in
            # autotuning stage
            log.debug("Found all %d timings in cache, returning no_op", len(timings))
            return no_op

        if config.search_autotune_cache and not (
            config.max_autotune or config.max_autotune_gemm
        ):
            return no_op

        precompile_key = create_precompile_key(name, inputs_key, choices)
        if precompile_func := self.precompile_cache.get(precompile_key):
            log.debug("Precompile function found in cache, returning it")
            return precompile_func

        log.info(
            "Multithreaded precompilation for %d choices using %d worker threads",
            len(choices),
            num_workers,
        )

        # In rare circumstances, because python threads inherit global state,
        # thread pool executor can race and leave stdout/stderr in a state
        # different than the original values. we explicitly restore the state
        # here to avoid this issue.

        def precompile_with_captured_stdout(choice) -> tuple[None, int]:
            log.debug("Precompiling choice with captured stdout: %s", choice)
            start_ns = time.time_ns()
            with restore_stdout_stderr():
                choice.precompile()
            elapsed_ns = time.time_ns() - start_ns
            # Return tuple as triton async compile (_worker_compile_triton)
            # returns tuple[CachingAutotuner, int]
            return None, elapsed_ns // 1000

        def on_complete(future):
            if not future.exception():
                _, precompile_elapsed_us = future.result()
                elapsed_seconds = precompile_elapsed_us / 1e6
                elapsed_times[future] = elapsed_seconds
                log.debug(
                    "Precompilation complete for future: %s, elapsed time: %.02fs",
                    future,
                    elapsed_seconds,
                )

        executor = ThreadPoolExecutor(max_workers=num_workers)
        async_compile = torch._inductor.async_compile.AsyncCompile()

        futures: dict[concurrent.futures.Future[Any], ChoiceCaller] = {}
        elapsed_times: dict[concurrent.futures.Future[Any], float] = {}

        # Some choices only differ in runtime arguments, so we
        # skip a choice if it has the same hash as a previously seen choice
        seen_choices: OrderedSet[str] = OrderedSet()
        for c in choices:
            # Skip choices which we have already issued a precompile
            if c.kernel_hash_key() in seen_choices:
                log.debug("Skipping already seen choice: %s", c)
                continue
            else:
                seen_choices.add(c.kernel_hash_key())

            if hasattr(c, "precompile"):
                triton_cuda_choice = isinstance(c, TritonTemplateCaller) and isinstance(
                    c.bmreq, TritonGPUBenchmarkRequest
                )
                if triton_cuda_choice and async_compile.use_process_pool():
                    with open(c.bmreq.module_path) as file:
                        source_code = file.read()
                    future = async_compile.triton(
                        kernel_name=c.bmreq.kernel_name, source_code=source_code
                    ).future
                    log.debug("Submitted triton async compile for choice: %s", c)
                else:
                    future = executor.submit(precompile_with_captured_stdout, c)
                    log.debug("Submitted precompile for choice: %s", c)

                future.add_done_callback(on_complete)
                futures[future] = c

        @functools.cache
        @restore_stdout_stderr()
        def wait_on_futures():
            log.debug("Waiting on futures")
            counters["inductor"]["select_algorithm_precompile"] += 1
            for future in as_completed(
                futures,
                timeout=precompilation_timeout_seconds,
            ):
                if e := future.exception():
                    from torch._inductor.codegen.cuda.cuda_kernel import (
                        CUDATemplateCaller,
                    )

                    if isinstance(e, CUDACompileError) and isinstance(
                        futures[future], CUDATemplateCaller
                    ):
                        log.debug(
                            "Exception %s for benchmark choice %s",
                            e,
                            futures[future],
                            exc_info=True,
                        )
                    else:
                        log.error(
                            "Exception %s for benchmark choice %s", e, futures[future]
                        )
                else:
                    counters["inductor"]["select_algorithm_num_precompiles"] += 1
                    log.info(
                        "Precompiling benchmark choice %s took %.02fs",
                        futures.get(future),
                        elapsed_times.get(future),
                    )

            executor.shutdown(wait=True)

        self.precompile_cache[precompile_key] = wait_on_futures

        return wait_on_futures

    @classmethod
    def get_inputs(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
    ) -> AutotuneArgs:
        """
        Factory method to create AutotuneArgs from a list of ChoiceCallers.
        """
        if input_gen_fns is None:
            input_gen_fns = {}

        # de-duplicate args
        unique_example_inputs = {
            x.get_name(): input_gen_fns.get(i, cls.benchmark_example_value)(x)
            for i, x in enumerate(input_nodes)
        }
        example_inputs = list(unique_example_inputs.values())
        example_inputs_extern = [
            (
                unique_example_inputs[input_node.get_name()]
                if unique_example_inputs[input_node.get_name()].is_mkldnn
                else torch.as_strided(
                    unique_example_inputs[input_node.get_name()],
                    V.graph.sizevars.size_hints(
                        input_node.get_size(),
                        fallback=config.unbacked_symint_fallback,
                    ),
                    V.graph.sizevars.size_hints(
                        input_node.get_stride(),
                        fallback=config.unbacked_symint_fallback,
                    ),
                    V.graph.sizevars.size_hint(
                        input_node.get_layout().offset,
                        fallback=config.unbacked_symint_fallback,
                    ),
                )
            )
            for input_node in input_nodes
        ]
        out = cls.benchmark_example_value(layout)
        out_extern = torch.as_strided(
            out, out.size(), out.stride(), V.graph.sizevars.size_hint(layout.offset)
        )
        expected = None
        if VERIFY:
            choices[0].benchmark(*example_inputs_extern, out=out_extern)
            expected = out_extern.clone()

        return AutotuneArgs.from_choice_args(
            example_inputs,
            example_inputs_extern,
            out,
            out_extern,
            expected,
        )

    @classmethod
    def benchmark_choice(
        cls, choice: ChoiceCaller, autotune_args: AutotuneArgs
    ) -> float:
        is_extern = isinstance(choice, (ExternKernelCaller, SubgraphChoiceCaller))
        benchmark_tensors = autotune_args.get_benchmark_tensors(is_extern)
        inpts, output = benchmark_tensors.unpack()
        output.zero_()
        result = choice.benchmark(*inpts, out=output)
        device_type = next(
            (tensor.device.type for tensor in inpts if is_gpu(tensor.device.type)),
            "cuda",
        )
        device_interface = get_interface_for_device(device_type)
        if device_interface.is_available():
            device_interface.synchronize()  # shake out any CUDA errors

        if VERIFY and autotune_args.expected is not None:
            autotune_args.verify(**VERIFY)
        return result

    @classmethod
    def benchmark_choices(
        cls,
        choices: Sequence[ChoiceCaller],
        autotune_args: AutotuneArgs,
    ) -> dict[ChoiceCaller, float]:
        timings = {}
        for choice in choices:
            try:
                timing = cls.benchmark_choice(choice, autotune_args)
            except CUDACompileError as e:
                from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

                if not isinstance(choice, CUDATemplateCaller):
                    log.error(
                        "CUDA compilation error during autotuning: \n%s. \nIgnoring this choice.",
                        e,
                    )
                timing = float("inf")
            except NotImplementedError as e:
                log.warning("Not yet implemented: %s", e)
                timing = float("inf")
            except RuntimeError as e:
                from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

                msg = str(e)
                if "invalid argument" in msg:
                    msg += "\n\nThis may mean this GPU is too small for max_autotune mode.\n\n"
                else:
                    if "illegal memory access" in msg:
                        msg += "\n\nEither error in template or triton bug.\n"

                if isinstance(choice, CUDATemplateCaller):
                    log.debug(
                        "Runtime error during autotuning: \n%s. \nIgnoring this choice.",
                        msg,
                        exc_info=True,
                    )
                else:
                    log.error(
                        "Runtime error during autotuning: \n%s. \nIgnoring this choice.",
                        msg,
                    )
                timing = float("inf")
            except AssertionError as e:
                raise AssertionError(  # noqa: B904
                    f"Incorrect result from choice {choice}\n\n{e}"
                )
            except Exception as e:
                try:
                    from triton.runtime.autotuner import OutOfResources

                    if isinstance(e, OutOfResources):
                        log.warning(e)
                        timing = float("inf")
                    else:
                        raise e
                except ImportError:
                    raise e from None

            timings[choice] = timing

        return timings

    @classmethod
    def benchmark_in_current_process(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
    ) -> dict[ChoiceCaller, float]:
        inputs = cls.get_inputs(choices, input_nodes, layout, input_gen_fns)
        return cls.benchmark_choices(choices, inputs)

    @classmethod
    def benchmark_in_sub_process(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
    ):
        from . import autotune_process

        # only benchmark triton kernel in sub process for now.
        # ATen/Extern kernel are still benchmarked in the current process.
        extern = [c for c in choices if isinstance(c, ExternKernelCaller)]
        triton = [c for c in choices if not isinstance(c, ExternKernelCaller)]

        timings = cls.benchmark_in_current_process(
            extern, input_nodes, layout, input_gen_fns
        )
        timings.update(autotune_process.benchmark_in_sub_process(triton))  # type: ignore[arg-type]
        return timings

    @classmethod
    def make_benchmark_fn(
        cls,
        choices: Sequence[ChoiceCaller],
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        input_gen_fns: Optional[dict[int, Callable[[ir.Buffer], torch.Tensor]]],
    ):
        if DEBUG:
            print(f"{len(choices)} tuning requests:")

        if config.autotune_in_subproc:
            return functools.partial(
                cls.benchmark_in_sub_process,
                input_nodes=input_nodes,
                layout=layout,
                input_gen_fns=input_gen_fns,
            )
        else:
            return functools.partial(
                cls.benchmark_in_current_process,
                input_nodes=input_nodes,
                layout=layout,
                input_gen_fns=input_gen_fns,
            )

    @staticmethod
    def prescreen_choices(
        choices: list[ChoiceCaller],
        name: str,
        inputs_key: str,
        prescreen_cache: dict[str, OrderedSet[str]],
    ) -> list[ChoiceCaller]:
        """
        Figure out what choices need to be prescreened before autotuning with runtime
        params.

        Prescreening is a process of reducing the number of autotuning for choices with
        runtime params via a two stage autotuning process. First, we fix a set of runtime
        params (here we use swizzle=2) and run autotuning to get a set of candidates.
        Then, we run autotuning again with the candidates and the full set of runtime
        params.

        Since have the concept of runtime params, we need to differentiate between
        choice's hash_key and choice's kernel_hash_key. The former includes information
        like runtime params, while the latter does not. prescreen_cache, if exists, stores
        the set of hash_key that should win the prescreening.

        Right now, only CUTLASS choices have runtime params.
        """
        # Create a cache key for prescreening results
        prescreen_key = f"{name}:{inputs_key}"

        # Check if we have cached prescreening results (prescreen_winners)
        if prescreen_key in prescreen_cache:
            prescreen_winners = [
                choice
                for choice in choices
                if choice.hash_key() in prescreen_cache[prescreen_key]
            ]
            return prescreen_winners

        # prescreen cutlass
        from .codegen.cuda.cuda_kernel import CUDATemplateCaller

        candidates = []
        if (
            config.cuda.cutlass_prescreening
            and len(config.cuda.cutlass_max_profiling_swizzle_options) > 1
        ):
            candidates.extend(
                [
                    c
                    for c in choices
                    if isinstance(c, CUDATemplateCaller)
                    # hardcoded to only look at swizzle=2
                    if c.info_dict().get("swizzle") == "2"
                ]
            )

        # skip prescreening if the number of candidates is too small
        if len(candidates) < 10:
            return []

        return candidates  # type: ignore[return-value]

    @staticmethod
    def prune_choices_postscreen(
        choices: list[ChoiceCaller],
        candidate_timings: dict[ChoiceCaller, float],
        name: str,
        inputs_key: str,
        prescreen_cache: dict[str, OrderedSet[str]],
    ) -> list[ChoiceCaller]:
        """
        Prune the choices after prescreening.
        """
        from .codegen.cuda.cuda_kernel import CUDATemplateCaller

        prescreen_key = f"{name}:{inputs_key}"

        # Check if we have cached postscreen results
        if prescreen_key in prescreen_cache:
            # candidate_timings are from choices that have won prescreening already
            winner_kernel_hashes = [
                candidate.kernel_hash_key() for candidate in candidate_timings
            ]

            pruned_choices = [
                choice
                for choice in choices
                if not isinstance(choice, CUDATemplateCaller)
                or choice.kernel_hash_key() in winner_kernel_hashes
            ]
            return pruned_choices

        log.debug("Before pruning using prescreening timings, %d choices", len(choices))
        sorted_candidates = sorted(
            candidate_timings.keys(), key=lambda choice: candidate_timings[choice]
        )

        # Print prescreening timings
        if (
            candidate_timings
            and PRINT_AUTOTUNE
            and config.autotune_num_choices_displayed != 0
        ):
            n = config.autotune_num_choices_displayed
            top_k = sorted_candidates[:n]
            best = top_k[0]
            best_time = candidate_timings[best]

            lines = ["PRESCREENING CANDIDATE TIMINGS"]
            for choice in top_k:
                result = candidate_timings[choice]
                if result:
                    lines.append(
                        f"  {choice.name} {result:.4f} ms {best_time / result:.1%} {choice.description}"
                    )
                else:
                    lines.append(
                        f"  {choice.name} {result:.4f} ms <DIVIDED BY ZERO ERROR>"
                    )

            log.info("\n".join(lines))
        num_to_keep = max(int(math.sqrt(len(choices)) / 4), 8)

        # prune choices based on prescreening timings
        candidates_to_prune = OrderedSet(
            candidate.kernel_hash_key() for candidate in sorted_candidates[num_to_keep:]
        )
        winner_hashes: OrderedSet[str] = OrderedSet()
        for candidate in sorted_candidates[:num_to_keep]:
            if candidate_timings[candidate] == float("inf"):
                candidates_to_prune.add(candidate.kernel_hash_key())
            else:
                winner_hashes.add(candidate.hash_key())
                if isinstance(candidate, CUDATemplateCaller):
                    candidate.bmreq.ensure_dll_loaded()

        pruned_choices = [
            choice
            for choice in choices
            if choice.kernel_hash_key() not in candidates_to_prune  # type: ignore[attr-defined]
        ]

        # Cache the hash_key of winners of prescreening
        prescreen_cache[prescreen_key] = winner_hashes

        log.debug(
            "After pruning using prescreening timings, %d choices", len(pruned_choices)
        )
        return pruned_choices

    @staticmethod
    def log_results(
        name: str,
        input_nodes: list[ir.IRNode],
        timings: dict[ChoiceCaller, float],
        elapse: float,
        precompile_elapse: float,
        prescreening_elapse: Optional[float] = None,
    ):
        V.debug.log_autotuning_results(
            name, input_nodes, timings, elapse, precompile_elapse
        )
        if not (config.max_autotune or config.max_autotune_gemm) or not PRINT_AUTOTUNE:
            return
        sizes = ", ".join(
            [
                "x".join(
                    map(
                        str,
                        V.graph.sizevars.size_hints(
                            n.get_size(),
                            fallback=config.unbacked_symint_fallback,  # type: ignore[arg-type]
                        ),
                    )
                )
                for n in input_nodes
            ]
        )

        strides = ", ".join([str(n.get_stride()) for n in input_nodes])
        dtypes = ", ".join([str(n.get_dtype()) for n in input_nodes])
        if config.autotune_num_choices_displayed == 0:
            return
        # when autotune_num_choices_displayed is None, [:None] means all
        n = config.autotune_num_choices_displayed
        top_k = sorted(timings, key=timings.__getitem__)[:n]

        best = top_k[0]

        def get_choice_info(choice):
            if isinstance(choice, torch._inductor.select_algorithm.ExternKernelCaller):
                return {"type": "cublas", "time": timings[choice]}

            assert isinstance(
                choice,
                torch._inductor.codegen.triton_templates.caller.TritonTemplateCaller,
            )

            info = choice.info_dict()
            tile = info["tile_shape"]

            tile_vals = eval(tile)  # type: ignore[arg-type]
            BLOCK_M = tile_vals[0]
            BLOCK_K = tile_vals[1]
            BLOCK_N = tile_vals[2]

            return {
                "type": "triton",
                "time": timings[choice],
                "BLOCK_M": BLOCK_M,
                "BLOCK_K": BLOCK_K,
                "BLOCK_N": BLOCK_N,
                "num_stages": info["num_stages"],
                "num_warps": info["num_warps"],
            }

        mm_filename = get_mm_log_filename()
        if mm_filename and "mm" in name:
            M, K = input_nodes[-2].get_size()[:2]
            N = input_nodes[-1].get_size()[-1]

            out_dict = {
                str((M, K, N)): [get_choice_info(choice) for choice in timings.keys()]
            }

            append_to_log(mm_filename, out_dict)

        best_time = timings[best]
        sys.stderr.write(f"AUTOTUNE {name}({sizes})\n")
        sys.stderr.write(f"strides: {strides}\n")
        sys.stderr.write(f"dtypes: {dtypes}\n")

        for choice in top_k:
            result = timings[choice]
            if result:
                kernel_description = choice.description
                sys.stderr.write(
                    f"  {choice.name} {result:.4f} ms {best_time / result:.1%} {kernel_description}\n"
                )
            else:
                sys.stderr.write(
                    f"  {choice.name} {result:.4f} ms <DIVIDED BY ZERO ERROR>\n"
                )

        autotune_type_str = (
            "SubProcess" if config.autotune_in_subproc else "SingleProcess"
        )
        prescreening_msg = (
            f" and {prescreening_elapse:.4f} seconds prescreening"
            if prescreening_elapse is not None
            else ""
        )
        sys.stderr.write(
            f"{autotune_type_str} AUTOTUNE benchmarking takes {elapse:.4f} seconds and {precompile_elapse:.4f}"
            f" seconds precompiling for {len(timings)} choices"
            + prescreening_msg
            + "\n"
        )

    @staticmethod
    def benchmark_example_value(node):
        """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
        if isinstance(node, ir.Layout):
            node = ir.Buffer(name="fake", layout=node)
        # triton templates want the base tensor.
        if isinstance(node, ir.BaseView):
            node = node.unwrap_view()

        # Inplace padding may reinterpret a tensor to a larger tensor if the
        # stride is large enough. The V.graph.get_allocation_size takes this into account.
        # So we need call as_strided in the end to 'view' the tensor with the correct
        # sizes/strides
        return AlgorithmSelectorCache.generate_example_value(
            V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            V.graph.sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            node.get_device(),
            node.get_dtype(),
            node.layout.offset,
            V.graph.sizevars.size_hints(
                V.graph.get_allocation_size(node),
                fallback=config.unbacked_symint_fallback,
            ),
        )

    @staticmethod
    def generate_example_value(
        size, stride, device, dtype, extra_size, allocation_size=None
    ):
        # preserve rng states to avoid the rand_strided call below changes
        # the rng states for the real model code.
        with preserve_rng_state():
            if allocation_size is None or allocation_size == size:
                return rand_strided(
                    size,
                    stride,
                    device=device,
                    dtype=dtype,
                    extra_size=extra_size,
                )
            else:
                return rand_strided(
                    allocation_size,
                    stride,
                    device=device,
                    dtype=dtype,
                    extra_size=extra_size,
                ).as_strided(size, stride)

    @staticmethod
    def key_of(node):
        """
        Extract the pieces of an ir.Buffer that we should invalidate cached
        autotuning results on.
        """
        sizevars = V.graph.sizevars
        return (
            node.get_device().type,
            str(node.get_dtype()),
            *sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            *sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            sizevars.size_hint(
                node.get_layout().offset,
                fallback=config.unbacked_symint_fallback,
            ),
        )

    def add_feedback_saver(self, fn: FeedbackFunction):
        self.feedback_saver_fns.append(fn)


_ALGORITHM_SELECTOR_CACHE: Optional[AlgorithmSelectorCache] = None


def autotune_select_algorithm(*args, **kwargs):
    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()

    if "return_multi_template" not in kwargs:
        kwargs["return_multi_template"] = (
            torch._inductor.config.benchmark_epilogue_fusion
        )

    if "precompilation_timeout_seconds" not in kwargs:
        kwargs["precompilation_timeout_seconds"] = config.precompilation_timeout_seconds

    return _ALGORITHM_SELECTOR_CACHE(*args, **kwargs)


def add_feedback_saver(
    fn: FeedbackFunction,
):
    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()
    _ALGORITHM_SELECTOR_CACHE.add_feedback_saver(fn)


def realize_inputs(*args):
    if len(args) == 1:
        return ir.ExternKernel.require_stride1(ir.ExternKernel.realize_input(args[0]))
    return [realize_inputs(x) for x in args]


class SymbolicGridFn:
    """
    Wrapper around a grid function that allows either int or sympy inputs.

        @SymbolicGridFn
        def grid(x, meta, *, cdiv):
            return cdiv(x, meta["BLOCK_X"])
    """

    def __init__(self, fn: Callable[..., tuple[Any, Any, Any]]):
        self.fn = fn
        self.kwargs_int = {}
        self.kwargs_sym = {}
        params = inspect.signature(fn).parameters
        for name, fn_sym, fn_int in [
            ("cdiv", CeilDiv, ceildiv),
            ("min", sympy.Min, min),
            ("max", sympy.Max, max),
        ]:
            if name in params:
                self.kwargs_int[name] = fn_int
                self.kwargs_sym[name] = fn_sym

    def __call__(self, *args, **kwargs) -> tuple[int, int, int]:
        return self.fn(*args, **kwargs, **self.kwargs_int)

    def sympy_call(self, *args, **kwargs):
        return self.fn(*args, **kwargs, **self.kwargs_sym)


# ensure lowering is imported so that `extern_kernels.*` is populated
from . import lowering  # noqa: F401
