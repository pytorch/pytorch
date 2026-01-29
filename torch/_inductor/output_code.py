"""
This provides an abstract class which parametrizes over an "output code" concept
for Inductor.  Intuitively, this represents the compiled callable which Inductor
produces which you can call to get optimized code.  However, this callable
has some other capabilities:

- It is serializable, so you can save/load this product from disk without
  having to do compilation again.

- (When using remote cache) it is addressable, so you can save just a key
  which you can use to load this product from remote cache later.

This class is abstract because we have several different implementations of
serialized format:

- Python wrapper (the default)

- AOTInductor (this produces ABI stable binaries which work across PyTorch
  versions)

"""

from __future__ import annotations

import dataclasses
import logging
import os
from functools import partial
from typing import Any, Optional, TYPE_CHECKING, TypeAlias, Union

import torch
from torch._dynamo.utils import counters, get_runtime_metrics_context
from torch._higher_order_ops.wrap import inductor_compiled_code
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    CudagraphCachedInfo,
    CudagraphMetadata,
    get_partition_cudagraph_metadata,
    get_placeholder_info,
    log_cudagraph_skip_and_bump_counter,
)
from torch._inductor.freezing_utils import has_frozen_params, is_frozen_param
from torch._inductor.utils import (
    _unstable_customized_partition_wrapper,
    align_inputs_from_check_idxs,
    BoxedBool,
    CUDAGraphWrapperMetadata,
    GraphPartitionMap,
    InputType,
    output_node,
    set_tracing_context_output_strides,
)
from torch.fx._graph_pickler import _ops_filter_safe
from torch.utils._ordered_set import OrderedSet
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from . import config
from .runtime.autotune_cache import AutotuneCacheBundler


if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable, Sequence

    from torch._inductor import metrics
    from torch._inductor.graph import GraphLowering
    from torch._library.fake_class_registry import FakeScriptObject
    from torch.export.pt2_archive._package_weights import Weights

    from .compile_fx import _CompileFxKwargs
    from .triton_bundler import TritonBundle

log = logging.getLogger(__name__)


@dataclasses.dataclass
class OutputCode:
    # TODO: Remove underscores here

    # None if the output is not remote cacheable
    _fx_graph_cache_key: Optional[str] = dataclasses.field(default=None, init=False)
    _fx_graph_cache_debug_lines: Optional[list[str]] = dataclasses.field(
        default=None, init=False
    )

    # How long it took to compile this OutputCode, end to end
    _time_taken_ns: Optional[int] = dataclasses.field(default=None, init=False)

    def __call__(self, inputs: Sequence[Any]) -> Any:
        raise NotImplementedError(type(self))

    def prepare_for_serialization(self) -> None:
        raise NotImplementedError(type(self))

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        raise NotImplementedError(type(self))

    # TODO: Get rid of this
    def set_triton_bundle(self, triton_bundle: Any) -> None:
        raise NotImplementedError(type(self))


_StrideExprStr: TypeAlias = str


# copy_ fails when trying to write to tensors with memory overlap,
# for expanded dimensions (a dimension which used to have size 1 -> ?)
# we can select one element from that dimension and write to it
# to achieve writing to all values of that dimension of the input tensor
def get_expanded_dims(t: torch.Tensor) -> list[int]:
    if not isinstance(t, torch.Tensor):
        # pyrefly: ignore [bad-return]
        return None
    return [i for i in range(t.ndim) if t.stride(i) == 0 and t.size(i) != 1]


def index_expanded_dims(t: torch.Tensor, expanded_dims: list[int]) -> torch.Tensor:
    for expanded_dim in expanded_dims:
        t = torch.ops.aten.slice(t, expanded_dim, 0, 1)
    return t


def complex_memory_overlap(t: torch.Tensor) -> bool:
    if config.always_complex_memory_overlap_TESTING_ONLY:
        return True

    # if torch._debug_has_internal_overlap thinks this tensor potentially has
    # memory overlap internally, let's dig deeper to find out whether it's true.
    #
    # Call squeeze() so that dimension with size 1 does not cause false positive.
    t = index_expanded_dims(t, get_expanded_dims(t)).squeeze()
    if torch._debug_has_internal_overlap(t) != 0:
        strides = t.stride()
        sizes = t.shape
        indices = list(range(len(strides)))
        indices = [x for _, x in sorted(zip(strides, indices))]
        for i in range(len(strides)):
            prev_stride = 1 if i == 0 else strides[indices[i - 1]]
            prev_size = 1 if i == 0 else sizes[indices[i - 1]]
            if strides[indices[i]] < prev_stride * prev_size:
                return True
    return False


def maybe_handle_backward_generation(
    compiled_graph: CompiledFxGraph,
    boxed_forward_device_index: Optional[BoxedDeviceIndex],
) -> None:
    assert compiled_graph.current_callable is not None
    is_backward = compiled_graph.fx_kwargs["is_backward"]

    # See [Backward Generation Handling]
    # if cudagraph'd the forward and set the device, we need to let the cudagraph manager
    # know we are we running the backward even if we will not run it in cudagraphs
    if is_backward and config.triton.cudagraph_trees:
        assert boxed_forward_device_index is not None
        assert boxed_forward_device_index.value is not None
        compiled_graph_callable = compiled_graph.current_callable

        manager = torch._inductor.cudagraph_trees.get_manager(
            boxed_forward_device_index.value, create_if_none_exists=False
        )
        # should already exist from forward
        assert manager is not None

        def compiled_artifact(new_inputs: list[Any]) -> Callable[..., Any]:
            manager.set_to_running_backward()  # type: ignore[union-attr]
            return compiled_graph_callable(new_inputs)

        compiled_graph.current_callable = compiled_artifact


def prepare_cudagraph_post_compile(
    compiled_graph: CompiledFxGraph,
    example_inputs: Sequence[InputType],
    boxed_forward_device_index: Optional[BoxedDeviceIndex],
) -> None:
    if not config.triton.cudagraph_trees:
        # Force specialize all inputs so that CUDA graphs will work
        for t in example_inputs:
            if isinstance(t, torch.SymInt):
                int(t)  # guard

    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]
    if boxed_forward_device_index is not None and not is_inference and not is_backward:
        boxed_forward_device_index.set(next(iter(compiled_graph.device_idxs)))


def cudagraph_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, Union[torch.Tensor, type]],
    boxed_forward_device_index: Optional[BoxedDeviceIndex],
) -> None:
    """
    Checks for any reasons not to run cudagraphs and then
    runs it on compiled_graph.
    Mutates the `compiled_graph.current_callable` and `cudagraphs`
    """
    from torch._inductor.compiler_bisector import CompilerBisector

    assert compiled_graph.current_callable is not None
    assert compiled_graph.cudagraph_info is not None
    cached_info = compiled_graph.cudagraph_info
    cudagraph_fail_reasons = cached_info.cudagraph_fail_reasons
    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]

    # Check if bisector wants to disable cudagraphs for this graph
    if CompilerBisector.disable_subsystem("inductor", "cudagraphs"):
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        log_cudagraph_skip_and_bump_counter("skipping cudagraphs due to bisector")
        return

    if not cudagraph_fail_reasons:
        fx_kwargs = compiled_graph.fx_kwargs
        static_input_idxs = fx_kwargs["static_input_idxs"]

        placeholders = cached_info.placeholders
        stack_traces = cached_info.stack_traces

        prepare_cudagraph_post_compile(
            compiled_graph, example_inputs, boxed_forward_device_index
        )

        from .compile_fx import cudagraphify

        current_callable = compiled_graph.current_callable
        assert current_callable is not None
        # Filter to only tensor constants (exclude opaque value type classes)
        tensor_constants = {
            k: v for k, v in constants.items() if isinstance(v, torch.Tensor)
        }
        compiled_graph.current_callable = cudagraphify(
            current_callable,
            static_input_idxs=static_input_idxs or (),
            device_index=next(iter(compiled_graph.device_idxs)),
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=tuple(tensor_constants.values()),
            placeholders=placeholders,
            mutated_input_idxs=tuple(compiled_graph.mutated_input_idxs),
        )

    else:
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)

        if "cuda" in compiled_graph.device_types:
            # prefer better disable_cudagraphs_reason bc stack trace
            # TODO: migrate all disable reasons to stack trace, refactor
            if compiled_graph.disabled_cudagraphs_reason:
                log_cudagraph_skip_and_bump_counter(
                    compiled_graph.disabled_cudagraphs_reason
                )
            else:
                log_cudagraph_skip_and_bump_counter(
                    f"skipping cudagraphs due to {cudagraph_fail_reasons}"
                )


def cudagraph_partition_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    constants: dict[str, Union[torch.Tensor, type]],
    boxed_forward_device_index: Optional[BoxedDeviceIndex],
) -> None:
    """
    Cudagraphify each partition functions, which first prepares the necessary
    metadata and then applies the cudagraphify function to each partition.

    Assuming all partition functions are cudagraphified and share the same order
    as `compiled_graph.partition_maps`. See [Note: Graph Partition Map for CUDAGraph].
    """
    from torch._inductor.compiler_bisector import CompilerBisector

    if CompilerBisector.disable_subsystem("inductor", "cudagraphs"):
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        log_cudagraph_skip_and_bump_counter("skipping cudagraphs due to bisector")
        return

    assert compiled_graph.cudagraph_info is not None
    cudagraph_fail_reasons = compiled_graph.cudagraph_info.cudagraph_fail_reasons

    if (
        cudagraph_fail_reasons
        or compiled_graph.partition_maps is None
        or len(compiled_graph.partition_maps) == 0
    ):
        # cudagraphify is not called if there are no partitions
        BoxedBool.disable(cudagraphs)
        maybe_handle_backward_generation(compiled_graph, boxed_forward_device_index)
        return

    from .compile_fx import cudagraphify

    assert compiled_graph.current_callable is not None
    assert compiled_graph.recursively_apply_fns is not None
    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]
    static_input_idxs = OrderedSet(compiled_graph.fx_kwargs["static_input_idxs"] or ())
    mutated_input_idxs = compiled_graph.mutated_input_idxs
    device_index = next(iter(compiled_graph.device_idxs))

    # Filter to only tensor constants (exclude opaque value type classes)
    tensor_constants = {
        k: v for k, v in constants.items() if isinstance(v, torch.Tensor)
    }

    graph_metadata = CudagraphMetadata(
        compiled_graph.cudagraph_info.placeholders,
        static_input_idxs,
        mutated_input_idxs,
        compiled_graph.cudagraph_info.stack_traces,
        tensor_constants,
    )

    prepare_cudagraph_post_compile(
        compiled_graph, example_inputs, boxed_forward_device_index
    )

    # cudagraphify each partition function, assuming every graph partition function
    # is cudagraphable. Non-cudagraphable ops (e.g., cpu ops) are inlined into
    # `call` function and not included in partition functions.
    cudagraphify_fns = []
    for partition_map in compiled_graph.partition_maps:
        partition_metadata = get_partition_cudagraph_metadata(
            partition_map,
            graph_metadata,
        )

        cudagraphify_fn = partial(
            cudagraphify,
            static_input_idxs=tuple(partition_metadata.static_input_idxs),
            device_index=device_index,
            stack_traces=partition_metadata.stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=tuple(partition_metadata.constants.values()),
            placeholders=partition_metadata.placeholders,
            mutated_input_idxs=tuple(partition_metadata.mutated_input_idxs),
        )
        cudagraphify_fns.append(cudagraphify_fn)

    compiled_graph.recursively_apply_fns(cudagraphify_fns)


def maybe_realign_inputs(
    ran_cudagraphs: BoxedBool,
    compiled_graph: CompiledFxGraph,
    inputs_to_check: Sequence[int],
    mutated_inputs_idxs: OrderedSet[int],
) -> None:
    """
    Realigns input strides from inputs_to_check if
    we didn't end up running cudagraphs. Mutates
    `compiled_graph.current_callable` if cudagraphs
    was run. Otherwise, does nothing.
    """
    if not ran_cudagraphs:
        assert compiled_graph.current_callable is not None
        new_callable = align_inputs_from_check_idxs(
            compiled_graph.current_callable, inputs_to_check, mutated_inputs_idxs
        )
        if new_callable is not compiled_graph.current_callable:
            compiled_graph.current_callable = new_callable


class CompiledFxGraphConstants:
    """Wrapper class that unwraps constants from a compiled fx graph. This
    version of the class only supports directly grabbing the saved constants off of
    a CompiledFxGraph.

    With freezing, FxGraphCache doesn't store the constants of the input
    GraphModule it gets from AOTAutograd. Instead, it saves just the **names**
    of those constants, and grabs the constant values directly from the graph module
    passed in at runtime.

    Thing is, we don't always *have* the graph module available at runtime, hence
    the existence of this class and its CompiledFxGraphConstantsWithGm counterpart.

    To support freezing, FXGraphCache gets passed a CompiledFxGraphConstantsWithGm during
    post compile. Otherwise, CompiledFxGraphConstants supports the basic case of loading
    the value of constants directly off of the original saved object.
    """

    def unwrap(self, g: CompiledFxGraph) -> dict[str, Union[torch.Tensor, type]]:
        assert g.constants is not None
        return {**g.constants, **g.opaque_value_type_classes}


class CompiledFxGraphConstantsWithGm(CompiledFxGraphConstants):
    """
    This version of CompiledFxGraphConstants, instead of grabbing constants
    directly saved on CompiledFxGraphs, will just grab their names. Then, it takes
    a second GraphModule to grab the corresponding constant values out of.

    This is necessary for supporting freezing in FxGraphCache.
    """

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.gm = gm

    def unwrap(self, g: CompiledFxGraph) -> dict[str, Union[torch.Tensor, type]]:
        frozen_params = {
            name: getattr(self.gm, orig_name)
            for name, orig_name in g.frozen_param_names.items()
        }
        constants = g.constants or {}
        return {**constants, **frozen_params, **g.opaque_value_type_classes}


@dataclasses.dataclass
class CompiledFxGraph(OutputCode):
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: Optional[Callable[..., Any]]
    recursively_apply_fns: Optional[Callable[..., Any]]
    compiled_fn_runner: Optional[Any]
    cache_key: str
    source_code: str = dataclasses.field(repr=False)  # Do not display source_code
    runnable_graph_str: str = dataclasses.field(repr=False)  # Do not display graph
    inductor_post_grad_graph_str: str = dataclasses.field(
        repr=False
    )  # Do not display graph
    cache_linemap: Optional[list[tuple[int, str]]]
    device_types: OrderedSet[str]
    device_idxs: OrderedSet[int]
    mutated_inputs: OrderedSet[str]
    mutated_input_idxs: OrderedSet[int]
    constants: Optional[dict[str, torch.Tensor]]
    frozen_param_names: dict[str, str]
    torchbind_constants: dict[str, torch._C.ScriptObject | FakeScriptObject]
    opaque_value_type_classes: dict[str, type]
    output_strides: Optional[list[Optional[tuple[_StrideExprStr, ...]]]]
    disabled_cudagraphs_reason: Optional[str]
    metrics_deltas: metrics.CachedMetricsDeltas
    counter_deltas: Counter[str]
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: Optional[str]
    inductor_provenance_mapping_str: Optional[str]
    inductor_provenance_stack_traces_str: Optional[str]

    cudagraph_info: Optional[CudagraphCachedInfo]
    partition_maps: Optional[list[GraphPartitionMap]]
    fx_kwargs: _CompileFxKwargs
    inputs_to_check: Sequence[int]

    _boxed_call: Optional[bool] = None
    _triton_bundle: Optional[TritonBundle] = None
    _wrap_compiled_regions: bool = False

    def __init__(
        self,
        current_callable: Optional[Callable[..., Any]],
        graph: GraphLowering,
        gm: torch.fx.GraphModule,
        output_strides: list[Optional[tuple[_StrideExprStr, ...]]],
        disabled_cudagraphs_reason: Optional[str],
        metrics_deltas: metrics.CachedMetricsDeltas,
        counter_deltas: Counter[str],
        cudagraphs: BoxedBool,
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
        runnable_graph_str: str,
        inductor_post_grad_graph_str: str,
        compiled_fn_runner: Optional[Any] = None,
        inductor_provenance_mapping_str: Optional[str] = None,
        inductor_provenance_stack_traces_str: Optional[str] = None,
    ) -> None:
        self.current_callable = current_callable
        self.compiled_fn_runner = compiled_fn_runner
        self.recursively_apply_fns = (
            compiled_fn_runner.recursively_apply_fns
            if compiled_fn_runner is not None
            else None
        )
        self.cache_key = graph.cache_key
        if graph.cache_path:
            with open(graph.cache_path) as f:
                self.source_code = f.read()
        self.runnable_graph_str = runnable_graph_str
        self.inductor_post_grad_graph_str = inductor_post_grad_graph_str
        self.inductor_provenance_mapping_str = inductor_provenance_mapping_str
        self.inductor_provenance_stack_traces_str = inductor_provenance_stack_traces_str
        self.cache_linemap = graph.cache_linemap
        # TODO - ordered set
        self.device_types = OrderedSet(graph.device_types)
        self.device_idxs = OrderedSet(graph.device_idxs)
        self.mutated_inputs = OrderedSet(graph.mutated_inputs)
        self.mutated_input_idxs = OrderedSet(graph.mutated_input_idxs)

        # We store the constant attributes in the cache entry and re-attach them
        # to the module created in PyCodeCache.load_by_key_path. In the case that
        # the graph has frozen parameters, we save the mapping from the attribute
        # names in the GraphLowering to the original name of the attribute in the
        # GraphModule. When we create the module from the cache entry, we then
        # look up the constants from the current GraphModule. This scheme allows
        # us to support caching with freezing.
        if not has_frozen_params(gm):
            self.constants = graph.constants
            self.frozen_param_names = {}
        else:
            self.constants = {}
            self.frozen_param_names = {}
            for k, v in graph.constants.items():
                if is_frozen_param(v):
                    self.frozen_param_names[k] = graph.allocated_constant_name[k]
                else:
                    self.constants[k] = v

        self.torchbind_constants = graph.torchbind_constants
        self.opaque_value_type_classes = graph.opaque_value_type_classes
        self.output_strides = output_strides
        self.disabled_cudagraphs_reason = disabled_cudagraphs_reason
        self.metrics_deltas = metrics_deltas
        self.counter_deltas = counter_deltas
        self.guards_expr = None
        self.cudagraph_info = None
        self.partition_maps = graph.partition_maps
        self.fx_kwargs = {}
        self.inputs_to_check = ()

        cudagraph_info = None
        if cudagraphs:
            # check cudagraph disabling reasons from inductor lowering
            if self.disabled_cudagraphs_reason:
                if "cuda" in self.device_types:
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraphs due to {self.disabled_cudagraphs_reason}"
                    )
                else:
                    counters["inductor"]["cudagraph_skips"] += 1
                BoxedBool.disable(cudagraphs)
            else:
                complex_memory_overlap_inputs = any(
                    complex_memory_overlap(t)
                    for t in example_inputs
                    if isinstance(t, torch.Tensor)
                )

                if not config.triton.cudagraph_support_input_mutation:
                    # Skip supports for cudagraph-managed tensors
                    from torch._inductor.cudagraph_utils import (
                        check_for_mutation_ignore_cuda_graph_managed_tensor,
                    )

                    has_mutation_str = (
                        check_for_mutation_ignore_cuda_graph_managed_tensor(
                            gm,
                            self.mutated_inputs,
                            self.mutated_input_idxs,
                            static_input_idxs,
                        )
                    )
                    has_mutation = has_mutation_str is not None

                    if has_mutation:
                        self.disabled_cudagraphs_reason = has_mutation_str
                else:
                    # Check mutation later to support cudagraph-managed tensors
                    has_mutation = None

                cudagraph_tests = [
                    (not has_mutation, "mutated inputs"),
                    (not complex_memory_overlap_inputs, "complex memory overlap"),
                    (
                        all(
                            isinstance(t, (torch.Tensor, torch.SymInt, torch.Generator))
                            for t in example_inputs
                        ),
                        "non-Tensor inputs",
                    ),
                ]
                output = output_node(gm)
                # output args are tuple of first argument
                assert len(output.args) == 1
                stack_traces = [
                    (arg.stack_trace if isinstance(arg, torch.fx.node.Node) else None)
                    for arg in output.args[0]  # type: ignore[union-attr]
                ]
                cudagraph_fail_reasons = [s for b, s in cudagraph_tests if not b]
                placeholders = tuple(get_placeholder_info(gm.graph))
                cudagraph_info = CudagraphCachedInfo(
                    placeholders, stack_traces, cudagraph_fail_reasons
                )

        self.cudagraph_info = cudagraph_info
        self.inputs_to_check = inputs_to_check
        self.fx_kwargs = fx_kwargs

        # aot autograd needs to know to pass in inputs as a list
        self._boxed_call = True

        # Store whether to wrap compiled regions in inductor_compiled_code HOP
        # This is set at compile time to avoid runtime overhead
        self._wrap_compiled_regions = config.wrap_inductor_compiled_regions

    def __del__(self) -> None:
        if self.compiled_fn_runner is not None:
            # For torch._inductor.config.graph_partition = True,
            # self.compiled_fn_runner.partitions hold cudagraphified functions
            # which prevents deallocation. When CompiledFxGraph is deleted,
            # self.compiled_fn_runner will not be called in the future so we
            # should also delete these partitions.
            del self.compiled_fn_runner.partitions

    def __call__(self, inputs: Sequence[Any]) -> Any:
        assert self.current_callable is not None

        if (
            torch._inductor.debug.RECORD_GRAPH_EXECUTION
            and torch._inductor.debug.GRAPH_EXECUTION_ORDER is not None
        ):
            graph_id = self.fx_kwargs.get("graph_id")
            compile_id = (
                torch._inductor.debug.GRAPH_COMPILE_IDS.get(graph_id)
                if graph_id is not None
                and torch._inductor.debug.GRAPH_COMPILE_IDS is not None
                else None
            )
            torch._inductor.debug.GRAPH_EXECUTION_ORDER.append(
                {
                    "compile_id": compile_id,
                }
            )
        try:
            # Checking the profiler directly is faster than nullcontext
            if torch.autograd.profiler._is_profiler_enabled:
                with torch._C._profiler._RecordFunctionFast(
                    f"## Call CompiledFxGraph {self._fx_graph_cache_key} ##",
                    keyword_values={"scope": "user_scope"},
                ):
                    return self.current_callable(inputs)
            else:
                return self.current_callable(inputs)
        finally:
            get_runtime_metrics_context().finish()
            AutotuneCacheBundler.end_compile()

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        """
        Run a set of post processing steps after loading from the cache. These involve:
         - Setting the tracing context output strides
         - Running cudagraphs if enabled
         - Realigning inputs

        This runs whether or not we have a cache hit, and always runs directly after we get a CompiledFxGraph.
        The results of this function are *not* saved in the cache itself.
        """
        if config.graph_partition and _unstable_customized_partition_wrapper.wrapper:
            # Mechanically apply user-specified cudagraph wrappers without modification
            assert self.recursively_apply_fns is not None
            assert self.compiled_fn_runner is not None
            num_partitions = len(self.compiled_fn_runner.partitions)
            wrapper_metadatas = [
                CUDAGraphWrapperMetadata(num_partitions, i)
                for i in range(num_partitions)
            ]
            customized_wrapper = _unstable_customized_partition_wrapper.wrapper
            customized_wrappers_with_metadata = [
                lambda f, m=metadata: customized_wrapper(f, m)
                for metadata in wrapper_metadatas
            ]
            self.recursively_apply_fns(customized_wrappers_with_metadata)
            return

        set_tracing_context_output_strides(example_inputs, self)
        assert graph_kwargs["cudagraphs"] is not None
        assert graph_kwargs["is_backward"] is not None
        is_backward = graph_kwargs["is_backward"]
        cudagraphs: BoxedBool = graph_kwargs["cudagraphs"]
        if cudagraphs:
            # It's possible that cudagraphs is enabled, but was disabled
            # during a previous compilation we're loading from the cache.
            # If so, we need to disable it on this new process too.
            if self.disabled_cudagraphs_reason:
                if "cuda" in self.device_types:
                    log_cudagraph_skip_and_bump_counter(
                        f"skipping cudagraphs due to {self.disabled_cudagraphs_reason}"
                    )
                else:
                    counters["inductor"]["cudagraph_skips"] += 1
                BoxedBool.disable(cudagraphs)
            else:
                if is_backward:
                    assert "boxed_forward_device_index" in graph_kwargs
                    boxed_forward_device_index = graph_kwargs[
                        "boxed_forward_device_index"
                    ]
                else:
                    # On the forward we don't know whether or not
                    # boxed_forward_device_index is set yet
                    boxed_forward_device_index = graph_kwargs.get(
                        "boxed_forward_device_index", None
                    )

                if config.graph_partition:
                    # with graph_partition=True, we skip some cudagraph checks if it's supported
                    # with partition. So we have to use cudagraph_partition_post_compile.
                    cudagraph_partition_post_compile(
                        example_inputs,
                        self,
                        cudagraphs,
                        constants.unwrap(self),
                        boxed_forward_device_index,
                    )
                else:
                    cudagraph_post_compile(
                        example_inputs,
                        self,
                        cudagraphs,
                        constants.unwrap(self),
                        boxed_forward_device_index,
                    )
        inputs_to_check = self.inputs_to_check
        # cudagraphs could have been disabled from the earlier conditions
        # so we still need to realign inputs if that happens
        maybe_realign_inputs(
            cudagraphs,
            self,
            inputs_to_check,
            self.mutated_input_idxs,
        )

        # Apply inductor_compiled_code HOP wrapper if configured
        # This is done in post_compile to ensure it works with cached artifacts
        if self._wrap_compiled_regions and self.current_callable is not None:
            original_callable = self.current_callable

            def wrapped_callable(inputs):
                if is_in_torch_dispatch_mode():
                    return inductor_compiled_code(original_callable, inputs)
                else:
                    return original_callable(inputs)

            self.current_callable = wrapped_callable

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        self._triton_bundle = triton_bundle

    def prepare_for_serialization(self) -> None:
        # We can't really serialize callables that may be C++/Triton/etc.,
        # so we serialize their PyCodeCache disk cache location instead.
        # TODO: This could be better if we're ever able to serialize compiled
        # models to disk.
        self.current_callable = None
        self.recursively_apply_fns = None
        self.compiled_fn_runner = None

    def write_to_disk(self) -> str:
        from torch._dynamo.utils import counters
        from torch._inductor.codecache import get_path, write_atomic

        # See _save_graph(); we don't store the callable in the cache entry so
        # recreate it here from the PyCodeCache disk cache.
        artifact_path = get_path(self.cache_key, "py")[2]
        code = self.source_code
        if not os.path.exists(artifact_path):
            counters["inductor"]["fxgraph_lookup_write_file"] += 1
            write_atomic(artifact_path, code, make_dirs=True)
        return artifact_path

    def after_deserialization(self, constants: CompiledFxGraphConstants) -> str:
        from torch._dynamo.utils import dynamo_timed
        from torch._inductor.codecache import PyCodeCache

        artifact_path = self.write_to_disk()

        try:
            with dynamo_timed(
                "PyCodeCache.load_by_key_path",
                log_pt2_compile_event=True,
            ):
                code_cache = PyCodeCache.load_by_key_path(
                    self.cache_key,
                    artifact_path,
                    self.cache_linemap,
                    constants.unwrap(self),
                )
                self.current_callable = code_cache.call
                self.recursively_apply_fns = getattr(
                    code_cache, "recursively_apply_fns", None
                )
                self.compiled_fn_runner = getattr(code_cache, "runner", None)
        except OSError:
            log.error("Failed to load artifact: %s", artifact_path)
            raise

        return artifact_path


@dataclasses.dataclass
class CompiledAOTI(OutputCode):
    """
    Class holding an AOTInductor compiled so.
    """

    filename: Union[str, list[Union[str, Weights]], torch.fx.GraphModule]
    device_type: str
    current_callable: Optional[Callable[..., Any]] = None
    _cached_files: dict[str, bytes] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not config.aot_inductor.link_libtorch:
            return

        if (
            torch._inductor.cpp_builder._IS_MACOS
            or torch._inductor.cpp_builder._IS_WINDOWS
        ):
            return

        if config.aot_inductor.cross_target_platform == "windows":
            return

        if config.aot_inductor.package_cpp_only:
            return

        if not config.enable_autograd_for_aot:
            return

        if isinstance(self.filename, list):
            current_callable = next(
                fn for fn in self.filename if isinstance(fn, str) and fn.endswith(".so")
            )
        else:
            current_callable = self.filename

        if isinstance(current_callable, torch.fx.GraphModule):
            # pyrefly: ignore [bad-assignment]
            self.current_callable = current_callable
            return

        if self.device_type.startswith("cuda"):
            current_callable = (
                torch._C._aoti.AOTIModelContainerRunnerCuda(  # type: ignore[call-arg]
                    current_callable,
                    1,
                    self.device_type,
                    "",
                    True,
                ).run  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        elif self.device_type == "cpu":
            current_callable = (
                torch._C._aoti.AOTIModelContainerRunnerCpu(  # type: ignore[call-arg]
                    current_callable, 1
                ).run  # type: ignore[attr-defined]
            )  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"unsupported device type {self.device_type}")
        self.current_callable = current_callable
        self._boxed_call = True
        for file in self._cached_files:
            if not os.path.exists(file):
                with open(file, "wb") as f:
                    f.write(self._cached_files[file])

    def __call__(self, inputs: Sequence[Any]) -> Any:
        if self.current_callable is None:
            raise RuntimeError("AOTInductor compiled so is not loaded")
        return self.current_callable(inputs)

    def prepare_for_serialization(self) -> None:
        self.current_callable = None
        self._cached_files = {}
        filenames: list[str] = []
        if isinstance(self.filename, list):
            filenames = self.filename  # type: ignore[assignment]
        elif isinstance(self.filename, str):
            filenames = [self.filename]
        for name in filenames:
            with open(name, "rb") as f:
                self._cached_files[name] = f.read()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["current_callable"] = None
        return state

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        if self.current_callable is None:
            self.__post_init__()

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        pass


@dataclasses.dataclass
class MockFXGraphCacheOutput(OutputCode):
    gm: Any = None

    def __post_init__(self) -> None:
        self._boxed_call = True

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        pass

    def __call__(self, inputs: Sequence[Any]) -> Any:
        return self.gm(inputs)

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        pass


@dataclasses.dataclass
class RegionalOutputCode(OutputCode):
    """
    OutputCode for regional inductor compilation results.

    Regional inductor returns a torch.fx.GraphModule that contains both
    compiled regions (via standalone_compile) and eager regions. This needs
    special serialization using GraphPickler instead of standard pickle.

    The serialization strategy stores the GraphModule as bytes using
    GraphPickler.dumps(), which handles FakeTensors, AOTCompiledArtifacts,
    and other special objects that standard pickle cannot handle.
    """

    # The serialized graph module as bytes (using GraphPickler)
    _serialized_graph_module: Optional[bytes] = None

    # The actual graph module (cleared during serialization)
    _graph_module: Optional[torch.nn.Module] = None

    # Optional filter for ops during serialization
    _ops_filter: Callable[[str], bool] | None = None

    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        ops_filter: Callable[[str], bool] = _ops_filter_safe,
    ):
        """
        Args:
            graph_module: The torch.fx.GraphModule returned by regional_inductor
            ops_filter: Optional filter function for op names during serialization.
                If provided, only ops whose name passes the filter will be serialized.
        """
        super().__init__()
        self._graph_module = graph_module
        self._serialized_graph_module = None
        self._serialized_wrappers = []
        self._boxed_call = True
        _, module = self._unwrap_graph_module()
        self._inner_boxed_call = isinstance(
            module.graph._codegen, torch.fx.graph._BoxedCodeGen
        )
        self._ops_filter = ops_filter

    def __call__(self, inputs: list[Any]) -> Any:
        """
        Execute the regional compiled graph.

        Args:
            inputs: A mutable list of inputs. Must be a list (not an arbitrary
                Sequence) because the boxed calling convention allows the callee
                to mutably clear the list when inputs become dead, enabling early
                memory deallocation.
        """
        if self._graph_module is None:
            raise RuntimeError(
                "RegionalOutputCode has no graph module loaded. "
                "Did you forget to call post_compile()?"
            )

        if self._inner_boxed_call:
            return self._graph_module(inputs)
        return self._graph_module(*inputs)

    @property
    def graph(self):
        _, module = self._unwrap_graph_module()
        return module.graph

    def _unwrap_graph_module(
        self,
    ) -> tuple[list[tuple[Callable, dict[str, Any]]], torch.fx.GraphModule]:
        module = self._graph_module
        serialized_wrappers = []
        if isinstance(module, torch._dynamo.OptimizedModule):
            dynamo_ctx = module.dynamo_ctx
            assert isinstance(dynamo_ctx, torch._dynamo.eval_frame.DisableContext)
            serialized_wrappers.append(
                (
                    torch._dynamo.disable,
                    {"reason": dynamo_ctx.msg, "wrapping": dynamo_ctx.wrapping},
                )
            )
            module = module._orig_mod
        assert isinstance(module, torch.fx.GraphModule)
        return serialized_wrappers, module

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        constants: CompiledFxGraphConstants,
        graph_kwargs: _CompileFxKwargs,
    ) -> None:
        """
        Post-compile processing for regional inductor.

        This deserializes the GraphModule from bytes using GraphPickler,
        extracting the fake_mode from example_inputs.
        """
        if self._graph_module is not None:
            return
        assert self._serialized_graph_module is not None
        # Get fake mode from example inputs
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode(example_inputs)
        if fake_mode is None:
            raise RuntimeError(
                "Could not detect fake mode from example inputs. "
                "Regional inductor requires fake mode for deserialization."
            )

        # Deserialize the graph module
        from torch.fx._graph_pickler import GraphPickler

        gm = GraphPickler.loads(self._serialized_graph_module, fake_mode)
        assert isinstance(gm, torch.fx.GraphModule)
        gm.recompile()
        for fn, kwargs in reversed(self._serialized_wrappers):
            gm = fn(gm, **kwargs)
        self._graph_module = gm

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        """Regional inductor doesn't use triton bundles directly."""

    def prepare_for_serialization(self) -> None:
        """
        Prepare for serialization by converting the GraphModule to bytes.

        This uses GraphPickler to serialize the graph module since it contains
        special objects like FakeTensors and AOTCompiledArtifacts that need
        custom pickling.
        """
        if self._graph_module is not None:
            from torch.fx._graph_pickler import GraphPickler, Options

            self._serialized_graph_module = None
            self._serialized_wrappers, graph_module = self._unwrap_graph_module()

            self._serialized_graph_module = GraphPickler.dumps(
                graph_module,
                options=Options(
                    ops_filter=self._ops_filter,
                ),
            )
            # Clear the graph module to avoid pickling it with standard pickle
            self._graph_module = None
