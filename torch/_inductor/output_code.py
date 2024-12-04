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
from typing import (
    Any,
    Callable,
    Counter,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import TypeAlias

import torch
from torch._dynamo.utils import counters
from torch._inductor.cudagraph_utils import (
    BoxedDeviceIndex,
    CudagraphCachedInfo,
    get_placeholder_info,
    log_cudagraph_skip_and_bump_counter,
)
from torch._inductor.utils import (
    align_inputs_from_check_idxs,
    BoxedBool,
    InputType,
    output_node,
    set_tracing_context_output_strides,
)

from . import config
from .runtime.autotune_cache import AutotuneCacheBundler


if TYPE_CHECKING:
    from torch._inductor import metrics
    from torch._inductor.graph import GraphLowering
    from torch.fx import GraphModule

    from .compile_fx import _CompileFxKwargs
    from .triton_bundler import TritonKernelArtifacts


@dataclasses.dataclass
class OutputCode:
    # TODO: Remove underscores here

    # None if the output is not remote cacheable
    _fx_graph_cache_key: Optional[str] = dataclasses.field(default=None, init=False)

    # How long it took to compile this OutputCode, end to end
    _time_taken_ns: Optional[int] = dataclasses.field(default=None, init=False)

    def __call__(self, inputs: Sequence[Any]) -> Any:
        raise NotImplementedError(type(self))

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        cudagraphs: BoxedBool,
        gm: GraphModule,
    ) -> None:
        raise NotImplementedError(type(self))

    # TODO: Get rid of this
    def set_triton_bundle(self, triton_bundle: Any) -> None:
        raise NotImplementedError(type(self))


_StrideExprStr: TypeAlias = str


def has_frozen_params(gm: torch.fx.GraphModule) -> bool:
    return getattr(gm, "_has_frozen_params", False)


# copy_ fails when trying to write to tensors with memory overlap,
# for expanded dimensions (a dimension which used to have size 1 -> ?)
# we can select one element from that dimension and write to it
# to achieve writing to all values of that dimension of the input tensor
def get_expanded_dims(t: torch.Tensor) -> List[int]:
    if not isinstance(t, torch.Tensor):
        return None
    return [i for i in range(t.ndim) if t.stride(i) == 0 and t.size(i) != 1]


def index_expanded_dims(t: torch.Tensor, expanded_dims: List[int]) -> torch.Tensor:
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


def cudagraph_post_compile(
    example_inputs: Sequence[InputType],
    compiled_graph: CompiledFxGraph,
    cudagraphs: BoxedBool,
    gm: Optional[torch.fx.GraphModule],
) -> None:
    """
    Checks for any reasons not to run cudagraphs and then
    runs it on compiled_graph.
    Mutates the `compiled_graph.current_callable` and `cudagraphs`
    """
    assert compiled_graph.current_callable is not None
    assert compiled_graph.cudagraph_info is not None
    cached_info = compiled_graph.cudagraph_info
    cudagraph_fail_reasons = cached_info.cudagraph_fail_reasons
    inputs_to_check = compiled_graph.inputs_to_check
    boxed_forward_device_index = compiled_graph.boxed_forward_device_index
    is_inference = compiled_graph.fx_kwargs["is_inference"]
    is_backward = compiled_graph.fx_kwargs["is_backward"]

    if not cudagraph_fail_reasons:
        fx_kwargs = compiled_graph.fx_kwargs
        static_input_idxs = fx_kwargs["static_input_idxs"]

        placeholders = cached_info.placeholders
        stack_traces = cached_info.stack_traces
        if not config.triton.cudagraph_trees:
            # Force specialize all inputs so that CUDA graphs will work
            for t in example_inputs:
                if isinstance(t, torch.SymInt):
                    int(t)  # guard

        if (
            boxed_forward_device_index is not None
            and not is_inference
            and not is_backward
        ):
            boxed_forward_device_index.set(next(iter(compiled_graph.device_idxs)))

        from .compile_fx import cudagraphify

        current_callable = compiled_graph.current_callable
        assert current_callable is not None
        compiled_graph.current_callable = cudagraphify(
            current_callable,
            static_input_idxs=static_input_idxs or (),
            device_index=next(iter(compiled_graph.device_idxs)),
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=tuple(compiled_graph.get_constants(gm).values()),
            placeholders=placeholders,
            mutated_input_idxs=tuple(compiled_graph.mutated_input_idxs),
        )

    else:
        BoxedBool.disable(cudagraphs)

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

            def compiled_artifact(new_inputs: List[Any]) -> Callable[..., Any]:
                manager.set_to_running_backward()  # type: ignore[union-attr]
                return compiled_graph_callable(new_inputs)

            compiled_graph.current_callable = compiled_artifact

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


def maybe_realign_inputs(
    ran_cudagraphs: BoxedBool,
    compiled_graph: CompiledFxGraph,
    inputs_to_check: Sequence[int],
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
            compiled_graph.current_callable, inputs_to_check
        )
        if new_callable is not compiled_graph.current_callable:
            compiled_graph.current_callable = new_callable


@dataclasses.dataclass
class CompiledFxGraph(OutputCode):
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: Optional[Callable[..., Any]]
    cache_key: str
    source_code: str = dataclasses.field(repr=False)  # Do not display source_code
    cache_linemap: Optional[List[Tuple[int, str]]]
    device_types: Set[str]
    device_idxs: Set[int]
    mutated_inputs: Set[str]
    mutated_input_idxs: Set[int]
    # We populate exactly one of the next two fields. In the common case, we store the
    # constant attirbutes in the cache entry and re-attach them to the module created in
    # PyCodeCache.load_by_key_path. In the case that the graph has frozen parameters,
    # however, we save the mapping from attribute names in the GraphLowering to the
    # original name of the attribute in the GraphModule. When we create the module from
    # the cache entry, we then look up the constants from the current GraphModule. This
    # scheme allows us to support caching with freezing.
    allocated_constant_name: Optional[Dict[str, str]]
    constants: Optional[Dict[str, torch.Tensor]]
    torchbind_constants: Dict[str, torch._C.ScriptObject]
    output_strides: Optional[List[Optional[Tuple[_StrideExprStr, ...]]]]
    disabled_cudagraphs_reason: Optional[str]
    metrics_deltas: metrics.CachedMetricsDeltas
    counter_deltas: Counter[str]
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: Optional[str]

    cudagraph_info: Optional[CudagraphCachedInfo]
    fx_kwargs: _CompileFxKwargs
    inputs_to_check: Sequence[int]
    boxed_forward_device_index: Optional[BoxedDeviceIndex]

    _boxed_call: Optional[bool] = None
    _triton_bundle: Optional[List[TritonKernelArtifacts]] = None

    def __init__(
        self,
        current_callable: Optional[Callable[..., Any]],
        graph: GraphLowering,
        gm: torch.fx.GraphModule,
        output_strides: List[Optional[Tuple[_StrideExprStr, ...]]],
        disabled_cudagraphs_reason: Optional[str],
        metrics_deltas: metrics.CachedMetricsDeltas,
        counter_deltas: Counter[str],
        cudagraphs: BoxedBool,
        example_inputs: Sequence[InputType],
        static_input_idxs: Sequence[int],
        fx_kwargs: _CompileFxKwargs,
        inputs_to_check: Sequence[int],
        boxed_forward_device_index: Optional[BoxedDeviceIndex],
    ) -> None:
        self.current_callable = current_callable
        self.cache_key = graph.cache_key
        if graph.cache_path:
            with open(graph.cache_path) as f:
                self.source_code = f.read()
        self.cache_linemap = graph.cache_linemap
        # TODO - ordered set
        self.device_types = set(graph.device_types)
        self.device_idxs = set(graph.device_idxs)
        self.mutated_inputs = set(graph.mutated_inputs)
        self.mutated_input_idxs = set(graph.mutated_input_idxs)
        if has_frozen_params(gm):
            self.allocated_constant_name = graph.allocated_constant_name
            self.constants = None
        else:
            self.allocated_constant_name = None
            self.constants = graph.constants
        self.torchbind_constants = graph.torchbind_constants
        self.output_strides = output_strides
        self.disabled_cudagraphs_reason = disabled_cudagraphs_reason
        self.metrics_deltas = metrics_deltas
        self.counter_deltas = counter_deltas
        self.guards_expr = None
        self.cudagraph_info = None
        self.fx_kwargs = {}
        self.inputs_to_check = ()
        self.boxed_forward_device_index = None

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
                            self,
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
                            isinstance(t, (torch.Tensor, torch.SymInt))
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
                    for arg in output.args[0]
                ]
                cudagraph_fail_reasons = [s for b, s in cudagraph_tests if not b]
                placeholders = tuple(get_placeholder_info(gm.graph))
                cudagraph_info = CudagraphCachedInfo(
                    placeholders, stack_traces, cudagraph_fail_reasons
                )

        self.cudagraph_info = cudagraph_info
        self.inputs_to_check = inputs_to_check
        self.fx_kwargs = fx_kwargs
        # TODO: should this be part of fx_kwargs
        self.boxed_forward_device_index = boxed_forward_device_index

        # aot autograd needs to know to pass in inputs as a list
        self._boxed_call = True

    def __call__(self, inputs: Sequence[Any]) -> Any:
        assert self.current_callable is not None
        try:
            return self.current_callable(inputs)
        finally:
            AutotuneCacheBundler.end_compile()

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        cudagraphs: BoxedBool,
        gm: GraphModule,
    ) -> None:
        """
        Run a set of post processing steps after loading from the cache. These involve:
         - Setting the tracing context output strides
         - Running cudagraphs if enabled
         - Realigning inputs

        This runs whether or not we have a cache hit, and always runs directly after we get a CompiledFxGraph.
        The results of this function are *not* saved in the cache itself.
        """
        set_tracing_context_output_strides(example_inputs, self)

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
                cudagraph_post_compile(
                    example_inputs,
                    self,
                    cudagraphs,
                    gm,
                )
        inputs_to_check = self.inputs_to_check
        # cudagraphs could have been disabled from the earlier conditions
        # so we still need to realign inputs if that happens
        maybe_realign_inputs(
            cudagraphs,
            self,
            inputs_to_check,
        )

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        self._triton_bundle = triton_bundle

    def get_constants(
        self, gm: Optional[torch.fx.GraphModule]
    ) -> Dict[str, torch.Tensor]:
        """
        Get the constant attributes.
        """
        # Normal case: The constants are stored in the entry.
        if self.constants is not None:
            return self.constants

        # Freezing case: Look up the constants from attributes on the GraphModule using
        # the allocated_constant_name map.
        assert gm is not None
        assert self.allocated_constant_name is not None
        constants = {
            name: getattr(gm, orig_name)
            for name, orig_name in self.allocated_constant_name.items()
        }
        return constants


def _typecheck_CompiledFxGraph(h: CompiledFxGraph) -> OutputCode:
    return h


@dataclasses.dataclass
class CompiledAOTI(OutputCode):
    """
    Class holding an AOTInductor compiled so.
    """

    filename: Union[str, List[str]]

    def __call__(self, inputs: Sequence[Any]) -> Any:
        raise NotImplementedError("NYI")

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        cudagraphs: BoxedBool,
        gm: GraphModule,
    ) -> None:
        pass

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        pass


def _typecheck_CompiledAOTI(h: CompiledAOTI) -> OutputCode:
    return h


@dataclasses.dataclass
class MockFXGraphCacheOutput(OutputCode):
    gm: Any = None

    def __post_init__(self) -> None:
        self._boxed_call = True

    def post_compile(
        self,
        example_inputs: Sequence[InputType],
        cudagraphs: BoxedBool,
        gm: GraphModule,
    ) -> None:
        pass

    def __call__(self, inputs: Sequence[Any]) -> Any:
        return self.gm(inputs)

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        pass
