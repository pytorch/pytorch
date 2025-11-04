# mypy: allow-untyped-defs
"""
This module provides result classes for AOT Autograd compilation.

Similar to how torch._inductor.output_code provides OutputCode classes for inductor
compilation results, this module provides AOTAutogradResult classes that represent
the compiled artifacts produced by AOT Autograd.

These results are:
- Serializable: can be saved/loaded from disk without recompilation
- Addressable: can be stored in caches with keys for later retrieval
- Reusable: can be used for both caching and ahead-of-time compilation (precompile)

The main result types are:
- GenericAOTAutogradResult: Abstract base for all AOT Autograd results
- AOTAutogradResult: Regular result that references FxGraphCache entries
- BundledAOTAutogradResult: Result that bundles the entire compiled code directly
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import Any, Generic, Optional, TYPE_CHECKING, TypeVar

import torch
from torch._dynamo.precompile_context import BackendCacheArtifact
from torch._inductor.codecache import FxGraphCache
from torch._inductor.output_code import (
    CompiledFxGraph,
    CompiledFxGraphConstants,
    OutputCode,
)
from torch._inductor.utils import should_use_remote_fx_graph_cache

from .runtime_wrappers import (
    AOTDispatchAutograd,
    AOTDispatchSubclassWrapper,
    CachedAutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    FunctionalizedRngRuntimeWrapper,
    post_compile,
    RuntimeWrapper,
    SerializableCompiledFunction,
    SubclassMeta,
)
from .schemas import AOTAutogradCacheInfo  # noqa: F401
from .utils import simple_wraps


if TYPE_CHECKING:
    from torch._inductor.compile_fx import _CompileFxKwargs

    from .schemas import AOTConfig, ViewAndMutationMeta

log = logging.getLogger(__name__)


TOut = TypeVar("TOut", bound=OutputCode)


class InductorOutput(ABC, Generic[TOut]):
    """
    Class representing a single inductor output
    """

    @abstractmethod
    def pre_save(self) -> None: ...

    @abstractmethod
    def load(self, example_inputs) -> TOut: ...

    @abstractmethod
    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut: ...


TOutputCode = TypeVar("TOutputCode", bound=OutputCode)


@dataclass
class BundledOutputCodeLoadable(InductorOutput[TOutputCode], Generic[TOutputCode]):
    """
    A generic wrapper for OutputCode objects that are bundled directly in the cache
    (rather than looked up via FxGraphCache).

    This works for any OutputCode subclass (CompiledFxGraph, RegionalOutputCode, etc.)
    """

    result: TOutputCode

    def pre_save(self) -> None:
        disk_result = copy(self.result)
        disk_result.prepare_for_serialization()
        self.result = disk_result
        return

    def load(self, example_inputs) -> TOutputCode:
        self.example_inputs = example_inputs
        return self.result

    def post_compile(
        self, result: TOutputCode, fx_config: _CompileFxKwargs
    ) -> TOutputCode:
        constants = CompiledFxGraphConstants()

        # Special handling for CompiledFxGraph - needs FxGraphCache.cache_hit_post_compile
        if isinstance(result, CompiledFxGraph):
            graph, cache_info = FxGraphCache.cache_hit_post_compile(
                result, {}, constants
            )
            if graph is None:
                raise RuntimeError("Failed to reload cache entry from disk")
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_bundled_cache_hit",  # always a hit
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )
            result = graph  # type: ignore[assignment]

        # Run normal post compile
        result.post_compile(self.example_inputs, constants, fx_config)
        return result


# Backwards compatibility alias
CompiledFxGraphLoadable: type[BundledOutputCodeLoadable[CompiledFxGraph]] = (
    BundledOutputCodeLoadable[CompiledFxGraph]
)


@dataclass
class FxGraphCacheLoadable(InductorOutput[CompiledFxGraph]):
    fx_graph_cache_info: tuple[str, list[str]]
    fx_graph_guard_expr: Optional[str]

    def pre_save(self):
        return

    def _is_backward(self) -> bool:
        return False

    def load(self, example_inputs) -> CompiledFxGraph:
        from .autograd_cache import FXGraphCacheMiss

        # [Note: AOTAutogradCache and FXGraphCache Guard interactions]
        # As mentioned, AOTAutograd takes in the symint inputs from dynamo's list of arguments.
        # FXGraphCache serializes guards that are needed in the shape_env based on these symint inputs to the graph.
        # The invariant that AOTAutograd uses here is that the sources for symints given to it by dynamo are exactly
        # the same as the ones it passes to inductor, for both the forward and backward passes.
        # (This does not mean that the tensor values passed in are the same: only that their symints are).
        # That is, AOTAutograd and Inductor never create new guards based on symints with different sources
        # than those passed to it by inductor.
        # We pass the post compile function, which sets various fx_config boxed values,
        # so we can call it only after we're sure both forward and backward have
        # Clear CompiledTritonKernels before loading from FXGraphCache
        torch._inductor.async_compile.CompiledTritonKernels.cache_clear()
        remote_cache = None
        constants = CompiledFxGraphConstants()
        if should_use_remote_fx_graph_cache():
            remote_cache = FxGraphCache.get_remote_cache()
        (cache_key, debug_lines) = self.fx_graph_cache_info

        def check_exact_guard_match(guard_expr, _hints):
            """
            AOTAutogradCache tracks its own guards, so we just need to treat these guard expressions as a second
            cache key of sorts: we just check for equality, i.e. the FXGraphCache entry with
            the exact same guards as we originally saved into the cache.
            """
            return guard_expr == self.fx_graph_guard_expr

        result, cache_info = FxGraphCache.load_with_key(
            cache_key,
            debug_lines,
            example_inputs,
            local=True,
            remote_cache=remote_cache,
            is_backward=self._is_backward(),
            constants=constants,
            evaluate_guards=check_exact_guard_match,
        )
        if result is None:
            log.info("FXGraphCache cache miss for key %s", self.fx_graph_cache_info)
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "fx_graph_cache_miss",  # always a hit
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )

            raise FXGraphCacheMiss

        # No need to log chromium event because AOTAutograd will log that immediately for us
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_cache_hit",  # always a hit
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(cache_info),
        )
        self.example_inputs = example_inputs
        self.constants = constants
        return result

    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph:
        """
        Called after FXGraphCacheLoadable.load, mutates fx_config
        """
        result.post_compile(self.example_inputs, self.constants, fx_config)
        return result


@dataclass
class CompiledForward(FxGraphCacheLoadable):
    """
    Cacheable entry for a forward function
    """

    def _is_backward(self) -> bool:
        return False


@dataclass
class GenericCompiledBackward(InductorOutput[TOut]):
    # Used by AOTDispatchAutograd.post_compile
    backward_state_indices: list[int]
    num_symints_saved_for_bw_: int


@dataclass
class CompiledBackward(GenericCompiledBackward[CompiledFxGraph], FxGraphCacheLoadable):
    """
    Cacheable entry for a forward function
    """

    def _is_backward(self) -> bool:
        return True

    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph:
        compiled_bw = super().post_compile(result, fx_config)
        # See note [Wrapping bw_compiler in disable]
        # This is done by _wrapped_bw_compiler in torch/_dynamo/backends/common.py
        # But since on cache hit we do not call the bw_compiler, we need to reapply the disable
        return torch._dynamo.disable(  # type: ignore[return-value]
            compiled_bw, reason="do not trace generated backwards pass"
        )


# Generic bundled forward/backward classes that work with any OutputCode type
@dataclass
class BundledCompiledForward(
    BundledOutputCodeLoadable[TOutputCode], Generic[TOutputCode]
):
    """
    Generic forward function for bundled compilation.
    Works with any OutputCode type (CompiledFxGraph, RegionalOutputCode, etc.)
    """


@dataclass
class BundledCompiledBackward(
    GenericCompiledBackward[TOutputCode],
    BundledOutputCodeLoadable[TOutputCode],
    Generic[TOutputCode],
):
    """
    Generic backward function for bundled compilation.
    Works with any OutputCode type (CompiledFxGraph, RegionalOutputCode, etc.)
    """

    def post_compile(
        self, result: TOutputCode, fx_config: _CompileFxKwargs
    ) -> TOutputCode:
        compiled_bw = super().post_compile(result, fx_config)
        # See note [Wrapping bw_compiler in disable]
        # This is done by _wrapped_bw_compiler in torch/_dynamo/backends/common.py
        # But since on cache hit we do not call the bw_compiler, we need to reapply the disable
        return torch._dynamo.disable(  # type: ignore[return-value]
            compiled_bw, reason="do not trace generated backwards pass"
        )


@dataclass
class SerializedGraphModule:
    fn: Callable[[dict[Any, Any], str], torch.nn.Module]
    args: tuple[Any, ...]

    def __init__(self, gm: torch.fx.GraphModule):
        self.fn, self.args = gm.__reduce__()

    def deserialize(self) -> torch.fx.GraphModule:
        gm = self.fn(*self.args)
        assert isinstance(gm, torch.fx.GraphModule)
        return gm


def serialize_graph_module(gm: torch.fx.GraphModule) -> SerializedGraphModule:
    # NOTE: mutates the graph module
    gm.meta = {}
    for node in gm.graph.nodes:
        node.meta = {}
    return SerializedGraphModule(gm)


TForward = TypeVar("TForward", bound=InductorOutput)
TBackward = TypeVar("TBackward", bound=GenericCompiledBackward)


@dataclass
class GenericAOTAutogradResult(Generic[TForward, TBackward]):
    """A single result from AOT Autograd compilation, genericized by Forward and Backward types.

    A TForward is always an InductorOutput of some sort, which represents the
    forward graph of the compile.
    A TBackward is an InductorOutput + metadata about the backward, useful for specific
    backward-only wrappers. This type is encapsulated by GenericCompiledBackward.

    Each AOTAutogradResult is essentially parameterized by 1. the method of loading
    from the cache (either Bundled or UnBundled), and 2. The type of the output. For now,
    the only type of output we support is Python Wrapper output, i.e. OutputCode.CompiledFxGraph,
    but the same technique works for C++ wrapper code; we'd just add an extra InductorOutput type.
    """

    # Forward and Backward info
    compiled_fw: TForward
    compiled_bw: Optional[TBackward]

    # Code of the joint graph using print_readable()
    # Used for logging purposes
    aot_joint_graph_str: Optional[str]
    aot_forward_graph_str: Optional[str]
    aot_backward_graph_str: Optional[str]

    # Runtime_metadata saved right before compilation
    runtime_metadata: ViewAndMutationMeta

    # Wrappers that run after each aot_dispatch_* function
    dispatch_wrappers: list[CompilerWrapper]

    # Used by AOTSubclassWrapper
    maybe_subclass_meta: Optional[SubclassMeta]
    num_fw_outs_saved_for_bw: Optional[int]

    # Used by RuntimeWrapper
    indices_of_inps_to_detach: list[int]

    # Time taken to trace/compile the forward
    # forward_time_taken includes AOTAutograd tracing time + inductor compilation time
    # backward_time_taken is essentially just the time inductor took to compile
    forward_time_taken_ns: int
    backward_time_taken_ns: int

    # Used by standalone_compile
    sanitized_aot_config: AOTConfig

    guards_expr: Optional[str]

    # Used by Compiled Autograd
    serialized_bw_module: Optional[SerializedGraphModule]

    def pre_save(self):
        """
        Perform any preparations to make the result ready for serialization.
        """
        self.compiled_fw.pre_save()
        if self.compiled_bw is not None:
            self.compiled_bw.pre_save()

    # Turn result into the original callable
    def wrap_post_compile(
        self,
        args: list[torch.Tensor],
        aot_config: AOTConfig,
        fx_config: _CompileFxKwargs,
    ) -> Callable:
        """
        This function takes a result and carefully reconstructs the original callable
        that AOTAutograd returned the first time it was run. It does this by running the various
        post compile steps that AOTAutograd runs on its compiled artifact after running the fw/bw compilers.

        In the inference path, this consists of the Subclass, FunctionalzedRngRuntime, and RuntimeWrappers.
        In the autograd path, this consists of AOTAutogradDispatch.post_compile.

        The steps here should match exactly the steps that are run in aot_dispatch_base and aot_dispatch_autograd.

        Notably absent from the cached path are:
        - DebugAssertWrapper
        - FakifiedOutWrapper

        Which we'll handle separately later on, if necessary.
        """
        from torch._dynamo.utils import CompileEventLogger, dynamo_timed

        # Log the output of AOTAutogradCache
        if aot_config.enable_log:
            # TODO: maybe also log to aot_graphs_log
            # Unfortunately aot_graphs_log uses
            # slightly different formatting though
            if self.aot_joint_graph_str is not None:
                torch._logging.trace_structured(
                    "aot_joint_graph", payload_fn=lambda: self.aot_joint_graph_str
                )

            if self.aot_forward_graph_str is not None:
                from torchgen.utils import dataclass_repr

                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "aot_forward_graph_fw_metadata",
                        "encoding": "string",
                    },
                    payload_fn=lambda: dataclass_repr(self.runtime_metadata),
                )
                if self.maybe_subclass_meta is not None:
                    torch._logging.trace_structured(
                        "artifact",
                        metadata_fn=lambda: {
                            "name": "aot_forward_graph_fw_subclass_metadata",
                            "encoding": "string",
                        },
                        payload_fn=lambda: dataclass_repr(self.maybe_subclass_meta),
                    )

                # It's called an inference graph if not running with autograd
                name = (
                    "aot_forward_graph"
                    if self.aot_backward_graph_str is not None
                    else "aot_inference_graph"
                )
                torch._logging.trace_structured(
                    name, payload_fn=lambda: self.aot_forward_graph_str
                )

            if self.aot_backward_graph_str is not None:
                torch._logging.trace_structured(
                    "aot_backward_graph", payload_fn=lambda: self.aot_backward_graph_str
                )
        with dynamo_timed("AOTAutogradCache.inductor_load"):
            compiled_fw_func = self.compiled_fw.load(args)
            compiled_bw_func = None
            if self.compiled_bw is not None:
                compiled_bw_func = self.compiled_bw.load(args)
                needs_autograd = True
                CompileEventLogger.try_add_pt2_compile(
                    "backend_compile", dispatch_mode="autograd"
                )
                # Now that we've loaded forward and backward, call post compile on both
                # This avoids setting things like BoxedBools in fx_config until
                # after both forward and backward cache hit
                fw_fx_config: _CompileFxKwargs = {
                    **fx_config,
                    "is_backward": False,
                }
                bw_fx_config: _CompileFxKwargs = {
                    **fx_config,
                    "is_backward": True,
                }
                compiled_fw_func = self.compiled_fw.post_compile(
                    compiled_fw_func, fw_fx_config
                )
                compiled_bw_func = self.compiled_bw.post_compile(
                    compiled_bw_func, bw_fx_config
                )
            else:
                inference_fx_config: _CompileFxKwargs = {
                    **fx_config,
                    "is_backward": False,
                }

                needs_autograd = False
                CompileEventLogger.try_add_pt2_compile(
                    "backend_compile", dispatch_mode="inference"
                )
                compiled_fw_func = self.compiled_fw.post_compile(
                    compiled_fw_func, inference_fx_config
                )

        # Wrap the forward function in post compile wrappers
        compiled_fw_func = AOTDispatchSubclassWrapper(
            trace_joint=needs_autograd,
            fw_only=None,
            maybe_subclass_meta=self.maybe_subclass_meta,
            num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
        ).post_compile(
            compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
        )

        req_subclass_dispatch = self.maybe_subclass_meta is not None
        CompileEventLogger.try_add_pt2_compile(
            "backend_compile", requires_subclass_dispatch=req_subclass_dispatch
        )

        # In autograd case, functionalizedRngWrapper should not modify outs
        return_new_outs = not needs_autograd
        compiled_fw_func = FunctionalizedRngRuntimeWrapper(
            return_new_outs=return_new_outs
        ).post_compile(
            compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
        )
        disable_amp = torch._C._is_any_autocast_enabled()

        if needs_autograd:
            assert self.compiled_bw is not None

            cached_lazy_backward = None
            if self.serialized_bw_module is not None:
                cached_lazy_backward = CachedAutogradLazyBackwardCompileInfo(
                    self.serialized_bw_module.deserialize
                )
            # This function is run on both cache miss and cache hit, either here
            # or in aot_dispatch_autograd. On a cache hit,
            # 1. the bw is already compiled
            # 2. we don't need to save to the cache again
            # so those corresponding arguments are set to None.
            compiled_function = AOTDispatchAutograd.post_compile(
                compiled_fw_func,
                compiled_bw_func,
                self.maybe_subclass_meta,
                self.compiled_bw.num_symints_saved_for_bw_,
                self.compiled_bw.backward_state_indices,
                disable_amp,
                self.indices_of_inps_to_detach,
                cached_lazy_backward,
                aot_config,
                fw_metadata=self.runtime_metadata,
                try_save_cache_entry=None,
            )

        else:
            compiled_function = RuntimeWrapper(
                indices_of_inps_to_detach=self.indices_of_inps_to_detach,
                trace_joint=False,
                disable_amp=disable_amp,
            ).post_compile(
                compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
            )

        # Add serialization function back onto object
        compiled_function, _ = post_compile(
            self.dispatch_wrappers,
            compiled_function,
            aot_config,
            runtime_metadata=self.runtime_metadata,
        )

        # Now that we're pretty sure it's a successful load, add guards
        # to the existing shape environment from the cache
        if self.guards_expr:
            from .autograd_cache import AOTAutogradCache

            symints = AOTAutogradCache._filter_backed_symints(args)
            check = bool(AOTAutogradCache.evaluate_guards(self.guards_expr, symints))
            assert check is True

        return compiled_function


class AOTAutogradResult(GenericAOTAutogradResult[CompiledForward, CompiledBackward]):
    """
    Regular AOTAutogradResult: saves the forward/backward FxGraphCache keys
    and looks them up in FxGraphCache on load
    """


class BundledAOTAutogradResult(
    GenericAOTAutogradResult[
        BundledCompiledForward[TOutputCode], BundledCompiledBackward[TOutputCode]
    ],
    Generic[TOutputCode],
):
    """
    Generic AOTAutogradResult where we bundle the entire OutputCode directly
    (rather than looking it up via FxGraphCache).

    This works with any OutputCode type:
    - CompiledFxGraph: Traditional inductor compilation
    - RegionalOutputCode: Regional inductor compilation with GraphPickler serialization
    - Any future OutputCode subclasses

    Type parameter:
        TOutputCode: The OutputCode subclass (e.g., CompiledFxGraph, RegionalOutputCode)

    Usage with CompiledFxGraph:
        entry = BundledAOTAutogradResult[CompiledFxGraph](
            compiled_fw=BundledCompiledForward(result=CompiledFxGraph(...)),
            compiled_bw=BundledCompiledBackward(
                result=CompiledFxGraph(...),
                backward_state_indices=[...],
                num_symints_saved_for_bw_=...,
            ),
            ...
        )

    Usage with RegionalOutputCode:
        entry = BundledAOTAutogradResult[RegionalOutputCode](
            compiled_fw=BundledCompiledForward(result=RegionalOutputCode(gm)),
            compiled_bw=BundledCompiledBackward(
                result=RegionalOutputCode(gm),
                backward_state_indices=[...],
                num_symints_saved_for_bw_=...,
            ),
            ...
        )
    """


def deserialize_bundled_cache_entry(entry: BundledAOTAutogradResult) -> Callable:
    from copy import deepcopy

    from torch._inductor.cudagraph_utils import BoxedDeviceIndex
    from torch._inductor.utils import BoxedBool

    # In the precompile use case, guards are already serialized
    # by dynamo, so we don't need to add them to the environment
    entry.guards_expr = None
    # TODO: this isn't exactly right, because cudagraphs needs to be a shared config
    # which is set by compile_fx. But in precompile, we never actually call compile_fx
    # so we don't have a place to track cudagraphs here.
    cudagraphs = BoxedBool(torch._inductor.config.triton.cudagraphs)
    boxed_forward_device_index = BoxedDeviceIndex(None)
    # We need to make a clean copy of the cache entry
    # in case it needs to be serialized again
    serializable_copy = deepcopy(entry)

    from torch._subclasses import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    context = torch._guards.TracingContext.try_get()
    if context is None:
        # Create a clean environment when running fx graph post compile
        # if one is not available
        context = torch._guards.TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
    with torch._guards.tracing(context):
        compiled_fn = entry.wrap_post_compile(
            [],
            entry.sanitized_aot_config,
            {
                "cudagraphs": cudagraphs,
                "boxed_forward_device_index": boxed_forward_device_index,
            },
        )
    # Ensure the deserialized cache entry is still serializable

    compiled_fn = SerializableCompiledFunction(compiled_fn, lambda: serializable_copy)

    # TODO: this ignores flat_params, which can exist
    # if inline_builtin_nn_modules=False
    @simple_wraps(compiled_fn)
    def forward(*runtime_args: tuple[Any]):
        return compiled_fn(list(runtime_args))

    assert hasattr(compiled_fn, "serialize")
    forward.serialize = compiled_fn.serialize  # type: ignore[attr-defined]

    return forward


@dataclass
class BundledAOTAutogradCacheArtifact(BackendCacheArtifact[Callable]):
    def after_deserialization(self) -> Callable:
        return deserialize_bundled_cache_entry(self.content)
