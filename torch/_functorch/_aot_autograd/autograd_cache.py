# mypy: allow-untyped-defs
"""
Utils for caching the outputs of AOTAutograd
"""

from __future__ import annotations

import base64
import contextlib
import functools
import json
import logging
import os
import pickle
import shutil
import time
import traceback
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import override

import torch
from torch._dynamo.precompile_context import PrecompileCacheArtifact, PrecompileContext
from torch._dynamo.trace_rules import torch_non_c_binding_in_graph_functions
from torch._dynamo.utils import (
    chromium_event_log_active,
    CompileEventLogger,
    counters,
    dynamo_timed,
)
from torch._functorch import config
from torch._inductor.codecache import (
    _ident,
    add_ephemeral_timeout_increase_for_distributed,
    BypassFxGraphCache,
    create_cache,
    extract_tensor_metadata_for_cache_key,
    FxGraphCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    GuardedCache,
    sha256_hash,
    write_atomic,
)
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.output_code import (
    CompiledFxGraph,
    CompiledFxGraphConstants,
    OutputCode,
)
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import should_use_remote_fx_graph_cache
from torch._logging import LazyString
from torch._utils_internal import log_cache_bypass
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.fx.experimental.symbolic_shapes import hint_int
from torch.utils._triton import has_triton_package
from torchgen.utils import dataclass_repr

from .runtime_wrappers import (
    AOTDispatchAutograd,
    AOTDispatchSubclassWrapper,
    CachedAutogradLazyBackwardCompileInfo,
    CompilerWrapper,
    FunctionalizedRngRuntimeWrapper,
    post_compile,
    RuntimeWrapper,
    SubclassMeta,
)
from .schemas import AOTAutogradCacheInfo, AOTConfig, ViewAndMutationMeta  # noqa: F401


if TYPE_CHECKING:
    from torch._inductor.compile_fx import _CompileFxKwargs
    from torch._inductor.remote_cache import JsonDataTy, RemoteCache
    from torch._inductor.utils import BoxedBool
    from torch.fx.node import Node

log = logging.getLogger(__name__)


class BypassAOTAutogradCache(Exception):
    pass


# Used to signify when FXGraphCache missed when AOTAutogradCache uses it
class FXGraphCacheMiss(BypassAOTAutogradCache):
    pass


def should_use_remote_autograd_cache():
    if torch._inductor.config.force_disable_caches:
        return False
    if config.enable_remote_autograd_cache is not None:
        return config.enable_remote_autograd_cache
    if not config.is_fbcode():
        return False

    if torch._utils_internal.is_fb_unit_test():
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    jk_name = "pytorch/remote_cache:aot_autograd_cache_version"

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(jk_name)


def should_use_local_autograd_cache():
    if torch._inductor.config.force_disable_caches:
        return False
    return config.enable_autograd_cache


def should_bundle_autograd_cache():
    return config.bundled_autograd_cache or torch._dynamo.config.caching_precompile


def check_node_safe(node: Node):
    """
    Checks that the node only uses supported operators. We are starting with very
    conservative cacheability constraints, and incrementally adding more support as we expand.

    [Note: AOTAutograd Cacheability checks]
    - Our cache key is computed from the FX graph produced by Dynamo and the input example values
    - A node is "safe" if the same cache key results in a compiled artifact that has the same behavior
        (i.e, the set of inputs that go into our cache key is sufficient to distinguish its behavior)

    To accomplish this safety check, we consider the following functions to be safe:
        - Public functions under modules torch, torch.functional, and torch.nn.functional: these are
        allowed in the graph by dynamo, so we can assume they are safe to cache.
        - method calls on base tensor types
        - Any call_module that dynamo deemed safe to allow AOTAutograd to trace
        - Non callable nodes, such as placeholder, output, get_attr

    The test suite test_aot_autograd_cache.py::AOTAutogradCachePicklerTests tries its best to fully cover/specify this behavior.
    """
    SAFE_TORCH_MODULES = ("torch.functional", "torch.nn.functional")
    SAFE_TORCH_FUNCTIONS = (
        "torch.Size",
        "torch.Tensor",
        "torch.sym_int",
        "torch._sym_sqrt",
        "torch.sym_float",
        "torch.sym_sum",
    )
    SAFE_NON_TORCH_FUNCTIONS = (
        "einops.einops.rearrange",
        "einops.einops.repeat",
    )

    def is_public_torch_api(target):
        # Don't blindly allow private functions in the torch namespace
        is_private = target.__name__.startswith("_")

        return (
            getattr(target, "__module__", None) in SAFE_TORCH_MODULES and not is_private
        )

    def is_safe_torch_function(target):
        """Allowlisted torch functions"""
        function_name = f"{target.__module__}.{target.__name__}"
        # Allow torch.autograd.function.FunctionCtx if custom autograd functions are allowed
        if function_name == "torch.autograd.function.FunctionCtx":
            return (
                torch._functorch.config.autograd_cache_allow_custom_autograd_functions
            )

        # Functions in torch_non_c_binding_in_graph_functions
        # are guaranteed to be cache safe.
        # See NOTE: [Cacheability of in-graph torch functions]
        return (
            function_name in torch_non_c_binding_in_graph_functions
            or function_name in SAFE_TORCH_FUNCTIONS
            or function_name in torch._inductor.config.unsafe_marked_cacheable_functions
        )

    def is_cacheable_function(target):
        if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
            return True
        if is_public_torch_api(target):
            return True
        # Technically, FXGraphCache._check_for_hop already checks this,
        # but better to error earlier anyway
        if isinstance(target, torch._ops.HigherOrderOperator):
            return target.cacheable()
        is_builtin_fun_or_type = type(target).__name__ == "builtin_function_or_method"
        if is_builtin_fun_or_type:
            return True
        if is_safe_torch_function(target):
            return True
        function_name = f"{target.__module__}.{target.__name__}"
        if function_name in SAFE_NON_TORCH_FUNCTIONS:
            return True
        return False

    def is_tensor(target):
        # Tensors always have example values in meta field
        return "example_value" in target.meta

    # I'd love to use a match statement here, but it wasn't introduced until py3.10
    if node.op == "call_function":
        if node.meta and node.meta.get("is_wrapped", False):
            # This is fx.wrap function
            # By default we BypassAOTAutogradCache for unknown functions,
            # But if user explicitly specified cache hash - allow to cache it.
            if node.meta.get("user_cache_hash", None):
                return

        if not is_cacheable_function(node.target):
            module = getattr(node.target, "__module__", None)
            name = getattr(node.target, "__name__", None)
            raise BypassAOTAutogradCache(
                f"Unsupported call_function target {node.target}. \n Function module: {module}, \nFunction name: {name}"
            )
    elif node.op == "call_method":
        method_name = node.target
        method_target = node.args[0]
        # Only support method calls on base tensors
        if not is_tensor(method_target):
            module = getattr(method_target, "__module__", None)
            name = getattr(method_target, "__name__", None)
            raise BypassAOTAutogradCache(
                f"Unsupported call_method target {method_target}. \nMethod module: {module}, \nMethod name: {name}"
            )
        if (
            type(method_name) != str
            and type(method_name).__name__ != "method_descriptor"
        ):
            raise BypassAOTAutogradCache(
                f"Unsupported call_method method {node.target}: {method_name}"
            )
    # Cache safe
    elif node.op in ("placeholder", "get_attr", "call_module", "output"):
        # Assumption today for call_module being a safe op:
        # (1) today the only call_module ops that can show up in a graph come from "built-in-nn-modules"
        # that dynamo assumes are safe to trace. If dynamo assumes they are safely to blindly trace, then
        # they should be safe to cache as well.
        # (2) in the steady-state (some time in H2?) we shouldn't see these anymore, once inline builtin nn modules by default
        # (3) We do not allow user made nn modules in the graph today, only function calls.
        pass
    else:
        raise BypassAOTAutogradCache(f"Unsupported node op {node.op}")


def check_cacheable(gm: torch.fx.GraphModule):
    """
    Checks that the graph module only uses supported operators
    """
    nodes = gm.graph.nodes
    if torch._inductor.config.freezing:
        raise BypassAOTAutogradCache("Cannot cache a graph with freezing enabled")

    if not (
        torch._inductor.config.fx_graph_cache or should_use_remote_fx_graph_cache()
    ):
        raise BypassAOTAutogradCache("FX graph cache is not enabled")

    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context and tracing_context.fakify_first_call:
        raise BypassAOTAutogradCache(
            "Won't cache a graph with fakify_first_call enabled"
        )
    for node in nodes:
        check_node_safe(node)

    # Saved tensors hooks are globally set subgraphs,
    # that are not used explicitly in the main graph.
    # They are inlined in aot_autograd graphs.
    # Subgraphs are only used for caching logic.
    if hasattr(gm, "saved_tensors_hooks_pack_0"):
        check_cacheable(gm.saved_tensors_hooks_pack_0)  # type: ignore[arg-type]
        # We have guarantee of unpack sugraph existence if pack subgraph exists
        check_cacheable(gm.saved_tensors_hooks_unpack_0)  # type: ignore[arg-type]


def check_metadata_cacheable(metadata: ViewAndMutationMeta):
    """
    When view replay is turned on, we bypass autograd cache if
    the output is aliased.
    """
    if config.view_replay_for_aliased_outputs:
        for info in metadata.output_info:
            if info.functional_tensor is not None:
                raise BypassAOTAutogradCache(
                    "Cannot cache a graph with functional tensor"
                )


class AOTAutogradCacheDetails(FxGraphHashDetails):
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs,
        aot_config: AOTConfig,
        fx_config: _CompileFxKwargs,
    ):
        # FxGraphHashDetails contains all the keys related to inductor. Also includes some system info
        self.aot_config = aot_config
        self.grad_enabled = torch.is_grad_enabled()
        self.disable_amp = torch._C._is_any_autocast_enabled()
        self.deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        self.autograd_config = config.save_config()
        self.saved_tensors_hooks_fx_wrap_cache_hashes: tuple[list[str], list[str]] = (
            [],
            [],
        )

        if hasattr(gm, "saved_tensors_hooks_pack_0"):

            def _add_wrapped_user_cache_hashes(_gm, _l):
                for node in _gm.graph.nodes:
                    if node.meta and node.meta.get("is_wrapped", False):
                        _l.append(node.meta["user_cache_hash"])

            _add_wrapped_user_cache_hashes(
                gm.saved_tensors_hooks_pack_0,
                self.saved_tensors_hooks_fx_wrap_cache_hashes[0],
            )
            _add_wrapped_user_cache_hashes(
                gm.saved_tensors_hooks_unpack_0,
                self.saved_tensors_hooks_fx_wrap_cache_hashes[1],
            )

        try:
            # FXGraphCache has constraints on what can be pickled in its inductor
            # config. Check that the gm is cacheable by inductor first,
            # and if it raises an exception, also bypass on our end.
            FxGraphCache._check_can_cache(gm)
            super().__init__(gm, example_inputs, fx_config, [])
        except BypassFxGraphCache as e:
            # Sometimes inductor configs are unpickleable and can fail
            raise BypassAOTAutogradCache(str(e)) from e


class AOTAutogradCachePickler(FxGraphCachePickler):
    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)
        self.dispatch_table: dict
        self.dispatch_table.update(
            {
                AOTConfig: functools.partial(self._reduce_aot_config),
                torch.Tensor: functools.partial(self._reduce_tensor),
            }
        )

    def _reduce_aot_config(self, aot_config: AOTConfig):
        """
        Reduce the config to a stable key for caching.
        """
        return (
            _ident,
            (
                aot_config.num_params_buffers,
                aot_config.keep_inference_input_mutations,
                aot_config.is_export,
                aot_config.no_tangents,
                aot_config.dynamic_shapes,
                aot_config.aot_autograd_arg_pos_to_source,
                aot_config.enable_log,
                aot_config.pre_dispatch,
            ),
        )

    def _reduce_tensor(self, tensor):
        """
        Reduce the tensor to a stable key for caching.
        """
        metadata = extract_tensor_metadata_for_cache_key(tensor)
        return (_ident, (metadata,))


@contextlib.contextmanager
def normalize_placeholder_names(gm: torch.fx.GraphModule):
    """
    Context manager that normalizes the placeholder names in the graph module.
    This is used while generating a cache key for AOTAutogradCache, so that two graphs
    that are isomorphic when normalizing names can hit the same cache entry.
    This is safe because nothing underneath AOTAutograd uses the node names on the
    original dynamo graph: AOTAutograd re-traces with its own nodes, and guards are
    in terms of original sources rather than placeholder names.
    """
    # Standalone inductor: we're bypassing AOTAutogradCache anyway, so return the graph
    # as-is
    if not config.autograd_cache_normalize_inputs or not hasattr(gm, "graph"):
        yield
        return

    # Track all the old state of placeholders
    old_placeholder_names = []
    old_used_names = copy(gm.graph._graph_namespace._used_names)
    i = 0
    for n in gm.graph.find_nodes(op="placeholder", sort=True):
        if n.type != torch.SymInt:
            # _rename renames the node in the body of the function,
            # but it doesn't change the raw name from node.target
            # So we also set the raw_name of node.target to a new placeholder name
            new_placeholder_name = f"p_{i}"
            old_placeholder_names.append((n.name, n.target))
            n.target = new_placeholder_name
            n._rename(new_placeholder_name)
            i += 1
    gm.recompile()
    try:
        yield
    finally:
        # Used_names contains all our old placeholder names,
        # so we clear it temporarily when we put them back
        gm.graph._graph_namespace._used_names = set()
        # Restore the placeholder names
        i = 0
        for n in gm.graph.find_nodes(op="placeholder", sort=True):
            if n.type != torch.SymInt:
                (name, target) = old_placeholder_names[i]
                n.target = target
                n._rename(name)
                i += 1
        assert i == len(old_placeholder_names)
        # Now restore the old namespace's used names
        gm.graph._graph_namespace._used_names = old_used_names
        gm.recompile()


def autograd_cache_key(
    gm: torch.fx.GraphModule,
    example_inputs,
    config: AOTConfig,
    fx_config: _CompileFxKwargs,
    # TODO: add args and parameters
) -> tuple[str, list[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """
    check_cacheable(gm)
    if has_triton_package():
        # Due to https://github.com/triton-lang/triton/issues/3729,
        # if triton is < 3.2.0, AOTAutogradCache may cause us to
        # attempt to load a cache entry without initializing
        # the CUDA context on the autograd thread.

        # Without caching, we naturally do this initialization when
        # tracing through the graph with the autograd engine.
        import triton

        if triton.__version__ < "3.2.0":
            raise BypassAOTAutogradCache("AOTAutogradCache requires triton 3.2.0")
    details = AOTAutogradCacheDetails(gm, example_inputs, config, fx_config)
    pickler = AOTAutogradCachePickler(gm)
    # The prefix distinguishes among the other kinds of objects we cache
    key = "a" + pickler.get_hash(details)
    debug_lines = pickler.debug_lines(details)
    log.debug(
        "Autograd graph cache hash details for key %s:\n%s",
        key,
        LazyString(lambda: "\n".join(debug_lines)),
    )
    return key, debug_lines


TOut = TypeVar("TOut", bound=OutputCode)


class InductorOutput(Generic[TOut], ABC):
    """
    Class representing a single inductor output
    """

    @abstractmethod
    def pre_save(self) -> None: ...

    @abstractmethod
    def load(self, example_inputs) -> TOut: ...

    @abstractmethod
    def post_compile(self, result: TOut, fx_config: _CompileFxKwargs) -> TOut: ...


@dataclass
class CompiledFxGraphLoadable(InductorOutput[CompiledFxGraph]):
    """
    A full compiled fx graph that doesn't need to lookup the FxGraphCache
    to run
    """

    result: CompiledFxGraph

    def pre_save(self) -> None:
        disk_compiled_graph = copy(self.result)
        disk_compiled_graph.prepare_for_serialization()
        self.result = disk_compiled_graph
        return

    def load(self, example_inputs) -> CompiledFxGraph:
        self.example_inputs = example_inputs

        return self.result

    def post_compile(
        self, result: CompiledFxGraph, fx_config: _CompileFxKwargs
    ) -> CompiledFxGraph:
        constants = CompiledFxGraphConstants()
        # Cache hit specific post compile
        graph, cache_info = FxGraphCache.cache_hit_post_compile(result, {}, constants)
        if graph is None:
            raise BypassAOTAutogradCache("Failed to reload cache entry from disk")
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_bundled_cache_hit",  # always a hit
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(cache_info),
        )
        # Run normal post compile
        graph.post_compile(self.example_inputs, constants, fx_config)
        return graph


@dataclass
class FxGraphCacheLoadable(InductorOutput[CompiledFxGraph]):
    fx_graph_cache_info: tuple[str, list[str]]
    fx_graph_guard_expr: Optional[str]

    def pre_save(self):
        return

    def _is_backward(self) -> bool:
        return False

    def load(self, example_inputs) -> CompiledFxGraph:
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


# Forward types don't have any extra parameters, so this is just a TypeAlias, in essence
class BundledCompiledForward(CompiledFxGraphLoadable):
    pass


@dataclass
class BundledCompiledBackward(
    GenericCompiledBackward[CompiledFxGraph], CompiledFxGraphLoadable
):
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
class GenericAOTAutogradCacheEntry(Generic[TForward, TBackward]):
    """A single entry into the cache, genericized by Forward and Backward types.

    A TForward is always an InductorOutput of some sort, which represents the
    forward graph of the compile.
    A TBackward is an InductorOutput + metadata about the backward, useful for specific
    backward-only wrappers. This type is encapsulated by GenericCompiledBackward.

    Each AOTAutogradCacheEntry is essentially parameterized by 1. the method of loading
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

    # Used by RuntimeWrapepr
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
        Perform any preparations to make the cache entry ready for serialization.
        """
        check_metadata_cacheable(self.runtime_metadata)
        self.compiled_fw.pre_save()
        if self.compiled_bw is not None:
            self.compiled_bw.pre_save()

    # Turn cache entry into the original callable
    def wrap_post_compile(
        self,
        args: list[torch.Tensor],
        aot_config: AOTConfig,
        fx_config: _CompileFxKwargs,
    ) -> Callable:
        """
        This function takes a cache entry and carefully reconstructs the original callable
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

        compiled_function, _ = post_compile(
            self.dispatch_wrappers,
            compiled_function,
            aot_config,
            runtime_metadata=self.runtime_metadata,
        )

        # Now that we're pretty sure it's a successful load, add guards
        # to the existing shape environment from the cache
        if self.guards_expr:
            symints = AOTAutogradCache._filter_backed_symints(args)
            check = bool(AOTAutogradCache.evaluate_guards(self.guards_expr, symints))
            assert check is True

        return compiled_function


class AOTAutogradCacheEntry(
    GenericAOTAutogradCacheEntry[CompiledForward, CompiledBackward]
):
    """
    Regular AOTAutogradCacheEntry: saves the forward/backward FxGraphCache keys
    and looks them up in FxGraphCache on load
    """


class BundledAOTAutogradCacheEntry(
    GenericAOTAutogradCacheEntry[BundledCompiledForward, BundledCompiledBackward]
):
    """
    AOTAutogradCacheEntry where we save the entire CompiledFxGraph instead
    of relying on cache keys from FxGraphCache
    """


@contextlib.contextmanager
def sanitize_gm_for_cache(gm: torch.fx.GraphModule):
    """
    Clears a few fields in a dynamo supplied Graph Module that are not stable between graph inputs, but don't
    affect inductor or aotdispatch correctness.

    These fields **can** be used by code calling into aotdispatch (namely, dynamo), so we can't null them out completely.

    To ensure that these fields are not accessed by inductor or aotdispatch, we clear them during AOTAutogradCache.load,
    and then put them back before returning. This way, we generate a cache key based off of a canonical graph
    without these fields, and also guarantee they aren't used to affect the cache's output.
    """
    # Mapping from each field to a default value
    IGNORED_FIELDS: dict[str, Any] = {
        "meta": {},  # metadata used by export
        "compile_subgraph_reason": None,  # Used by dynamo only for logging, no change in inductor/autograd behavior
        "_param_name_to_source": None,  # Encapsulated by aot_config.aot_autograd_arg_pos_to_source
        "_backend_id": None,
    }
    saved_fields = {}
    for field, default_value in IGNORED_FIELDS.items():
        saved_fields[field] = getattr(gm, field, None)
        # Clear the field
        setattr(gm, field, default_value)
    try:
        with normalize_placeholder_names(gm):
            yield
    finally:
        for field, value in saved_fields.items():
            setattr(gm, field, value)


@CacheArtifactFactory.register
class AOTAutogradCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self):
        AOTAutogradCache._write_to_local_cache(self.key, self.content)

    @override
    @staticmethod
    def type():
        return "aot_autograd"


@CacheArtifactFactory.register
class BundledAOTAutogradCacheArtifact(PrecompileCacheArtifact[Callable]):
    @override
    @staticmethod
    def type():
        return "precompile_aot_autograd"

    @override
    def after_deserialization(self) -> Callable:
        entry = pickle.loads(self.content)
        # In the precompile use case, guards are already serialized
        # by dynamo, so we don't need to add them to the environment
        entry.guards_expr = None
        # TODO: this isn't exactly right, because cudagraphs needs to be a shared config
        # which is set by compile_fx. But in precompile, we never actually call compile_fx
        # so we don't have a place to track cudagraphs here.
        cudagraphs = torch._inductor.config.triton.cudagraphs
        boxed_forward_device_index = BoxedDeviceIndex(None)
        compiled_fn = entry.wrap_post_compile(
            [],
            entry.sanitized_aot_config,
            {
                "cudagraphs": cudagraphs,
                "boxed_forward_device_index": boxed_forward_device_index,
            },
        )

        # TODO: this ignores flat_params, which can exist
        # if inline_builtin_nn_modules=False
        def forward(*runtime_args: tuple[Any]):
            return compiled_fn(list(runtime_args))

        return forward


class AOTAutogradCache(GuardedCache[GenericAOTAutogradCacheEntry]):
    """
    Caches the results of running AOTAutograd. This class mostly handles the save and load logic, whereas
    AOTAutogradCacheEntry handles the wrapping/unwrapping logic.

    Cache Inputs (AOTAutogradCacheDetails)
    - AOTAutogradCache takes in the following inputs, which are analogous to inputs given
        to AOTAutograd by dynamo:
        - A fx graph module generated by dynamo
        - A list of args, which consists of:
            - Symint inputs to the graph, generated by dynamo
            - The **real tensor** inputs, which inductor uses for cudagraphs
            - Notably, the real tensor inputs don't have symints in their metadata.
        AOTAutograd then retraces those real tensor arguments into FakeTensors later during execution.
        - A set of global configurations that affect AOTAutograd or Inductor behavior.

    It then generates a cache key given these values. Notably, this means AOTAutogradCache currently
    specializes on the sizes and strides of the real tensor inputs when dynamic shapes are turned on.
    In a later PR, we'll likely generate the cache key based on the FakeTensors AOTAutograd generates
    based on the real tensor inputs, which can contain symints.

    # Cache Outputs (AOTAutogradCacheEntry)
    - AOTAutogradCache caches the following values:
        - The compiled forward and backward functions from inductor, via keys to the FXGraphCache
        - Metadata to reconstruct the AOTModule from the compiled inductor artifacts
        - See AOTAutogradCacheEntry for more info

    [Note: Caching guards generated by AOTAutograd and Inductor]
    AOTAutograd and inductor both can introduce new guards to the shape environment. FXGraphCache saves guards with each
    compiled graph inductor generates. On a cache hit, AOTAutograd reloads the compiled forward and backward functions
    from FXGraphCache, giving it new symint arguments from the input args.
    FXGraphCache uses those symints and its saved guards to repopulate the ShapeEnv with guards.
    **No new guards are generated into the shape env after inductor finishes compiling**, so the guards
    saved by inductor are sufficient for correctness for both AOTAutograd and Inductor's caches.
    """

    @staticmethod
    def clear():
        """Clear the cache"""
        try:
            shutil.rmtree(AOTAutogradCache._get_tmp_dir())
        except FileNotFoundError:
            pass

    @staticmethod
    def try_load(
        mod: Union[torch.fx.GraphModule, torch._dynamo.utils.GmWrapper],
        args,
        aot_config: AOTConfig,
        cudagraphs: BoxedBool,
        boxed_forward_device_index: Optional[BoxedDeviceIndex],
        local: bool,
        remote: bool,
    ) -> Optional[Callable]:
        """
        Load a result from the cache, and reconstruct a runtime wrapper around the object
        """
        gm = mod.gm if isinstance(mod, torch._dynamo.utils.GmWrapper) else mod
        with sanitize_gm_for_cache(gm):
            compiled_fn = None
            cache_info: dict[str, Any] = {}
            cache_key = None
            debug_lines: list[str] = []
            cache_event_time = time.time_ns()
            cache_state = None
            fx_config: _CompileFxKwargs = {
                "cudagraphs": cudagraphs,
                "boxed_forward_device_index": boxed_forward_device_index,
            }
            try:
                cache_key, debug_lines = autograd_cache_key(
                    gm, args, aot_config, fx_config
                )
                entry: Optional[GenericAOTAutogradCacheEntry] = (
                    AOTAutogradCache._lookup(
                        cache_key, local, remote, args, cache_info, aot_config
                    )
                )
                if entry is not None:
                    compiled_fn = entry.wrap_post_compile(args, aot_config, fx_config)
                    log.info("AOTAutograd cache hit for key %s", cache_key)

                    counters["aot_autograd"]["autograd_cache_hit"] += 1
                    cache_state = "hit"
                    cache_event_time = time.time_ns()
                    forward_time_saved = entry.forward_time_taken_ns // 1e6
                    backward_time_saved = entry.backward_time_taken_ns // 1e6
                    cache_info.update(
                        {
                            "forward_time_saved_ms": forward_time_saved,
                            "backward_time_saved_ms": backward_time_saved,
                            "time_saved_ms": forward_time_saved + backward_time_saved,
                        }
                    )
                    time_saved_ns = (
                        entry.forward_time_taken_ns + entry.backward_time_taken_ns
                    )
                    # TODO: should we use the same field for remote cache time saved for both
                    # FXGraphCache and AOTAutogradCache?
                    # get_metrics_context().increment(...)
                    if (
                        ephemeral_increase
                        := add_ephemeral_timeout_increase_for_distributed(time_saved_ns)
                    ) != 0:
                        cache_info["ephemeral_timeout_increase"] = ephemeral_increase

                if compiled_fn is None:
                    log.info("AOTAutograd cache miss for key %s", cache_key)
                    counters["aot_autograd"]["autograd_cache_miss"] += 1
                    cache_state = "miss"
                    cache_event_time = time.time_ns()
            # Count missing the FXGraphCache as a miss not a bypass
            except FXGraphCacheMiss as e:
                counters["aot_autograd"]["autograd_cache_miss"] += 1
                cache_state = "miss"
                if (
                    config.strict_autograd_cache
                    or torch._dynamo.config.caching_precompile
                ):
                    raise e
            # Most often this is BypassAOTAutogradCache, but
            # if there's ever different reason we can't cache,
            # we still never want to hard throw an exception, since
            # we can always fallback to a cache bypass.
            # As an example, if the user calls autograd via
            # standalone inductor, we will sometimes get a GraphModule
            # that doesn't actually have a `.graph` on it. Instead
            # of checking every single case, we safely catch the exception
            # in those cases.
            except Exception as e:
                cache_key = None
                counters["aot_autograd"]["autograd_cache_bypass"] += 1
                log.info("Bypassing autograd cache due to: %s", e)
                cache_state = "bypass"
                cache_event_time = time.time_ns()
                cache_info["cache_bypass_reason"] = str(e)
                cache_info["cache_bypass_exception_type"] = type(e).__name__
                cache_info["cache_bypass_traceback"] = traceback.format_exc().split(
                    "\n"
                )
                # TODO: this gets logged implicitly by cache_bypass_reason,
                # and here we explicitly log it into tlparse.
                # We may want to log this as an extra column in Scuba, though.
                cache_info["cache_bypass_hard_exception"] = not isinstance(
                    e, BypassAOTAutogradCache
                )
                if remote:
                    log_cache_bypass("bypass_aot_autograd", str(e))
                if (
                    config.strict_autograd_cache
                    or torch._dynamo.config.caching_precompile
                ):
                    raise e
            if compiled_fn is None:
                # Set the cache key so we can save a cache result later
                symints = AOTAutogradCache._filter_backed_symints(args)
                if cache_key is not None:
                    aot_config.cache_info = AOTAutogradCacheInfo(
                        cache_key,
                        time.time_ns(),
                        forward_symints=symints,
                    )

            cache_info.update(
                {
                    "key": cache_key,
                    "cache_state": cache_state,
                    "components": debug_lines,
                }
            )
            if chromium_event_log_active():
                CompileEventLogger.instant(
                    f"autograd_cache_{cache_state}",
                    metadata=cache_info,
                    time_ns=cache_event_time,
                )
                CompileEventLogger.try_add_pt2_compile(
                    "backend_compile",
                    cache_state=cache_state,
                    cache_event_time=cache_event_time,
                    key=cache_info.get("key"),
                    components=cache_info.get("components"),
                    cache_bypass_reason=cache_info.get("cache_bypass_reason"),
                    remote_cache_enabled=remote,
                    local_cache_enabled=local,
                )

            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": f"aotautograd_cache_{cache_state}",
                    "encoding": "json",
                },
                payload_fn=lambda: json.dumps(cache_info),
            )

            return compiled_fn

    @classmethod
    def generate_guards_expression(
        cls: type[AOTAutogradCache], cache_info: AOTAutogradCacheInfo
    ) -> Optional[str]:
        shape_env = cls._get_shape_env()
        assert shape_env is not None
        symints = cache_info.forward_symints
        guards = shape_env.get_pruned_guards(symints)
        return shape_env.produce_guards_expression(placeholders=symints, guards=guards)

    @classmethod
    def _get_tmp_dir(cls: type[AOTAutogradCache]) -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), "aotautograd")

    @classmethod
    def _get_tmp_dir_for_key(cls: type[AOTAutogradCache], key) -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cls._get_tmp_dir(), key)

    @staticmethod
    def evaluate_guards(guard_expr: str, hints: Union[list[int], list[torch.SymInt]]):
        if torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards:
            return True
        shape_env = AOTAutogradCache._get_shape_env()
        assert shape_env is not None
        result = shape_env.evaluate_guards_expression(guard_expr, hints)
        return result

    @staticmethod
    def _lookup(
        key: str,
        local: bool,
        remote: bool,
        args: list[Any],
        cache_info: dict[str, Any],
        aot_config: Optional[AOTConfig],
    ) -> Optional[GenericAOTAutogradCacheEntry]:
        """Given a key generated by AOTAutogradCachePickler, look up its location in the cache."""
        remote_cache: Optional[RemoteCache[JsonDataTy]] = None
        if remote:
            remote_cache = AOTAutogradCache.get_remote_cache()

        symints = AOTAutogradCache._filter_backed_symints(args)
        hints = [hint_int(s) for s in symints]
        entry = None
        try:
            (
                entry,
                pickled_content,
                guard_info,
            ) = AOTAutogradCache.find_guarded_entry(
                key, local, remote_cache, AOTAutogradCache.evaluate_guards, hints
            )

            if entry is None and guard_info["cache_status_detailed"] == "guard_miss":
                counters["aot_autograd"]["autograd_cache_guard_miss"] += 1
            cache_info.update(guard_info)
            if pickled_content is not None:
                CacheArtifactManager.record_artifact(
                    AOTAutogradCacheArtifact.type(), key, pickled_content
                )
                if (
                    should_bundle_autograd_cache()
                    and aot_config is not None
                    and aot_config.precompile_backend_id is not None
                ):
                    # NB: We don't want to use the cached aot_config.precompile_backend_id
                    # 1. because we set it to None on save 2. even if we didn't, this new run
                    # that cache hit has a *new* backend id associated with it.
                    PrecompileContext.record_artifact(
                        BundledAOTAutogradCacheArtifact.type(),
                        aot_config.precompile_backend_id,
                        pickled_content,
                    )
        except Exception as e:
            log.info("AOTAutograd cache unable to load compiled graph: %s", e)
            if config.strict_autograd_cache:
                raise e
        return entry

    @staticmethod
    def _write_to_local_cache(key: str, content: bytes):
        """Write an entry to the local cache."""
        subdir = AOTAutogradCache._get_tmp_dir_for_key(key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)

        # Use a hash of the serialized entry to get a unique file
        # name. The specific name doesn't matter since a lookup involves
        # iterating over all entries in the parent subdir.
        path = os.path.join(subdir, sha256_hash(content))
        log.info("Writing AOTAutograd cache entry to %s", path)
        write_atomic(path, content)

    @staticmethod
    def save(key: str, entry: GenericAOTAutogradCacheEntry, remote: bool):
        """Save a single entry into the cache."""
        try:
            entry.pre_save()
            content = pickle.dumps(entry)
            CacheArtifactManager.record_artifact(
                AOTAutogradCacheArtifact.type(), key, content
            )
            if (
                should_bundle_autograd_cache()
                and entry.sanitized_aot_config.precompile_backend_id is not None
            ):
                precompile_key = entry.sanitized_aot_config.precompile_backend_id
                # Now that we're saving it, the precompile_backend_id field is no longer
                # useful, remove it from the entry.
                entry.sanitized_aot_config.precompile_backend_id = None
                PrecompileContext.record_artifact(
                    BundledAOTAutogradCacheArtifact.type(),
                    precompile_key,
                    entry,
                    editable=True,
                )
            AOTAutogradCache._write_to_local_cache(key, content)
            counters["aot_autograd"]["autograd_cache_saved"] += 1
        except BypassAOTAutogradCache as e:
            counters["aot_autograd"]["autograd_cache_bypass"] += 1
            log.info("Bypassing autograd cache due to: %s", e)
            if remote:
                log_cache_bypass("bypass_aot_autograd", str(e))
            return None
        except Exception as e:
            log.info("AOTAutograd cache unable to serialize compiled graph: %s", e)
            if remote:
                log_cache_bypass(
                    "bypass_aot_autograd", "Unable to serialize: " + str(e)
                )
            if config.strict_autograd_cache:
                raise e
            return None

        if remote:
            remote_cache: Optional[RemoteCache[JsonDataTy]] = (
                AOTAutogradCache.get_remote_cache()
            )
            if remote_cache is not None:
                time_taken_ms = int(
                    (entry.forward_time_taken_ns + entry.backward_time_taken_ns) // 1e6
                )
                cache_data: JsonDataTy = {
                    "data": base64.b64encode(content).decode("ascii"),
                    "time_taken_ms": time_taken_ms,
                }
                remote_cache.put(key, cache_data)

    @staticmethod
    @functools.cache
    def get_remote_cache() -> Optional[RemoteCache[JsonDataTy]]:
        """
        Attempts to load the remote cache, returns None on error.
        """
        cache_id = "autograd-experimental"
        return create_cache(
            cache_id,
            config.is_fbcode(),
            "FbRemoteAOTAutogradCache",
            "RemoteAOTAutogradCache",
        )

    @staticmethod
    def make_entry(
        compiled_fw_func: CompiledFxGraph,
        compiled_bw_func: Optional[CompiledFxGraph],
        aot_joint_graph_str: Optional[str],
        aot_forward_graph_str: Optional[str],
        aot_backward_graph_str: Optional[str],
        runtime_metadata: ViewAndMutationMeta,
        dispatch_wrappers: list[CompilerWrapper],
        maybe_subclass_meta: Optional[SubclassMeta],
        num_fw_outs_saved_for_bw: Optional[int],
        indices_of_inps_to_detach: list[int],
        forward_time_taken_ns: int,
        backward_time_taken_ns: int,
        sanitized_aot_config: AOTConfig,
        guards_expr: Optional[str],
        backward_state_indices: Optional[list[int]],
        num_symints_saved_for_bw: Optional[int],
        serialized_bw_module: Optional[SerializedGraphModule],
    ) -> GenericAOTAutogradCacheEntry:
        if should_bundle_autograd_cache():
            # Helper function to unwrap all the wrappers we added during aotdispatch
            # They get reapplied on cache load
            def unwrap_compiled_fx_graph(obj):
                while hasattr(obj, "__wrapped__"):
                    obj = obj.__wrapped__
                assert isinstance(obj, CompiledFxGraph)
                return obj

            compiled_fw_graph = unwrap_compiled_fx_graph(compiled_fw_func)
            bundled_compiled_forward = BundledCompiledForward(compiled_fw_graph)
            bundled_compiled_backward = None
            if compiled_bw_func is not None:
                assert backward_state_indices is not None
                assert num_symints_saved_for_bw is not None
                compiled_bw_graph = unwrap_compiled_fx_graph(compiled_bw_func)
                bundled_compiled_backward = BundledCompiledBackward(
                    compiled_bw_graph, backward_state_indices, num_symints_saved_for_bw
                )

            return BundledAOTAutogradCacheEntry(
                compiled_fw=bundled_compiled_forward,
                compiled_bw=bundled_compiled_backward,
                aot_joint_graph_str=aot_joint_graph_str,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=aot_backward_graph_str,
                runtime_metadata=runtime_metadata,
                dispatch_wrappers=dispatch_wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
                indices_of_inps_to_detach=indices_of_inps_to_detach,
                forward_time_taken_ns=forward_time_taken_ns,
                backward_time_taken_ns=backward_time_taken_ns,
                sanitized_aot_config=sanitized_aot_config,
                guards_expr=guards_expr,
                serialized_bw_module=serialized_bw_module,
            )

        else:
            fw_key = getattr(compiled_fw_func, "_fx_graph_cache_key", None)
            fw_debug_lines = getattr(
                compiled_fw_func, "_fx_graph_cache_debug_lines", []
            )

            assert fw_key is not None
            compiled_forward = CompiledForward(
                fx_graph_cache_info=(fw_key, fw_debug_lines),
                fx_graph_guard_expr=getattr(compiled_fw_func, "guards_expr", None),
            )
            compiled_backward = None
            if compiled_bw_func is not None:
                bw_key = getattr(compiled_bw_func, "_fx_graph_cache_key", None)
                bw_debug_lines = getattr(
                    compiled_bw_func, "_fx_graph_cache_debug_lines", []
                )
                assert bw_key is not None
                assert backward_state_indices is not None
                assert num_symints_saved_for_bw is not None
                compiled_backward = CompiledBackward(
                    fx_graph_cache_info=(bw_key, bw_debug_lines),
                    fx_graph_guard_expr=getattr(compiled_bw_func, "guards_expr", None),
                    backward_state_indices=backward_state_indices,
                    num_symints_saved_for_bw_=num_symints_saved_for_bw,
                )

            return AOTAutogradCacheEntry(
                compiled_fw=compiled_forward,
                compiled_bw=compiled_backward,
                aot_joint_graph_str=aot_joint_graph_str,
                aot_forward_graph_str=aot_forward_graph_str,
                aot_backward_graph_str=aot_backward_graph_str,
                runtime_metadata=runtime_metadata,
                dispatch_wrappers=dispatch_wrappers,
                maybe_subclass_meta=maybe_subclass_meta,
                num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw,
                indices_of_inps_to_detach=indices_of_inps_to_detach,
                forward_time_taken_ns=forward_time_taken_ns,
                backward_time_taken_ns=backward_time_taken_ns,
                sanitized_aot_config=sanitized_aot_config,
                guards_expr=guards_expr,
                serialized_bw_module=serialized_bw_module,
            )
