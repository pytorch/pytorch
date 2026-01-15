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
import random
import shutil
import time
import traceback
from copy import copy
from typing import Any, Optional, TYPE_CHECKING, Union
from typing_extensions import override

import torch
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.trace_rules import torch_non_c_binding_in_graph_functions
from torch._dynamo.utils import chromium_event_log_active, CompileEventLogger, counters
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
from torch._inductor.output_code import OutputCode
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import BoxedBool, should_use_remote_fx_graph_cache
from torch._logging import LazyString
from torch._utils_internal import log_cache_bypass
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
)
from torch.fx.experimental.symbolic_shapes import size_hint
from torch.utils._triton import has_triton_package

from .aot_autograd_result import (
    AOTAutogradResult,
    BundledAOTAutogradCacheArtifact,
    BundledAOTAutogradResult,
    BundledCompiledBackward,
    BundledCompiledForward,
    CompiledBackward,
    CompiledForward,
    GenericAOTAutogradResult,
    SerializedGraphModule,
)
from .runtime_wrappers import (
    CompilerWrapper,
    SerializableCompiledFunction,
    SubclassMeta,
)
from .schemas import AOTAutogradCacheInfo, AOTConfig, ViewAndMutationMeta  # noqa: F401


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.compile_fx import _CompileFxKwargs
    from torch._inductor.cudagraph_utils import BoxedDeviceIndex
    from torch._inductor.remote_cache import JsonDataTy, RemoteCache
    from torch.fx.node import Node


log = logging.getLogger(__name__)


class BypassAOTAutogradCache(Exception):
    pass


# Used to signify when FXGraphCache missed when AOTAutogradCache uses it
class FXGraphCacheMiss(BypassAOTAutogradCache):
    pass


def should_use_remote_autograd_cache():
    if torch.compiler.config.force_disable_caches:
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
    if torch.compiler.config.force_disable_caches:
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
        "torch.autograd.grad",
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
            type(method_name) is not str
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


def _get_context_fn_cache_hash(context_fn):
    """
    Extract a cache hash from a context_fn used for selective activation checkpointing (SAC).

    The context_fn determines which ops are saved vs recomputed in the SAC region.
    Since context_fn can be an arbitrary Python function, we cannot reliably pickle
    it for cache key generation (pickle only captures the function name, not the code).

    Users must provide a stable hash by setting a `cache_hash` attribute on the context_fn.
    For functools.partial objects, set the cache_hash on the partial object itself, not on
    the underlying function.

    Returns:
        The cache hash if found
        None: If no hash is provided (caller should bypass caching)
    """
    if hasattr(context_fn, "cache_hash"):
        return context_fn.cache_hash

    return None


def _collect_context_fn_hashes(gm: torch.fx.GraphModule) -> list:
    """
    Collect cache hashes from all context_fn used in SAC HOPs within the graph module.

    Returns a list of hashes. Raises BypassAOTAutogradCache if any context_fn
    lacks a cache_hash attribute.
    """
    hashes = []
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        context_fn = module.meta.get("_checkpoint_context_fn")
        if context_fn is not None:
            cache_hash = _get_context_fn_cache_hash(context_fn)
            if cache_hash is None:
                raise BypassAOTAutogradCache(
                    "SAC context_fn does not have a cache_hash attribute. "
                    "To enable caching with selective activation checkpointing, "
                    "add a 'cache_hash' attribute to your context_fn. This can be "
                    "a string or any hashable value that uniquely identifies the checkpointing "
                    "behavior (e.g., based on source code hash and closed-over globals). "
                    "For functools.partial objects, set cache_hash on the partial itself."
                )
            hashes.append(cache_hash)
    return hashes


class AOTAutogradCacheDetails(FxGraphHashDetails):
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """

    def get_triton_source_codes_from_gm(
        self,
        gm: torch.fx.GraphModule,
    ):
        assert has_triton_package(), "Triton is not available"

        triton_kernels = []
        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if isinstance(node.target, torch._ops.OpOverloadPacket):
                    attrs = node.target._dir
                    for attr in attrs:
                        if custom_op := getattr(node.target, attr, None):
                            kernels = torch._library.triton.get_triton_kernels_for_op(
                                custom_op._name
                            )
                            triton_kernels.extend(kernels)
                elif isinstance(node.target, torch._ops.OpOverload):
                    kernels = torch._library.triton.get_triton_kernels_for_op(
                        node.target._name
                    )
                    triton_kernels.extend(kernels)

        triton_kernel_source_codes = []
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        for kernel in triton_kernels:
            from triton.runtime.autotuner import Autotuner

            if isinstance(kernel, Autotuner):
                # Grab the Inner JITFunction
                kernel = kernel.fn
            source_codes = user_defined_triton_kernel_transitive_closure_source_code(
                kernel
            )
            triton_kernel_source_codes.append(source_codes)

        return triton_kernel_source_codes

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
        if has_triton_package():
            self.triton_kernel_source_codes = self.get_triton_source_codes_from_gm(gm)

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

        self.sac_context_fn_hashes: list = _collect_context_fn_hashes(gm)

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
        # pyrefly: ignore [bad-override]
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

    try:
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
    except Exception:
        # If enable_aot_compile is set, we're in AOT precompile mode where we always
        # want to use fallback nonce keys. Unlike caching, it's fine if we can't generate
        # a proper key because we are guaranteed in an AOT precompile world users are in
        # complete control of distributing and loading artifacts.
        if torch._functorch.config.bypass_autograd_cache_key:
            log.info(
                "Failed to generate AOTAutograd cache key; falling back to nonce due to enable_aot_compile",
                exc_info=True,
            )
            return str(random.random()), []
        else:
            raise


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


class AOTAutogradCache(GuardedCache[GenericAOTAutogradResult]):
    """
    Caches the results of running AOTAutograd. This class mostly handles the save and load logic, whereas
    AOTAutogradResult handles the wrapping/unwrapping logic.

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

    # Cache Outputs (AOTAutogradResult)
    - AOTAutogradCache caches the following values:
        - The compiled forward and backward functions from inductor, via keys to the FXGraphCache
        - Metadata to reconstruct the AOTModule from the compiled inductor artifacts
        - See AOTAutogradResult for more info

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
                result: Optional[tuple[GenericAOTAutogradResult, bytes]] = (
                    AOTAutogradCache._lookup(
                        cache_key, local, remote, args, cache_info, aot_config
                    )
                )
                if result is not None:
                    (entry, pickled_content) = result
                    compiled_fn = entry.wrap_post_compile(args, aot_config, fx_config)
                    # Make the compiled_fn serializable, where the serialize function just
                    # makes a copy of the original entry before post compile via the pickled content
                    compiled_fn = SerializableCompiledFunction(
                        compiled_fn, lambda: pickle.loads(pickled_content)
                    )
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
                    or torch._dynamo.config.strict_precompile
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
                log.info("Bypassing autograd cache due to: %s", e)  # noqa: G200
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
                    or torch._dynamo.config.strict_precompile
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

        if shape_env is None:
            return None

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

    @classmethod
    def _record_result(
        cls: type[AOTAutogradCache],
        _key: str,
        local_hit: bool,
        local_miss: bool,
        remote_hit: bool,
        remote_miss: bool,
    ) -> None:
        """
        Called by GuardedCache to record hit/miss statistics.
        """
        if local_hit:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_local_cache_hit_count",
            )
        if remote_hit:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_remote_cache_hit_count",
            )
        if local_miss:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_local_cache_miss_count",
            )
        if remote_miss:
            CompileEventLogger.try_(
                CompileEventLogger.increment_toplevel,
                "aotautograd_remote_cache_miss_count",
            )

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
    ) -> Optional[tuple[GenericAOTAutogradResult, bytes]]:
        """Given a key generated by AOTAutogradCachePickler, look up its location in the cache."""
        remote_cache: Optional[RemoteCache[JsonDataTy]] = None
        if remote:
            remote_cache = AOTAutogradCache.get_remote_cache()

        symints = AOTAutogradCache._filter_backed_symints(args)
        hints = [size_hint(s) for s in symints]
        entry = None
        pickled_content = None
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
                        BundledAOTAutogradCacheArtifact(
                            aot_config.precompile_backend_id, entry
                        ),
                    )
        except Exception as e:
            log.info("AOTAutograd cache unable to load compiled graph: %s", e)  # noqa: G200
            if config.strict_autograd_cache:
                raise e
        if entry is not None:
            assert pickled_content is not None
            return (entry, pickled_content)
        else:
            return None

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
    def save(key: str, entry: GenericAOTAutogradResult, remote: bool):
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
                artifact = BundledAOTAutogradCacheArtifact(precompile_key, entry)
                # Now that we're saving it, the precompile_backend_id field is no longer
                # useful, remove it from the entry.
                entry.sanitized_aot_config.precompile_backend_id = None
                PrecompileContext.record_artifact(artifact)
            AOTAutogradCache._write_to_local_cache(key, content)
            counters["aot_autograd"]["autograd_cache_saved"] += 1
        except BypassAOTAutogradCache as e:
            if config.strict_autograd_cache:
                raise
            counters["aot_autograd"]["autograd_cache_bypass"] += 1
            log.info("Bypassing autograd cache due to: %s", e)  # noqa: G200
            if remote:
                log_cache_bypass("bypass_aot_autograd", str(e))
            return None
        except Exception as e:
            log.info("AOTAutograd cache unable to serialize compiled graph: %s", e)  # noqa: G200
            if remote:
                log_cache_bypass(
                    "bypass_aot_autograd", "Unable to serialize: " + str(e)
                )
            if config.strict_autograd_cache:
                raise
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
        compiled_fw_func: OutputCode,
        compiled_bw_func: Optional[OutputCode],
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
    ) -> GenericAOTAutogradResult:
        if should_bundle_autograd_cache():
            # Helper function to unwrap all the wrappers we added during aotdispatch
            # They get reapplied on cache load
            def unwrap_output_code(obj):
                while hasattr(obj, "__wrapped__"):
                    obj = obj.__wrapped__
                assert isinstance(obj, OutputCode)
                return obj

            compiled_fw_graph = unwrap_output_code(compiled_fw_func)
            bundled_compiled_forward = BundledCompiledForward(compiled_fw_graph)
            bundled_compiled_backward = None
            if compiled_bw_func is not None:
                assert backward_state_indices is not None
                assert num_symints_saved_for_bw is not None
                compiled_bw_graph = unwrap_output_code(compiled_bw_func)
                bundled_compiled_backward = BundledCompiledBackward(
                    compiled_bw_graph, backward_state_indices, num_symints_saved_for_bw
                )

            return BundledAOTAutogradResult(
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

            return AOTAutogradResult(
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
