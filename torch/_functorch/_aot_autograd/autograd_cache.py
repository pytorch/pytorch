"""
Utils for caching the outputs of AOTAutograd
"""
from __future__ import annotations

import functools
import logging
import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch._dynamo.utils import counters
from torch._inductor.codecache import (
    _ident,
    FxGraphCache,
    FxGraphCachePickler,
    get_code_hash,
    get_inductor_root,
    write_atomic,
)
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._prims_common import CUDARngStateHelper

from torch.fx.node import Node
from .runtime_wrappers import (
    aot_dispatch_subclass_wrapper,
    create_runtime_wrapper,
    functionalized_rng_runtime_epilogue,
)

from .schemas import AOTConfig, SubclassMeta, ViewAndMutationMeta

from .utils import make_boxed_func

log = logging.getLogger(__name__)


class BypassAOTAutogradCache(Exception):
    pass


def check_node_safe(node: Node):
    """
    Checks that the node only uses supported operators.
    """

    def is_torch_function(target):
        if isinstance(target, str):
            name = target
        elif callable(target):
            return type(target).__name__ == "builtin_function_or_method"
        else:
            return False
        # TODO: is this the right check?
        return name.startswith("torch.")

    def is_tensor(target):
        # Tensors always have example values in meta field
        return "example_value" in target.meta

    # I'd love to use a match statement here, but it wasn't introduced until py3.10
    if node.op == "call_function":
        # We support only torch.* functions for now
        # We can probably add an allowlist of safe non-torch implementations as well
        if not is_torch_function(node.target):
            raise BypassAOTAutogradCache(
                f"Unsupported call_function target {node.target}"
            )
    elif node.op == "call_method":
        method_target = node.args[0]
        if not is_tensor(method_target):
            # We support only method calls on tensors and symints
            raise BypassAOTAutogradCache(
                f"Unsupported call_method target {method_target}"
            )
    # Cache safe
    elif node.op in ("placeholder", "call_module", "get_attr", "output"):
        # TODO: not all call_modules may be safe
        pass
    else:
        raise BypassAOTAutogradCache(f"Unsupported node op {node.op}")


@functools.lru_cache(None)
def get_autograd_code_hash():
    autograd_root = os.path.dirname(__file__)
    inductor_root = get_inductor_root()
    return get_code_hash([autograd_root, inductor_root])


def check_cacheable(gm: torch.fx.GraphModule):
    """
    Checks that the graph module only uses supported operators
    """
    if not torch._inductor.config.fx_graph_cache:
        raise BypassAOTAutogradCache("fx_graph_cache is required for AOTAutogradCache")
    nodes = gm.graph.nodes
    for node in nodes:
        check_node_safe(node)


class AOTAutogradCacheDetails:
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """

    def __init__(self, gm: torch.fx.GraphModule, config: AOTConfig):
        self.gm = gm  # TODO: we'll handle different parts of the graph module
        # TODO: We'll want to handle the full_args passed in as well
        self.config = config  # Gets reduced by the Pickler
        self.code_hash = get_autograd_code_hash()

    def debug_str(self) -> str:
        return AOTAutogradCachePickler.debug_str(self)


def _reduce_aot_config(config: AOTConfig):
    """
    Reduce the config to a stable key for caching.
    """
    return (
        _ident,
        (
            config.num_params_buffers,
            config.keep_inference_input_mutations,
            config.is_export,
            config.no_tangents,
            config.dynamic_shapes,
            config.aot_autograd_arg_pos_to_source,
            config.enable_log,
            config.pre_dispatch,
        ),
    )


class AOTAutogradCachePickler(FxGraphCachePickler):
    dispatch_table = FxGraphCachePickler.dispatch_table.copy()
    dispatch_table[AOTConfig] = _reduce_aot_config


def autograd_cache_hash(
    gm: torch.fx.GraphModule,
    config: AOTConfig,
    # TODO: add args and parameters
) -> str:
    """
    Generate a unique hash of the FX graph for caching.
    """
    details = AOTAutogradCacheDetails(gm, config)
    # The prefix distinguishes among the other kinds of objects we cache
    key = "a" + AOTAutogradCachePickler.get_hash(details)
    log.debug("Autograd cache hash details for key %s:\n%s", key, details.debug_str())
    return key


@dataclass
class AOTAutogradCacheEntry:
    """
    An entry in the AOTAutograd cache. We store the fw and backward compiled modules by storing cache keys to FXGraphCache.
    """

    fw_cache_key: str
    bw_cache_key: Optional[str]
    fw_metadata: ViewAndMutationMeta
    maybe_subclass_meta: Optional[SubclassMeta]


class AOTAutogradCache:
    """
    Cache for results of AOTAutograd. AOTAutogradCache is an extension of FXGraphCache that
    caches extra metadata to reconstruct the autograd generated wrapper around the compiled FXGraph.
    """
    @staticmethod
    def clear():
        """
        Clear out the on-disk cache.
        """
        try:
            shutil.rmtree(AOTAutogradCache._get_tmp_dir())
        except FileNotFoundError:
            pass

    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), "aotautograd")

    @staticmethod
    def _lookup(key: str):
        """Given a key generated by AOTAutogradCachePickler, look up its location in the cache."""
        subdir = os.path.join(AOTAutogradCache._get_tmp_dir(), key)
        if not os.path.exists(subdir):
            return None
        path = os.path.join(subdir, "entry")
        try:
            with open(path, "rb") as f:
                entry: AOTAutogradCacheEntry = pickle.load(f)
            return entry
        except Exception as e:
            log.warning("AOTAutograd cache unable to load compiled graph: %s", e)
            raise BypassAOTAutogradCache("Error loading compiled graph")

    @staticmethod
    def save(key: str, fw_module, fw_metadata, maybe_subclass_meta, bw_module=None):
        """Save forward and backward modules to the cache."""

        if fw_module._fx_cache_key is None:
            raise BypassAOTAutogradCache("fw_module has no cache key")
        if bw_module is not None and bw_module._fx_cache_key is None:
            raise BypassAOTAutogradCache("bw_module has no cache key")

        fw_key = fw_module._fx_cache_key
        bw_key = bw_module._fx_cache_key if bw_module is not None else None

        entry = AOTAutogradCacheEntry(fw_key, bw_key, fw_metadata, maybe_subclass_meta)
        try:
            content = pickle.dumps(entry)
        except Exception as e:
            log.warning("AOTAutograd cache unable to serialize compiled graph: %s", e)
            raise BypassAOTAutogradCache("Error serializing compiled graph")
        subdir = os.path.join(AOTAutogradCache._get_tmp_dir(), key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, "entry")
        log.info("Writing AOTAutograd cache entry to %s", path)
        write_atomic(path, content)

    @staticmethod
    def _wrap_base_result(entry: AOTAutogradCacheEntry, aot_config: AOTConfig):
        """
        Wrap cached entry when bw_module is None (i.e., result of aot_dispatch_base)
        """
        # TODO: handle shape env guards in AOTAutogradCache
        compiled_fw = FxGraphCache._lookup_graph(
            entry.fw_cache_key, [], ignore_guards=True
        )
        if not compiled_fw:
            return None
        fw_metadata = entry.fw_metadata
        maybe_subclass_meta = entry.maybe_subclass_meta
        disable_amp = torch._C._is_any_autocast_enabled()
        compiled_fw._boxed_call = True

        # Create a wrapper to set up the rng functionalize bits
        @functools.wraps(compiled_fw)
        def rng_functionalization_wrapper(args: List[Any]):
            if fw_metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                args.extend([seed, offset])
                out = compiled_fw(args)
                out = functionalized_rng_runtime_epilogue(fw_metadata, out)
                return out
            else:
                return compiled_fw(args)

        if maybe_subclass_meta is not None:
            compiled_fw_func = aot_dispatch_subclass_wrapper(
                rng_functionalization_wrapper,
                subclass_metas=fw_metadata.subclass_fw_graph_out_meta,
                num_fw_outs_saved_for_bw=None,
            )
        else:
            compiled_fw_func = rng_functionalization_wrapper

        if not hasattr(compiled_fw_func, "_boxed_call"):
            compiled_fw_func = make_boxed_func(compiled_fw_func)

        compiled_fn = create_runtime_wrapper(
            compiled_fw_func,
            runtime_metadata=fw_metadata,
            indices_of_inps_to_detach=[],
            trace_joint=False,
            keep_input_mutations=aot_config.keep_inference_input_mutations,
            disable_amp=disable_amp,
        )

        return compiled_fn

    @staticmethod
    def load(gm: torch.fx.GraphModule, _example_inputs, aot_config: AOTConfig):
        """
        Load a result from the cache, and reconstruct a runtime wrapper around the object
        """
        check_cacheable(gm)
        key = autograd_cache_hash(gm, aot_config)
        # Set the AOTConfig cache key

        aot_config.cache_key = key
        entry: AOTAutogradCacheEntry = AOTAutogradCache._lookup(key)
        if not entry:
            result = None
        elif entry.bw_cache_key is None:
            result = AOTAutogradCache._wrap_base_result(entry, aot_config)
        else:
            result = None

        if result is not None:
            counters["aot_autograd"]["autograd_cache_hit"] += 1
        else:
            counters["aot_autograd"]["autograd_cache_miss"] += 1
        return result
