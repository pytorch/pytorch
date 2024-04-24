"""
Utils for caching the outputs of AOTAutograd
"""
from __future__ import annotations

import functools
import logging
import os

import torch
from torch._inductor.codecache import (
    _ident,
    FxGraphCachePickler,
    get_code_hash,
    get_inductor_root,
)
from torch.fx.node import Node

from .schemas import AOTConfig  # noqa: F401

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
            name = target.__name__
        else:
            return False
        # TODO: is this the right check?
        return name.startswith("torch.")

    def is_tensor(target: Node):
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
                f"Unsupported call_method target {node.target}"
            )
    # Cache safe
    elif node.op in  ("placeholder", "call_module", "get_attr",  "output"):
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
        check_cacheable(gm)
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
    log.debug("FX graph cache hash details for key %s:\n%s", key, details.debug_str())
    return key
