"""
Utils for caching the outputs of AOTAutograd
"""
from __future__ import annotations

import logging

import torch
from torch._inductor.codecache import _ident, FxGraphCachePickler

from torch._subclasses.fake_tensor import extract_tensor_metadata
from .schemas import AOTConfig  # noqa: F401

log = logging.getLogger(__name__)


class AOTAutogradCacheDetails:
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """

    def __init__(self, gm: torch.fx.GraphModule, config: AOTConfig):
        self.gm = gm  # TODO: we'll handle different parts of the graph module
        # TODO: We'll want to handle the full_args passed in as well
        self.config = config  # Gets reduced by the Pickler

    def debug_str(self) -> str:
        """
        Get a printable string describing in more detail all the attributes
        comprising this object. Useful for debugging when one graph hashes
        to a different value than another.
        """

        def get_str(obj) -> str:
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata(obj))
            elif isinstance(obj, bytes):
                return "<bytes>"
            else:
                return str(obj)

        lines = []
        for attr, obj in vars(self).items():
            if isinstance(obj, list):
                for ii in range(len(obj)):
                    h = AOTAutogradCachePickler.get_hash(obj[ii])
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    h = AOTAutogradCachePickler.get_hash(v)
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                h = AOTAutogradCachePickler.get_hash(obj)
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        return "\n".join(lines)


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
