"""
Utils for caching the outputs of AOTAutograd
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING

from torch._inductor.codecache import _ident, FxGraphCachePickler

from .runtime_wrappers import (
    AOTDispatchAutograd,
    AOTDispatchSubclassWrapper,
    CompilerWrapper,
    FunctionalizedRngRuntimeWrapper,
    post_compile,
    RuntimeWrapper,
    SubclassMeta,
)

from .schemas import AOTConfig, ViewAndMutationMeta  # noqa: F401

if TYPE_CHECKING:
    import torch

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


@dataclass
class CompiledForward:
    """
    Cacheable entry for a forward function
    """

    fw_key: str  # FXGraphCache hash key

    def load(self):
        # TODO: do we save the entire CompiledFXGraph or just the key to FXGraphCache?
        pass


@dataclass
class CompiledBackward:
    """
    Cacheable entry for a forward function
    """

    bw_key: str  # FXGraphCache hash key
    backward_state_indices: List[int]
    num_symints_saved_for_bw_: int

    def load(self):
        # TODO: do we save the entire CompiledFXGraph or just the key to FXGraphCache?
        pass


@dataclass
class AOTAutogradCacheEntry:
    """A single entry into the cache"""

    compiled_fw: CompiledForward
    compiled_bw: Optional[CompiledBackward]
    runtime_metadata: ViewAndMutationMeta

    # Wrappers that run after each aot_dispatch_* function
    dispatch_wrappers: List[CompilerWrapper]

    # Used by AOTSubclassWrapper
    maybe_subclass_meta: Optional[SubclassMeta]
    num_fw_outs_saved_for_bw: int

    # Used by RuntimeWrapepr
    indices_of_inps_to_detach: List[int]

    # Turn cache entry into the original callable
    def wrap_post_compile(self, aot_config: AOTConfig) -> Callable:
        compiled_fw_func = self.compiled_fw.load()
        compiled_bw_func = None
        if self.compiled_bw is not None:
            compiled_bw_func = self.compiled_bw.load()
            needs_autograd = True
        else:
            needs_autograd = False

        # Wrap the forward function in post compile wrappers
        compiled_fw_func = AOTDispatchSubclassWrapper(
            trace_joint=needs_autograd,
            fw_only=None,
            maybe_subclass_meta=self.maybe_subclass_meta,
            num_fw_outs_saved_for_bw=self.num_fw_outs_saved_for_bw,
        ).post_compile(
            compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
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
            compiled_function = AOTDispatchAutograd.post_compile(
                compiled_fw_func,
                compiled_bw_func,
                self.maybe_subclass_meta,
                self.compiled_bw.num_symints_saved_for_bw_,
                self.compiled_bw.backward_state_indices,
                disable_amp,
                self.indices_of_inps_to_detach,
                None,  # lazy_backward_info
                aot_config,
                fw_metadata=self.runtime_metadata,
            )
        else:
            compiled_function = RuntimeWrapper(
                indices_of_inps_to_detach=self.indices_of_inps_to_detach,
                trace_joint=False,
                disable_amp=disable_amp,
            )

        compiled_function = post_compile(
            self.dispatch_wrappers,
            compiled_function,
            aot_config,
            runtime_metadata=self.runtime_metadata,
        )

        return compiled_function
