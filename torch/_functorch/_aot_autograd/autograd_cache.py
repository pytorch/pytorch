"""
Utils for caching the outputs of AOTAutograd
"""
from __future__ import annotations

import contextlib
import copyreg
import io

import logging
import os
import pickle
import shutil

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
from torch._dynamo.utils import counters
from torch._guards import detect_fake_mode

from torch._inductor.codecache import (
    _ident,
    CompiledFxGraph,
    FxGraphCache,
    FxGraphCachePickler,
    write_atomic,
)
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._subclasses.fake_tensor import (
    extract_tensor_metadata,
    FakeTensor,
    FakeTensorConverter,
    in_kernel_invocation_manager,
    TensorMetadata,
)

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

log = logging.getLogger(__name__)


class BypassAOTAutogradCache(Exception):
    pass


class AOTAutogradCacheDetails:
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """

    def __init__(self, gm: torch.fx.GraphModule, config: AOTConfig):
        self.gm = gm  # TODO: we'll handle different parts of the graph module
        # TODO: We'll want to handle the full_args passed in as well
        self.config = config  # Gets reduced by the Pickler
        self.autograd_enabled = torch.is_grad_enabled()

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


def autograd_cache_key(
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

    def load(self, example_inputs) -> CompiledFxGraph:
        # TODO: do we save the entire CompiledFXGraph or just the key to FXGraphCache?
        result = FxGraphCache._lookup_graph(self.fw_key, example_inputs, True, False)

        if result is None:
            raise BypassAOTAutogradCache("Failed to load from FXGraphCache")
        result._boxed_call = True
        return result


@dataclass
class CompiledBackward:
    """
    Cacheable entry for a forward function
    """

    bw_key: str  # FXGraphCache hash key

    # Used by AOTDispatchAutograd.post_compile
    backward_state_indices: List[int]
    num_symints_saved_for_bw_: int

    def load(self, example_inputs: List[torch.Tensor]):
        # TODO: do we save the entire CompiledFXGraph or just the key to FXGraphCache?
        result = FxGraphCache._lookup_graph(self.bw_key, example_inputs, True, False)
        if result is None:
            raise BypassAOTAutogradCache("Failed to load from FXGraphCache")
        result._boxed_call = True
        return result


@dataclass
class AOTAutogradCacheEntry:
    """A single entry into the cache."""

    # Forward and Backward info
    compiled_fw: CompiledForward
    compiled_bw: Optional[CompiledBackward]

    # Runtime_metadata saved right before compilation
    runtime_metadata: ViewAndMutationMeta

    # Wrappers that run after each aot_dispatch_* function
    dispatch_wrappers: List[CompilerWrapper]

    # Used by AOTSubclassWrapper
    maybe_subclass_meta: Optional[SubclassMeta]
    num_fw_outs_saved_for_bw: Optional[int]

    # Used by RuntimeWrapepr
    indices_of_inps_to_detach: List[int]

    # Turn cache entry into the original callable
    def wrap_post_compile(
        self, args: List[torch.Tensor], aot_config: AOTConfig
    ) -> Callable:
        compiled_fw_func = self.compiled_fw.load(args)
        compiled_bw_func = None
        if self.compiled_bw is not None:
            compiled_bw_func = self.compiled_bw.load(args)
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
            ).post_compile(
                compiled_fw_func, aot_config, runtime_metadata=self.runtime_metadata
            )

        compiled_function, _ = post_compile(
            self.dispatch_wrappers,
            compiled_function,
            aot_config,
            runtime_metadata=self.runtime_metadata,
        )

        return compiled_function


def _fake_tensor_from_meta(metadata: TensorMetadata):
    """
    Given a fake tensor metadata, reconstruct the fake tensor.
    This should be used only on TensorMetadata that was serialized/unserialized by AOTAutogradCache.
    """
    # Synthesize a new FakeTensor with the cached metadata.
    # Based around FakeTensor._output_from_cache_entry
    assert not metadata.is_sparse
    fake_mode = detect_fake_mode()
    empty = torch.empty_strided(
        metadata.shape,
        metadata.stride,
        dtype=metadata.dtype,
        layout=metadata.layout,
        device="meta",
        requires_grad=metadata.requires_grad,
    )

    if metadata.is_conj:
        torch._C._set_conj(empty, True)
    if metadata.is_neg:
        torch._C._set_neg(empty, True)

    # TODO: can traced tangents ever have a storage offset or storage bytes?
    maybe_suppress: Callable[[], Any] = contextlib.nullcontext
    if fake_mode is not None and fake_mode.shape_env is not None:
        maybe_suppress = fake_mode.shape_env.suppress_guards

    if metadata.storage_offset != 0:
        storage = empty.untyped_storage()
        with in_kernel_invocation_manager(fake_mode), maybe_suppress():
            empty.set_(
                storage, metadata.storage_offset, metadata.shape, metadata.stride
            )
    if metadata.storage_bytes == 0:
        empty.untyped_storage().resize_(0)

    return FakeTensorConverter().from_meta_and_device(fake_mode, empty, metadata.device)


def _reduce_fake_tensor(t):
    """
    Allows us to serialize and deserialize FakeTensors, which show up in various metadata in our cache entries
    """
    metadata = extract_tensor_metadata(t)
    if metadata.is_sparse:
        raise BypassAOTAutogradCache(
            "Sparse tensors in the FW metadata are not yet supported"
        )
    return (_fake_tensor_from_meta, (metadata,))


class AOTAutogradCacheEntryPickler(pickle.Pickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[FakeTensor] = _reduce_fake_tensor

    @staticmethod
    def dumps(obj) -> bytes:
        """
        Pickle an object using the FxGraphCachePickler.
        """
        with io.BytesIO() as stream:
            pickler = AOTAutogradCacheEntryPickler(stream)
            pickler.dump(obj)
            return stream.getvalue()


class AOTAutogradCacheEntryUnpickler(pickle.Unpickler):
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[FakeTensor] = _reduce_fake_tensor


class AOTAutogradCache:
    """
    Caches the results of running AOTAutograd. This class mostly handles the save and load logic, whereas
    AOTAutogradCacheEntry handles the wrapping/unwrapping logic.
    """

    @staticmethod
    def clear():
        """Clear the cache"""
        try:
            shutil.rmtree(AOTAutogradCache._get_tmp_dir())
        except FileNotFoundError:
            pass

    @staticmethod
    def load(
        dispatch_and_compile: Callable,
        gm: torch.fx.GraphModule,
        args,
        aot_config: AOTConfig,
    ) -> Callable:
        """
        Load a result from the cache, and reconstruct a runtime wrapper around the object
        """
        compiled_fn = None
        try:
            cache_key = autograd_cache_key(gm, aot_config)
            entry: Optional[AOTAutogradCacheEntry] = AOTAutogradCache._lookup(cache_key)
            if entry is not None:
                compiled_fn = entry.wrap_post_compile(args, aot_config)
                log.info("AOTAutograd cache hit for key %s", cache_key)
                counters["aot_autograd"]["autograd_cache_hit"] += 1
        except BypassAOTAutogradCache:
            cache_key = None

        if compiled_fn is None:
            log.info("AOTAutograd cache miss for key %s", cache_key)
            counters["aot_autograd"]["autograd_cache_miss"] += 1
            # Set the cache key so we can save a cache result later
            aot_config.cache_key = cache_key
            compiled_fn = dispatch_and_compile()
        return compiled_fn

    @staticmethod
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        return os.path.join(cache_dir(), "aotautograd")

    @staticmethod
    def _lookup(key: str) -> Optional[AOTAutogradCacheEntry]:
        """Given a key generated by AOTAutogradCachePickler, look up its location in the cache."""
        subdir = os.path.join(AOTAutogradCache._get_tmp_dir(), key)
        if not os.path.exists(subdir):
            return None
        path = os.path.join(subdir, "entry")
        try:
            with open(path, "rb") as f:
                entry: AOTAutogradCacheEntry = AOTAutogradCacheEntryUnpickler(f).load()
            return entry
        except Exception as e:
            log.warning("AOTAutograd cache unable to load compiled graph: %s", e)
            return None

    @staticmethod
    def save(key: str, entry: AOTAutogradCacheEntry):
        """Save a single entry into the cache."""
        try:
            content = AOTAutogradCacheEntryPickler.dumps(entry)
        except Exception as e:
            log.warning("AOTAutograd cache unable to serialize compiled graph: %s", e)
            raise e
        subdir = os.path.join(AOTAutogradCache._get_tmp_dir(), key)
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, "entry")
        log.info("Writing AOTAutograd cache entry to %s", path)
        write_atomic(path, content)
