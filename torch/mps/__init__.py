# mypy: allow-untyped-defs
r"""
This package enables an interface for accessing MPS (Metal Performance Shaders) backend in Python.
Metal is Apple's API for programming metal GPU (graphics processor unit). Using MPS means that increased
performance can be achieved, by running work on the metal GPU(s).
See https://developer.apple.com/documentation/metalperformanceshaders for more details.
"""

from contextlib import contextmanager

import torch
from torch import Tensor

from .._utils import _dummy_type


if not hasattr(torch._C, "_MetalGraph"):
    # Built without MPS: keep this module importable. Constructing a MetalGraph
    # then raises RuntimeError from the dummy type, matching torch.cuda.graphs.
    torch._C.__dict__["_MetalGraph"] = _dummy_type("_MetalGraph")

_is_in_bad_fork = getattr(torch._C, "_mps_is_in_bad_fork", lambda: False)
# Absent on a non-MPS build, where nothing can be capturing.
_is_current_stream_capturing = getattr(
    torch._C, "_mps_isCurrentStreamCapturing", lambda: False
)
_default_mps_generator: torch._C.Generator = None  # type: ignore[assignment]


# local helper function (not public or exported)
def _get_default_mps_generator() -> torch._C.Generator:
    global _default_mps_generator
    if _default_mps_generator is None:
        _default_mps_generator = torch._C._mps_get_default_generator()
    return _default_mps_generator


def device_count() -> int:
    r"""Returns the number of available MPS devices."""
    return int(torch._C._has_mps and torch._C._mps_is_available())


def synchronize() -> None:
    r"""Waits for all kernels in all streams on a MPS device to complete."""
    return torch._C._mps_deviceSynchronize()


def get_rng_state(device: int | str | torch.device = "mps") -> Tensor:
    r"""Returns the random number generator state as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    return _get_default_mps_generator().get_state()


def set_rng_state(new_state: Tensor, device: int | str | torch.device = "mps") -> None:
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mps'`` (i.e., ``torch.device('mps')``, the current MPS device).
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    _get_default_mps_generator().set_state(new_state_copy)


def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    # the torch.mps.manual_seed() can be called from the global
    # torch.manual_seed() in torch/random.py. So we need to make
    # sure mps is available (otherwise we just return without
    # erroring out)
    if not torch._C._has_mps:
        return
    seed = int(seed)
    _get_default_mps_generator().manual_seed(seed)


def seed() -> None:
    r"""Sets the seed for generating random numbers to a random number."""
    _get_default_mps_generator().seed()


def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications.
    """
    torch._C._mps_emptyCache()


def set_per_process_memory_fraction(fraction) -> None:
    r"""Set memory fraction for limiting process's memory allocation on MPS device.
    The allowed value equals the fraction multiplied by recommended maximum device memory
    (obtained from Metal API device.recommendedMaxWorkingSetSize).
    If trying to allocate more than the allowed value in a process, it will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~2. Allowed memory equals total_memory * fraction.

    .. note::
       Passing 0 to fraction means unlimited allocations
       (may cause system failure if out of memory).
       Passing fraction greater than 1.0 allows limits beyond the value
       returned from device.recommendedMaxWorkingSetSize.
    """

    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 2:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~2")

    torch._C._mps_setMemoryFraction(fraction)


def current_allocated_memory() -> int:
    r"""Returns the current GPU memory occupied by tensors in bytes.

    .. note::
       The returned size does not include cached allocations in
       memory pools of MPSAllocator.
    """
    return torch._C._mps_currentAllocatedMemory()


def driver_allocated_memory() -> int:
    r"""Returns total GPU memory allocated by Metal driver for the process in bytes.

    .. note::
       The returned size includes cached allocations in MPSAllocator pools
       as well as allocations from MPS/MPSGraph frameworks.
    """
    return torch._C._mps_driverAllocatedMemory()


def recommended_max_memory() -> int:
    r"""Returns recommended max Working set size for GPU memory in bytes.

    .. note::
       Recommended max working set size for Metal.
       returned from device.recommendedMaxWorkingSetSize.
    """
    return torch._C._mps_recommendedMaxMemory()


def compile_shader(source: str):
    r"""Compiles compute shader from source and allows one to invoke kernels
    defined there from the comfort of Python runtime
    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MPS)
        >>> lib = torch.mps.compile_shader(
        ... "kernel void full(device float* out, constant float& val, uint idx [[thread_position_in_grid]]) { out[idx] = val; }"
        ...  )
        >>> x = torch.zeros(16, device="mps")
        >>> lib.full(x, 3.14)
    """
    from pathlib import Path

    from torch._utils_internal import get_file_path
    from torch.utils._cpp_embed_headers import _embed_headers

    if not hasattr(torch._C, "_mps_compileShader"):
        raise RuntimeError("MPS is not available")
    # Resolve the header directory the same way cpp_extension does. Deriving it
    # from `__file__` breaks under an editable install, where the package is
    # redirected to the source checkout while the headers are staged next to
    # the installed distribution.
    source = _embed_headers(
        [l + "\n" for l in source.split("\n")],
        [Path(get_file_path("torch")) / "include"],
        set(),
    )
    return torch._C._mps_compileShader(source)


def load_metallib(source):
    r"""Loads a precompiled Metal library (.metallib) and returns a shader
    library object that allows invoking kernels defined in it.

    Args:
        source: Either raw metallib bytes (``bytes``/``bytearray``) or a
            filesystem path (``str``/``os.PathLike``) to a ``.metallib`` file.

    This is useful for loading Metal libraries compiled ahead of time or
    generated by external tools (e.g. Triton, MetalASM).

    Example::

        >>> # xdoctest: +SKIP("requires external .metallib file")
        >>> lib = torch.mps.load_metallib("kernels.metallib")
        >>> x = torch.ones(16, device="mps")
        >>> lib.square(x)
    """
    import os

    if isinstance(source, (bytes, bytearray)):
        if not hasattr(torch._C, "_mps_loadMetalllib"):
            raise RuntimeError("MPS is not available")
        return torch._C._mps_loadMetalllib(bytes(source))
    elif isinstance(source, (str, os.PathLike)):
        if not hasattr(torch._C, "_mps_loadMetallibFromPath"):
            raise RuntimeError("MPS is not available")
        return torch._C._mps_loadMetallibFromPath(str(source))
    else:
        raise TypeError(f"expected bytes or path, got {type(source).__name__}")


def is_available() -> bool:
    return device_count() > 0


def _host_alias_storage(storage: "torch.UntypedStorage") -> "torch.UntypedStorage":
    r"""Returns a CPU :class:`torch.UntypedStorage` that aliases the
    host-visible contents of the MTLBuffer backing ``storage``.

    The returned storage shares memory with ``storage``: writes through the
    CPU alias land directly in the MPS-allocated MTLBuffer, avoiding a
    CPU->MPS staging copy. This is intended for advanced interop with bulk
    loaders (e.g. safetensors) that already know how to write into CPU
    memory.

    The alias storage retains a reference to the source MPS storage, so the
    host pointer remains valid for the alias's lifetime even if the original
    tensor is freed.

    Raises an exception if ``storage`` is not backed by a shared-storage
    ``id<MTLBuffer>`` allocated by the MPS allocator.

    .. warning::
        Use with caution. This bypasses the cache-coherence guarantees that
        the higher-level PyTorch APIs (:meth:`torch.Tensor.cpu`,
        :meth:`torch.Tensor.to`, ``copy_``) provide for you, and makes the
        caller responsible for ordering CPU and GPU accesses to the same
        memory. You should **always** call :func:`torch.mps.synchronize`
        both **before** issuing host reads/writes through the alias (to
        drain any in-flight GPU work that may still be touching the
        buffer) and **after** (before launching GPU work that depends on
        the host writes). Failure to do so can produce stale reads,
        torn writes, or data corruption.
    """
    return torch._C._mps_host_alias_storage(storage)


class MetalGraph:
    r"""Wraps a captured sequence of MPS operations for repeated replay.

    Mirrors :class:`torch.cuda.CUDAGraph`. The captured resources (compiled
    executables and retained buffers) are owned by this object and released when
    it is garbage collected, so dropping the graph is sufficient to free them.
    :meth:`reset` releases them eagerly.

    Every op run inside the :func:`metal_graph` block is encoded normally, so the
    capture pass produces valid outputs, and its executable plus buffer bindings
    are recorded. :meth:`replay` re-encodes all recorded ops inside a single
    ``dispatch_sync``, collapsing N per-op dispatches into one.

    .. warning::

        This API is in beta and may change in future releases.

    .. note::

        Replay re-binds the exact buffers seen during capture, so the graph keeps
        every device buffer it bound reserved: the allocator will not hand that
        storage to another tensor until the graph is released. Freeing a captured
        tensor is therefore safe, but its memory is not reclaimed until then.
        Call :meth:`reset` (or drop the graph) to give it back. This reserves
        more than :class:`torch.cuda.CUDAGraph` does, which pools only the
        allocations made during capture and can reuse them across replays.

    Constraints:

    * Tensor shapes must not change between the capture pass and replays.
    * Input data must be updated **in-place** via ``.copy_()`` before each
      replay - do **not** create new tensors or reassign variables. This only
      propagates for buffer-backed inputs. Python scalar operands (``x * 2``)
      are baked into the recorded dispatch and are **frozen** on replay; pass
      them as MPS tensors if they need to vary. 0-dim MPS tensors are already
      buffer-backed and do update in place.
    * Copies between MPS and CPU are recorded and re-issued on replay. The graph
      holds the CPU tensor's storage for its lifetime, so the copy stays valid
      even if you drop your own reference, but the host memory is not reclaimed
      until the graph is released.
    * Ops that read device data on the host to decide an output shape - and so
      call ``.item()`` internally, such as :func:`torch.nonzero`,
      :func:`torch.bincount`, :func:`torch.unique` and
      :func:`torch.repeat_interleave` - do **not** raise, but the shape they
      computed during capture is baked into the graph: a replay with different
      data keeps the capture pass's shape. :class:`torch.cuda.CUDAGraph` rejects
      these outright, because a host sync is illegal during a CUDA capture. Keep
      them outside the capture block.
    * Random number generation is not supported inside a capture: the philox seed
      and offset are recorded as fixed bytes, so replays would repeat the capture
      pass's values. Ops that consume the MPS generator raise inside a capture.
      This is a deliberate divergence from :class:`torch.cuda.CUDAGraph`, which
      re-seeds captured generators on every replay, so code that graphs cleanly
      on CUDA can still hard-error here.
    * MPS profiling must be disabled during capture.
    * Ops that encode opaque MPS-framework kernels or fall back to CPU cannot be
      recorded, and raise inside a capture block rather than silently producing a
      graph that omits them.

    Multiple graphs may be alive at once and are fully independent. Recording is
    exclusive per stream, so beginning a capture while another is recording on the
    same stream raises ``RuntimeError``. :meth:`replay` always runs on the stream
    the graph was captured on, not the current one.

    Example::

        g = torch.mps.MetalGraph()
        x = torch.randn(batch, seq, d_model, device="mps")

        with torch.mps.metal_graph(g):
            out = model(x)  # runs once; ops recorded

        for data in loader:
            x.copy_(data)  # update inputs in-place
            g.replay()
            results.append(out.cpu())
    """

    def __init__(self) -> None:
        self._graph = torch._C._MetalGraph()

    def capture_begin(self) -> None:
        r"""Begins recording. Prefer the :func:`metal_graph` context manager,
        which pairs this with :meth:`capture_end` even if the block raises."""
        self._graph.capture_begin()

    def capture_end(self) -> None:
        r"""Stops recording."""
        self._graph.capture_end()

    def replay(self) -> None:
        r"""Re-encodes every recorded op in a single dispatch. Inputs must have
        been updated in-place beforehand."""
        self._graph.replay()

    def reset(self) -> None:
        r"""Releases the captured resources now instead of waiting for this
        object to be collected. Safe to call more than once."""
        self._graph.reset()

    def step_count(self) -> int:
        r"""Number of recorded ops, or 0 if nothing has been captured."""
        return self._graph.step_count()

    def is_captured(self) -> bool:
        r"""Whether this graph currently holds a capture."""
        return self._graph.is_captured()


def is_current_stream_capturing() -> bool:
    r"""Returns True if a :class:`MetalGraph` capture is currently recording on
    the current MPS stream.

    Mirrors :func:`torch.cuda.is_current_stream_capturing`. Useful for code that
    must take a capture-safe path, since ops that cannot be recorded raise while
    a capture is in progress. If MPS has not been initialized, returns False
    without initializing it.
    """
    return _is_current_stream_capturing()


@contextmanager
def metal_graph(g: "MetalGraph"):
    r"""Context manager that records everything executed inside it into ``g``.

    Mirrors :func:`torch.cuda.graph`. See :class:`MetalGraph` for constraints and
    a full example. If the block raises, the partial capture is discarded, so
    ``g`` is left with nothing captured rather than a replayable prefix.

    .. warning::

        This API is in beta and may change in future releases.

    Example::

        g = torch.mps.MetalGraph()
        with torch.mps.metal_graph(g):
            out = model(x)
        g.replay()
    """
    g.capture_begin()
    try:
        yield g
    except BaseException:
        # The block did not finish, so whatever was recorded is a truncated
        # prefix of the intended graph. Replaying it would silently run partial
        # work, so drop it: reset() stops the recording and releases the steps,
        # after which replay() raises the same way it does for a graph that
        # never captured. This is deliberately stricter than torch.cuda.graph,
        # whose __exit__ ends the capture regardless and can leave a replayable
        # partial graph behind.
        g.reset()
        raise
    try:
        g.capture_end()
    except BaseException:
        g.reset()
        raise


from . import profiler
from .event import Event


__all__ = [
    "compile_shader",
    "load_metallib",
    "device_count",
    "get_rng_state",
    "MetalGraph",
    "metal_graph",
    "is_current_stream_capturing",
    "manual_seed",
    "seed",
    "set_rng_state",
    "synchronize",
    "empty_cache",
    "set_per_process_memory_fraction",
    "current_allocated_memory",
    "driver_allocated_memory",
    "Event",
    "profiler",
    "recommended_max_memory",
    "is_available",
]
