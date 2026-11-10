# Shared operand descriptors and tvm-ffi launch glue. Compile against fake operands, then pass
# torch tensors directly (5.98us versus 19.16us with per-call wrapping). Keep the stream explicit:
# ENV detection misses tensors inside lists and packed-id lookup deadlocks capture. read_only()
# exports COW inputs through const_data_ptr() to avoid forbidden materialization in backward.

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, NamedTuple, overload, TYPE_CHECKING

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]

import cutlass
import cutlass.cute as cute

import torch
from torch._native.utils.tensor import const_data_ptr
from torch.utils._pytree import tree_leaves
from torch.utils.dlpack import ReadOnlyTensorWrapper


if TYPE_CHECKING:
    from tvm_ffi import Function  # pyrefly: ignore[missing-import]


def sym(divisibility: int = 1) -> cute.SymInt:
    """A dynamic extent or stride divisible by divisibility, allowing one wide-load
    kernel to serve every matching value.
    """
    return cute.sym_int(divisibility=divisibility)


def fake_compact(
    dtype: type[cutlass.Numeric],
    shape: Sequence[int | cute.SymInt],
    *,
    stride_order: tuple[int, ...] | None = None,
    align: int | None = None,
) -> cute.Tensor:
    """Describe a compact compile-time operand. `stride_order` gives each mode's
    compactness rank (0 is stride-1); the pointer must satisfy `align`.
    """
    return cute.runtime.make_fake_compact_tensor(
        dtype, tuple(shape), stride_order=stride_order, assumed_align=align
    )


def supported_alignment(tensor: torch.Tensor, maximum: int) -> int:
    """Return the largest power-of-two alignment up to `maximum` supported by `tensor`."""
    ptr = const_data_ptr(tensor)
    alignment = maximum
    while alignment > tensor.element_size() and ptr % alignment:
        alignment //= 2
    return alignment


@overload
def read_only(t: torch.Tensor) -> ReadOnlyTensorWrapper: ...


@overload
def read_only(t: None) -> None: ...


def read_only(t: torch.Tensor | None) -> ReadOnlyTensorWrapper | None:
    """Export an input through const_data_ptr() without materializing COW storage.
    Outputs must remain writable. Non-DLPack operations are rejected, so wrap only
    the final-shape tensor.
    """
    if t is None:
        return None
    with torch._C.DisableTorchFunctionSubclass():
        return ReadOnlyTensorWrapper(t)


def compile_kernel(op: Callable[..., Any], *args: Any, options: str = "") -> "Function":
    """Compile `op` against FAKE operands, for the fast tvm-ffi arg convention."""
    return cute.compile(op, *args, options=f"--enable-tvm-ffi {options}".rstrip())


def stream(device_index: int | None = None) -> cuda.CUstream:
    # Read the live stream each call because callers can change device or capture stream.
    if device_index is None:
        device_index = torch.cuda.current_device()
    return cuda.CUstream(torch._C._cuda_getCurrentRawStream(device_index))


class _View(NamedTuple):
    slot: int  # -1 for the recorded input, else a buffer index
    whole: bool  # the operand is the slot's tensor itself
    size: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int


_RECORD: ContextVar["LaunchRecord | None"] = ContextVar("_RECORD", default=None)


def _storage_id(t: torch.Tensor) -> int:
    # The StorageImpl address: data_ptr() would materialize a COW input.
    return t.untyped_storage()._cdata


def _geometry(t: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    return tuple(t.shape), t.stride(), t.storage_offset()


def launch(fn: "Function", device: int, ins: Any, *args: Any) -> None:
    """Call compiled `fn` on `device`'s current stream. `ins`, a tensor or a list of
    tensors, is exported read-only; tensors in `args`, bare or in lists, are outputs.
    This calls fn(read-only ins, *args, stream), so kernels are compiled with the stream
    as their last parameter and callers do not pass it.

    Compiled kernels launched through here must treat outputs as write-only scratch: a
    replay allocates them uninitialized and repeats only the launches. While recording,
    every compiled-kernel call must come through here; a direct call is missing from
    the replay.
    """
    record = _RECORD.get()
    if record is not None:
        record.add(fn, device, ins, args)
    ins = [read_only(t) for t in ins] if isinstance(ins, list) else read_only(ins)
    fn(ins, *args, stream(device))


class LaunchRecord:
    """The launches that computed outputs from `x`, replayable on any input that matches
    x's dtype, device, sizes, strides, storage offset, storage size and pointer alignment.

    Callers must ensure that no torch op writes data to a buffer a launch has seen
    (resizes that do not grow the storage are fine), that launch outputs are write-only
    scratch (a replay allocates them uninitialized and repeats only the launches), and
    that every compiled-kernel call goes through launch(): finish() catches a direct
    call only when it writes a storage no launch has seen and a launch then reads that
    storage, or that storage is returned without a launch writing it.
    finish() returns None unless every operand views `x` or a buffer whose storage first
    appeared in a launch's output position, before any input position, and every output
    views a distinct such buffer. Torch-op results that a launch reads, or that are
    returned without a launch writing them, therefore decline; the write-only-scratch
    contract is what protects tensors first passed as launch outputs. Buffers are keyed
    by storage, since views made below autograd do not record their base.
    """

    def __init__(self, x: torch.Tensor) -> None:
        self._x = x
        self._slots = {_storage_id(x): -1}
        # Holding the buffers keeps their storage ids unique while recording.
        self._buffers: list[torch.Tensor] = []
        self._allocs: list[tuple[int, torch.dtype]] = []
        self._launches: list[tuple[Function, int, Any, tuple[Any, ...]]] = []
        self._ok = True

    @staticmethod
    def input_key(x: torch.Tensor, max_align: int) -> tuple[Any, ...]:
        """The metadata of `x` a replay must match, alignment capped at `max_align`."""
        # Routes and compiled kernels assume the pointer's alignment, and a storage base
        # need not be allocator-aligned (e.g. a sliced UntypedStorage).
        return (
            x.dtype,
            x.get_device(),
            x.shape,
            x.stride(),
            x.storage_offset(),
            x.untyped_storage().nbytes(),
            supported_alignment(x, max_align),
        )

    def _view(self, t: torch.Tensor, written: bool) -> _View | None:
        slot = self._slots.get(_storage_id(t))
        if slot is None:
            if not written:
                return None
            slot = self._slots[_storage_id(t)] = len(self._buffers)
            self._buffers.append(t)
            numel = t.untyped_storage().nbytes() // t.element_size()
            self._allocs.append((numel, t.dtype))
        if slot < 0:
            dtype, whole = self._x.dtype, _geometry(self._x)
        else:
            numel, dtype = self._allocs[slot]
            whole = ((numel,), (1,), 0)
        if t.dtype != dtype:
            return None
        return _View(slot, _geometry(t) == whole, *_geometry(t))

    def _template(self, value: Any, written: bool) -> Any:
        if isinstance(value, list):
            return [self._template(v, written) for v in value]
        if isinstance(value, torch.Tensor):
            view = self._view(value, written)
            self._ok = self._ok and view is not None
            return view
        # Replays substitute bare tensors and tensors in lists only; others go stale.
        if any(isinstance(leaf, torch.Tensor) for leaf in tree_leaves(value)):
            self._ok = False
        return value

    def add(self, fn: "Function", device: int, ins: Any, args: tuple[Any, ...]) -> None:
        ins = self._template(ins, written=False)
        args = tuple(self._template(a, written=True) for a in args)
        self._launches.append((fn, device, ins, args))

    def finish(self, outs: Sequence[torch.Tensor]) -> "LaunchReplay | None":
        views = [self._view(out, written=False) for out in outs]
        slots = [-1 if view is None else view.slot for view in views]
        if not self._ok or min(slots) < 0 or len(set(slots)) < len(slots):
            return None
        return LaunchReplay(self._allocs, self._launches, views)


@contextmanager
def recording(x: torch.Tensor) -> Iterator[LaunchRecord]:
    """Record the launches made from `x` in this context."""
    record = LaunchRecord(x)
    token = _RECORD.set(record)
    try:
        yield record
    finally:
        _RECORD.reset(token)


class LaunchReplay:
    """Fresh buffers and the recorded launches on them; see LaunchRecord."""

    def __init__(
        self,
        allocs: list[tuple[int, torch.dtype]],
        launches: list[tuple["Function", int, Any, tuple[Any, ...]]],
        outs: list[_View],
    ) -> None:
        self._allocs = allocs
        self._launches = launches
        self._outs = outs

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        bufs = [
            torch.empty(numel, dtype=dtype, device=x.device)
            for numel, dtype in self._allocs
        ]

        def resolve(value: Any) -> Any:
            if isinstance(value, _View):
                base = x if value.slot < 0 else bufs[value.slot]
                if value.whole:
                    return base
                return base.as_strided(value.size, value.stride, value.offset)
            if isinstance(value, list):
                return [resolve(v) for v in value]
            return value

        for fn, device, ins, args in self._launches:
            launch(fn, device, resolve(ins), *(resolve(a) for a in args))
        outs = []
        for view in self._outs:
            out = bufs[view.slot]
            if not view.whole:
                # Outputs own their storage rather than viewing a buffer.
                out.as_strided_(view.size, view.stride, view.offset)
            outs.append(out)
        return tuple(outs)
