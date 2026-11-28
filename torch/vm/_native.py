"""Passes written in Rust, loaded from a library (see ``load_passes``).

The library is built on the crate ``torch-vm-pass`` (``torch/_rust/sdk``),
which links against neither Python nor PyTorch: a pass sees a region through a
table of C functions (the crate's ``Api``, interface version 1) that this
module fills in over the region's graph. To the pass, the graph's calls are
the instructions of a block, in order: each has the operator it calls
(``aten::mul.Tensor``), the value it defines, the values it uses and whether
it can be run again; a run of them can be replaced by a call of a Python
callable (``fuse``), or by a kernel that is checked against the run it
replaces as :func:`torch.vm.swap` checks it (``swap``).
"""

import ctypes
import operator
import os
from typing import Any

import torch
import torch.fx
from torch.vm import _graph


ABI_VERSION = 1

_LEN = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)
_TEXT = ctypes.CFUNCTYPE(
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
)
_DST = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
_USES = ctypes.CFUNCTYPE(
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t,
)
_REPLAYABLE = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t)
_FUSE = ctypes.CFUNCTYPE(
    ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p
)
_SWAP = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_uint32,
)
_ERROR = ctypes.CFUNCTYPE(None, ctypes.c_char_p)


class _Api(ctypes.Structure):
    """The crate's ``Api``: same fields, same order."""

    _fields_ = [
        ("version", ctypes.c_uint32),
        ("len", _LEN),
        ("op", _TEXT),
        ("kernel", _TEXT),
        ("dst", _DST),
        ("uses", _USES),
        ("replayable", _REPLAYABLE),
        ("fuse", _FUSE),
        ("swap", _SWAP),
        ("error", _ERROR),
    ]


_RUN = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(_Api),
    ctypes.c_void_p,
)


class _RawPass(ctypes.Structure):
    """The crate's ``RawPass``."""

    _fields_ = [("name", ctypes.c_char_p), ("run", _RUN), ("state", ctypes.c_void_p)]


class _Region:
    """A region's graph as a pass of a library sees it: its calls, in order."""

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        self.gm = gm
        # What a callback raised, or the pass's own error: raised once the
        # pass has returned (an exception cannot cross the library).
        self.error: BaseException | None = None
        self.changed()

    def changed(self) -> None:
        nodes = list(self.gm.graph.nodes)
        self.ids = {n: i for i, n in enumerate(nodes)}
        self.calls = [n for n in nodes if n.op == "call_function"]

    def kernel(self, n: torch.fx.Node) -> str:
        return (
            _graph._operator(n) if isinstance(n.target, torch._ops.OpOverload) else ""
        )

    def replayable(self, n: torch.fx.Node) -> bool:
        if isinstance(n.target, torch._ops.OpOverload):
            return not _graph._writes_or_draws(n)
        return n.target is operator.getitem

    def replace(self, start: int, end: int, target_of: Any) -> int:
        """Replaces calls ``start..end`` by one call of ``target_of(segment)``;
        1 if replaced, 0 if the range cannot be."""
        nodes = self.calls[start:end]
        if not nodes or not all(map(self.replayable, nodes)):
            return 0
        seg = _graph._segment_of(self.gm, nodes)
        if seg is None:
            return 0
        _graph.replace_segment(self.gm, seg, target_of(seg))
        self.changed()
        return 1


# The regions passes are running on, by the handle they were given.
_regions: dict[int, _Region] = {}


def _guarded(failed: int):
    """A callback of the table: what it raises is kept for the region, and
    ``failed`` returned to the library."""

    def wrap(fn):
        def callback(handle, *args):
            region = _regions[handle]
            try:
                return fn(region, *args)
            except BaseException as e:
                region.error = e
                return failed

        return callback

    return wrap


def _text(text: str, buf, cap: int) -> int:
    data = text.encode()
    for i, b in enumerate(data[:cap]):
        buf[i] = b
    return len(data)


@_guarded(0)
def _len(region: _Region) -> int:
    return len(region.calls)


@_guarded(0)
def _op(region: _Region, index: int, buf, cap: int) -> int:
    return _text("call", buf, cap)


@_guarded(0)
def _kernel(region: _Region, index: int, buf, cap: int) -> int:
    return _text(region.kernel(region.calls[index]), buf, cap)


@_guarded(0xFFFFFFFF)
def _dst(region: _Region, index: int) -> int:
    return region.ids[region.calls[index]]


@_guarded(0)
def _uses(region: _Region, index: int, buf, cap: int) -> int:
    used = [region.ids[a] for a in region.calls[index].all_input_nodes]
    for i, u in enumerate(used[:cap]):
        buf[i] = u
    return len(used)


@_guarded(0)
def _replayable(region: _Region, index: int) -> int:
    return int(region.replayable(region.calls[index]))


def _object(pointer: int) -> Any:
    return ctypes.cast(pointer, ctypes.py_object).value


@_guarded(-1)
def _fuse(region: _Region, start: int, end: int, callable_: int) -> int:
    fn = _object(callable_)
    name = getattr(fn, "__name__", type(fn).__name__)

    def target_of(seg):
        def fused(*values):
            return tuple(fn(*values))

        return _graph._as_target(fused, f"fuse_{name}")

    return region.replace(start, end, target_of)


@_guarded(-1)
def _swap(region: _Region, start: int, end: int, kernel: int, checks: int) -> int:
    fn = _object(kernel)

    def target_of(seg):
        _graph._replacements(region.gm).append(
            (fn, _graph.SWAP_ATOL, _graph.SWAP_RTOL, "same")
        )
        return _graph._checked(
            fn,
            checks,
            _graph.SWAP_RTOL,
            _graph.SWAP_ATOL,
            seg.as_callable(),
            len(seg.outputs),
        )

    return region.replace(start, end, target_of)


# The message of the error the pass running has set (it has no region at
# hand when it sets one).
_message: list[str] = []


def _error(message: bytes) -> None:
    _message.append(message.decode(errors="replace"))


_API = _Api(
    ABI_VERSION,
    _LEN(_len),
    _TEXT(_op),
    _TEXT(_kernel),
    _DST(_dst),
    _USES(_uses),
    _REPLAYABLE(_replayable),
    _FUSE(_fuse),
    _SWAP(_swap),
    _ERROR(_error),
)


class NativePass:
    """A pass of a library loaded with :func:`torch.vm.load_passes`: a block
    pass, an entry for ``passes=``. ``bind(obj)`` gives the pass with an
    object that is handed to it on every run (the kernel it puts in, as a
    rule)."""

    def __init__(self, library: Any, raw: _RawPass, bound: Any = None) -> None:
        self._library = library
        self._raw = raw
        self._bound = bound
        self.name = self.__name__ = (raw.name or b"?").decode()

    def bind(self, obj: Any) -> "NativePass":
        return NativePass(self._library, self._raw, obj)

    def __call__(
        self, gm: torch.fx.GraphModule, input_types: list[Any]
    ) -> torch.fx.GraphModule:
        region = _Region(gm)
        handle = id(region)
        _regions[handle] = region
        del _message[:]
        try:
            bound = None if self._bound is None else id(self._bound)
            status = self._raw.run(self._raw.state, handle, ctypes.byref(_API), bound)
        finally:
            del _regions[handle]
        if region.error is not None:
            raise region.error
        if status < 0:
            raise RuntimeError(
                _message[-1] if _message else f"the pass {self.name} failed"
            )
        return gm

    def __repr__(self) -> str:
        return f"<pass {self.name} of {self._library._name}>"


def load_passes(path: str | os.PathLike[str]) -> dict[str, NativePass]:
    """Loads passes written in Rust from a library, by name.

    The library is a ``cdylib`` crate that depends on the crate
    ``torch-vm-pass`` (``torch/_rust/sdk`` in the source tree), implements its
    ``Pass`` trait and ends with ``export_passes!``. Such a pass is a block
    pass: it sees a region's graph as a run of instructions (the operator
    each calls, the values they define and use) and can replace a run of them
    by a call. A library links against neither Python nor PyTorch, so one
    that was built once keeps working::

        passes = torch.vm.load_passes("target/release/libmine.so")
        fuse = passes["fuse_rmsnorm"].bind(kernel)  # what the pass puts in
        torch.vm(fn, passes=[*torch.vm.DEFAULT_PASSES, fuse])

    ``bind`` returns the pass with an object that is handed to it on every
    run: the kernel, as a rule, which stays a Python callable.
    """
    path = os.fspath(path)
    try:
        library = ctypes.CDLL(path)
    except OSError as e:
        raise RuntimeError(f"cannot load {path}: {e}") from e
    try:
        abi, table = library.torch_vm_abi, library.torch_vm_passes
    except AttributeError:
        raise RuntimeError(
            f"{path} exports no passes (it has no torch_vm_abi and torch_vm_passes)"
        ) from None
    abi.restype = ctypes.c_uint32
    abi.argtypes = []
    found = abi()
    if found != ABI_VERSION:
        raise RuntimeError(
            f"{path} was built for version {found} of the pass interface; this torch.vm has version {ABI_VERSION}"
        )
    table.restype = ctypes.c_size_t
    table.argtypes = [ctypes.POINTER(ctypes.POINTER(_RawPass))]
    first = ctypes.POINTER(_RawPass)()
    count = table(ctypes.byref(first))
    passes = [NativePass(library, first[i]) for i in range(count)]
    return {p.name: p for p in passes}
