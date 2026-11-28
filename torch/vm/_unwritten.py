# mypy: allow-untyped-defs
"""``vm.Unwritten``: a pass keeps what it derived from a weight's content.

Given: a weight of a block (an input of the graph whose type ``is_const``), on
which a block pass puts the guard with ``vm.guard(gm, weight,
vm.Unwritten())`` before it reads the weight.

A pass list that uses it is a fast-mode list: the guard is not part of
``vm.DEFAULT_PASSES``, whose results are eager's without an exception,
because of the last paragraph here.

What it sees: every change of the tensor's content made through PyTorch
after the guard was placed, wherever it is made (inside the VM or not, on
any thread): an in-place operation on the tensor or on anything over its
memory (a view, ``.data``, ``detach()``, a tensor put on its storage),
``param.data = ...``, a parameter bound to another tensor, and compiled
code that reports its write (it bumps the version counter).

It also fails, without a write, when the memory's address is handed out
after it was placed (``data_ptr()``, ``.numpy()``, DLPack, compiled code
that reads the tensor): from then on what is done through that address
can't be seen. ``vm.address(t)`` reads where a tensor is without that,
and ``t.data_ptr()`` is how a pass ends a guard on purpose. For the same
reason the guard holds once: after it failed for a tensor it can't be
placed on that tensor again (``vm.guard`` raises ``vm.CannotGuard``),
unless all that happened was a reported write. It also can't be placed
on memory that isn't PyTorch's alone (shared memory, memory from NumPy,
DLPack or a buffer), on a tensor that isn't dense, or off the CPU and
CUDA.

A CUDA graph capture ends the guards of the CUDA tensors: PyTorch refuses
a capture that reads a tensor carrying the guard's mark, so the marks are
cleared when a capture begins (a capture begun from C++, not through
``torch.cuda.CUDAGraph``, is not seen, and raises if it reads one). A
lowering whose own capture reads none of the guarded tensors keeps them
with ``vm.keep_unwritten()``.

What it can't see, and nothing in PyTorch can: a write through an address
that left PyTorch before the guard was placed. A NumPy array or a DLPack
consumer made earlier, a CUDA graph captured earlier that writes the
tensor and is replayed without a version bump, a pointer kept by C code,
another process on the same memory. With such a write the block goes on
computing with what the pass derived.

How: the tensor's version counter, and a copy-on-write mark on its
storage that the first request for a mutable pointer clears (nothing is
copied: the mark's only other owner is dropped at once).

The parts: ``Unwritten`` is the check; ``watched`` says, for the region's
checks (exec/reads.rs, through torch._rust_inductor) and for the memo of
the passes' output, which version counter a guarded tensor is to keep;
``keep_unwritten`` is for a lowering that captures CUDA graphs.
"""

import contextlib
import dataclasses
import weakref
from typing import Any

import torch
from torch.vm._cfg import CannotGuard
from torch.vm._examples import raw
from torch.vm._guards import Check


class Unwritten(Check):
    """A pass's guard on a weight: it has not been written to since the
    guard was placed (see the top of torch/vm/_unwritten.py for what that
    covers and what it does not). ``vm.guard(gm, weight, vm.Unwritten())``
    places it, or raises ``vm.CannotGuard``; when it fails, the block is
    traced again and the pass runs again."""

    def holds(self, t, value_of):
        return watched(t) is not None

    def place(self, gm, node):
        # Now, before the pass reads the weight.
        placeholders = list(gm.graph.find_nodes(op="placeholder"))
        examples = raw(gm)
        weights = gm.meta.get("vm_const_inputs", ())
        if node not in placeholders or node.name not in weights:
            raise CannotGuard(f"{self!r} on {node.name}: it goes on a weight")
        at = placeholders.index(node)
        if at >= len(examples):
            raise CannotGuard(f"{self!r} on {node.name}: placed while the pass runs")
        why = _watch(examples[at])
        if why is not None:
            raise CannotGuard(f"{self!r} on {node.name}: {why}")

    def __repr__(self):
        return "Unwritten()"


@dataclasses.dataclass
class _Watch:
    """A tensor under vm.Unwritten: the version counter it is to keep, and
    whether the guard has failed for good."""

    tensor: Any
    version: int
    spent: bool = False


# By id(tensor); the references have no callback (see
# torch._rust_inductor._made).
_watches: dict[int, _Watch] = {}
# While positive, a CUDA graph capture leaves the marks (see keep_unwritten).
_keeping = 0


def _watch(t: torch.Tensor) -> str | None:
    """Places vm.Unwritten on ``t``: None, or why it can't be."""
    if type(t) not in (torch.Tensor, torch.nn.Parameter) or t.layout != torch.strided:
        return "it is no plain dense tensor"
    if t.device.type not in ("cpu", "cuda") or t.is_inference():
        return "it is an inference tensor, or neither on the CPU nor on CUDA"
    # is_shared() is True of every CUDA tensor.
    shared = t.device.type == "cpu" and t.is_shared()
    if shared or not t.untyped_storage().resizable():
        return "its memory is shared, or came from NumPy, DLPack or a buffer"
    watch = _watches.get(id(t))
    if watch is not None and watch.tensor() is t:
        if watch.spent or not torch._C._is_cow_tensor(t):
            watch.spent = True
            return "it was written to, or its address handed out, under the guard"
        # At most a reported write: nothing asked for the memory.
        watch.version = t._version
        return None
    with torch.no_grad():
        # The clone is dropped here: the mark stays, with this storage alone
        # on the memory, so that clearing it copies nothing.
        torch._lazy_clone(t.detach())
    if not torch._C._is_cow_tensor(t):
        return "its storage takes no copy-on-write mark"
    if _clear_marks not in torch.cuda.graphs._before_capture_hooks:
        torch.cuda.graphs._before_capture_hooks.append(_clear_marks)
    for key in [key for key, old in _watches.items() if old.tensor() is None]:
        del _watches[key]
    _watches[id(t)] = _Watch(weakref.ref(t), t._version)
    return None


def watched(t: torch.Tensor) -> int | None:
    """The version counter ``t`` is to keep under a vm.Unwritten that holds
    now; None if it is under none, or was written to."""
    watch = _watches.get(id(t))
    if watch is None or watch.tensor() is not t or watch.spent:
        return None
    if not torch._C._is_cow_tensor(t):
        watch.spent = True
        return None
    return watch.version if t._version == watch.version else None


def _clear_marks() -> None:
    """Before a CUDA graph capture: PyTorch refuses a capture that reads a
    marked CUDA tensor, so the marks go, and with them the guards."""
    if _keeping:
        return
    for watch in _watches.values():
        t = watch.tensor()
        if t is not None and not watch.spent and t.is_cuda:
            watch.spent = True
            # A request for a mutable pointer: what clears the mark.
            t.data_ptr()


@contextlib.contextmanager
def keep_unwritten():
    """Inside it a CUDA graph capture does not end the ``vm.Unwritten``
    guards. For a lowering that captures a graph of its own which reads none
    of the guarded tensors (it reads what the pass derived from them); if the
    capture does read one, PyTorch raises."""
    global _keeping
    _keeping += 1
    try:
        yield
    finally:
        _keeping -= 1
