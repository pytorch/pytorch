# mypy: allow-untyped-defs
"""What a traced run did to memory that outlives it, and how that is taken
back when the trace is given up and its block runs again from the start.

A traced run has ONE record, an ``Undo``: its writes to the memory of
tensors and its draws from random generators, in the order they happened,
each with what was there before. ``back()`` undoes them, the last first.
Every byte is then what it was when the first write that could change it
was told, which is what it was when the run began: whatever two tensors
that were written have to do with each other (one object, a view and its
base, views that overlap, another dtype, two storages over one buffer), no
question about it is asked, only the order.

Whoever knows that a call is about to write tells the record BEFORE the
call runs (``write``): the row of the engine's rule table for a C callee
that was measured to write an operand (exec/rules/torch.rs), and the
trace's dispatch mode for an operator (stays.written: what its schema
marks, and the writes no schema marks). A write that is told twice, or
that is to memory an earlier entry holds whole, adds nothing.

What an entry holds: the BYTES of the elements of the tensor that is about
to be written, as they are in its storage then; a copy the size of the
tensor, as the copy of the tensor was that this record replaces. It is
taken, and written back, through a view of bytes over the storage that is
the record's own, so that nothing comes between the bytes and their place:
not what the run does to the tensor afterwards (sizes, strides, offset or
storage changed in place), not what an operator checks of a tensor it
writes (a leaf that requires grad, an inference tensor, elements that
repeat: ``fill_`` writes an expanded tensor, ``copy_`` does not), not a
version counter. An ``out=`` is resized to its result before it is written:
its entry holds its storage from its first element on. A tensor that is not
strided has no such bytes: it is kept whole, once, and copied back.

The version counters the run's writes moved are set back to what each
tensor that was told of counted then, the first told last.

``empty()`` is what "the run left nothing behind" reads (a region whose
graph raises is a miss only then), ``undeclared`` what "a schema marks
every write of the run" reads (no host prologue otherwise). Imports
nothing of this package.
"""

import torch
from torch.utils._python_dispatch import _disable_current_modes


_BYTES, _WHOLE, _DREW = "bytes", "whole", "drew"


class Undo:
    def __init__(self) -> None:
        # In the order told: (_BYTES, the view of bytes, their copy),
        # (_WHOLE, the tensor, its copy), (_DREW, the generator, its state).
        self._log: list = []
        # By id, each held: a tensor told of, with the version it counted
        # then (None: an inference tensor counts none); a generator.
        self._told: dict = {}
        # What the entries hold of a storage (by its address; an entry's
        # view holds it): the ranges of bytes held whole, and the views.
        self._whole: dict = {}
        self._views: set = set()
        # Whether a write was told that no schema marks.
        self.undeclared = False

    def write(self, t, declared=True, grows=False) -> None:
        """Told before a call that may write the memory of the tensor ``t``
        (``grows``: as an ``out=``, which may be resized first)."""
        if not declared:
            self.undeclared = True
        # The record's own tensors are none of the trace's, none of a mode
        # of the program's, and no inference tensors (the program may have
        # left an inference mode when they are written back).
        with (
            _disable_current_modes(),
            torch._C.DisableTorchFunction(),
            torch._C._InferenceMode(False),
        ):
            first = id(t) not in self._told
            if first:
                self._told[id(t)] = (t, None if t.is_inference() else t._version)
            if t.layout is not torch.strided:
                if first:
                    self._log.append((_WHOLE, t, t.detach().clone()))
                return
            size = t.element_size()
            storage = t.untyped_storage()
            at = t.storage_offset() * size
            if grows:
                sizes, strides = (storage.nbytes() - at,), (1,)
            elif t.numel() == 0:
                return
            else:
                # A dimension that repeats its elements is one of memory.
                sizes = (
                    *(1 if s == 0 else n for n, s in zip(t.shape, t.stride())),
                    size,
                )
                strides = (*(s * size for s in t.stride()), 1)
            if min(sizes) <= 0:
                return
            end = at + sum((n - 1) * s for n, s in zip(sizes, strides)) + 1
            view = (storage._cdata, at, sizes, strides)
            whole = self._whole.setdefault(storage._cdata, [])
            if view in self._views or any(a <= at and end <= b for a, b in whole):
                return
            held = torch.empty(0, dtype=torch.uint8, device=t.device)
            held.set_(storage, at, sizes, strides)
            self._log.append((_BYTES, held, held.clone()))
            self._views.add(view)
            if held.is_contiguous():
                whole.append((at, end))

    def draw(self, generator) -> None:
        """Told before a call that may draw from ``generator``."""
        if id(generator) not in self._told:
            self._told[id(generator)] = (generator, None)
            self._log.append((_DREW, generator, generator.get_state()))

    def empty(self) -> bool:
        """Whether the run was told of no write and no draw."""
        return not self._told

    def nbytes(self) -> int:
        """How many bytes of tensors the record holds copies of."""
        copies = [saved for kind, _, saved in self._log if kind is not _DREW]
        return sum(saved.numel() * saved.element_size() for saved in copies)

    def kept(self) -> list:
        """What the record holds of the program's: the tensors and the
        generators it was told of, and its views of their storages."""
        return [*self._told.values(), *self._log]

    def clear(self) -> None:
        """Forgets the writes and the draws (not ``undeclared``): what the
        run did stands."""
        self._log.clear()
        self._told.clear()
        self._whole.clear()
        self._views.clear()

    def back(self) -> None:
        """Takes back what the run did, the last first, and forgets it."""
        with (
            _disable_current_modes(),
            torch._C.DisableTorchFunction(),
            torch.no_grad(),
        ):
            for kind, x, saved in reversed(self._log):
                if kind is _DREW:
                    x.set_state(saved)
                else:
                    x.copy_(saved)
            # The run's writes are none of the program's (and writing a
            # tensor back whole counted one more).
            for x, version in reversed(self._told.values()):
                if version is not None:
                    torch._C._autograd._unsafe_set_version_counter((x,), (version,))
        self.clear()
