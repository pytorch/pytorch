# mypy: allow-untyped-defs
"""A lowering bound to a block's constants: ``vm.lower_to(gm, bind=make)``.

Given: ``make``, a pass's function of the block's constants (its weights: the
inputs of the graph whose type ``is_const``, in order) that returns the
callable of the block's other inputs.

Guarantees: ``make`` is called once for each region that uses the lowering,
and again before a call at which the data of a constant is at another address
than at the call before. Between two calls of ``make`` the constants are the
same tensors at the same addresses, with the same shape, dtype and device, so
what ``make`` returned may read them where they are (a CUDA graph) without
looking. A constant written in place stays where it is: no call of ``make``.

May not assume: that a constant's content is what it was (that is
``vm.Unwritten``); that ``make`` is called once in all (each region binds for
its own constants: a layer, an input shape, a block traced again); that the
callable is the only way in (``BoundLowering.__call__`` takes every input).

The parts, in the order a region meets them: ``BoundLowering`` is the call
``lower_to`` puts in the pass's graph; ``bind_versions`` hands a region's
constants to its versions (torch/vm/_pipeline.py); ``Binding`` is what a
region keeps (torch/_rust_inductor.py's trace_region) and what the executor
has bound again when a constant's data moved (exec/reads.rs).
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.vm import _guards
from torch.vm._guards import address


class BoundLowering:
    """The call in a graph from ``lower_to(gm, bind=make)``. Called with
    every input (by ``verify``, by a pass that wraps it), it looks at the
    constants itself on each call: it binds again when they are other
    tensors, or elsewhere, than at its last call."""

    def __init__(self, make: Callable, constants: list[int], inputs: int) -> None:
        self.make = make
        self.constants = constants
        self.rest = [i for i in range(inputs) if i not in set(constants)]
        # FX names a call by its target's __name__ and __module__.
        self.__name__ = getattr(make, "__name__", "bound")
        self.__module__ = getattr(make, "__module__", None) or __name__
        self._key: list[tuple[int, int]] | None = None
        self._call: Callable | None = None

    def bind(self, consts: Sequence[torch.Tensor]) -> Callable:
        return self.make(list(consts))

    def __call__(self, *inputs):
        consts = [inputs[i] for i in self.constants]
        key = [(id(c), address(c)) for c in consts]
        if self._call is None or key != self._key:
            self._call, self._key = self.make(consts), key
        return self._call(*[inputs[i] for i in self.rest])


def bind_versions(base, run_base, guarded, consts: list[torch.Tensor]) -> Callable:
    """A region's function of its inputs without the constants. ``base`` and
    ``guarded`` are its versions with what each runs (see _pipeline._dispatch,
    whose ``run`` this is with the constants taken out): a version lowered
    with ``lower_to(bind=)`` is bound to ``consts`` here, the others are
    handed them on every call."""

    def bound(call):
        if isinstance(call, BoundLowering):
            return call.bind(consts)
        return lambda *inputs: call(*inputs, *consts)

    calls = [(v, bound(call)) for v, call in guarded]
    call_base = bound(run_base)

    def run(*inputs):
        for v, call in calls:
            failed = _guards.failing(v.checks, (*inputs, *consts))
            if failed is None:
                v.hits += 1
                return call(*inputs)
            key = repr(failed[1])
            v.misses[key] = v.misses.get(key, 0) + 1
        base.hits += 1
        return call_base(*inputs)

    return run


class Binding:
    """What a region keeps of a lowering bound to its constants: ``call``
    takes the region's other inputs. ``rebind`` is for the executor, before
    an entry at which the data of a constant is somewhere else."""

    def __init__(self, bind: Callable, consts: list[torch.Tensor]) -> None:
        self.bind = bind
        self.consts = consts
        self.call: Any = bind(consts)

    def rebind(self) -> None:
        self.call = self.bind(self.consts)
