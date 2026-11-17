"""Passes that are made from what the user hands over.

``swap`` is the pass an author of a kernel needs: it finds a run of tensor
operations in the function, replaces it by one call of the kernel, and checks
the kernel against the operations it replaced.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch


class Mismatch(RuntimeError):
    """A kernel gave something else than the operations it replaced."""


# Instructions that only make a value out of nothing the function computed: a
# constant, a global, an attribute of one. In front of a run they belong to
# it, so that the run is handed tensors and not ``torch.rsqrt``.
_SETUP = ("const", "load_global", "load_attr", "load_method")


def _is_setup(inst: dict[str, Any]) -> bool:
    return inst["op"] in _SETUP and inst["replayable"]


def _names(operator: str) -> tuple[str, str]:
    """``aten::mul.Tensor`` as ("mul.Tensor", "mul")."""
    full = operator.split("::", 1)[-1]
    return full, full.split(".", 1)[0]


def _same(wanted: str, found: str) -> bool:
    full, base = _names(found)
    return _names(wanted)[0] in (full, base)


class _Swapped:
    """What stands in for the run: the kernel, and the run to check it by."""

    def __init__(
        self,
        kernel: Callable[..., Any],
        replay: Any,
        span: tuple[int, int],
        check: int,
        tolerance: tuple[float, float],
        name: str,
    ) -> None:
        self.kernel = kernel
        self.replay = replay
        self.span = span
        self.inputs, self.outputs = replay.io(*span)
        self.checks_left = check
        self.tolerance = tolerance
        self.name = name
        self.calls = 0

    def __call__(self, *values: Any) -> tuple[Any, ...]:
        self.calls += 1
        if self.checks_left > 0:
            self.checks_left -= 1
            # The run first: it is the reference, and it gets the inputs as
            # the function had them.
            env = dict(zip(self.inputs, values))
            self.replay.run(*self.span, env, False)
            expected = tuple(env[v] for v in self.outputs)
            found = self.give(values)
            for i, (want, have) in enumerate(zip(expected, found)):
                self.compare(i, want, have)
            return found
        return self.give(values)

    def give(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        out = self.kernel(*values)
        if len(self.outputs) == 1:
            return (out,)
        out = tuple(out)
        if len(out) != len(self.outputs):
            raise Mismatch(
                f"{self.name} gave {len(out)} values where the operations it "
                f"replaced give {len(self.outputs)}"
            )
        return out

    def compare(self, i: int, want: Any, have: Any) -> None:
        if isinstance(want, torch.Tensor):
            rtol, atol = self.tolerance
            same = (
                isinstance(have, torch.Tensor)
                and have.shape == want.shape
                and have.dtype == want.dtype
                and have.device == want.device
                and torch.allclose(have, want, rtol=rtol, atol=atol, equal_nan=True)
            )
            if same:
                return
            if isinstance(have, torch.Tensor) and have.shape == want.shape:
                what = f"differs by up to {(have - want).abs().max().item():.3g}"
            elif isinstance(have, torch.Tensor):
                what = f"has shape {tuple(have.shape)}, not {tuple(want.shape)}"
            else:
                what = f"is a {type(have).__name__}, not a tensor"
        elif type(want) is type(have) and want == have:
            return
        else:
            what = f"is {have!r}, not {want!r}"
        raise Mismatch(
            f"{self.name}: result {i} {what} on call {self.calls} "
            "(the operations it replaced were run on the same inputs)"
        )


def fuse_checked(
    region: Any,
    start: int,
    end: int,
    kernel: Callable[..., Any],
    checks: int = 8,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> bool:
    """Replaces instructions ``start:end`` of a region by a call of ``kernel``.

    The kernel takes what the instructions take from before them and returns
    what they give to what follows (one value, or a tuple). For its first
    ``checks`` calls the instructions are run on the same inputs, and a
    kernel that gives something else raises :class:`Mismatch`. Returns
    whether the instructions were replaced.
    """
    name = getattr(kernel, "__name__", type(kernel).__name__)
    replay = region.replayer(start, end)
    stand_in = _Swapped(kernel, replay, (start, end), checks, (rtol, atol), name)
    return region.fuse(start, end, stand_in)


def swap(
    ops: Iterable[str],
    kernel: Callable[..., Any],
    *,
    check: bool | int = True,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Callable[[Any], bool]:
    """A pass that replaces a run of tensor operations by a call of ``kernel``.

    The pass looks, in every basic block, for instructions that were seen to
    be exactly these ATen operators, in this order, with nothing between them
    that calls a kernel or has an effect, and replaces the run by one call of
    ``kernel``. Where it goes is decided by what the function does, not by
    how it is written: the run can come from one line or from several
    inlined functions, and every place that has it is found.

    Args:
        ops: the operators, as ``region.instructions()`` names them in its
            ``kernel`` field: ``"aten::mul.Tensor"``, or without the overload
            or the namespace (``"mul"``), which matches every overload.
        kernel: called with what the run takes from before it, in the order
            the run first uses it: its tensors, and whatever else it reads
            that is not a constant or a global. It returns what the run gives
            to what follows: one value, or a tuple of them in the order they
            are computed.
        check: run the operations that were replaced on the same inputs, for
            the first calls (8 with ``True``, or as many as given), and raise
            :class:`Mismatch` if the kernel gives something else: a tensor of
            another shape, dtype or device, or one that is not close
            (``rtol``, ``atol``). The operations must not write to their
            inputs. ``False`` trusts the kernel.

    The pass goes in front of the passes that rewrite single instructions::

        passes = [torch.rewrite.swap(["pow", "mean", "add", "rsqrt", "mul"], kernel)]
        torch.rewrite(fn, passes=[*passes, *torch.rewrite.canonical])
    """
    wanted = list(ops)
    if not wanted:
        raise ValueError("swap needs at least one operator")
    checks = 8 if check is True else int(check)
    name = getattr(kernel, "__name__", type(kernel).__name__)

    def swap_pass(region: Any) -> bool:
        insts = region.instructions()
        found = [i for i in insts if i["kernel"] is not None]
        for at in range(len(found) - len(wanted) + 1):
            run = found[at : at + len(wanted)]
            if not all(_same(w, i["kernel"]) for w, i in zip(wanted, run)):
                continue
            start, end = run[0]["index"], run[-1]["index"] + 1
            if not all(i["replayable"] for i in insts[start:end]):
                continue
            # What only sets a call up comes along (a constant, a global, the
            # method of a tensor the run has anyway): the setup in front of
            # the run, as far back as it takes nothing new from the function
            # and as far as the run uses it.
            low = start
            while low > 0 and _is_setup(insts[low - 1]):
                low -= 1
            made = {i["dst"] for i in insts[start:end]}
            taken = {u for i in insts[start:end] for u in i["uses"]} - made
            first = start
            for k in range(low, start):
                known = taken | {i["dst"] for i in insts[k:start]}
                if all(use in known for i in insts[k:start] for use in i["uses"]):
                    first = k
                    break
            used = {use for i in insts[first:end] for use in i["uses"]}
            while first < start and insts[first]["dst"] not in used:
                first += 1
            if fuse_checked(region, first, end, kernel, checks, rtol, atol):
                return True
        return False

    swap_pass.__name__ = f"swap_{name}"
    return swap_pass
