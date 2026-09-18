"""Plain-data types the multi-graph precompile capture reports through.

A leaf module on purpose. ``import torch`` loads ``torch.compiler``, and through
it ``torch._precompile`` (and, once ``torch.compiler.precompile`` is a module
later in this stack, this module too), without loading ``torch._dynamo``, so a
type that public surface exports cannot live in the Dynamo-side internals,
``torch/_dynamo/precompile_package.py`` (which imports this module at load time
later in this stack), without an import cycle. Import-wise the types could live
in ``torch/_precompile.py``; keeping them out of it is layering: the Dynamo
internals must not depend on the make_fx capture module, which the follow-up
capture session makes an importer of those internals. The types are frozen
dataclasses of immutable fields, so they pickle, compare by value and hash.
"""

import collections
import dataclasses
from collections.abc import Sequence


def _count_types(pairs: Sequence[tuple[str, str]]) -> dict[str, int]:
    return dict(collections.Counter(guard_type for guard_type, _ in pairs))


@dataclasses.dataclass(frozen=True, kw_only=True)
class GuardFact:
    """One guard observed while compiling a frame variant.

    Attributes:
        guard_type: The Dynamo guard type, e.g. ``"TENSOR_MATCH"``.
        source: The guarded source, spelled as ``GuardFilterEntry.name``, i.e.
            the ``Guard.name`` with local scope stripped (``L['x']`` -> ``x``),
            the same spelling as the ``(guard_type, source)`` slots of
            ``PrecompileSummary``: ``"x"``, ``"self.eps"``, ``"G['CFG'].width"``.
            Empty for a guard checked against no source.
        code: The rendered check parts, with the addresses Dynamo interpolates
            scrubbed by the producer, e.g.
            ``("___check_type_id(L['x'], <id>), type=<class 'int'>",)``; empty
            when the guard renders none.
        value: A rendered fragment for what the check compares that its code does
            not show: a tensor's dtype and shape line, or ``"is <callable>"`` for
            an identity guard. Empty when the code says it all.
        enforced: Whether the artifact still checks this guard (it was serialized).
    """

    guard_type: str
    source: str
    code: tuple[str, ...]
    value: str
    enforced: bool


@dataclasses.dataclass(frozen=True)
class PrecompileSummary:
    """Coverage and guard information from an observed precompile capture."""

    frames: int
    resume_functions: int
    guarded_codes: int
    backend_graphs: int
    bypassed: tuple[str, ...] = ()
    truncated: tuple[str, ...] = ()
    uncovered_frames: tuple[str, ...] = ()
    wont_generalize: tuple[str, ...] = ()
    dropped_guards: tuple[tuple[str, str], ...] = ()
    kept_guards: tuple[tuple[str, str], ...] = ()
    risky_dropped_guards: tuple[tuple[str, str], ...] = ()
    # Guards that COULD have been serialized and were not, because they held identically
    # across every captured variant. Reported apart from dropped_guards ("could not be
    # serialized") since the reason and the remedy differ, but reported: a capture that
    # discards a precondition must not look like one that had none.
    policy_dropped_guards: tuple[tuple[str, str], ...] = ()
    # (guard_type, source, rendered check) for each dropped slot that HAS a rendered
    # check. Some do not: EMPTY_NN_MODULE_HOOKS_DICT installs nothing under the default
    # skip_nnmodule_hook_guards, and the global-state guards are checked in C++ against no
    # source, so those appear in the drop lists with no entry here. A slot's type and
    # SOURCE alone can be too little to judge the drop: a dropped
    # ``('HASATTR', "counts['pixel']")`` may be the benign companion of a kept TENSOR_MATCH
    # on the same source, or the only thing guarding an optional attribute. The rendered
    # check tells them apart; it is reported alongside the three lists rather than folded
    # into them, so the slot tuples stay the identity the policy compares on.
    dropped_guard_code: tuple[tuple[str, str, str], ...] = ()
    capture_errors: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether the capture covers everything it exercised.

        False if the capture produced NO guarded code at all, if any frame hit the
        recompile limit, was bypassed or was left uncovered (never reached by the calls),
        or if a capture call raised. ``backend_graphs`` is checked too, because
        ``guarded_codes`` alone cannot tell a real capture from an empty one:
        ``allow_empty_graphs`` lets a frame that compiled nothing still count as one
        guarded code, so a model whose every graph sits behind a recursive
        ``torch._dynamo.disable`` reported complete while carrying no compiled compute.
        """
        return (
            not self.bypassed
            and not self.truncated
            and not self.uncovered_frames
            and not self.capture_errors
            and self.guarded_codes > 0
            and self.backend_graphs > 0
        )

    def dropped_guard_types(self) -> dict[str, int]:
        """Count omitted guards by guard type."""
        return _count_types(self.dropped_guards)

    def kept_guard_types(self) -> dict[str, int]:
        """Count serialized guards by guard type."""
        return _count_types(self.kept_guards)

    def __str__(self) -> str:
        base = (
            f"{self.frames} frames ({self.resume_functions} from graph breaks), "
            f"{self.guarded_codes} guarded codes, "
            f"{self.backend_graphs} backend graphs"
        )
        if self.dropped_guards:
            base += f", dropped guards {self.dropped_guard_types()}"
        if self.risky_dropped_guards:
            # Type AND source: the histogram one label over counts types only, and two
            # risky drops of different types on one source are not the same drop.
            risky = [f"{t} on {s}" for t, s in self.risky_dropped_guards]
            base += f", RISKY drops {risky}"
        if self.policy_dropped_guards:
            base += f", {len(self.policy_dropped_guards)} policy drops"
        if self.uncovered_frames:
            base += (
                f", {len(self.uncovered_frames)} UNCOVERED: "
                f"{list(self.uncovered_frames)}"
            )
        if self.wont_generalize:
            base += f", {len(self.wont_generalize)} value-pinned guards"
        if self.truncated:
            base += f", >={len(self.truncated)} TRUNCATED: {list(self.truncated)}"
        if self.bypassed:
            base += f", {len(self.bypassed)} BYPASSED: {list(self.bypassed)}"
        if self.capture_errors:
            base += f", {len(self.capture_errors)} CAPTURE ERROR(S)"
        return base
