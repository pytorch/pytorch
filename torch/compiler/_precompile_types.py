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


@dataclasses.dataclass(frozen=True, kw_only=True)
class PrecompileSummary:
    """Coverage and guard information from an observed precompile capture.

    ``str(summary)`` renders a one-line digest. Everything here describes the
    calls that ran, not every possible input or unexecuted branch, and once
    ``truncated`` is non-empty every count and frame list is a lower bound: the
    frames called beneath a truncated one ran untraced, so nothing here saw them.

    The guard fields hold ``(guard_type, source)`` slots, the source spelled as
    ``GuardFilterEntry.name``, i.e. the ``Guard.name`` with local scope stripped
    (``L['self'].act`` -> ``self.act``; ``G['CFG'].width`` unchanged). Each list
    holds a slot once, however many frames or variants carried it, so
    ``dropped_guard_types`` counts distinct slots, not occurrences. The relations
    between the lists hold within one frame variant, where the producer decides
    them, and are stated here rather than checked:

    * ``kept_guards`` and ``dropped_guards`` are disjoint: a guard is
      serialized or it is not.
    * ``risky_dropped_guards`` is drawn from ``dropped_guards``.
    * ``policy_dropped_guards`` is disjoint from ``dropped_guards``.
    * ``dropped_guard_code`` draws its slots from ``dropped_guards`` and
      ``policy_dropped_guards``.

    The lists aggregate every captured frame and a slot names no frame, so two
    frames' ``self.act`` are one slot: a slot one frame kept and another dropped
    is in both ``kept_guards`` and ``dropped_guards`` (the digest's ``(N kept)``
    is the kept count beside the drops, not the other half of a total), and a
    slot one frame's filter rejected and another frame's invariance policy
    dropped is in both drop lists. Nothing is enforced, as for the frame lists
    below: a report that raised on its own bookkeeping would lose the coverage
    it exists to describe.

    The frame lists (``bypassed``, ``truncated``, ``uncovered_frames``) hold
    bare ``co_name``s, which are not unique: every ``nn.Module`` has a
    ``forward``, so ``['forward', 'forward']`` is two frames, not a repeat, and
    the lists can neither identify a frame nor be checked disjoint.

    Attributes:
        frames: Captured frames.
        resume_functions: Of those, the graph-break continuations.
        guarded_codes: Guarded code objects across all frames.
        backend_graphs: Compiled backend graphs.
        bypassed: ``co_name``s of frames the package refused to record: every compile
            of the frame was bypassed because its guards could not be
            serialized, or a backend artifact was missing when the package was
            saved. Not an eager fallback: the frame ran compiled during capture
            (Dynamo installed its guarded code, it just went unrecorded), the
            artifact carries no variant of it, and an install re-traces it
            rather than skipping it as trivial.
        truncated: ``co_name``s of frames that hit the recompile limit. A lower bound,
            which is why the digest prints it as ``>=``: Dynamo runs a frame
            that hit the limit, and every frame called beneath it, without
            tracing (its ``FrameExecStrategy`` is ``RUN_ONLY`` for the frame and
            its callees alike), so a limit hit below the first one is never seen.
        uncovered_frames: ``co_name``s of frames that ended with no guarded code and
            were not bypassed, so the artifact cannot serve them: a thin wrapper
            whose graphs all landed in an inner frame, or a frame Dynamo gave up
            on. The remainder after ``bypassed``, which also holds no guarded
            code but has a different cause and remedy.
        wont_generalize: Guard *sources* (not frame names) that every captured
            variant pins to one value, so no variant will serve another value.
        dropped_guards: Slots the serialized copy's guard filter rejected: the
            guards the serializer refuses (the identity guards, plus
            ``DICT_VERSION`` and ``WEAKREF_ALIVE``), plus whatever a
            caller-supplied filter dropped.
        kept_guards: Slots the artifact still checks.
        risky_dropped_guards: The subset of ``dropped_guards`` observed to tell
            captured variants apart, or flagged by the risky-drop lint as a
            configuration-chosen binding.
        policy_dropped_guards: Serializable slots dropped because they held
            identically across every variant. Reported apart from
            ``dropped_guards`` because the remedy differs, and reported at all
            because a capture that discards a precondition should not look like
            one that had none.
        dropped_guard_code: ``(guard_type, source, rendered_check)``, one per slot
            of ``dropped_guards`` or ``policy_dropped_guards`` whose guard
            rendered a check, i.e. whose ``GuardBuilder`` method set
            ``Guard.code_list``. How the check is installed does not predict
            that (``AUTOGRAD_SAVED_TENSORS_HOOKS`` checks through a lambda and
            ``DEFAULT_DEVICE`` through a C++ leaf, and both export code); the
            methods that render none today include ``GLOBAL_STATE``,
            ``TORCH_FUNCTION_STATE``, ``DISPATCH_KEY_SET_MATCH`` and
            ``TENSOR_SUBCLASS_METADATA_MATCH``, and ``CLOSURE_MATCH`` on a plain
            function, which guards its ``__code__`` through a separate guard
            (on any other callable it falls through to ``ID_MATCH`` and renders
            that check). Such a slot has no entry here. The entry holds one
            rendering however many variants dropped the slot: a check that
            embeds the guarded value (``EQUALS_MATCH`` renders ``L['n'] == 3``)
            differs between variants that differ in the value, and the entry
            then shows one variant's, the producer's pick rather than a merge,
            so it tells the form of the check and not the value; that the slot
            varied at all is what ``risky_dropped_guards`` records.
            A slot alone can be ambiguous: a dropped ``('HASATTR', "counts['pixel']")``
            is either the benign companion of a kept ``TENSOR_MATCH`` on the same
            source or the only guard on an optional attribute, and only the
            rendered check tells them apart. Kept beside the slot lists rather
            than folded into them, so the slots stay the identity the policy
            compares on; for programmatic consumers, not the digest.
        capture_errors: Messages from capture calls that raised.
    """

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
    policy_dropped_guards: tuple[tuple[str, str], ...] = ()
    dropped_guard_code: tuple[tuple[str, str, str], ...] = ()
    capture_errors: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether the capture covers everything it exercised.

        Coverage only: False if a frame was ``bypassed``, hit the recompile
        limit (``truncated``) or ended with no guarded code
        (``uncovered_frames``), if a capture call raised (``capture_errors``),
        or if nothing was compiled. Both ``guarded_codes`` and ``backend_graphs``
        must be non-zero for the last check, because ``allow_empty_graphs`` lets
        a frame that compiled nothing still count as one guarded code, so
        ``guarded_codes`` alone cannot tell a real capture from an empty one.
        The guard fields never enter: ``risky_dropped_guards`` is a lint
        finding, not a proof, and ``wont_generalize`` follows from any value the
        capture pinned, so both describe how far a complete capture generalizes
        and the digest reports them on their own.
        """
        return (
            not self.bypassed
            and not self.truncated
            and not self.uncovered_frames
            and not self.capture_errors
            and self.guarded_codes > 0
            and self.backend_graphs > 0
        )

    @property
    def dropped_guard_types(self) -> dict[str, int]:
        """The distinct dropped slots counted by guard type."""
        return dict(collections.Counter(t for t, _ in self.dropped_guards))

    @property
    def kept_guard_types(self) -> dict[str, int]:
        """The distinct kept slots counted by guard type."""
        return dict(collections.Counter(t for t, _ in self.kept_guards))

    def __str__(self) -> str:
        def count(k: int, noun: str) -> str:
            return f"{k} {noun}" if k == 1 else f"{k} {noun}s"

        base = (
            f"{count(self.frames, 'frame')} ({self.resume_functions} from graph breaks), "
            f"{count(self.guarded_codes, 'guarded code')}, "
            f"{count(self.backend_graphs, 'backend graph')}"
        )
        if self.dropped_guards:
            kept = len(self.kept_guards)
            base += f", dropped guards {self.dropped_guard_types} ({kept} kept)"
        if self.risky_dropped_guards:
            base += f", RISKY drops {[src for _, src in self.risky_dropped_guards]}"
        if self.policy_dropped_guards:
            policy = len(self.policy_dropped_guards)
            base += f", {count(policy, 'policy-dropped guard')}"
        if self.wont_generalize:
            base += f", {count(len(self.wont_generalize), 'value-pinned source')}"
        if self.uncovered_frames:
            base += (
                f", {len(self.uncovered_frames)} UNCOVERED: "
                f"{list(self.uncovered_frames)}"
            )
        if self.truncated:
            base += f", >={len(self.truncated)} TRUNCATED: {list(self.truncated)}"
        if self.bypassed:
            base += f", {len(self.bypassed)} BYPASSED: {list(self.bypassed)}"
        if self.capture_errors:
            errors = len(self.capture_errors)
            base += f", {errors} CAPTURE ERROR{'' if errors == 1 else 'S'}"
        return base
