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
