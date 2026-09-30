"""
Helpers behind the multi-graph precompile: the capture of a callable into one
artifact holding every frame Dynamo produces while the caller's calls run --
the entry frame, the ``torch_dynamo_resume_in_*`` continuations graph breaks
create, and the recompiled variants of each -- stored through CompilePackage
(``torch/_dynamo/package.py``), a low-level component not meant to be used
directly. It is not ``torch.compiler.precompile``, the ahead-of-time capture
API this repository already has (``torch/_precompile.py``), which does not call
into this module; nor ``torch._dynamo.config.caching_precompile``, which caches
``torch.compile`` artifacts transparently without an explicit capture and, when
set, wraps every guard filter, this module's included (see
``default_guard_filter_fn``).

``default_guard_filter_fn`` is the guard filter a capture's serialized guards
are written under. The rest of this module, added by the following commits of
this stack, is the guard tooling that reports what that filter dropped and the
configuration a capture runs under. The filter lives here, with that tooling,
rather than beside the serializer's pre-check in ``guards.py``: it is the
capture's policy over that pre-check, not part of it. Everything here is
internal; the filter alone is unprefixed because the capture session passes it
as the default a caller may name. The multi-graph Dynamo capture session that
drives it is a follow-up stack; nothing under ``torch/`` calls into this module
yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .guards import CheckFunctionManager


if TYPE_CHECKING:
    from collections.abc import Sequence

    from .types import GuardFilterEntry


def default_guard_filter_fn(guard_entries: Sequence[GuardFilterEntry]) -> list[bool]:
    """
    Drop every guard ``CheckFunctionManager.serialize_guards`` would refuse for
    its type or a derived type, and keep everything else.

    The refused types are ``UNSUPPORTED_SERIALIZATION_GUARD_TYPES``: the
    identity guards ID_MATCH, FUNCTION_MATCH, MODULE_MATCH, NN_MODULE,
    CLASS_MATCH and CLOSURE_MATCH (a function by its ``__code__`` id), plus
    DICT_VERSION and WEAKREF_ALIVE. Dropping one gives up on noticing that the
    guarded object was rebound, mutated or collected: rebind a global function
    between capture and load and the artifact serves the graph traced against
    the old one, with no error
    (``test_default_guard_filter_through_serialize_guards``). Every dropped
    slot is reported in ``PrecompileSummary.dropped_guards``, once however many
    variants dropped it.

    The criterion is the pre-check's own: a guard is dropped if its type is
    refused or a derived type is (a CONSTANT_MATCH on a code object runs
    through ID_MATCH), and TYPE_MATCH and BUILTIN_MATCH are kept whatever they
    derive, as the pre-check accepts them before it looks at derived types.
    That keeps BUILTIN_MATCH, an ``id_match_unchecked`` deriving ID_MATCH that
    the loaded artifact still checks against the loading process's builtins.

    A DICT_VERSION derived by a DICT_KEYS_MATCH is ignored. That compensates
    for a ``CheckFunctionManager`` artifact, not a property of the serializer:
    a filter sees the derived types of the pre-filter build, which runs with
    ``save_guards=False`` and so promotes the DICT_KEYS_MATCH on
    ``torch.utils._pytree.SUPPORTED_NODES`` to DICT_VERSION, while the save
    build keeps the DICT_KEYS_MATCH the pre-check accepts
    (``test_default_guard_filter_keeps_the_pytree_registry_keys_match``). A
    ``guards.py`` fix giving both builds one verdict would delete the two
    lines below. A DICT_KEYS_MATCH deriving another refused type, and any other
    type deriving DICT_VERSION, are dropped. Neither this keep nor the
    accepted-by-type branch holds under
    ``torch._dynamo.config.caching_precompile``: its wrapper around every guard
    filter trips over the same artifact and drops, with a warning, any guard
    of type ID_MATCH, CLOSURE_MATCH, WEAKREF_ALIVE or DICT_VERSION or deriving
    ID_MATCH or DICT_VERSION.

    Passing this filter does not mean the artifact serializes: the pre-check
    also refuses a kept TYPE_MATCH on a local-scope type, which cannot be
    pickled by name. The filter keeps such a guard on purpose, so serialization
    fails loudly; dropping the TYPE_MATCH on an ``nn.Module`` of a local class
    would ship an artifact that serves a module of any class whose guarded
    attributes match
    (``test_default_guard_filter_keeps_local_type_guards_for_a_loud_refusal``).
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    keep = []
    for g in guard_entries:
        derived = g.derived_guard_types
        if g.guard_type == "DICT_KEYS_MATCH":
            derived = tuple(d for d in derived if d != "DICT_VERSION")
        keep.append(
            # The pre-check's accepted-by-type pair, a literal in serialize_guards,
            # in test_aot_compile.py's keep_builtin_guards and in
            # test_precompile_package.py's _pre_check_accepts too; a type added
            # to one is not seen by the others.
            g.guard_type in ("TYPE_MATCH", "BUILTIN_MATCH")
            or (
                g.guard_type not in unsupported
                and not any(d in unsupported for d in derived)
            )
        )
    return keep
