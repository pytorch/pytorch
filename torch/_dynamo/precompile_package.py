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

Over the stack that adds it, this module comes to hold the guard filter for
the serialized guards (``default_guard_filter_fn``), the lint over the identity
guards it drops (``_is_risky_drop``), the fingerprints and the guard-type
classification behind the ``PrecompileSummary`` report -- which also decides
the only guards an invariance policy may drop
(``_INVARIANT_DROPPABLE_GUARD_TYPES``) -- the per-frame comparison of captured
variants and the summary builder (``_varying_guard_slots``, ``_summarize``),
and the compiler configuration and frame converter a capture runs under
(``_capture_config``, ``_AllowEmptyGraphsConvertFrame``). The filter lives
here, with the rest of the capture's guard tooling, rather than beside the
serializer's pre-check in ``guards.py``: it is the capture's policy over that
pre-check, not part of it.
Everything here is internal; the filter alone is unprefixed because the
capture session passes it as the default a caller may name. The multi-graph
Dynamo capture session that drives them is a follow-up stack; nothing under
``torch/`` calls into this module yet.
"""

from __future__ import annotations

import functools
import os
import site
import sys
import sysconfig
from typing import TYPE_CHECKING

from .guards import CheckFunctionManager


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .types import GuardFilterEntry


def default_guard_filter_fn(guard_entries: Sequence[GuardFilterEntry]) -> list[bool]:
    """
    Drop every guard ``CheckFunctionManager.serialize_guards`` would refuse for
    its type or a derived type, and keep everything else.

    The refused types are ``UNSUPPORTED_SERIALIZATION_GUARD_TYPES``: ID_MATCH,
    FUNCTION_MATCH, MODULE_MATCH, NN_MODULE and CLASS_MATCH, which check the
    guarded object's identity, CLOSURE_MATCH, which checks a function by its
    ``__code__`` id, plus DICT_VERSION and WEAKREF_ALIVE. Dropping one gives up
    on noticing that the guarded object was rebound, mutated or collected:
    rebind a global function between capture and load and the artifact serves
    the graph traced against the old one, with no error
    (``test_default_guard_filter_through_serialize_guards``). Every dropped
    slot is reported in ``PrecompileSummary.dropped_guards``, once however many
    variants dropped it.

    The criterion is the serializer's own pre-check over the entry's type and
    derived types: a guard is dropped if its type is refused or one of its
    derived types is (a CONSTANT_MATCH on a code object runs through
    ID_MATCH), and TYPE_MATCH and BUILTIN_MATCH are kept whatever they derive,
    as the pre-check accepts them before it looks at derived types. That is
    what keeps BUILTIN_MATCH, an ``id_match_unchecked`` deriving ID_MATCH; the
    loaded artifact checks the builtin against the loading process's builtins,
    so it still notices one swapped after load. Neither that accepted-by-type
    branch nor the DICT_KEYS_MATCH keep below holds under
    ``torch._dynamo.config.caching_precompile``: ``CheckFunctionManager``
    wraps every guard filter under that setting and drops, with a warning, any
    guard of type ID_MATCH, CLOSURE_MATCH, WEAKREF_ALIVE or DICT_VERSION and
    any guard deriving ID_MATCH or DICT_VERSION. The one departure from the
    pre-check is a DICT_VERSION derived by a DICT_KEYS_MATCH, which is
    ignored: the entries this filter sees carry the derived types of the build
    ``CheckFunctionManager`` runs before filtering, with ``save_guards=False``,
    where a DICT_KEYS_MATCH on ``torch.utils._pytree.SUPPORTED_NODES`` is
    promoted to a DICT_VERSION, while the save build pins it to the keys-match
    the pre-check accepts
    (``test_default_guard_filter_keeps_the_pytree_registry_keys_match``). That
    pair only: a DICT_KEYS_MATCH deriving another refused type, and any other
    type deriving DICT_VERSION, are dropped as the pre-check would refuse them
    (``test_default_guard_filter_drops_the_unserializable_types``).

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


def _norm(path: str) -> str:
    """
    realpath then normcase. A relative path resolves against the process cwd,
    so a recorded ``__file__`` is gated with isabs before it gets here and every
    root candidate comes through ``_norm_absolute``; this module's own
    ``__file__`` is taken as read because it is always the path finder's
    spelling, absolute since bpo-43105 (3.10+): the finder that keeps a relative
    one, zipimport, cannot load torch (torch._C is an extension module), and a
    frozen torch has no ``__file__`` at all. os is different: frozen since 3.11,
    its ``__file__`` is spelled from sys._stdlib_dir, relative under a relative
    home until site.abs_paths() re-anchors it at startup, which -S skips.
    """
    return os.path.normcase(os.path.realpath(path))


def _norm_absolute(paths: Iterable[object]) -> set[str]:
    """
    The ``_norm`` of every absolute str among ``paths``; anything else is
    dropped rather than resolved. The interpreter's own metadata can be
    relative: a venv whose pyvenv.cfg ``home`` is relative or a relative
    PYTHONHOME leaves sys.base_prefix, sys._stdlib_dir and every sysconfig
    path relative (os.__file__ alone is re-anchored, by site.abs_paths()),
    a relative PYTHONUSERBASE the user site. Resolved, such a root would sit
    wherever the process cwd was at the first call and stay cached there,
    so a writer and a reader on one interpreter could classify differently.
    """
    return {_norm(p) for p in paths if isinstance(p, str) and os.path.isabs(p)}


@functools.cache
def _stdlib_roots() -> tuple[str, ...]:
    """
    Where this interpreter's own library lives, sorted, so the first root a path
    lies under is the outermost (``_classify_file`` reads it that way). os is
    unquestionably stdlib, so its directory is the direct evidence and the only
    one that stays right when the stdlib is a zip, where that root is the whole
    archive and a third party bundled into it is waived with the stdlib;
    sysconfig and sys._stdlib_dir cover a build where os is frozen with no
    __file__ (``_classify_file`` skips the stdlib arm under ``sys.frozen``, so
    an app bundle does not use them). An install root can nest inside one of
    these (see ``_install_roots``), so a path under both is third party: an
    install root wins over a stdlib root.
    """
    roots: list[object] = []
    os_file = getattr(os, "__file__", None)
    if isinstance(os_file, str) and os.path.isabs(os_file):
        # The directory the file resolves into, not the one it was imported
        # from: in a venv over a symlink-farm prefix (a Nix, Guix or Spack
        # profile) os.py is a per-file link into the store, sysconfig and
        # sys._stdlib_dir already name the farm, and every consumer normalizes
        # the file it asks about.
        roots.append(os.path.dirname(_norm(os_file)))
    roots.append(getattr(sys, "_stdlib_dir", None))  # 3.11+
    paths = sysconfig.get_paths()
    roots += [paths["stdlib"], paths["platstdlib"]]
    if sys.platform == "win32":
        # The stdlib's C extensions live beside Lib, not under it.
        roots.append(os.path.join(sys.base_prefix, "DLLs"))
    return tuple(sorted(_norm_absolute(roots)))


@functools.cache
def _install_roots() -> tuple[str, ...]:
    """
    Where a third party lands. This is the load-bearing exclusion: purelib is
    NESTED inside stdlib in a conda layout and inside platstdlib in a venv (on
    a --with-platlibdir=lib64 build it is platlib that nests, purelib living
    under lib instead), so without it every pip-installed package is under a
    stdlib root.
    """
    paths = sysconfig.get_paths()
    roots: list[object] = [paths["purelib"], paths["platlib"]]
    for name in ("getsitepackages", "getusersitepackages"):
        try:
            got = getattr(site, name)()
            roots += [got] if isinstance(got, str) else list(got)
        except Exception:
            continue  # an old-virtualenv site.py lacks it, or it cannot answer
    # On Windows getsitepackages() lists the bare prefix, which the whole stdlib
    # sits under; a directory a stdlib root lies strictly under is not an
    # install root. A candidate that IS a stdlib root stays one, on purpose:
    # dropping it would leave a third party installed into the stdlib directory
    # itself under no install root and waived with the stdlib, while keeping it
    # only reads the stdlib as third party.
    stdlib = _stdlib_roots()
    normed = _norm_absolute(roots)
    above = {r for r in normed for s in stdlib if s.startswith(r + os.sep)}
    return tuple(sorted(normed - above))


@functools.cache
def _torch_roots() -> tuple[str, ...]:
    """
    Every directory torch's own submodules come from. An editable build splits
    them -- torch/__init__.py out of the source tree, _C.so and version.py out
    of site-packages -- and torch.__path__ is exactly that set. The gate rules
    out a substituted sys.modules['torch'] only: its __path__ is ignored unless
    the torch package directory this file sits under (two levels up, past
    _dynamo) is among the entries, and then every absolute entry is adopted,
    one a third party appended to the real torch's included; a relative entry
    is dropped (``_norm_absolute``). That directory has two spellings, both of
    them roots: resolved as a directory, and two levels up from where this file
    resolves. They differ in a per-file symlink farm (see ``_stdlib_roots``),
    where torch.__path__ names the farm and every consumer asks about a file
    that resolves into the store. The second is taken only while the file still
    resolves to <root>/_dynamo/<file>: a link that flattens the depth would make
    an ancestor of unrelated code a torch root, and then the resolved file lies
    under no root rather than under too wide a one.
    """
    own_file = globals().get("__file__")
    if not own_file:
        return ()  # frozen torch: no directory to anchor to
    own_dir = os.path.dirname(own_file)
    own = os.path.dirname(own_dir)
    roots = {_norm(own)}
    resolved = _norm(own_file)
    tail = os.path.join(os.path.basename(own_dir), os.path.basename(own_file))
    if resolved.endswith(os.sep + os.path.normcase(tail)):
        roots.add(os.path.dirname(os.path.dirname(resolved)))
    search = getattr(sys.modules.get("torch"), "__path__", None) or ()
    listed = _norm_absolute(search)
    if roots & listed:
        roots |= listed
    return tuple(sorted(roots))


def _within(path: str, roots: tuple[str, ...]) -> bool:
    """Prefix test over ``_norm``-ed paths; the caller normalizes both sides."""
    return any(path == r or path.startswith(r + os.sep) for r in roots)
