"""
Helpers behind the multi-graph precompile: the capture of a callable into one
artifact holding every frame Dynamo produces while the caller's calls run --
the entry frame, the ``torch_dynamo_resume_in_*`` continuations graph breaks
create, and the recompiled variants of each -- stored through CompilePackage
(``torch/_dynamo/package.py``), a low-level component not meant to be used
directly.

This module holds the guard filter for the serialized guards
(``default_guard_filter_fn``), the lint over the guards it drops
(``_is_risky_drop``), the guard-type classification and fingerprints behind
the ``PrecompileSummary`` report, the per-frame comparison of captured
variants and the summary builder (``_varying_guard_slots``, ``_summarize``),
and the compiler configuration and frame converter a capture runs under
(``_capture_config``, ``_AllowEmptyGraphsConvertFrame``). Everything here is
internal. The capture session that drives them, ``torch.compiler.precompile``,
is a follow-up; nothing under ``torch/`` calls into this module yet. It is
distinct from ``torch._dynamo.config.caching_precompile``, which caches
``torch.compile`` artifacts transparently without an explicit capture.
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
    from collections.abc import Sequence

    from .types import GuardFilterEntry


def default_guard_filter_fn(entries: Sequence[GuardFilterEntry], /) -> Sequence[bool]:
    """
    Drop every guard ``CheckFunctionManager.serialize_guards`` would refuse for
    its type or a derived type, and keep everything else.

    The refused types are ``UNSUPPORTED_SERIALIZATION_GUARD_TYPES``: the
    identity guards ID_MATCH, FUNCTION_MATCH, CLOSURE_MATCH, MODULE_MATCH,
    NN_MODULE and CLASS_MATCH, plus DICT_VERSION and WEAKREF_ALIVE. Dropping
    one gives up on noticing that the guarded object was rebound, mutated or
    collected: rebind a global function between capture and load and the
    artifact serves the graph traced against the old one, with no error
    (``test_default_guard_filter_through_serialize_guards``). Every drop is
    reported in ``PrecompileSummary.dropped_guards``.

    The test is the serializer's own pre-check over the entry's type and
    derived types: a guard is dropped if its type is refused or one of its
    derived types is (a CONSTANT_MATCH on a code object runs through
    ID_MATCH), and TYPE_MATCH and BUILTIN_MATCH are kept whatever they derive,
    as the pre-check accepts them before it looks at derived types. That is
    what keeps BUILTIN_MATCH, an ``id_match_unchecked`` deriving ID_MATCH; the
    loaded artifact checks the builtin against the loading process's builtins,
    so it still notices one swapped after load. The one departure from the
    pre-check is a DICT_VERSION derived by a DICT_KEYS_MATCH, which is
    ignored: the entries this filter sees carry the derived types of the build
    ``CheckFunctionManager`` runs before filtering, with ``save_guards=False``,
    where a DICT_KEYS_MATCH on ``torch.utils._pytree.SUPPORTED_NODES`` is
    promoted to a DICT_VERSION, while the save build pins it to the keys-match
    the pre-check accepts
    (``test_default_guard_filter_keeps_the_pytree_registry_keys_match``).

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
    for g in entries:
        derived = g.derived_guard_types
        if g.guard_type == "DICT_KEYS_MATCH":
            derived = tuple(d for d in derived if d != "DICT_VERSION")
        keep.append(
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
    so a recorded ``__file__`` or a ``__path__`` entry is gated with isabs
    before it gets here.
    """
    return os.path.normcase(os.path.realpath(path))


@functools.cache
def _stdlib_roots() -> tuple[str, ...]:
    """
    Where this interpreter's own library lives, sorted, so the first root a path
    lies under is the outermost (``_classify_file`` reads it that way). os is
    unquestionably stdlib, so its directory is the direct evidence and the only
    one that stays right when the stdlib is a zip, where that root is the whole
    archive and a third party bundled into it is waived with the stdlib;
    sysconfig and sys._stdlib_dir cover a build where os is frozen with no
    __file__. An install root can nest inside one of these (see
    ``_install_roots``), so a path under both is third party: an install root
    wins over a stdlib root.
    """
    roots = []
    os_file = getattr(os, "__file__", None)
    if os_file:
        # The directory the file resolves into, not the one it was imported
        # from: in a venv over a symlink-farm prefix (a Nix, Guix or Spack
        # profile) os.py is a per-file link into the store, sysconfig and
        # sys._stdlib_dir already name the farm, and every consumer normalizes
        # the file it asks about.
        roots.append(os.path.dirname(_norm(os_file)))
    frozen_dir = getattr(sys, "_stdlib_dir", None)  # 3.11+
    if frozen_dir:
        roots.append(frozen_dir)
    paths = sysconfig.get_paths()
    roots += [paths["stdlib"], paths["platstdlib"]]
    if sys.platform == "win32":
        # The stdlib's C extensions live beside Lib, not under it.
        roots.append(os.path.join(sys.base_prefix, "DLLs"))
    return tuple(sorted({_norm(p) for p in roots}))


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
    roots = [paths["purelib"], paths["platlib"]]
    for name in ("getsitepackages", "getusersitepackages"):
        try:
            got = getattr(site, name)()
            found = [got] if isinstance(got, str) else list(got)
        except Exception:
            continue  # an old-virtualenv site.py lacks it, or it cannot answer
        roots += [p for p in found if isinstance(p, str)]
    # On Windows getsitepackages() lists the bare prefix, which the whole stdlib
    # sits under; a directory a stdlib root lies under is not an install root.
    stdlib = _stdlib_roots()
    normed = {_norm(p) for p in roots}
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
    would resolve against the process cwd and make it a torch root.
    """
    own_file = globals().get("__file__")
    if not own_file:
        return ()  # frozen torch: no directory to anchor to
    own = _norm(os.path.dirname(os.path.dirname(own_file)))
    roots = {own}
    search = getattr(sys.modules.get("torch"), "__path__", None) or ()
    listed = {_norm(p) for p in search if isinstance(p, str) and os.path.isabs(p)}
    if own in listed:
        roots |= listed
    return tuple(sorted(roots))


def _within(path: str, roots: tuple[str, ...]) -> bool:
    """Prefix test over ``_norm``-ed paths; the caller normalizes both sides."""
    return any(path == r or path.startswith(r + os.sep) for r in roots)
