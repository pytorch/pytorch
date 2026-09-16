"""
Ahead-of-time precompilation of a callable into MANY graphs: the multi-graph
counterpart of ``torch.compile(fn, fullgraph=True).aot_compile(...)``. Every
frame Dynamo produces while the caller's calls run -- the entry
frame, each ``torch_dynamo_resume_in_*`` continuation created by a graph break,
and every recompiled variant of each -- is captured into one serializable
artifact on top of CompilePackage.

Everything here is internal; the capture session and the
``torch.compiler.precompile.capture`` / ``accumulate`` / ``load`` entry points
that build on it follow in later commits. This is distinct from
``torch._dynamo.config.caching_precompile``, which caches ``torch.compile``
artifacts transparently without an explicit capture.

Capture is by execution, and the caller drives it: the session hands back a
callable, the caller invokes it with real inputs inside their own loop, and
every frame Dynamo produces is recorded. Runtime guards stay intact during
capture, so later calls trigger the same recompilations as ordinary
``torch.compile``; ``guard_filter_fn`` applies only to the serialized copy, and
every dropped guard is reported in ``PrecompileSummary.dropped_guards``.

    with torch.compiler.precompile.capture(
        step, artifact_path="m.py", cache_path="m.cache", backend="inductor"
    ) as cap:
        y1 = cap(model, x1)  # runs step(model, x1), returns its result
        y2 = cap(model, x2)  # exercises another variant

    # later, in a fresh process
    compiled = torch.compiler.precompile.load("m.py", "m.cache")
    with compiled, torch.no_grad():
        compiled(model, x1)

``precompile.accumulate`` is the same model, rewriting the artifact on every
call instead of once at block exit.

If serialization drops a configuration-dependent guard, the artifact is refused
by default rather than written with variants whose dispatch would be ambiguous
after load. ``invariants`` writes a readable report
that separates, per frame, the guards holding in EVERY variant from the ones
that differed: the first are preconditions the artifact is only valid under,
the second are what tell its graphs apart. Guards from different frames are not
comparable -- an entry frame guards its arguments, a resume frame guards
whatever crossed the break -- so the intersection is per frame.

Because capture is by execution, a resume function only exists once the frame
ahead of it has actually run, so every variant must be exercised. Whatever you
do not run is not in the artifact, and ``summary().complete`` means complete
only for the observed capture, not for every possible input to the callable. A
captured call that raises marks the session incomplete even if caller code
catches it.

Know these before relying on an artifact in production:

* Calls run with the grad mode the caller sets; capture forces neither
  ``no_grad()`` nor ``enable_grad()``. For an inference artifact run the calls
  under ``torch.no_grad()``. For a training artifact pass ``training=True``,
  which lowers the backward eagerly so the artifact carries one and a served
  output can be backpropagated -- without it, AOTAutograd defers the backward
  to the first ``.backward()`` call, so a grad-enabled capture that never makes
  one records no backends and cannot be written. No loss is needed: the joint
  trace synthesizes tangents from the forward outputs' own metadata.
* A non-tensor argument, and any value that crosses a graph break, is guarded
  by equality, so an int/bool/str argument or a break coming from ``.item()``
  yields an artifact that only serves calls reproducing those exact values.
  ``summary().wont_generalize`` lists them; exercise every value you need to
  serve with a ``cap(...)`` call, or expect poor coverage on new data.
  ``dynamic=True`` helps with shapes but not with pinned values.
* Identity guards cannot be serialized, so precompiling gives up on noticing
  that a guarded object was rebound. ``summary().dropped_guards`` is the
  authoritative list. ``summary().risky_dropped_guards`` includes every drop
  observed to distinguish captured variants plus a lint for configuration-like
  sources; it is still not a proof for unobserved deployments. See
  ``_is_risky_drop``. The public ``torch.compiler.precompile`` facade rejects
  the RISKY subset by default. Refusing every drop is opt-in: every model drops
  the identity guards precompile cannot serialize, so
  ``require_no_dropped_guards=True`` refuses essentially every real artifact.
  Some models trip the lint on library internals. Measured on stock models
  when this was written, torchvision resnet18 and
  mobilenet_v3 report none, timm's ViT reports one (a re-exported
  ``torch._assert``) and transformers' Qwen2 reports 33 built from a two-layer
  config, 55 for the pretrained 24-layer, of which only the
  attention-implementation registry looks genuinely config-selected. Report
  counts are per model, not per library: torchvision's efficientnet_b0 reports
  2 and timm's swin reports 5, one of which is a real config slot. Audit the
  list before relying on the relaxed dropped-guard default, and before
  relaxing the risky-drop rail on top of it.
* Some models do not capture yet. For example, T5 raises ``PackageError: Cannot
  find module for code <code object __init__`` from ``_get_code_source``, which
  is byte-identical to base and which plain ``caching_precompile`` also raises.
* The model must live in an importable module. Source is checksummed, so a
  class defined in ``__main__`` or a REPL cannot be loaded elsewhere.
* ``install()`` writes compiled and resume functions into module globals, but
  guarded dispatch is scoped to the isolated compile region owned by the
  returned callable. Call the returned object rather than another instance of
  the same class. Multiple loaded artifacts can share entry, inner, and resume
  code objects without taking each other's entries; ``unload()`` removes only
  its own region and the globals it still owns.

This wraps CompilePackage, which is the low-level component and is not meant to
be used directly.
"""

from __future__ import annotations

import functools
import os
import site
import sys
import sysconfig
import types
from typing import TYPE_CHECKING

from torch._guards import ChainedSource, Source

from .guards import CheckFunctionManager
from .source import DictGetItemSource, GlobalSource, LocalSource


if TYPE_CHECKING:
    import traceback
    from collections.abc import Callable, Sequence

    from .types import GuardFilterEntry


def _compose_with_default(
    user: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]],
) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
    """AND a caller's filter with the default rather than replacing it.

    ``default_guard_filter_fn`` is not a default in the "sensible starting point"
    sense -- it is what drops the identity guards that CANNOT be serialized at
    all. Replacing it means a caller who wanted to drop three of their own guards
    silently re-admits every unserializable one, and the failure surfaces as
    "ID_MATCH guard cannot be serialized" in frames that have nothing to do with
    their filter. A custom filter can only ever want to drop MORE, so composing
    is the only reading that makes sense.
    """

    def composed(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
        chosen = user(entries)
        if len(chosen) != len(entries):
            raise ValueError(
                f"guard_filter_fn returned {len(chosen)} decisions for "
                f"{len(entries)} guards; it must return one per entry."
            )
        base = default_guard_filter_fn(entries)
        return [bool(a) and bool(b) for a, b in zip(base, chosen)]

    return composed


def default_guard_filter_fn(
    guard_entries: Sequence[GuardFilterEntry],
) -> Sequence[bool]:
    """
    Drop every guard whose type, or whose derived type, is one the serializer
    refuses, and keep everything else.

    Read this before trusting an artifact. The refused set is the IDENTITY
    guards -- ID_MATCH, FUNCTION_MATCH, CLOSURE_MATCH, MODULE_MATCH, NN_MODULE,
    CLASS_MATCH, DICT_VERSION, WEAKREF_ALIVE -- so precompiling inherently gives
    up on noticing that a guarded object was REBOUND to a different object of
    the same shape. Most such guards are on modules and builtins and are stable
    in practice, but one on a global holding a function is not: rebind it
    between capture and load and the artifact serves the graph traced against
    the old one, with no error.

    This is not exactly ``CheckFunctionManager.serialize_guards``'s test. It is
    stricter in one direction: the serializer accepts a TYPE_MATCH or
    BUILTIN_MATCH whatever its derived types, so the BUILTIN_MATCH whose
    derived ID_MATCH this drops would have serialized. It is looser in the
    other: a TYPE_MATCH on a local-scope class passes here and the serializer
    still refuses it, because the type cannot be pickled.

    Keeping the identity guards makes serialization raise for essentially every
    function, so every drop is recorded with its source name in
    ``PrecompileSummary.dropped_guards``. A caller is not refused for having
    them, because requiring none would refuse essentially every model. The rail
    that is on is the risky-drop lint, and a lint is not a proof. See
    ``_is_risky_drop``.
    """
    unsupported = CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
    return [
        g.guard_type not in unsupported
        and not any(d in unsupported for d in g.derived_guard_types)
        for g in guard_entries
    ]


def _owning_module(value: object) -> str | None:
    if isinstance(value, types.ModuleType):
        return value.__name__
    owner = getattr(value, "__module__", None)
    return owner if isinstance(owner, str) else None


def _source_root(source: Source) -> Source:
    return source.get_base() if isinstance(source, ChainedSource) else source


# Locals Dynamo synthesizes when a resume function is itself nested, passed
# positionally into the continuation (resume_execution.py builds the list).
# They name generated code, not a slot any config chooses, so an identity guard
# lost on one cannot diverge.
_DYNAMO_SYNTHESIZED = ("__nested_resume_fns", "__nested_frame_values")


def _is_dynamo_synthesized(source: Source) -> bool:
    root = _source_root(source)
    return isinstance(root, LocalSource) and root.local_name in _DYNAMO_SYNTHESIZED


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


@functools.cache
def _stdlib_roots() -> tuple[str, ...]:
    """
    Where this interpreter's own library lives. os is unquestionably stdlib, so
    its directory is the direct evidence and the only one that stays right when
    the stdlib is a zip; sysconfig and sys._stdlib_dir cover a build where os is
    frozen with no __file__.
    """
    roots = []
    os_file = getattr(os, "__file__", None)
    if os_file:
        roots.append(os.path.dirname(os_file))
    frozen_dir = getattr(sys, "_stdlib_dir", None)  # 3.11+
    if frozen_dir:
        roots.append(frozen_dir)
    paths = sysconfig.get_paths()
    roots += [p for p in (paths.get("stdlib"), paths.get("platstdlib")) if p]
    if sys.platform == "win32":
        # The stdlib's C extensions live beside Lib, not under it.
        roots.append(os.path.join(sys.base_prefix, "DLLs"))
    return tuple(sorted({_norm(p) for p in roots}))


@functools.cache
def _install_roots() -> tuple[str, ...]:
    """
    Where a third party lands. This is the load-bearing exclusion: purelib is
    NESTED inside stdlib in a conda layout and inside platstdlib in a venv, so
    without it every pip-installed package is under a stdlib root.
    """
    paths = sysconfig.get_paths()
    roots = [p for p in (paths.get("purelib"), paths.get("platlib")) if p]
    for name in ("getsitepackages", "getusersitepackages"):
        # getattr, not a direct reference: the "old virtualenv site.py" this
        # guards against does not DEFINE these, so naming them here would raise
        # the very AttributeError the except is for -- out of a lint, aborting
        # the capture it was asked to check.
        get = getattr(site, name, None)
        try:
            got = get() if get is not None else None
            found = [got] if isinstance(got, str) else list(got or ())
        except Exception:
            continue  # -S, or a site.py that defines it but cannot answer
        roots += [p for p in found if isinstance(p, str)]
    return tuple(sorted({_norm(p) for p in roots}))


@functools.cache
def _torch_roots() -> tuple[str, ...]:
    """
    Every directory torch's own submodules come from. An editable build splits
    them -- torch/__init__.py out of the source tree, _C.so and version.py out
    of site-packages -- and torch.__path__ is exactly that set. It is only
    trusted if the directory this file is running from is in it, so a
    sys.modules['torch'] that is not us cannot nominate its own roots.
    """
    own_file = globals().get("__file__")
    if not own_file:
        return ()  # frozen torch: no directory to anchor to
    own = _norm(os.path.dirname(os.path.dirname(own_file)))
    roots = {own}
    search = getattr(sys.modules.get("torch"), "__path__", None) or ()
    listed = {_norm(p) for p in search if isinstance(p, str)}
    if own in listed:
        roots |= listed
    return tuple(sorted(roots))


def _within(path: str, roots: tuple[str, ...]) -> bool:
    """Prefix test over ``_norm``-ed paths; the caller normalizes both sides."""
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def _defined_where_read(
    value: object, user_stack: traceback.StackSummary | None
) -> bool:
    """
    Whether the def lives in the file of the frame that read it.

    The caller has already matched the value's ``__name__`` to the global it
    was read from; this adds the WHERE. A def bound to its own name in its OWN
    module takes an edit there to repoint. ``from impl_a import op`` takes only
    a conditional import in the reader, which is not an edit at all and which
    no checksum covers.
    """
    home = sys.modules.get(_owning_module(value) or "")
    file = getattr(home, "__file__", None)
    if not user_stack or not isinstance(file, str):
        return False
    return _norm(file) == _norm(user_stack[-1].filename)


def _dynamo_alias_module(global_name: str) -> types.ModuleType | None:
    """
    The module behind an ``__import_a_dot_b`` alias, mirroring import_source.

    The OutputGraph's import_sources table is authoritative, but a guard entry
    does not carry it; unmangling collides only for a module literally named
    ``a_dot_b``.
    """
    prefix = "__import_"
    if not global_name.startswith(prefix):
        return None
    return sys.modules.get(global_name[len(prefix) :].replace("_dot_", "."))


# Dynamo's own handle on the builtins dict, minted by
# OutputGraph.install_builtins_dict_in_fglobals.
_BUILTINS_DICT_PREFIX = "__builtins_dict__"


def _reads_a_builtin(source: Source, value: object) -> bool:
    """
    ``len`` or ``sorted`` reached the ordinary way, through the builtins dict
    Dynamo installs to resolve them. No binding sits in front of those, so
    nothing can repoint them.

    A builtin parked in a slot -- ``self.act = abs``, straight out of an
    ACT2FN-style table -- is a slot like any other, so this deliberately keys
    on where the read comes FROM rather than on who owns the value.
    """
    return (
        isinstance(source, DictGetItemSource)
        and isinstance(source.base, GlobalSource)
        and source.base.global_name.startswith(_BUILTINS_DICT_PREFIX)
        and _owning_module(value) == "builtins"
    )
