"""
Helpers behind the multi-graph precompile: the capture of a callable into one
artifact holding every frame Dynamo produces while the caller's calls run --
the entry frame, the ``torch_dynamo_resume_in_*`` continuations graph breaks
create, and the recompiled variants of each -- stored through CompilePackage
(``torch/_dynamo/package.py``), a low-level component not meant to be used
directly, the multi-graph counterpart of
``torch.compile(fn, fullgraph=True).aot_compile(...)``. It is not
``torch.compiler.precompile``, the ahead-of-time capture API this repository
already has (``torch/_precompile.py``), which does not call into this module
yet; nor ``torch._dynamo.config.caching_precompile``, which caches
``torch.compile`` artifacts transparently without an explicit capture and, when
set, wraps every guard filter, this module's included (see
``default_guard_filter_fn``).

``default_guard_filter_fn`` is the guard filter a capture's serialized guards
are written under. Around it sits the guard tooling that reports what that
filter dropped and the configuration a capture runs under. The filter lives
here, with that tooling, rather than beside the serializer's pre-check in
``guards.py``: it is the capture's policy over that pre-check, not part of it.
Everything here is internal; the filter alone is unprefixed because the capture
session passes it as the default a caller may name. ``PrecompileSession``, the
multi-graph Dynamo capture session that drives it, is added here; the
``torch.compiler.precompile.capture(..., tracer=DynamoTracer())`` entry point
that reaches it lands later in this stack, so nothing under ``torch/`` calls
into the session yet.

Capture is by execution, and the caller drives it: the session hands back a
callable, the caller invokes it with real inputs inside their own loop, each
``cap(...)`` returns the callable's own result, and every frame, break
continuation and guarded variant the call exercises is recorded.

    with torch.compiler.precompile.capture(
        step, artifact_path="m.py", cache_path="m.cache", backend="inductor"
    ) as cap:
        y1 = cap(model, x1)  # runs step(model, x1), returns its result
        y2 = cap(model, x2)  # exercises another variant

    # later, in a fresh process
    compiled = torch.compiler.precompile.load("m.py", "m.cache")
    with compiled, torch.no_grad():
        compiled(model, x1)

Calls run with the grad mode the caller sets -- capture does not force
``no_grad()`` or ``enable_grad()``. ``training=True`` lowers the backward
eagerly so the artifact carries one and a served output can be backpropagated.
No loss is needed for that: the joint trace synthesizes tangents from the
forward outputs' own metadata.

``guard_filter_fn`` rides on the optimize context rather than on the
serializer, so the guards it drops leave the live check too and a capture
recompiles less often than ordinary ``torch.compile`` would, and every dropped
guard is reported in ``PrecompileSummary.dropped_guards``. If serialization
drops a guard that told the captured variants apart, the artifact is refused by
default rather than written with variants whose dispatch would be ambiguous
after load.
``invariants`` writes a readable report that separates, per frame, the guards
holding in EVERY variant from the ones that differed: the first are
preconditions the artifact is only valid under, the second are what tell its
graphs apart. Guards from different frames are not comparable -- an entry frame
guards its arguments, a resume frame guards whatever crossed the break -- so
the intersection is per frame.

A resume function only exists once the frame ahead of it has actually run, so
every variant must be exercised. Whatever you do not run is not in the
artifact, and ``summary().complete`` means complete only for the observed
capture, not for every possible input to the callable. A captured call that
raises marks the session incomplete even if caller code catches it.

Know these before relying on an artifact in production:

* An inference artifact is the default: the caller runs the calls under
  ``torch.no_grad()``. For a training artifact pass ``training=True``, which
  traces with grad on and lowers the backward eagerly -- without it, AOTAutograd
  defers the backward to the first ``.backward()`` call, so a grad-enabled
  capture that never makes one records no backends and cannot be written.
* A non-tensor argument, and any value that crosses a graph break, is guarded
  by equality, so an int/bool/str argument or a break coming from ``.item()``
  yields an artifact that only serves calls reproducing those exact values.
  Exercise every value you need to serve with a ``cap(...)`` call, or expect
  poor coverage on new data.
  ``dynamic=True`` helps with shapes but not with pinned values.
* Identity guards cannot be serialized, so precompiling gives up on noticing
  that a guarded object was rebound. ``summary().dropped_guards`` is the
  authoritative list. ``risky_dropped_guards`` includes every drop observed to
  distinguish captured variants plus the ones a custom filter added; it is still
  not a proof for unobserved deployments.
  Refusing every drop is opt-in: every model drops the identity guards
  precompile cannot serialize, so ``require_no_dropped_guards=True`` refuses
  essentially every real artifact. Audit the list before relying on the relaxed
  dropped-guard default, and before relaxing the risky-drop rail on top of it.
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

The public surface is ``torch.compiler.precompile.capture(...)``, a caller-driven
capture used as a context manager: the caller's own calls inside the block drive
the capture, and the ``(python_code, cache)`` artifact is written to the given files
when the block exits (its default ``tracer=DynamoTracer()`` records many calls;
``tracer=MakeFxTracer()`` produces a self-contained Python source artifact from one
call); and ``torch.compiler.precompile.load``.
The helpers in this module, including the capture session, implement that surface
and remain internal.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import functools
import hashlib
import importlib.machinery
import logging
import os
import re
import site
import sys
import sysconfig
import threading
import types
from typing import Any, TYPE_CHECKING

import torch
import torch._functorch.config as functorch_config
from torch._guards import ChainedSource
from torch.compiler._precompile_types import (
    FrameInvariants,
    GuardFact as _GuardFact,
    PrecompileSummary,
)
from torch.utils._config_module import ConfigModule

from .aot_compile import _BUILTINS_DICT_PREFIX, _IMPORT_ALIAS_PREFIX
from .convert_frame import CatchErrorsWrapper, ConvertFrame
from .exc import PackageError
from .guards import CheckFunctionManager
from .package import CompilePackage
from .source import (
    AttrSource,
    DictGetItemSource,
    GetItemSource,
    GlobalSource,
    LocalSource,
)
from .types import FrameAction


if TYPE_CHECKING:
    import traceback
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from typing import NoReturn

    from torch._guards import Source

    from .convert_frame import ConvertFrameReturn
    from .eval_frame import OptimizeContext
    from .hooks import Hooks
    from .package import _BackendId, _DynamoCacheEntry, _DynamoCodeCacheEntry
    from .repro.after_dynamo import WrapBackendDebug
    from .types import CacheEntry, DynamoFrameType, GuardFilterEntry
    from .variables.builder import FrameStateSizeEntry


# Built once: config.patch() allocates a class and a ContextVar each time it is
# called, and this runs on every frame Dynamo compiles for a package.
_ALLOW_EMPTY_GRAPHS = torch._dynamo.config._make_closure_patcher(
    allow_empty_graphs=True
)


@contextlib.contextmanager
def _capture_config(training: bool) -> Iterator[None]:
    """The compiler configuration a multi-graph capture's calls run under.

    Backends serialize into the artifact (``bundled_autograd_cache``) rather
    than the process-local inductor cache. A training capture lowers its
    backward eagerly: AOTAutograd otherwise defers it to the first
    ``.backward()``, and a capture that never makes one records no backend.
    ``allow_empty_graphs`` keeps an empty graph as a compiled frame so its
    guards reach the artifact; it also extends the lifetime of objects the frame
    holds -- with it on, a weakref callback on a value the frame captured does
    not fire when the caller drops its reference (test/dynamo/test_repros.py
    ReproTests.test_weakref_callback).

    Every scope patches and restores for itself: ``config.patch`` is re-entrant
    and per-thread, so nested scopes unwind in order (an inner ``training=True``
    still lowers the backward) and a worker thread sees only its own.
    """
    if torch.compiler.config.force_disable_caches:
        raise PackageError(
            "Cannot precompile with torch.compiler.config.force_disable_caches=True: "
            "compiled backends reach the artifact through the AOTAutograd cache, "
            "which that setting turns off, so the capture would record no graphs"
        )
    functorch_patch = {
        "bundled_autograd_cache": True,
        # AOTAutogradCache refuses to KEY a graph it cannot address soundly -- a
        # graph calling anything outside its allowlist -- and a refusal means it
        # never saves, so the bundled artifact precompile needs is never
        # recorded and the capture ends with nothing to serialize. That gate
        # asks whether the key tells this graph's behaviour apart from
        # another's, which a precompile artifact does not depend on: it is
        # addressed by backend id and pinned to one torch build, so fall back to
        # a nonce key rather than declining, as torch._dynamo.aot_compile and
        # aot_compile_joint_with_descriptors already do.
        "bypass_autograd_cache_key": True,
    }
    if training:
        functorch_patch["force_non_lazy_backward_lowering"] = True
    # AOTAutogradCache honours strict_precompile when it loads but not when it
    # saves: a bundled entry that fails to pickle is dropped with a warning and
    # the capture is short one backend, so raise where the pickle fails instead.
    if torch._dynamo.config.strict_precompile:
        functorch_patch["strict_autograd_cache"] = True
    with (
        functorch_config.patch(functorch_patch),
        torch._dynamo.config.patch(allow_empty_graphs=True),
    ):
        yield


class _AllowEmptyGraphsConvertFrame(ConvertFrame):
    """The package's frame converter, compiling its frames with allow_empty_graphs.

    An uncovered no-op branch must become a guarded variant rather than Dynamo's
    ordinary eager-only SkipFrame: a skip applies to the code object, so one
    fallback call permanently skips that frame and no later call can capture
    its other variants. Applied here as well as in _capture_config so the
    package's frames get it -- with the object-lifetime side effect noted there
    -- whenever this converter compiles them, including recompiles of a loaded
    artifact outside any capture-config scope. Beneath CatchErrorsWrapper
    rather than replacing it: frames its skipfile checks reject never pay the
    patch, and this frame stays out of the user stack dynamo_start reports.
    A frame under DistributedDataParallel with optimize_ddp="ddp_optimizer" is
    refused by name when a package is attached; _clone_with_backend below, the
    hook CatchErrorsWrapper calls only for that frame, says why. Without a
    package the DDP clone keeps this subclass, so the flag survives it.
    """

    @property
    def _clone_with_backend(self) -> Callable[[WrapBackendDebug], ConvertFrame]:
        # CatchErrorsWrapper asks for this clone only for a frame under an
        # active DDP module in ddp_optimizer mode (the default, optimize_ddp=True);
        # the other optimize_ddp modes never reach it. DDPOptimizer compiles the
        # graph one bucket at a time and no bucket carries the backend id the
        # package records, so the artifact could never be completed; the base
        # clone would drop the package silently instead. The wrapper probes the
        # attribute with hasattr before calling it, and hasattr swallows only
        # AttributeError, so a getter that raised would raise PackageError out of
        # the probe itself; raising from the returned callable puts the error at
        # the call the wrapper actually makes. A package-less converter is what
        # the tests build, and what a capture session may build before it
        # attaches a package; its clone keeps the subclass, hooks and limit. The
        # stance path (eval_frame._create_wrapped_callback, behind set_stance)
        # rebuilds a plain ConvertFrame with no package and is outside a capture
        # session's contract.
        if self._inner_convert._package is None:
            return lambda backend: type(self)(
                backend, self._hooks, recompile_limit=self._recompile_limit
            )

        def refuse(backend: WrapBackendDebug) -> NoReturn:
            raise PackageError(
                "Cannot precompile a DistributedDataParallel forward with "
                f"torch._dynamo.config.optimize_ddp={torch._dynamo.config.optimize_ddp!r}: "
                "DDPOptimizer compiles the graph one bucket at a time and no bucket "
                "records a backend under the id the package saves. Set "
                'torch._dynamo.config.optimize_ddp="no_optimization" to precompile '
                "a DDP forward (this disables DDPOptimizer's comm/compute overlap); "
                'the "python_reducer" and "python_reducer_without_compiled_forward" '
                "modes bypass DDPOptimizer as well"
            )

        return refuse

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        hooks: Hooks,
        frame_state: dict[str, int | FrameStateSizeEntry],
        skip: int = 0,
    ) -> ConvertFrameReturn:
        revert = _ALLOW_EMPTY_GRAPHS()
        try:
            return super().__call__(
                frame, cache_entry, hooks, frame_state, skip=skip + 1
            )
        finally:
            revert()


log = logging.getLogger(__name__)


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


def _owning_module(value: object) -> str | None:
    if isinstance(value, types.ModuleType):
        return value.__name__
    owner = getattr(value, "__module__", None)
    return owner if isinstance(owner, str) else None


def _source_root(source: Source) -> Source:
    return source.get_base() if isinstance(source, ChainedSource) else source


# The list of Dynamo-generated resume functions every generated resume function
# takes as its first parameter (resume_execution.py and
# comprehension_graph_break.py mint the name; codegen_call_resume in
# symbolic_convert.py builds the list). Its entries are generated code, not a
# slot any config chooses, so an identity guard lost on one cannot diverge.
# Only the list and an entry read straight off it are covered: a resume
# function's closure cells carry the resumed frame's cell variables, so a guard
# Dynamo mints past an entry (type(__nested_resume_fns[0].__closure__[0]
# .cell_contents).__call__ for a callable an inner def captured) is the user's
# and is judged like any other value. Its sibling __nested_frame_values is NOT
# here: it carries the live stack and locals of the frames nested INSIDE the
# one resuming, which pops the last entry off it for the callee it resumes, so
# a guard rooted there is judged like the value it stands for.
_DYNAMO_SYNTHESIZED = ("__nested_resume_fns",)


def _is_dynamo_synthesized(source: Source) -> bool:
    if isinstance(source, GetItemSource) and isinstance(source.index, int):
        source = source.base
    return isinstance(source, LocalSource) and source.local_name in _DYNAMO_SYNTHESIZED


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


# Belt and braces for an install directory none of the roots name: whichever
# layout put it there, a pip target still ends in one of these. Matched below
# the stdlib root a file was found under, not on the whole path, so a stdlib
# that itself sits under a site-packages directory keeps its waiver.
_INSTALL_DIR_NAMES = frozenset({"site-packages", "dist-packages"})


@functools.cache
def _classify_file(file: str, stdlib: bool) -> bool | None:
    """
    Shipped here (True), shipped elsewhere (False), or no evidence (None), for
    one ``__file__`` judged against the torch roots (``stdlib=False``) or the
    stdlib roots minus the install roots (``stdlib=True``). None is a path that
    cannot be resolved, or any path asked about for the stdlib in a frozen app;
    past those gates the torch arm never answers None: with no torch root (a
    frozen torch) every path is elsewhere, and ``_is_library_module`` waives
    torch names before asking. Cached on the (file, flag) pair rather than on a
    module name: everything else this reads, the roots and ``sys.frozen``, is
    fixed once the process is running, so the answer for a path never changes,
    while the module a name resolves to can (a test that patches either clears
    the cache on both sides of the patch).
    """
    # Before 3.13 ntpath.isabs accepts a driveless \Lib\x.py (its LEGACY BUG
    # comment), which realpath then resolves against the current drive.
    driveless = sys.platform == "win32" and not os.path.splitdrive(file)[0]
    if driveless or not os.path.isabs(file):
        # Resolving it would be against a cwd that is not the one it was
        # recorded under, so it is evidence in neither direction.
        return None
    if "\x00" in file:
        # No file has one. posixpath.realpath raises ValueError on it from lstat;
        # from gh-106242 on, ntpath.realpath swallows that and returns the path
        # unresolved, whose prefix _within would then match.
        return None
    path = _norm(file)
    if not stdlib:
        # Not gated on sys.frozen: the torch root is torch's own package
        # directory, taken from where this file sits, so in a bundle nothing
        # outside torch lies under it, whereas the stdlib root below is the
        # bundle root itself.
        return _within(path, _torch_roots())
    if getattr(sys, "frozen", False):
        # A frozen app bundles the stdlib and every third party under one root
        # with no site-packages component between them, and PyInstaller names
        # that root in sys._stdlib_dir and in the __file__ it gives the
        # CPython-frozen modules (its _fixup_frozen_stdlib), so every bundled
        # file would lie under a stdlib root: no path there says which of the
        # two a file is. _located's loader arm still waives what the interpreter
        # binary itself carries. Truthiness, as multiprocessing.spawn reads it:
        # PyInstaller and cx_Freeze set the flag to True, py2exe to a string
        # ("console_exe", "windows_exe" or "dll").
        return None
    if _within(path, _install_roots()):
        return False
    # The roots are sorted, so the first match is the outermost and the part
    # below it the longest: the strictest reading of _INSTALL_DIR_NAMES. Case
    # folded because normcase is the identity on posix, and a macOS filesystem
    # is case-insensitive.
    for root in _stdlib_roots():
        if _within(path, (root,)):
            below = path[len(root) + 1 :].lower().split(os.sep)
            return _INSTALL_DIR_NAMES.isdisjoint(below)
    return False


def _located(module: object, name: str, stdlib: bool) -> bool | None:
    """
    Shipped here (True), shipped elsewhere (False), or neither (None), for a
    sys.modules entry, which need not be a module. None covers no ``__file__``
    and no known loader, a ``__file__`` that ``_classify_file`` cannot place
    (relative, non-string, NUL, or any stdlib path in a frozen app) with no
    known loader, and a read below that raised; ``_is_library_module`` refuses
    None for a top-level name and keeps the package's waiver for an inner one,
    so an unplaceable inner ``__file__`` counts as no evidence there. A
    ``__file__`` that ``_classify_file`` can place decides, and only then is
    the loader read: with importlib on disk, importlib/__init__.py gives
    importlib._bootstrap a
    stdlib ``__file__`` next to a FrozenImporter whose table knows it only as
    _frozen_importlib (with importlib frozen too it has no ``__file__``, and
    the loader arm answers False under the dotted name: a lost waiver, not a
    wrong drop). The built-in and frozen arms answer on the caller's ``name``
    alone, not the module's ``__name__``, under either ``stdlib``: both importers
    precede the path finder, so no sys.path entry can shadow such a module,
    and its code is fixed by the interpreter binary, which is what either
    waiver needs. The ``stdlib=False`` case is an embedding that registers
    torch._C through PyImport_AppendInittab while torch's Python files stay on
    disk, so ``_torch_roots`` is non-empty and ``_is_library_module`` does not
    waive torch on its name.
    """
    # The module dict rather than getattr: a class attribute says nothing about
    # where the module came from (torch.ops is a ModuleType subclass carrying
    # the class's relative "_ops.py" as its __file__), and on an entry whose
    # dict is missing one of these dunders getattr would fall through to a PEP
    # 562 module __getattr__, user code a lint must not run (a raise is caught
    # below, its side effects are not). And object's read of the dict, not the
    # module's: importlib.util.LazyLoader leaves a _LazyModule whose
    # __getattribute__ executes the module body on ANY attribute read, __dict__
    # included, while its dict already carries the seeded __file__, __spec__ and
    # __loader__. That read defeats a __getattr__ or a __getattribute__
    # override, not a __dict__ descriptor defined on the type. sys.modules can
    # hold any object, so such a descriptor, and spec.loader on a hand-rolled
    # spec, are user code, and are caught, as is the AttributeError object
    # itself raises for a slotted entry with no __dict__ at all. The try also
    # spans _classify_file and the loader arms, so a raise out of the
    # classifier's own gates or out of FrozenImporter.find_spec (ImportError on
    # an excluded or invalid frozen table entry) lands here as None rather than
    # escaping a lint.
    try:
        attrs = object.__getattribute__(module, "__dict__")
        file = attrs.get("__file__")
        # Only a str is classifiable: a bytes or PathLike __file__ would pass
        # isabs and raise from the NUL check instead.
        if isinstance(file, str):
            verdict = _classify_file(file, stdlib)
            if verdict is not None:
                return verdict
        # The loader rather than spec.origin: both importers' find_spec pass
        # origin=cls._ORIGIN to spec_from_loader, so the two never disagree,
        # and the class is the stronger signal. A __loader__ in the dict skips
        # the spec's loader, the only user-code half of this; compared with None
        # rather than for truth, so no __bool__ of user code runs. A hand-made
        # ModuleType has the None its __init__ seeds and falls through (an
        # imported module's dict carries spec.loader instead, copied in by
        # _init_module_attrs).
        loader = attrs.get("__loader__")
        if loader is None:
            loader = getattr(attrs.get("__spec__"), "loader", None)
        # Both arms compare by identity: under == a __loader__ whose __eq__
        # answers true for anything would take the waiver.
        if loader is importlib.machinery.BuiltinImporter:
            # Statically linked, and BuiltinImporter precedes PathFinder on
            # sys.meta_path, so on import no file on sys.path is reachable
            # under this name; a spec assigned straight into sys.modules is
            # taken at its word. The inittab is keyed on the full dotted name.
            return name in sys.builtin_module_names
        if loader is importlib.machinery.FrozenImporter:
            # frozen also precedes the path finder
            return importlib.machinery.FrozenImporter.find_spec(name) is not None
    except Exception:
        return None
    return None  # namespace package, or exec'd in memory with no loader


def _is_library_module(module_name: str | None) -> bool:
    """
    Owned by torch or the stdlib, so config on the serving machine does not
    choose between implementations. NB this trusts the OWNER, not the binding:
    a third party that monkeypatches ``F.gelu`` at import time still diverges,
    and that is called out in ``_is_risky_drop``'s KNOWN GAP.

    sys.stdlib_module_names is a list of NAMES, and a waiver keyed on a name is
    a collision away from being wrong: graphlib, queue, code and distutils are
    all stdlib names a third party can and does supply. Worse, the name can be
    right and the code still not be the stdlib's -- in a default setuptools
    install ``import distutils`` gets site-packages/setuptools/_distutils, and
    SETUPTOOLS_USE_DISTUTILS picks which one, which is exactly the
    config-chooses-the-implementation shape this lint exists to catch. So the
    module has to RESOLVE to code shipped with the interpreter: located under a
    stdlib root and not under an install root (purelib nests inside stdlib in
    conda and inside platstdlib in a venv, so the exclusion is what does the
    work), or with no file at all because it is built in or frozen, which the
    path finder cannot shadow. That is required of the TOP-LEVEL name: not
    imported, or without location evidence (a namespace package has none), it
    is untrusted. An imported inner name only has to not be located ELSEWHERE:
    the package it was found in is already located, and a real submodule can
    carry no evidence of its own: pyexpat.errors, which pyexpat's C init
    registers with neither ``__file__`` nor ``__spec__``, and torch.ops, a
    ModuleType subclass whose ``__file__`` is a class attribute the module dict
    never sees. The torch arm alone has an escape hatch: a frozen torch gives
    ``_torch_roots`` no directory to anchor to, and then every torch name is
    waived on its name, with no location evidence at all.
    """
    if module_name is None:
        return False
    top = module_name.partition(".")[0]
    if top == "torch":
        if not _torch_roots():
            return True
        stdlib = False
    elif top in sys.stdlib_module_names:
        stdlib = True
    else:
        return False
    root = sys.modules.get(top)
    if root is None or _located(root, top, stdlib) is not True:
        return False
    parts = module_name.split(".")
    for i in range(2, len(parts) + 1):
        name = ".".join(parts[:i])
        module = sys.modules.get(name)
        # Unimported or unlocatable, an inner name has nothing to check, and
        # the package it would have to be found in has already been located.
        if module is not None and _located(module, name, stdlib) is False:
            return False
    return True


def _defined_where_read(
    value: object, global_name: str, user_stack: traceback.StackSummary | None
) -> bool:
    """
    Whether ``global_name`` is a def or class statement of that name living in
    the file that read it.

    ``global_name`` is the read's ``GlobalSource.global_name``; the
    ``GuardFilterEntry.name`` spelling keeps its ``G[...]`` wrapper and is not
    it. The reading file is the OUTERMOST frame of the guard's ``user_stack``:
    a bare GlobalSource denotes the root frame's globals (an inlined frame with
    other globals reads through an ``__import_`` alias or an
    ``___unnamed_scope`` dict entry instead), while the stack is stamped at
    first use, so its innermost frame can be a helper inlined from another
    file. A def bound under its own name in the reading file is the one binding
    the inlined-source checksum of that file covers. ``from impl_a import op``
    takes only a conditional import in the reader, which no checksum sees, and
    ``act = _impl_a if cfg.fast else _impl_b`` is a slot however close to home
    the def is; so are ``op = Ops.op`` and a def returned by a factory, which is
    why the name compared is ``__qualname__``, backed by the code object's own
    name: functools.wraps copies ``__qualname__`` onto a wrapper but cannot
    forge its ``co_qualname``, so a same-file ``wraps`` decorator a flag turns
    on is a slot in both arms (before 3.11 only ``co_name`` exists, so a
    wrapper def named after what it wraps slips through there). Only a plain
    function or a class is judged; a bound method or any other proxy that
    forwards ``__qualname__`` and ``__code__`` is refused before an attribute
    is read, which also keeps a proxy that answers reads by raising, such as
    ``torch.classes.<ns>``, out of the value slot. The file is read off the
    code object, not off ``__module__``: functools.wraps copies ``__module__``
    along with ``__name__`` and ``__qualname__``, so ``op = torch.compile(op)``
    behind a flag claims the reader's module while its code lives in
    eval_frame.py. The object does not tell that shape from an unconditional
    cross-file decorator, so ``@torch.no_grad()`` on a same-file def is not
    waived either.
    A class has no code object, and its ``__module__`` is no better: namedtuple
    and ``type()`` stamp it from the calling frame (make_dataclass does from
    3.12) under a BARE ``__qualname__`` (a def or class statement inside the
    factory would carry ``factory.<locals>.``), so a ``namedtuple("Point", ...)``
    called in the reader looks exactly like a class statement -- the frame
    stamped is the caller's, so the collision needs a factory called from the
    reading file, a cross-file ``lib.make_point()`` coming back stamped ``lib``.
    Its methods can tell: a class statement compiles its defs in its own file
    under its own ``__qualname__`` prefix, so a class is waived when at least one
    function in its own ``__dict__``, stored under key ``k`` with
    ``__qualname__`` ``Cls.k``
    (a staticmethod or classmethod is unwrapped through ``__func__`` and a
    property through ``fget``, by type rather than by ``getattr``, which a proxy
    attribute such as ``torch.classes.<ns>`` answers by raising; a
    cached_property keeps its function under ``.func`` and does not count), was
    compiled in the reading file. A function attached afterwards keeps its bare
    qualname, and one attached through functools.wraps keeps its code object's
    own name, so an imported class the reader extends (``Point.extra =
    _extra``) or patches (``Point.norm = wraps(Point.norm)(_norm)``) and a
    factory fed same-file methods (``type(name, bases, {"area": _area})``,
    ``make_dataclass(..., namespace=...)``) are refused, and ``class Marker:
    pass`` fails closed. So does a class statement whose only functions are
    generated: a fields-only ``@dataclass``'s and a ``NamedTuple``'s are defs
    of a factory (dataclasses' ``__create_fn__``, ``namedtuple``), so their
    code objects' own qualname carries the factory's ``<locals>.`` prefix (on
    3.10, where only ``co_name`` exists, the ``<string>`` or stdlib file they
    compile in refuses them instead), and an ``Enum``'s arrive in the subclass
    ``__dict__`` under ``Enum.`` qualnames the key rule refuses; so a plain
    config dataclass read as a global is reported. The one function the
    compiler itself puts in a class ``__dict__``, the PEP 649 annotate function
    3.14 stores for an annotated class body, compiles in the reading file, but
    under key ``__annotate_func__`` with ``__qualname__`` ``Cls.__annotate__``,
    so the key rule refuses it and the verdict is the same on every version;
    both annotate keys are skipped outright as well, against a version that
    stores it under its own name. Nothing else without a code object is
    waived, because a C-implemented wrapper such as functools.lru_cache claims
    the reader's module the same way. A ``co_filename`` is not always a path:
    an exec records ``<string>``, a REPL ``<stdin>``, and ``_norm`` would
    resolve either against the cwd, so a def exec'd under ``<string>`` and read
    from an exec-generated frame would collide with it and be waived (on 3.10 a
    fields-only dataclass too, whose ``__init__`` compiles in ``<string>``
    under a ``co_name`` that cannot tell it from a class statement's def). Only
    absolute filenames on both sides compare; anything else fails closed. What this
    cannot see is a same-name fork inside the reading file -- ``try: from x
    import impl as op`` / ``except ImportError: def op``, or a class statement
    under the same ``if`` -- which binds a different def per machine under one
    checksum; that is the conditional-bind KNOWN GAP recorded in
    ``_is_risky_drop``.
    """
    if not user_stack or not isinstance(value, (type, types.FunctionType)):
        return False
    if value.__qualname__ != global_name:
        return False

    # functools.wraps copies __qualname__ but not the code object's own name:
    # co_qualname from 3.11, before that co_name, its last component only.
    def compiled_as(fn: types.FunctionType, qualname: str) -> bool:
        if sys.version_info >= (3, 11):
            return fn.__code__.co_qualname == qualname
        return fn.__code__.co_name == qualname.rpartition(".")[2]

    if isinstance(value, type):
        # 3.14 stores the PEP 649 annotate function under __annotate_func__
        # with qualname Cls.__annotate__, which the key rule refuses; the skip
        # covers a version that stores it under its own name.
        skip = ("__annotate__", "__annotate_func__")
        files: list[str] = []
        for key, attr in vars(value).items():
            if isinstance(attr, (staticmethod, classmethod)):
                attr = attr.__func__
            elif isinstance(attr, property):
                attr = attr.fget
            if not isinstance(attr, types.FunctionType) or key in skip:
                continue
            qualname = f"{global_name}.{key}"
            if attr.__qualname__ == qualname and compiled_as(attr, qualname):
                files.append(attr.__code__.co_filename)
    else:
        files = [value.__code__.co_filename] if compiled_as(value, global_name) else []
    read = user_stack[0].filename
    if not os.path.isabs(read):
        return False
    try:
        here = _norm(read)
        return any(os.path.isabs(f) and _norm(f) == here for f in files)
    except ValueError:  # an embedded NUL, which posixpath.realpath lets through
        return False


def _dynamo_alias_module(global_name: str) -> types.ModuleType | None:
    """
    The module behind an ``__import_a_dot_b`` alias, mirroring the ordinary
    branch of import_source; a torch_package module is aliased without the
    prefix and comes back None here, which fails closed.

    The OutputGraph's import_sources table is authoritative, but a guard entry
    does not carry it; unmangling collides only for a module whose name
    contains ``_dot_`` (``pkg.sub_dot_mod`` is aliased as
    ``__import_pkg_dot_sub_dot_mod``, which comes back as ``pkg.sub.mod`` when
    that module is loaded too), and that collision fails open (the callers judge
    the module returned).
    """
    if not global_name.startswith(_IMPORT_ALIAS_PREFIX):
        return None
    tail = global_name[len(_IMPORT_ALIAS_PREFIX) :]
    return sys.modules.get(tail.replace("_dot_", "."))


def _module_namespaces(
    entries: Sequence[GuardFilterEntry],
) -> dict[str, types.ModuleType]:
    """
    Sources holding a module whose binding config cannot repoint, mapped to the
    module itself. Dynamo guards every module it walks through, so the path
    down to ``F.gelu`` is guarded module by module, which is what lets an
    attribute read be recognised as coming off a namespace rather than off an
    object a config could have swapped. The module comes back with the name
    because whether a read off a namespace is safe depends on which module it
    is -- see ``_is_risky_drop``.

    TRUSTED is the load-bearing half and is deliberately narrow. A module is
    that if torch or the stdlib owns it, if it is bound under its own name --
    ``import mypkg``, and the ``__import_x`` alias Dynamo installs to reach an
    inlined function's own globals -- or if it is an attribute of a trusted
    module under a name that module already owns: its own ``__name__`` (a plain
    ``import own_sub`` inside the parent) or the parent's plus the attribute
    (``from . import sub``, and ``import mypkg.layers``, which Dynamo guards as
    ``G['mypkg'].layers``). An ALIASED user module is none of those:
    ``if flag: import impl_b as impl`` picks what ``impl.op`` resolves to per
    machine, and so does the same alias spelled ``from . import impl_b as
    impl`` in a package __init__. Inheriting the parent's trust without
    checking the name is what let that shape through before.

    Library ownership is trusted under ANY binding, ``import torch.nn.functional
    as F`` included; the price, taken deliberately because flagging ``F`` would
    flag every model, is that an alias config picks between two torch modules
    is waived too (see ``_is_risky_drop``'s KNOWN GAP). A config module is the
    one namespace whose bindings config chooses by definition, so it is never
    trusted, whoever owns it and however it is reached (a recovered
    ``__import_x`` alias included). Keys are source names, and the consumer
    looks a read's ``source.base.name`` up exactly, never by prefix.
    """
    modules = {
        e.orig_guard.originating_source.name: (e.orig_guard.originating_source, e.value)
        for e in entries
        if isinstance(e.value, types.ModuleType)
        and isinstance(_source_root(e.orig_guard.originating_source), GlobalSource)
    }
    # Dynamo guards the attributes it reads off an import alias but never the
    # bare alias, so a real model produces G['__import_torch'].Tensor with no
    # module-valued entry for G['__import_torch'] to anchor it. The alias name
    # encodes its module, so recover it rather than treating torch.Tensor as a
    # config-swappable slot.
    for e in entries:
        root = _source_root(e.orig_guard.originating_source)
        if isinstance(root, GlobalSource) and root.name not in modules:
            aliased = _dynamo_alias_module(root.global_name)
            if aliased is not None:
                modules[root.name] = (root, aliased)
    trusted: dict[str, bool] = {}

    def is_trusted(name: str) -> bool:
        if name in trusted:
            return trusted[name]
        found = modules.get(name)
        ok = False
        if found is not None:
            source, module = found
            if isinstance(module, ConfigModule):
                ok = False
            elif _is_library_module(module.__name__):
                ok = True
            elif isinstance(source, GlobalSource):
                ok = (
                    source.global_name == module.__name__
                    or _dynamo_alias_module(source.global_name) is module
                )
            elif isinstance(source, AttrSource):
                outer = modules.get(source.base.name)
                ok = (
                    outer is not None
                    and is_trusted(source.base.name)
                    and module.__name__
                    in (source.member, f"{outer[1].__name__}.{source.member}")
                )
        trusted[name] = ok
        return ok

    return {name: module for name, (_, module) in modules.items() if is_trusted(name)}


def _reads_a_builtin(source: Source, value: object) -> bool:
    """
    ``len`` or ``sorted`` reached the ordinary way, through the builtins dict
    Dynamo installs to resolve them. That dict is the frame's live
    ``builtins.__dict__``, not a table of the real builtins, so a shim's
    ``builtins.py2_sum = sum`` is a binding under it that another machine's
    shim can point elsewhere; only a builtin ``builtins`` itself owns, read
    under its own name, is waived (``IOError`` and ``EnvironmentError``,
    CPython's aliases of ``OSError``, fail closed), and only one CPython built:
    functools.wraps copies ``__module__`` and ``__name__`` onto ``builtins.sum =
    wraps(sum)(logged_sum)``, a Python function whose CLOSURE_MATCH is dropped,
    and a class statement exec'd with the builtins namespace as its globals (or
    handed ``__module__ = "builtins"``) claims the module outright, so the value
    must be a builtin function or a static type as well, one carrying
    ``Py_TPFLAGS_IMMUTABLETYPE``, which a class statement or a ``type()`` call
    does not carry. The flag is read through ``type``'s own descriptor, so a
    metaclass can neither shadow it nor make the read raise, and a static
    type's ``__module__`` is derived from its C name; a builtin function's
    ``__module__`` is a writable member, though, so ``builtins.getcwd =
    os.getcwd`` plus ``os.getcwd.__module__ = "builtins"`` is waived -- evidence
    rather than proof on that branch, which an advisory lint over a
    dropped-guard set does not defend against. Every callable in
    ``builtins.__dict__`` is one of the two kinds apart from the
    ``_sitebuiltins`` objects, instances of neither, and beside the aliases
    (``WindowsError`` is a third on Windows) three of them fail closed:
    ``open``, the one builtin function ``builtins`` does not own (its
    ``__module__`` is ``_io``, ``io`` before 3.12, so a dropped guard on one of
    the most mainstream builtins here is reported), and the heap types
    ``ExceptionGroup`` (3.11+) and ``__loader__``, the latter refused under a
    name that is not its own as well. The exposure is
    narrow either way: a registered builtin is id-matched into a BUILTIN_MATCH
    the serializer keeps, so only a deregistered, polyfilled one (``sum``,
    ``enumerate``, ``all``, ``any``) or a shim reaches the dropped set this
    lint examines.

    A builtin parked in a slot -- ``self.act = abs``, straight out of an
    ACT2FN-style table -- is a slot like any other, so this deliberately keys
    on where the read comes FROM rather than on who owns the value.
    """
    return (
        isinstance(source, DictGetItemSource)
        and isinstance(source.base, GlobalSource)
        and source.base.global_name.startswith(_BUILTINS_DICT_PREFIX)
        and (
            isinstance(value, types.BuiltinFunctionType)
            # Py_TPFLAGS_IMMUTABLETYPE: a static type CPython built, not a
            # class statement or type() call, whose __module__ is writable. Read
            # through type's descriptor, past any metaclass shadowing __flags__.
            or (
                isinstance(value, type)
                and bool(vars(type)["__flags__"].__get__(value) & (1 << 8))
            )
        )
        and _owning_module(value) == "builtins"
        and getattr(value, "__name__", None) == source.index
    )


def _is_risky_drop(
    entry: GuardFilterEntry, namespaces: dict[str, types.ModuleType]
) -> bool:
    """
    Whether losing this identity guard can plausibly change results.

    Intersect the binding SITE with who owns the value; either test alone is
    wrong. Site alone: ``self.act = getattr(F, cfg.activation)`` and
    ``self.act = cfg.act_fn`` are the same swappable slot, so calling the first
    benign because ``F.gelu`` is torch-owned waves through the exact divergence
    this check exists for -- capture gelu, serve silu, get the gelu graph and no
    error. Ownership alone: ``if flag: import impl_b as impl`` then ``impl.op``
    is a read off a module namespace exactly like ``F.gelu``, and the module is
    user code an env var chose; so is ``self.act = abs``, where the value is a
    builtin nothing can repoint but the attribute holding it is a slot. The site
    survives the name stripping that makes source spelling useless -- a guard on
    ``self.act`` arrives as ``'self.act'`` -- because the structured source is
    still on the guard.

    For a capture-here / serve-there deployment the concern is not in-process
    rebinding but DIVERGENCE: the serving machine runs the same source but picks
    a different object because config, a flag, or an env var differs. Three
    bindings are waived -- a builtin read the ordinary way (see
    ``_reads_a_builtin``), a read off a TRUSTED namespace (see
    ``_module_namespaces``) that torch or the stdlib owns or that owns a def
    statement of that name, and a global bound to a def of that same name when
    torch or the stdlib owns the def or it lives in the file doing the reading.
    Both def-name tests compare ``__qualname__``, as ``_defined_where_read``
    does, so a def lifted off a class or returned by a factory is a slot. The
    rest are slots whose occupant config chooses: instance attributes, closure
    cells, aliased imports, cross-module ``from x import op``, registry
    lookups.

    Trusting a namespace is not trusting everything read off it. ``F.gelu`` is
    waived because torch owns torch.nn.functional and there is only one of it;
    ``own_helpers.call`` is waived because own_helpers owns a def of that same
    name, subject to the gap below. ``mypkg.op`` re-exported from
    ``mypkg.impl_b``, ``dispatch.op``, ``own_helpers.act`` bound to some other
    def, ``own_helpers.op`` bound to a staticmethod lifted off a class, and
    ``mypkg.impl.op`` where ``mypkg/__init__`` did ``from . import impl_b as
    impl`` are not waived: the import or assignment that chose the
    implementation lives in a file the inlined-source checksum never sees, so
    capture and serve can disagree with every other rail passing.
    ``test_risky_drop_decision_table`` in test_precompile_package.py pins these
    shapes and every other one found so far.

    KNOWN GAP, and it is a wrong-answer one. EVERY waiver above judges the
    object capture happened to bind, not the statement that bound it, so any
    name bound CONDITIONALLY is waived whenever the branch taken on the capture
    machine is one of the waived shapes. This is not specific to the def-name
    arm and it is not limited to the file being read:

    - def-name arm. ``if HAVE_FAST: from fastops import gelu`` / ``else: from
      torch.nn.functional import gelu``, captured without the flag, drops
      ``G['gelu']`` and reports nothing; a serving machine that has fastops
      runs torch's gelu instead, with no error. A def the reading file itself
      redefines under an ``if`` is the same shape.
    - namespace-owns-the-value arm. A module that binds a name under an ``if``
      and is read as a namespace by an ordinary ``import mod`` elsewhere is
      also waived, because at capture time the module really does own whichever
      def the branch produced. The reading file binds nothing conditionally,
      so it looks covered and is not.
    - library-namespace arm, and this one needs no conditional at all. The
      waiver trusts the owner, not the binding, so a third party that rebinds a
      torch or stdlib attribute -- ``F.gelu = _fast_gelu`` executed at import
      by a package that happens to be installed on the serving host -- is
      waived even though the model itself reads ``F.gelu`` unconditionally.
      ``functools.wraps(F.gelu)(user_fn)`` reaches the same waiver by a
      different route, since it copies ``__name__`` and ``__module__`` off the
      torch function it wraps. So does an alias config picks between two torch
      modules: ``F`` bound to torch.nn.functional or torch._refs.nn.functional
      is trusted either way, and ``F.gelu`` names a different function.

    An ``allow_in_graph`` function passes too, and Dynamo traces it opaquely so
    the inlined-source checksum never covers it either. Nothing at capture time
    distinguishes a conditional bind from an unconditional one -- only the
    resulting object is visible -- so this is a limit of the approach rather
    than a missing check. ``dropped_guards`` is the authoritative list; this
    predicate is a lint over it, not a proof of safety.
    """
    source = entry.orig_guard.originating_source
    value = entry.value
    stack = entry.orig_guard.user_stack
    if _is_dynamo_synthesized(source):
        return False
    if source.name in namespaces:
        return False
    if _reads_a_builtin(source, value):
        return False
    if not entry.has_value:
        return True  # nothing to judge ownership by
    if isinstance(source, AttrSource):
        namespace = namespaces.get(source.base.name)
        if namespace is not None:
            return not (
                _is_library_module(namespace.__name__)
                or (
                    _owning_module(value) == namespace.__name__
                    and getattr(value, "__qualname__", None) == source.member
                )
            )
    if (
        isinstance(source, GlobalSource)
        and getattr(value, "__name__", None) == source.global_name
    ):
        return not (
            _is_library_module(_owning_module(value))
            or _defined_where_read(value, source.global_name, stack)
        )
    return True


# The guards that pin a Python value by equality. CONSTANT_MATCH is the front
# door: it delegates a bool to BOOL_MATCH, None to NONE_MATCH, a code object to
# ID_MATCH and everything else -- ints included -- to EQUALS_MATCH, and it is
# the only guard method that derives into another one of these, so a slot's
# top-level guard type is enough here. CONSTANT_SUBCLASS_MATCH pins the base
# value of an int/float/str subclass argument; the two iterator guards pin an
# iterator argument's exact position. Length and key-set guards
# (SEQUENCE_LENGTH, TUPLE_ITERATOR_LEN, MAPPING_KEYS_CHECK) pin a container's
# structure rather than a value and are deliberately not counted; the set keys
# on guard type alone, so an EQUALS_MATCH installed on a container argument
# itself (a dict_keys or a frozenset of torch ops, variables/builder.py) IS
# counted, its contents being the value the artifact then serves only for.
_VALUE_EQUALITY_GUARD_TYPES = frozenset(
    {
        "CONSTANT_MATCH",
        "CONSTANT_SUBCLASS_MATCH",
        "COUNT_ITERATOR_MATCH",
        "EQUALS_MATCH",
        "RANGE_ITERATOR_MATCH",
    }
)


def _pins_a_value(guard_type: str, name: str) -> bool:
    """
    Whether this kept guard makes the artifact serve only the value it saw.

    Two things have to line up, and keying on either one alone is wrong.

    The guard has to be a value-equality one. TENSOR_MATCH, SHAPE_ENV and the
    global-state guards are what every capture has and they generalize fine --
    a TENSOR that crosses a graph break gets TENSOR_MATCH on a ``___stackN``
    source and is emphatically not a pin.

    And the source has to be a BARE name -- a plain local of some traced frame,
    or the ``___stackN`` Dynamo gives a value crossing a graph break. Anything
    dotted or subscripted (``self.eps``, ``model._modules['ln'].eps``,
    ``G['CFG'].width``) is reached THROUGH an argument rather than being one,
    which is where model config lives: every LayerNorm and Dropout contributes
    a CONSTANT_MATCH there, so counting those would flag every model and make
    the field noise. The EMPTY source of a guard checked against no source
    (``GuardFact.source`` for SHAPE_ENV, GLOBAL_STATE) is not a name either.

    KNOWN GAP: a constant inside a container argument is guarded on a
    subscripted source (``dims[0]`` for ``x.sum(dim=[0])``) and is not counted.
    ``kept_guards`` is the authoritative list; this is a lint over it.
    """
    return (
        bool(name)
        and guard_type in _VALUE_EQUALITY_GUARD_TYPES
        and not any(c in name for c in ".[")
    )


# Object addresses differ every run, so they are scrubbed from rendered guard
# facts. Keep these anchored to the call shapes that carry addresses: a bare
# \b\d{9,}\b also eats a user constant (a dict key, a slice bound), so two
# variants guarding different values render the same fact and invent an
# invariant neither holds.
_OBJ_ID = re.compile(r"(?<=, )\d+(?=\), type=)")
# There is deliberately no rule for the saved-tensors-hooks ids the guard
# renders ("... top_saved_tensors_hooks ids == (139, 140)"): that rendering is
# not an expression, so _mask_values below drops it whole and _normalize never
# sees the ids. _value_fingerprint is what names those hooks.
# Dynamo appends a per-process counter to the builtins dict it installs, so the
# same guard reads __builtins_dict___6 in one compilation and ___8 in the next.
# Of the globals Dynamo mints, only the three families in
# aot_compile._MINTED_GLOBAL_PREFIXES can root a serializable guard: this one and
# the two install_global_by_id shapes the pattern below covers. __compiled_fn_*
# and __resume_at_* are codegen-only LOAD_GLOBAL targets, never guard subjects.
_DYNAMO_COUNTER = re.compile(re.escape(_BUILTINS_DICT_PREFIX) + r"_\d+")
# OutputGraph.install_global_by_id names a global "<prefix>_<id(value)>_c<n>",
# so a guard reading one carries BOTH an address and a compile counter inside
# an identifier, where neither pattern above can see it. Real models reach this
# -- transformers' Qwen2 installs three -- and the report then differs run to
# run, which is exactly what the "commit and diff" contract rules out. The
# prefix can be empty (torch itself is installed as "_<id>_c<n>"), so the digit
# width is the anchor: it is what keeps a user identifier such as w_1_c2 intact.
_DYNAMO_GLOBAL_BY_ID = re.compile(r"_\d{9,}_c\d+\b")


def _normalize(text: str) -> str:
    text = _DYNAMO_GLOBAL_BY_ID.sub("_<id>_c<n>", _OBJ_ID.sub("<id>", text))
    return _DYNAMO_COUNTER.sub(_BUILTINS_DICT_PREFIX + "_<n>", text)


# A guard that pins a string pins it BY VALUE, so guards.py writes the value
# into the check it renders: an EQUALS_MATCH on a system prompt renders as
# L['self'].prompt == 'you are ...', and a dict keyed by a checkpoint path
# renders the path inside the guard's own name. The report those renderings go
# into is meant to be committed to a file and diffed, so every literal is masked
# by type where the fact is recorded. What makes the report auditable is the
# SHAPE of the check and the slot it names; that a value told two variants apart
# is reported as a varying slot, never by printing the value.
#
# The mask is a string CONSTANT ('<str>', '<list:3>'), so a masked check still
# parses and a caller that re-renders a recorded fact cannot collapse it into one
# placeholder. What a placeholder is not is READ BACK: a user string can be
# spelled exactly like one -- '<pad>' is an ordinary tokenizer token, and the
# user is who chooses it -- so there is no shape a string can be trusted by here.
# Every string constant is masked, and a second pass over a masked check rewrites
# a typed placeholder as '<str>', losing the type and never a value. Nothing this
# pass wrote is re-masked WITHIN a pass either, and not by its text: a masked
# node is a fresh constant put in where the traversal has already been. The bare
# <id> and <n> _normalize interpolates are the opposite case -- they run after
# the parse, on text nothing reads back.
#
# A subscript key is data unless what it subscripts is keyed by NAME. L, G and
# the builtins dict Dynamo installs are how a check spells a scope, and an
# nn.Module attribute is read through the module's own name dicts (mod.lin is
# rendered mod._modules['lin'], see GuardBuilder's __dict__ accessors), so those
# keys spell a source too. A user dict does not: in self.cfg['/home/me/w.pt']
# the value IS the key, and a secret field name is identifier-shaped exactly as
# 'lin' is, so the shape of the key cannot decide this. The BASE decides, and a
# kept key has to be name-shaped on top of that -- <> is allowed in it for the
# caller that masks an ALREADY NORMALIZED name, which is the one that records a
# slot, where a global reads as _<id>_c<n> inside the brackets; on the
# _render_code path masking runs first, so a key there is still the raw
# identifier.
#
# ONE implementation, on the tree: a source NAME is a Python expression too, so
# _mask_keys parses it and masks it with the same pass rather than approximating
# this rule over text, where a base is whatever precedes a bracket and every
# spelling of it has to be guessed.
_NAME_KEYED_SCOPES = frozenset({"L", "G"})
_NAME_KEYED_DICTS = frozenset({"__dict__", "_modules", "_parameters", "_buffers"})
_SLOT_KEY = re.compile(r"\A[A-Za-z_][\w<>]*\Z")
# The other place a rendering spells part of the SOURCE rather than a value: the
# argument of a call that carries an attribute NAME, by callable and position.
# HASATTR renders hasattr(L['x'], 'act'), and NOT_PRESENT_IN_GENERIC_DICT renders
# not ___dict_contains('act', L['x'].__dict__), which Dynamo installs once per
# attribute name on ONE source -- so masking the name would collapse the several
# facts that slot holds into one, and a slot that told two variants apart would
# be reported invariant. ___dict_contains is kept only where the dict beside the
# name is keyed by name, because DICT_CONTAINS renders the same helper over a
# USER dict, where argument 0 is the key itself. __import__('torch') is
# deliberately absent: a module name is a value like any other, and the ID_MATCH
# a rendered import carries names the module in GuardFact.value anyway.
_DICT_CONTAINS = "___dict_contains"
_NAME_ARGUMENT = {"getattr": 1, "hasattr": 1, _DICT_CONTAINS: 0}
# ___check_type_id renders as "<expr>, type=<class 'int'>", which is not one
# expression, so the annotation comes off before the parse and goes back where
# it was. Tolerant of a quote inside the class repr: an annotation left in the
# body costs the whole check, not just the annotation.
_CHECK_ANNOTATION = re.compile(r", type=<class '.*?'>")
# What a check that does not parse is reported as. Its own text cannot go in:
# masking needs the shape of an expression to tell a source from a value, and an
# arbitrary __repr__ can carry a path with no quote anywhere in it.
_UNPARSED_CHECK = "<unparsed check>"
# The same for a source name, where it lands in a slot rather than in a check.
_UNPARSED_SOURCE = "<unparsed source>"


def _keyed_by_name(base: ast.expr) -> bool:
    """Whether a subscript of ``base`` inside a check is keyed by a name."""
    if isinstance(base, ast.Name):
        return base.id in _NAME_KEYED_SCOPES
    if isinstance(base, ast.Attribute):
        return base.attr in _NAME_KEYED_DICTS
    if isinstance(base, ast.Subscript):
        # The builtins dict Dynamo installs, read out of a scope that is itself
        # keyed by name: G['__builtins_dict___6']['print']. The prefix alone
        # would hand the rule to whoever owns the dict, since a user dict may
        # hold a key spelled that way, and a base whose own key was masked must
        # not be able to keep the key under it.
        key = base.slice
        return (
            _keyed_by_name(base.value)
            and isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value.startswith(_BUILTINS_DICT_PREFIX)
        )
    return False


def _mask_expr(text: str) -> tuple[str, bool] | None:
    """``(text with its values masked, whether anything was masked)``.

    ``None`` when ``text`` is not one expression, which is the fail-closed
    case: the shape of an expression is what tells a source from a value here,
    so a text whose shape cannot be read is reported by its caller as a
    placeholder rather than patched.
    """
    mask = _MaskValues()
    try:
        tree = mask.visit(ast.parse(text, mode="eval"))
    except Exception:
        # Not a SyntaxError alone: a pinned container nests without limit, so a
        # deep enough one exhausts the stack in the parse or in this traversal,
        # and reporting a fact must never be the thing that breaks a capture.
        return None
    if not mask.masked:
        # Unparsing rewrites a text it has nothing to hide in -- it drops
        # redundant parentheses and respells a string -- so one without a
        # literal comes back exactly as its producer wrote it.
        return text, False
    try:
        return ast.unparse(tree), True
    except Exception:
        # Defensive, and for the same reason: ast.unparse raises on trees this
        # pass does not build (a non-string constant inside an f-string is one)
        # and on a tree too deep to walk.
        return None


def _mask_keys(name: str) -> str:
    """Mask the data keys a source name interpolates (cfg['/home/me/w.pt']).

    The rule is the one a rendered check goes through, run by the same code: a
    name is an expression, so it is parsed and masked as one, which is also
    what makes a key the text could not read -- a tuple, a number, a nested
    display -- masked rather than kept. A name that does not parse is reported
    as ``<unparsed source>``, so two such names read as one slot instead of
    reaching the report as their own text.
    """
    if not name:
        # A guard checked against no source (SHAPE_ENV, GLOBAL_STATE). Not a
        # name whose shape could not be read.
        return name
    masked = _mask_expr(name)
    return _UNPARSED_SOURCE if masked is None else masked[0]


class _MaskValues(ast.NodeTransformer):
    """Replace the values a rendered check embeds with their type.

    A node that is kept is mutated and returned, as ``generic_visit`` does; a
    node that is masked is replaced by a fresh constant.
    """

    def __init__(self) -> None:
        self.masked = False

    def _mask(self, node: ast.expr, kind: str) -> ast.expr:
        self.masked = True
        return ast.copy_location(ast.Constant(value=f"<{kind}>"), node)

    def _visit_expr(self, node: ast.expr) -> ast.expr:
        visited = self.visit(node)
        if not isinstance(visited, ast.expr):
            raise AssertionError(f"masking produced a {type(visited).__name__}")
        return visited

    def _visit_children(self, node: ast.expr) -> ast.expr:
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        if isinstance(node.value, str):
            # Every string, one shaped like a placeholder included: what this
            # pass wrote and what a user pinned read alike, and keeping the ones
            # that read alike would hand the rule to whoever picks the string.
            return self._mask(node, "str")
        if isinstance(node.value, bytes):
            return self._mask(node, "bytes")
        # A number, a bool and None stay: the number IS the check for a length
        # or a shape, and neither is a value a report can leak.
        return node

    def _mask_display(
        self, node: ast.List | ast.Set | ast.Dict | ast.Tuple
    ) -> ast.expr:
        # The whole display goes, not its elements: the names in a pinned list
        # of names are the value, and a display nests without limit -- a list of
        # dicts of names -- so a rule that had to look inside to stay safe is
        # one an element shape it does not model defeats. Its LENGTH stays, so
        # two variants pinning containers of different size still read
        # differently; two of the same size read alike, and a display of NUMBERS
        # (a marked-dims set, a pinned shape) reads as its length where a bare
        # number would have been kept. That is what not looking inside costs.
        items = node.keys if isinstance(node, ast.Dict) else node.elts
        return self._mask(node, f"{type(node).__name__.lower()}:{len(items)}")

    visit_List = visit_Set = visit_Dict = visit_Tuple = _mask_display

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.expr:
        # One value, masked whole. What a check compares against an f-string is
        # the joined string, so rewriting the pieces in place would report a
        # structure the value does not have -- and not every piece is a node a
        # placeholder can stand in for, since a format spec is an f-string too.
        return self._mask(node, "str")

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        key = node.slice
        if (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and _SLOT_KEY.match(key.value)
            and _keyed_by_name(node.value)
        ):
            node.value = self._visit_expr(node.value)
            return node
        return self._visit_children(node)

    def visit_Call(self, node: ast.Call) -> ast.expr:
        name = node.func.id if isinstance(node.func, ast.Name) else ""
        kept = _NAME_ARGUMENT.get(name)
        if (
            kept is None
            or len(node.args) < 2
            or (name == _DICT_CONTAINS and not _keyed_by_name(node.args[1]))
        ):
            return self._visit_children(node)
        # The attribute name is part of the source being read, not a value being
        # compared. That one argument only -- getattr's DEFAULT is a value like
        # any other, as is the key a containment check on a user dict compares.
        node.args = [
            arg if i == kept else self._visit_expr(arg)
            for i, arg in enumerate(node.args)
        ]
        for keyword in node.keywords:
            keyword.value = self._visit_expr(keyword.value)
        return node


def _mask_values(text: str) -> str:
    """Name what a rendered check compares by type instead of by value.

    What it writes parses, so a caller that masks a recorded check again reads
    it as a check rather than collapsing it whole; the placeholders in it are
    masked again, a typed one reading as ``'<str>'``, because a user string can
    be spelled like a placeholder and recognizing one by its shape is what let
    such a string through. A display is masked whole rather than element by element
    because a display can nest strings arbitrarily and a pass that stays safe
    only by inspecting elements is defeated by an element shape it does not
    model; it is named by its type and its length alone, so two variants pinning
    containers of the same length -- two pinned shapes, two sets of marked dims
    -- render the same check, and what told them apart has to come from
    ``GuardFact.value`` or from the slot. The ``, type=<class 'int'>`` tail
    ``___check_type_id`` appends is not part of the expression, so it comes off
    before the parse and goes back where it was; a rendering carrying one
    anywhere but at its end is reported as ``<unparsed check>``, since the
    unparse regenerates the whole text and the position cannot be restored.
    """
    annotations = list(_CHECK_ANNOTATION.finditer(text))
    masked = _mask_expr(_CHECK_ANNOTATION.sub("", text))
    if masked is None:
        # Fail closed. A rendering that is not an expression gives nothing to
        # tell a source from a value: the saved-tensors-hooks guard renders
        # prose, and an EQUALS_MATCH on a type pytree.register_constant admits
        # renders that class's __repr__, which can carry a path with no quote
        # anywhere in it for a textual pass to find. The text goes whole, and
        # for the hooks guard _value_fingerprint still tells two hook sets
        # apart. For an EQUALS_MATCH on a repr nothing does: its fingerprint is
        # "" and _pins_a_value counts bare names only, so two variants pinning
        # different objects on one slot report one fact and the slot reads
        # invariant. A digest of the text would tell them apart and is
        # deliberately not written: a stable fingerprint of the value this pass
        # exists to hide, in a file meant to be committed, is an oracle for
        # guessing that value.
        return _UNPARSED_CHECK
    body, anything_masked = masked
    if not anything_masked:
        # Nothing to hide, so the check is returned exactly as guards.py wrote
        # it -- annotation included, in the place guards.py put it.
        return text
    if len(annotations) > 1 or (annotations and annotations[0].end() != len(text)):
        # An annotation the unparse cannot put back where it was. Reordering it
        # would change what the check reads as, and a ", type=<class '...'>"
        # inside the body is likelier a false match on the pattern than an
        # annotation, so this fails closed like any other shape the pass cannot
        # read. No in-tree rendering puts one anywhere but at the end.
        return _UNPARSED_CHECK
    return body + (annotations[0].group(0) if annotations else "")


def _render_code(code_list: Sequence[str] | None) -> tuple[str, ...]:
    # Masked BEFORE normalizing: _normalize interpolates <id> and <n>
    # placeholders that no longer parse as Python.
    return tuple(_normalize(_mask_values(part)) for part in (code_list or ()))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# Guards whose check IS object identity, directly or through a derived guard,
# which is the same test default_guard_filter_fn drops on.
_IDENTITY_GUARD_TYPES = frozenset(
    CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
)


# Ellipsis and NotImplemented repr by name, so they are as stable as a literal.
_STABLE_CONST_TYPES = (
    str,
    int,
    float,
    complex,
    bytes,
    bool,
    type(None),
    type(Ellipsis),
    type(NotImplemented),
)


def _stable_consts(consts: tuple[object, ...]) -> tuple[object, ...]:
    """
    co_consts reduced to the part that reprs the same in every process.

    A nested code object reprs with its ADDRESS, so it cannot go into a digest
    that ends up in a file meant to be committed and diffed. Containers are
    filtered recursively rather than dropped whole: two lambdas differing only
    in a tuple or frozenset constant -- ``x * (1, 2)`` against ``x * (1, 3)`` --
    are genuinely different variants, and dropping the container is what let
    them collide. A const of any other type keeps its SLOT as a type marker:
    ``x[..., 0]`` and ``x[0, ...]`` fold to one const tuple at the same index,
    so with the slot dropped they would collide the same way.
    """
    out: list[object] = []
    for c in consts:
        if isinstance(c, _STABLE_CONST_TYPES):
            out.append(c)
        elif isinstance(c, types.CodeType):
            # A nested code object reprs with its ADDRESS, so it cannot go in
            # verbatim -- but dropping it merges two lambdas that differ only in
            # a comprehension or an inner lambda, which is this same bug one
            # level down. Recurse into its own fingerprint instead.
            out.append(_code_fingerprint(c))
        elif isinstance(c, tuple):
            out.append(_stable_consts(c))
        elif isinstance(c, frozenset):
            # Sorted by repr so the digest does not inherit set iteration order.
            out.append(tuple(sorted(_stable_consts(tuple(c)), key=repr)))
        else:
            out.append(f"<{type(c).__name__}>")
    return tuple(out)


def _code_fingerprint(code: types.CodeType) -> str:
    """
    Name a code object by its body, for callables a definition site cannot tell
    apart -- an ACT2FN table written on one source line makes every lambda in it
    agree on file AND lineno.

    Everything hashed is derived from the source, so the digest is identical in
    another process. It is NOT stable across Python versions, since co_code is
    version-specific bytecode: a committed invariants file churns wholesale on
    an interpreter upgrade even with unchanged source.
    """
    return _hash_text(
        repr(
            (
                code.co_code,
                code.co_names,
                code.co_varnames,
                # LOAD_DEREF addresses a cell by INDEX, so two closures that
                # capture different variables have identical co_code and are
                # told apart only by the names they close over.
                code.co_freevars,
                code.co_cellvars,
                _stable_consts(code.co_consts),
            )
        )
    )


def _object_identity(value: object) -> str:
    """
    A stable stand-in for the id ``_normalize`` stripped.

    A qualname alone does not separate the case this exists for: an ACT2FN-style
    table whose entries are all ``<lambda>`` in one module, where two variants
    holding different entries would render identically and the CLOSURE_MATCH
    that split them would be reported as an invariant of both. So a callable is
    also named by where it is DEFINED (basename, so no checkout path) and by a
    digest of its body, both source-derived and so stable across processes.

    The discriminating part goes FIRST: truncation bounds this string, and a
    transformers lambda nested in a long module path exceeds the limit on the
    qualname alone, so a digest appended at the end would be cut off exactly on
    the names that need it most.

    Only the code object is named, not the data bound to it. Two closures from
    one factory (``make(2)`` and ``make(3)``), two functions differing only in
    ``__defaults__``, bound methods of two instances, and every
    ``functools.partial`` render the same, as do two instances of one class.
    """
    if isinstance(value, types.ModuleType):
        return f"is module {value.__name__}"
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if isinstance(name, str):
        code = getattr(value, "__code__", None)
        where = ""
        if isinstance(code, types.CodeType):
            filename = os.path.basename(code.co_filename or "?")
            where = f"@{filename}:{code.co_firstlineno}#{_code_fingerprint(code)} "
        return _normalize(f"is {where}{_owning_module(value) or '?'}.{name}")[:160]
    return f"is a {type(value).__module__}.{type(value).__qualname__}"[:160]


# Guard types that pin an input's shape, value or kind. An invariance policy
# never drops one, and the report compares them across variants.
_SHAPE_BEARING_GUARD_TYPES = frozenset(
    {
        "BOOL_MATCH",
        "CONSTANT_MATCH",
        "CONSTANT_SUBCLASS_MATCH",
        "COUNT_ITERATOR_MATCH",
        "COW_TENSOR_MATCH",
        "DICT_CONTAINS",
        "DICT_KEYS_MATCH",
        "DICT_NOT_CONTAINS",
        "DUPLICATE_INPUT",
        "EMPTY_NN_MODULE_HOOKS_DICT",
        "EQUALS_MATCH",
        "FAKE_SCRIPT_TYPE_MATCH",
        "HASATTR",
        "MAPPING_KEYS_CHECK",
        "NONE_MATCH",
        "NOT_NONE_MATCH",
        "NOT_PRESENT_IN_GENERIC_DICT",
        "RANGE_ITERATOR_MATCH",
        "SEQUENCE_LENGTH",
        "SET_CONTAINS",
        "SET_NOT_CONTAINS",
        "TENSOR_MATCH",
        "TUPLE_ITERATOR_LEN",
        "TYPE_MATCH",
    }
)


# Guard types on ambient or process state that the guard leaf checks for itself
# and nothing in this module fingerprints. Never dropped, never compared.
_UNMODELLED_GUARD_TYPES = frozenset(
    {
        "AUTOGRAD_SAVED_TENSORS_HOOKS",
        "DEFAULT_DEVICE",
        "DISPATCH_KEY_SET_MATCH",
        "DTENSOR_SPEC_MATCH",
        "DUAL_LEVEL",
        "FSDP_TRAINING_STATE",
        "FUNCTORCH_STACK_MATCH",
        "GLOBAL_STATE",
        "OPAQUE_OBJ_GUARD_FN_MATCH",
        "SHAPE_ENV",
        "TENSOR_SUBCLASS_METADATA_MATCH",
        "TORCH_FUNCTION_STATE",
    }
)


# Guard types whose GuardBuilder method is `pass`: the guard is a marker, and
# the check it names is made by GLOBAL_STATE's leaf. Nothing about them is
# serialized or dropped, so they never appear in a dropped-guard report --
# listing GRAD_MODE as "a precondition nothing checks" would be false, since
# GlobalStateGuard checks it on every call. Their facts ARE compared, from the
# same process state GlobalStateGuard snapshots (see _value_fingerprint).
_NOOP_GUARD_TYPES = frozenset({"DETERMINISTIC_ALGORITHMS", "GRAD_MODE"})


def _is_noop_guard_type(guard_type: str) -> bool:
    # EMPTY_NN_MODULE_HOOKS_DICT is classified shape-bearing for the config
    # where it emits a check; under skip_nnmodule_hook_guards, the default,
    # GuardBuilder emits nothing for it, so a report must not call it a
    # precondition.
    return guard_type in _NOOP_GUARD_TYPES or (
        guard_type == "EMPTY_NN_MODULE_HOOKS_DICT"
        and torch._dynamo.config.skip_nnmodule_hook_guards
    )


# The ONLY guard types the invariance policy may drop, and only when proven
# invariant across every captured variant: the identity guards the default
# filter drops anyway as unserializable, plus BUILTIN_MATCH, an identity match
# on a builtin that the default filter keeps. The four sets are a total,
# disjoint classification of GuardBuilder's guard methods, pinned by
# test_guard_policy_classification_is_total: a guard type in none of them --
# any type added to GuardBuilder after this list -- is KEPT unconditionally
# until someone classifies it, so a new value-pinning guard can never become
# silently droppable. Guards installed outside GuardBuilder (the root manager's
# DuplicateInputs and StorageOverlap exprs, the dimension-marking lambda) never
# reach the guard filter and are outside the policy as well.
_INVARIANT_DROPPABLE_GUARD_TYPES = _IDENTITY_GUARD_TYPES | frozenset({"BUILTIN_MATCH"})


def _saved_hooks_fingerprint() -> str:
    """
    Name the installed saved-tensors hooks the way the guard compares them.

    The guard stores ``tuple(map(id, hooks))`` when both hooks are fx
    GraphModules and ``None`` otherwise, so plain-Python hooks, and no hooks at
    all, are one value to it and must be one value here, or the report shows a
    'varies' line for a guard that passes either way. Inlineable hooks are
    named by their rendered graph rather than by address, since an id cannot
    go in a committed, diffable file. KNOWN TRADEOFF: two distinct GraphModules
    with identical code read as one hook set here while the guard tells them
    apart, so the report may call that guard invariant when it is not.
    """
    try:
        from torch._functorch._aot_autograd.utils import (
            saved_tensors_hooks_are_inlineable,
            top_saved_tensors_hooks,
        )

        hooks = top_saved_tensors_hooks()
        if not saved_tensors_hooks_are_inlineable(hooks):
            return "hooks=None"
        return "hooks=(" + ", ".join(_hash_text(hook.code) for hook in hooks) + ")"
    except Exception:
        # Distinct from the "" that means "the rendered code already names the
        # check": a failed read must not merge two variants.
        return "hooks=<unreadable>"


def _value_fingerprint(entry: GuardFilterEntry) -> str:
    """
    What the guard checks, when the rendered code does not say.

    TENSOR_MATCH is the case that matters: its code_list carries only the
    _dynamo_*_indices hasattr checks, while everything it really compares lives
    in the C++ leaf. Without those two specializations of one frame look
    identical and wrongly land in the intersection, so this mirrors TensorCheck
    -- python type and the full dispatch key set included, since a Parameter
    against a Tensor, a conjugated view against a plain one, or an
    inference-mode tensor against a no_grad one, splits a compilation exactly
    as dtype does. KNOWN GAP: that leaf checks nothing for a dim the compile
    made dynamic, so under ``dynamic=True`` the concrete shape here is narrower
    than the guard and a shape-generic TENSOR_MATCH is reported as varying
    rather than invariant.

    An identity guard needs one too, because ``_normalize`` strips the id its
    code renders: without a name for the object, two variants holding different
    callables at one source collapse into one fact and are reported as an
    invariant neither of them holds.

    Every other guard takes its value from its own rendered code, which names
    it, so fingerprinting it again SPLITS identical guards: TYPE_MATCH on an
    unspecialized int checks only that the int is an int, and stamping 1 on one
    variant and 2 on the next demotes a real invariant into two identical
    'varies' lines. So every branch dispatches on the guard type, never on the
    value's: the NOT_NONE_MATCH Dynamo installs on an optimizer's .grad, or a
    TYPE_MATCH on a tensor attribute, holds a tensor whose shape and dtype it
    never checks.
    """
    if entry.guard_type == "AUTOGRAD_SAVED_TENSORS_HOOKS":
        # Its code renders "... ids == tuple(map(id, hooks))", which is not an
        # expression, so _mask_values drops it whole -- and dropping it alone
        # would merge two variants
        # that differ ONLY in their hooks and report the guard that split them
        # as an invariant. Put back a discriminator derived from what the hooks
        # ARE rather than where they live, which is both stable across
        # processes and still telling.
        return _saved_hooks_fingerprint()
    if entry.guard_type == "GRAD_MODE":
        # Global-state guards carry no name, code or value, so two variants of
        # one frame that differ only in grad mode render identically and land
        # in the intersection. The filter runs in the traced frame's own state.
        return f"grad_enabled={torch.is_grad_enabled()}"
    if entry.guard_type == "DETERMINISTIC_ALGORITHMS":
        # Same as GRAD_MODE: the two fields GlobalStateGuard snapshots for it.
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        return f"deterministic={torch.are_deterministic_algorithms_enabled()}, warn_only={warn_only}"
    if not entry.has_value:
        return ""
    value = entry.value
    if entry.guard_type == "TENSOR_MATCH" and isinstance(value, torch.Tensor):
        # Render exactly what TensorCheck stores (notably the TLS-adjusted
        # dispatch key set, not the tensor's own) rather than reconstructing it.
        # The fact's source already names the tensor, so no name goes in.
        from .guards import convert_to_concrete_values, get_tensor_guard_code_part

        try:
            return get_tensor_guard_code_part(
                value,
                "<value>",
                convert_to_concrete_values(value.size()),
                convert_to_concrete_values(value.stride()),
                type(value),
                torch._C._dispatch_keys(value),
            )
        except Exception:
            # A subclass whose __torch_function__ refuses attribute reads got
            # here; type() is the one read that cannot raise.
            return f"type={type(value).__name__}, <unrenderable>"
    if entry.guard_type in _IDENTITY_GUARD_TYPES or any(
        d in _IDENTITY_GUARD_TYPES for d in entry.derived_guard_types
    ):
        return _object_identity(value)
    return ""


def _fact_order(fact: _GuardFact) -> tuple[str, str, str, str]:
    # value is part of the key: once the boilerplate code parts are filtered a
    # TENSOR_MATCH renders no code, so two shape specializations would otherwise
    # tie and sort unstably, making the file differ run to run.
    return (fact.source, fact.guard_type, " ".join(fact.code), fact.value)


def _wont_generalize(
    kept: set[tuple[str, str]],
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
) -> tuple[str, ...]:
    """Sources no captured variant will serve a new value for.

    A source pinned in ONE variant is not pinned for the artifact: the union of
    kept guards says "some graph equality-matched this", while what the warning
    claims is "no graph will take anything else". A frame whose other variant
    guards the same source generically -- the ordinary shape once two examples
    are captured -- serves the new value fine, and warning about it tells the
    caller to enumerate values that already work.

    Cancellation is per frame, like every comparison in this module: a bare
    name means nothing across frames, and ``___stack0`` names whatever crossed
    the break in EVERY resume frame -- a tensor in one, an ``.item()`` int in
    the next -- so a generic mention elsewhere must not erase a real pin here.
    ``GuardFact.source`` and a kept slot's name share one spelling, the
    ``GuardFilterEntry.name`` with local scope stripped (``L['x']`` -> ``x``).
    A fact whose guard the filter dropped (``enforced`` False) checks nothing,
    so its variant counts as serving the source generically, not as pinning it;
    the drop itself is what ``dropped_guards`` reports, so the case changes
    fields rather than disappearing. A variant's OWN companion guards on a
    source it pins (the SEQUENCE_LENGTH beside a dict_keys EQUALS_MATCH) are
    not a generic mention, hence the ``- here`` below.

    KNOWN GAP: "reached it without pinning it" is read as "serves other
    values", and an unspecialized int breaks that: its variant carries only a
    TYPE_MATCH on the name, and if the trace then specializes the symbol the
    ``L['scale'] == 3`` check is a SHAPE_ENV fact with an EMPTY source, so the
    variant looks generic and cancels a sibling's pin that nothing serves.
    """
    pinned = {n for t, n in kept if _pins_a_value(t, n)}
    if not pinned:
        return ()
    # A source survives if SOME frame pins it and never serves it generically;
    # a frame that does both cancels only its own pin, not another frame's.
    survivors: set[str] = set()
    for variants in guard_sets.values():
        pins: set[str] = set()
        generic: set[str] = set()
        for facts in variants:
            here = {
                f.source
                for f in facts
                if f.enforced and _pins_a_value(f.guard_type, f.source)
            }
            pins |= here
            # A variant that reached the source without pinning it is the
            # graph that serves other values.
            generic |= {f.source for f in facts} - here
        survivors |= pins - generic
    return tuple(sorted(pinned & survivors))


def _varying_guard_slots(
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
) -> frozenset[tuple[str, str]]:
    """The guard slots that actually discriminate between captured variants.

    A slot is ``(guard_type, source)``, the source as ``GuardFact.source``
    spells it: the ``GuardFilterEntry.name`` with local scope stripped
    (``L['x']`` -> ``x``). It varies when two variants of one frame
    recorded DIFFERENT facts for it, and also when it is present in some
    variants and absent in others -- a guard only one variant carries is what
    tells that variant apart, and comparing values alone would call it
    invariant and drop it. That present-in-some case is the majority of what is
    kept, not an edge. A fact is its rendered code and value; ``enforced`` says
    whether the serialized copy keeps the guard, not what it checks, so two
    variants that differ only there agree. One variant can hold several facts
    on one slot (a ``HASATTR`` per attribute name, all on the parent source),
    so what is compared across variants is each variant's SET of facts for the
    slot, never one fact against another inside a variant.

    Everything else held identically in every variant, which is what licenses a
    caller to leave it out of the serialized copy.
    """
    varying: set[tuple[str, str]] = set()
    for variants in guard_sets.values():
        seen: dict[tuple[str, str], list[frozenset[tuple[tuple[str, ...], str]]]] = {}
        for facts in variants:
            rendered: dict[tuple[str, str], set[tuple[tuple[str, ...], str]]] = {}
            for f in facts:
                slot = (f.guard_type, f.source)
                rendered.setdefault(slot, set()).add((f.code, f.value))
            for slot, facts_here in rendered.items():
                seen.setdefault(slot, []).append(frozenset(facts_here))
        for slot, per_variant in seen.items():
            if len(per_variant) != len(variants) or len(set(per_variant)) > 1:
                varying.add(slot)
    return frozenset(varying)


def _summarize(
    entry: _DynamoCacheEntry,
    *,
    dropped: set[tuple[str, str]],
    kept: set[tuple[str, str]],
    policy_dropped: set[tuple[str, str]],
    risky: set[tuple[str, str]],
    truncated: frozenset[str],
    capture_errors: Sequence[str],
    guard_sets: Mapping[tuple[str, str, int], Sequence[frozenset[_GuardFact]]],
    dropped_code: Mapping[tuple[str, str], str],
) -> PrecompileSummary:
    """Assemble the report. ``dropped_code`` maps a dropped slot to the one
    rendering the report carries for it, the caller's pick among the variants
    that dropped it; a slot without one gets no ``dropped_guard_code`` entry.

    ``bypassed`` and ``uncovered_frames`` are read off the entry, one bare
    ``co_name`` per frame, so a repeated name is two frames and both lists are
    subsets of ``frames`` by construction. Uncovered is the coverage gap: the
    frame entered Dynamo (``has_compile_id``) and holds no guarded code, and was
    not bypassed. ``install()`` ``skip_code()``s a superset, every entry that is
    not bypassed and has no guarded codes whether or not it entered Dynamo, so a
    generated-but-never-executed resume entry is skipped there and is not a gap
    here. ``backend_graphs`` counts the backend ids of the entries that are not
    bypassed, the ones ``install()`` loads: a save-time bypass
    (``PrecompileCacheEntry.from_cache_entry``, backend artifact missing) marks
    the entry and leaves its ids in place. ``truncated`` comes from the compile
    path, which sees a frame hit the limit once and records it as ``co_name
    (filename:firstlineno)``, so that set cannot merge two frames.
    """
    return PrecompileSummary(
        frames=len(entry.codes),
        resume_functions=sum(1 for c in entry.codes if c.install_to_global),
        guarded_codes=sum(len(c.guarded_codes) for c in entry.codes),
        backend_graphs=len(
            {b for c in entry.codes if not c.bypassed for b in c.backend_ids}
        ),
        bypassed=tuple(c.python_code.co_name for c in entry.codes if c.bypassed),
        truncated=tuple(sorted(truncated)),
        uncovered_frames=tuple(
            c.python_code.co_name
            for c in entry.codes
            if c.has_compile_id and not c.guarded_codes and not c.bypassed
        ),
        wont_generalize=_wont_generalize(kept, guard_sets),
        dropped_guards=tuple(sorted(dropped)),
        dropped_guard_code=tuple(
            (gtype, name, dropped_code[(gtype, name)])
            for gtype, name in sorted(dropped | policy_dropped)
            if (gtype, name) in dropped_code
        ),
        kept_guards=tuple(sorted(kept)),
        risky_dropped_guards=tuple(sorted(risky)),
        policy_dropped_guards=tuple(sorted(policy_dropped)),
        capture_errors=tuple(capture_errors),
    )


def _compose_with_default(
    user: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]] | None,
) -> Callable[[Sequence[GuardFilterEntry]], tuple[Sequence[bool], Sequence[bool]]]:
    """AND a caller's filter with the default rather than replacing it.

    ``default_guard_filter_fn`` is not a default in the "sensible starting point"
    sense -- it is what drops the identity guards that CANNOT be serialized at
    all. Replacing it means a caller who wanted to drop three of their own guards
    silently re-admits every unserializable one, and the failure surfaces as
    "ID_MATCH guard cannot be serialized" in frames that have nothing to do with
    their filter. A custom filter can only ever want to drop MORE, so composing
    is the only reading that makes sense. ``user=None`` is no custom filter at
    all, where the default's decisions ARE the composition.

    Both are returned: the recorder judges a drop against the default's own
    verdict, and returning it here is what keeps the default to one call per
    compile.
    """

    def composed(
        entries: Sequence[GuardFilterEntry],
    ) -> tuple[Sequence[bool], Sequence[bool]]:
        base = default_guard_filter_fn(entries)
        if user is None:
            return base, base
        chosen = user(entries)
        if len(chosen) != len(entries):
            raise ValueError(
                f"guard_filter_fn returned {len(chosen)} decisions for "
                f"{len(entries)} guards; it must return one per entry."
            )
        return [bool(a) and bool(b) for a, b in zip(base, chosen)], base

    return composed


# A compiled frame, as _varying_guard_slots and _wont_generalize key it: what
# the report prints for it. Two frames whose code objects share a name, a file
# and a first line -- two lambdas on one line, or two frames exec'd under
# "<string>" -- read as variants of one frame, which is the tradeoff those
# helpers take.
_FrameKey = tuple[str, str, int]

# What one variant checked for a slot: the rendered check and the value it
# compared, which is what changes when the value behind a dropped slot changes.
_SlotCheck = tuple[tuple[str, ...], str]


@dataclasses.dataclass(frozen=True)
class _RecordedCompile:
    """What one guard-filter call decided, held until that compile's outcome is
    known.

    A compile that bypasses -- static-address parameters, a guard that cannot be
    serialized -- lands no guarded code, so no artifact enforces the guards the
    filter kept and no artifact was widened by the ones it dropped. The filter
    runs before either is decidable, so nothing is published when it runs: the
    facts are recorded per compile here and _confirmed_compiles keeps the ones
    whose guarded code reached the package entry.

    ``guarded_codes_before`` is the length of ``entry.guarded_codes`` when the
    filter ran, which is what makes "this compile landed" readable off the entry
    afterwards: the list grew.
    """

    frame: _FrameKey
    entry: _DynamoCodeCacheEntry
    guarded_codes_before: int
    kept: frozenset[tuple[str, str]]
    dropped: frozenset[tuple[str, str]]
    risky: frozenset[tuple[str, str]]
    # slot -> the check this compile rendered for it, in the order the guards
    # came in; summary() takes the first rendering across compiles.
    dropped_code: tuple[tuple[tuple[str, str], str], ...]
    facts: frozenset[_GuardFact]
    # The facts of the guards nothing here can compare, kept apart rather than
    # intersected with the rest: see _UNMODELLED_GUARD_TYPES.
    undetermined: frozenset[_GuardFact]


def _fact_key(fact: _GuardFact) -> tuple[str, str, str, str, bool]:
    # _fact_order leaves out enforced, which is the one field two otherwise
    # identical facts of one frame can differ in: a config read by the filter
    # (skip_nnmodule_hook_guards) toggled between two calls flips whether a slot
    # is checked and nothing else. Tied lines would keep frozenset iteration
    # order, which follows the hash of their strings, so the file would differ
    # between runs of the same capture -- the one property it has to have.
    return (*_fact_order(fact), fact.enforced)


def _render_fact(fact: _GuardFact) -> str:
    """Render one guard as a stable, human-readable line for the report."""
    body = " ; ".join(fact.code) if fact.code else f"<{fact.guard_type}>"
    if fact.value:
        body = f"{body} {fact.value}"
    where = f" on {fact.source}" if fact.source else ""
    label = "enforced" if fact.enforced else "dropped"
    return f"[{label:<8}] {body}{where}"


def _entry_fn_of(fn: object) -> Callable[..., object]:
    if isinstance(fn, torch.nn.Module):
        forward = fn.forward
        if not hasattr(forward, "__code__"):
            raise TypeError(
                f"{type(fn).__name__}.forward is a {type(forward).__name__}, which "
                f"has no __code__ for Dynamo to capture or to load an artifact onto. "
                f"Binding it in __init__ -- self.forward = functools.partial(...) -- "
                f"shadows the class method and lands here; keep forward a method."
            )
        return forward
    if not callable(fn):
        raise TypeError(f"expected a callable or nn.Module, got {type(fn).__name__}")
    if not hasattr(fn, "__code__"):
        raise TypeError(
            f"expected a function or nn.Module, got {type(fn).__name__}, which "
            f"has no __code__ for Dynamo to capture or to load an artifact "
            f"onto. Pass partial.func for a functools.partial, or obj.__call__ "
            f"for an object that only defines __call__."
        )
    return fn  # type: ignore[return-value]


def _identify_graph(gm: torch.fx.GraphModule) -> str:
    """Name a graph well enough to find it, from inside a backend.

    The module class alone does not: every graph a model produces reports the
    same one, so a capture that recompiled nine graphs at serve time said
    "GraphModule" nine times. The compile id keys tlparse, the backend id keys
    the artifact, and the first node carrying a stack trace names the user line
    -- including, for a continuation, the resume frame Dynamo minted for it,
    which is the only thing that tells one break in a chain from another.
    """
    parts: list[str] = []
    compile_id = torch._guards.CompileContext.current_compile_id()
    if compile_id is not None:
        parts.append(f"compile id {compile_id}")
    backend_id = gm.meta.get("backend_id") or getattr(gm, "_backend_id", None)
    if backend_id is not None:
        parts.append(f"backend id {backend_id}")
    for node in gm.graph.nodes:
        if node.op not in ("placeholder", "output") and node.stack_trace:
            first = node.stack_trace.strip().splitlines()[0].strip()
            parts.append(f"first traced at {first}")
            break
    return f" Graph: {'; '.join(parts)}." if parts else ""


class _PrecompileBackend:
    """One session's own object wrapped around the inner backend.

    It is not what gives the session a distinct cache identity: CacheEntry
    stores get_backend(backend), which follows every _torchdynamo_orig_backend
    link (torch/csrc/dynamo/cache_entry.cpp), so the entry ends up holding the
    same inner eager/inductor function every other session gets; the isolation
    comes from isolate_recompiles=True in _optimize_isolated. What the wrapper
    provides is a per-session object on the compile path, for bookkeeping that
    has to count or hold what the inner backend was handed.
    """

    def __init__(
        self, backend: str, keep_graphs: bool = False, serving: bool = False
    ) -> None:
        inner = torch._dynamo.lookup_backend(backend)
        self._torchdynamo_orig_backend = inner
        # Named the way get_compiler_fn derives a name, because a wrapper object
        # has none: without it the minifier repro, the compile log and
        # BackendCompilerFailed all report an unknown backend.
        self.compiler_name = getattr(inner, "compiler_name", backend)
        self.backend_ctx_ctor = getattr(
            inner, "backend_ctx_ctor", contextlib.nullcontext
        )
        # Rendering a subgraph as source needs the graph, which only exists
        # here. Kept only where something will render it (see the caller):
        # retaining deepcopies every compiled graph for the session.
        self._keep_graphs = keep_graphs
        self.graphs: dict[str, tuple[torch.fx.GraphModule, list[Any]]] = {}
        # Serving an INSTALLED artifact answers a guard miss by compiling,
        # because a frame reachable only through the frame evaluator has no
        # other way to run. Counted, and said out loud once per graph: an
        # artifact that quietly compiles more of itself on every batch looks
        # exactly like one that is serving.
        self.serving = serving
        self.serve_time_compiles = 0

    # Forwarded, as _TorchCompileWrapper and AotAutograd do, so the inner
    # backend's one-time init still fires through the wrapper; read at fire time
    # so the hook can be set after the session was built.
    @property
    def _dynamo_backend_init(self) -> Any | None:
        return getattr(self._torchdynamo_orig_backend, "_dynamo_backend_init", None)

    def __call__(
        self, gm: torch.fx.GraphModule, inputs: list[torch.Tensor], **kwargs: Any
    ) -> Any:
        if self.serving:
            self.serve_time_compiles += 1
            log.warning(
                "precompile: serving compiled a NEW graph -- no captured variant "
                "matched this call, so the artifact is serving less than it was "
                "measured to. Recapture with an example that covers it.%s",
                _identify_graph(gm),
            )
        if self._keep_graphs:
            backend_id = gm.meta.get("backend_id")
            if backend_id is not None and str(backend_id) not in self.graphs:
                # Deep-copy before the inner backend runs: inductor lowering
                # mutates the graph it is handed, and a rendered copy has to be
                # the graph Dynamo produced, not the leftovers.
                placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
                # Render against the placeholders' FAKES, never the real inputs
                # below: compile_fx re-fakifies real tensors into a fresh symbol
                # set and dedups by value, which silently unifies a batch dim
                # with any other dim of the same size.
                fakes = [
                    n.meta.get("example_value", n.meta.get("val")) for n in placeholders
                ]
                if all(f is not None for f in fakes):
                    self.graphs[str(backend_id)] = (copy.deepcopy(gm), fakes)
        return self._torchdynamo_orig_backend(gm, inputs, **kwargs)

    def get_compiler_config(self) -> Any:
        getter = getattr(self._torchdynamo_orig_backend, "get_compiler_config", None)
        return None if getter is None else getter()


class _ReportLimitConvertFrame(_AllowEmptyGraphsConvertFrame):
    """The package's frame converter, reporting a recompile-limit hit.

    Hitting the cap truncates the capture: Dynamo refuses the variant, runs the
    frame eagerly from then on, and says so only in a log warning, so a caller
    would get an artifact with fewer variants than they exercised and no sign of
    it. The converter's return value is the one in-process signal --
    convert_frame._compile puts a RUN_ONLY strategy on the Unsupported it
    raises for the cap, and nothing else in Dynamo sets FrameAction.RUN_ONLY --
    so a RUN_ONLY return here means the cap was hit. Reported, never raised:
    the calls keep running, as the limit's contract says.
    """

    def __init__(
        self,
        *args: Any,
        # Optional because the base class's package-less DDP clone rebuilds this
        # type positionally; a session's converter always carries a package, so
        # it takes the refusing clone instead.
        on_recompile_limit: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._on_recompile_limit = on_recompile_limit

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: CacheEntry | None,
        hooks: Hooks,
        frame_state: dict[str, int | FrameStateSizeEntry],
        skip: int = 0,
    ) -> ConvertFrameReturn:
        result = super().__call__(frame, cache_entry, hooks, frame_state, skip=skip + 1)
        if (
            self._on_recompile_limit is not None
            and result.frame_exec_strategy.cur_action == FrameAction.RUN_ONLY
        ):
            self._on_recompile_limit()
        return result


def _optimize_isolated(
    backend: _PrecompileBackend,
    package: CompilePackage,
    *,
    recompile_limit: int,
    dynamic: bool | None,
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]] | None,
    on_recompile_limit: Callable[[], None],
) -> OptimizeContext:
    from .eval_frame import OptimizeContext

    optimize_ctx = torch._dynamo.optimize(
        backend,
        package=package,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        guard_filter_fn=guard_filter_fn,
        isolate_recompiles=True,
    )
    if not isinstance(optimize_ctx, OptimizeContext):
        raise PackageError("torch.compiler.precompile requires Dynamo to be enabled")
    callback = optimize_ctx.callback
    if not isinstance(callback, CatchErrorsWrapper):
        raise AssertionError(f"expected a CatchErrorsWrapper, got {type(callback)}")
    converter = callback._torchdynamo_orig_backend
    if not isinstance(converter, ConvertFrame):
        raise AssertionError(f"expected a ConvertFrame, got {type(converter)}")
    # Swap the wrapper's converter for the package's own, which patches
    # allow_empty_graphs around every frame it compiles, refuses a DDPOptimizer
    # frame by name and reports a recompile-limit hit. Rebuilt from the converter
    # optimize() made so the backend, hooks and limit stay the ones it derived.
    optimize_ctx.callback = CatchErrorsWrapper(
        _ReportLimitConvertFrame(
            converter._torchdynamo_orig_backend,
            converter._hooks,
            package=package,
            recompile_limit=converter._recompile_limit,
            on_recompile_limit=on_recompile_limit,
        ),
        callback.hooks,
    )
    return optimize_ctx


def _warn_risky_drops(risky: Sequence[tuple[str, str]]) -> None:
    """Report accepted risky drops, shape-bearing ones first.

    Ordering by type only puts the candidates where they can be seen; whether a
    guarded VALUE can differ at serve time is not something the type answers.
    """
    by_type: dict[str, list[str]] = {}
    for guard_type, name in sorted(risky):
        by_type.setdefault(guard_type, []).append(name)

    # Grouped rather than a flat cut, and capped PER TYPE: a flat list is
    # dominated by whichever type happens to be most numerous, which can bury a
    # lone SEQUENCE_LENGTH behind a crowd of CONSTANT_MATCH and CLOSURE_MATCH.
    def render(types: list[str], per_type: int) -> str:
        parts = []
        for t in types:
            names = by_type[t]
            shown = ", ".join(names[:per_type])
            more = f", +{len(names) - per_type} more" if len(names) > per_type else ""
            parts.append(f"{t} x{len(names)}: {shown}{more}")
        return "; ".join(parts)

    shape_types = [t for t in by_type if t in _SHAPE_BEARING_GUARD_TYPES]
    other_types = [t for t in by_type if t not in _SHAPE_BEARING_GUARD_TYPES]
    # Says "could" rather than "can", and points at the distinction that
    # actually decides it. This is a classification by guard TYPE, and the
    # question a reader has is whether the guarded VALUE can differ at serve
    # time -- which the type does not answer. The first four this ordering
    # surfaced on a real model were all reached through a class or function
    # definition (__mro__ walks to __defaults__, __code__) and were therefore
    # compile-time constants that no batch could change. Distinguishing those
    # properly needs the structured source, not the name.
    shape_report = (
        f" COULD BEAR ON SHAPE ({sum(len(by_type[t]) for t in shape_types)}), "
        f"unlike the rest, so check these first -- but check whether each one "
        f"can actually differ at serve time: a guard reached through a class or "
        f"function definition (an __mro__ walk, __defaults__, __code__) is a "
        f"compile-time constant and cannot: {render(shape_types, 3)}."
        if shape_types
        else ""
    )
    log.warning(
        "precompile: %d dropped guard(s) can affect dispatch, so nothing checks "
        "them at load.%s The remaining dropped slots to audit: %s. "
        "summary().risky_dropped_guards has all of them; this warning appears "
        "only because require_no_risky_drops=False explicitly accepted them.",
        len(risky),
        shape_report,
        render(other_types, 2) or "none",
    )


def _missing_backends_message(
    total: int, missing: Sequence[object], backend: str = "inductor"
) -> str:
    """Why some compiled subgraphs never reached the artifact.

    Reports the recorded/total split rather than asserting nothing was
    recorded: a single missing id is fatal here, and saying so as "never
    recorded" reads as total failure when most of the capture succeeded.
    """
    shown = ", ".join(str(b) for b in missing[:8])
    if len(missing) > 8:
        shown += f", ... ({len(missing) - 8} more)"
    if backend not in ("inductor", "eager"):
        # A session takes any backend Dynamo can resolve, but only these two
        # leave something a served artifact can run: "eager" keeps the fx
        # graphs and "inductor" bundles compiled code. Anything else captures
        # cleanly and records nothing, so say so here rather than let it read
        # as a defect in the model. aot_eager is the one people reach for,
        # since it is how you isolate AOTAutograd.
        return (
            f"Precompilation recorded {total - len(missing)} of {total} "
            f"compiled backend(s) because backend={backend!r} does not produce "
            f"anything serializable; precompile can record only 'inductor' or "
            f"'eager'. To isolate AOTAutograd without inductor, use plain "
            f"torch.compile(backend='aot_eager') -- that needs no precompile."
        )
    from torch._dynamo.utils import counters

    # Rendering re-enters AOTAutograd outside the pinned bypass_autograd_cache_key
    # config and bypasses there routinely, so this is a diagnostic, not a diagnosis.
    bypasses = counters["aot_autograd"].get("autograd_cache_bypass", 0)
    bypass_note = (
        f" It bypassed {bypasses} time(s) here, which rendering does for any "
        "graph the cache cannot key, and which does not by itself explain a gap."
        if bypasses
        else ""
    )
    return (
        f"Precompilation recorded {total - len(missing)} of {total} compiled "
        f"backend(s), so {len(missing)} graph(s) would reach the artifact with "
        f"no code behind them: {shown}. Capture pins functorch's "
        "bypass_autograd_cache_key, so AOTAutograd keys every graph it lowers "
        "and no longer declines to record one it cannot address."
        + bypass_note
        + " A gap therefore means a graph whose backward never compiled, which "
        "is a forward-only capture with grad enabled. Pass training=True to "
        "lower the backward eagerly (the joint trace synthesizes tangents, so "
        "no loss is needed), capture under torch.no_grad() / "
        "torch.inference_mode() for an inference artifact, or run .backward() "
        "inside the capture block. Re-run with "
        "TORCH_LOGS=+torch._functorch._aot_autograd to see each graph as it "
        "lowers."
    )


class PrecompileSession:
    """
    A caller-driven capture in progress. Enter as a context manager to get the
    callable to exercise and invoke it with real inputs inside the block. The
    compiled region stays alive for the whole block, so every call reuses the
    variants the earlier ones produced.
    """

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        backend: str = "inductor",
        guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
        | None = None,
        recompile_limit: int = 256,
        dynamic: bool | None = None,
        training: bool = False,
        keep_graphs: bool = False,
        invariants: str | None = None,
    ) -> None:
        self._fn = fn
        self._backend = backend
        # A training capture traces with grad on and lowers the backward
        # eagerly, so the artifact carries AOTAutograd's CompiledFunction and
        # calling .backward() on a served output runs precompiled code.
        self._training = training
        # Retaining a graph deepcopies it for the session; the guard probe and
        # captures whose graphs are never rendered leave it off.
        self._keep_graphs = keep_graphs
        self._invariants_path = invariants
        self._backend_obj: _PrecompileBackend | None = None
        # One record per guard-filter call, whether or not that compile went
        # on to land anything; the report covers the ones that did. See
        # _RecordedCompile.
        self._compiles: list[_RecordedCompile] = []
        self._capture_errors: list[str] = []
        self._recorded_exception_keys: set[tuple[type[BaseException], str]] = set()
        self._guard_filter_fn = self._recording_filter(
            _compose_with_default(guard_filter_fn)
        )
        self._recompile_limit = recompile_limit
        self._dynamic = dynamic
        self._entry_fn = _entry_fn_of(fn)
        # The guard filter rides on the optimize context rather than the
        # package, so it applies to the live guards as well as the serialized
        # ones, exactly as caching_precompile does today.
        self._package = CompilePackage(self._entry_fn)
        self._backend_artifacts: dict[_BackendId, Any] = {}
        # The ids whose compiled callable _take_backend_artifacts decided to
        # leave on the package for a render to serialize off it; _release drops
        # every other copy the package holds.
        self._kept_backend_ids: set[_BackendId] = set()
        self._entered = False
        self._compiled: Callable[..., object] | None = None
        self._state = threading.Condition()
        self._active_calls = 0
        # thread id -> its in-flight calls, so summary() can tell "another
        # thread is compiling, wait for it" from "the caller is inside the
        # block's own callable", which it could only wait on forever.
        self._active_call_threads: dict[int, int] = {}
        self._closing = False
        self._finished = False

    def _take_backend_artifacts(self) -> None:
        """Collect what each backend id the entry names actually filed.

        Single-threaded by contract: this reads the package's cache entry and
        pops out of the process-global staging area, both of which a compile
        still running is using (see _drain_then_close). A caller must either
        hold self._state or be the thread that has drained the session, which is
        what _close is; a render collecting mid-block runs on the same terms.
        """
        from torch._dynamo.output_graph import noop_graph_call
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
            reduces_to_graph_source,
        )

        backend_ids = self._package.cache_entry().backend_ids
        unfiled: list[_BackendId] = []
        kept: set[_BackendId] = set()
        for backend_id in backend_ids:
            if backend_id in self._backend_artifacts:
                # Already collected. A render can collect mid-block and again at
                # exit, and take_artifact hands an artifact out once, so a second
                # pass must not read an id it already holds as one that filed
                # nothing.
                continue
            artifact = PrecompileContext.take_artifact(backend_id)
            compiled = self._package.cached_backends.get(backend_id)
            if artifact is not None:
                self._backend_artifacts[backend_id] = artifact
            elif compiled is noop_graph_call:
                # output_graph short-circuits an empty graph to noop_graph_call
                # without filing anything under its id, which the bytecode still
                # names. Record the no-op so the served frame dispatches to it
                # rather than running eager. Done here rather than at
                # render time because _release drops the package's copy of every
                # id whose artifact this pass took.
                self._backend_artifacts[backend_id] = EagerCacheArtifact(
                    key=backend_id, content=noop_graph_call
                )
            elif compiled is None:
                # An id the bytecode names that no compile ever reached: a resume
                # frame the capture never exercised has nothing to keep and
                # nothing to file.
                continue
            elif reduces_to_graph_source(compiled):
                # Filed nothing, but what it left on the package is the shape
                # EagerCacheArtifact.__reduce__ carries, so a render serializes
                # it straight off the package. Decided from the object rather
                # than from the backend's name, which does not determine the
                # shape: eager hands back a bound GraphModule.forward only while
                # force_autograd_cache is off, and a caller's own backend may
                # hand one back too.
                kept.add(backend_id)
            else:
                unfiled.append(backend_id)
        # Recomputed per pass rather than accumulated: an id that filed an
        # artifact after an earlier pass left its callable here has no further
        # use for a second copy on the package.
        self._kept_backend_ids = kept
        if unfiled:
            # Only what was observed: these ids filed nothing and what they left
            # behind is not the shape a render serializes, so the capture is
            # short those graphs. Why they filed nothing is not checked here, so
            # the causes stay a list of possibilities. Deduplicated on the
            # condition rather than on the wording, which names the ids and so
            # differs between a mid-block collection and the one at exit.
            self._record_capture_error(
                PackageError(
                    "the capture recorded no artifact for backend id(s) "
                    f"{', '.join(unfiled)}, and the callable backend "
                    f"{self._backend!r} left on the package for each is not a "
                    "bound GraphModule.forward, the one shape a render can "
                    "serialize off the package; the usual causes are a "
                    "grad-enabled capture without training=True, which leaves "
                    "the backward lowering deferred past the end of the "
                    "capture, caches turned off through force_disable_caches, "
                    "and a backend that never files its compiled code -- any of "
                    "them may apply, since nothing here diagnoses which"
                ),
                dedup_on="backend ids that filed no artifact",
            )

    def _record_recompile_limit(self) -> None:
        # Deduplicated like every other capture error, so a cap hit once per
        # frame and per call records one entry naming both limits rather than
        # one per refused variant.
        self._record_capture_error(
            PackageError(
                "the capture hit a recompile limit, so it holds fewer variants "
                "than the calls exercised: Dynamo caps a code object at "
                f"recompile_limit={self._recompile_limit} and at "
                "torch._dynamo.config.accumulated_recompile_limit"
                f"={torch._dynamo.config.accumulated_recompile_limit} counted "
                "across every isolated region on it, whichever it reaches "
                "first, and runs the frame eagerly from then on"
            )
        )

    def _record_capture_error(
        self, error: BaseException, *, dedup_on: str | None = None
    ) -> None:
        """Record one capture error, at most once per kind.

        dedup_on names the stable part of a message whose text varies with how
        far the capture has got, so that one condition observed twice records one
        entry rather than one per wording.
        """
        message = str(error)
        key = (type(error), dedup_on if dedup_on is not None else message)
        # Under _state: the check-then-add IS the once-only invariant, so two
        # concurrent calls raising the same exception must not both append. No
        # caller holds _state when it gets here, and a Condition's default lock
        # is an RLock, so a later re-entrant caller would not deadlock either.
        with self._state:
            if key in self._recorded_exception_keys:
                return
            self._recorded_exception_keys.add(key)
            self._capture_errors.append(f"{type(error).__name__}: {message}")

    def _release(self) -> None:
        # The compiled variants stay in the entry's ordinary Dynamo cache, as
        # they would after torch.compile; clearing them per capture needs the
        # region-scoped cache entries that are not part of this build. What goes
        # is the package's own copies, all but the ids _take_backend_artifacts
        # decided to leave behind: it took an artifact for the rest or recorded a
        # capture error naming them, so a second copy here serves no render.
        for backend_id in list(self._package.cached_backends):
            if backend_id not in self._kept_backend_ids:
                del self._package.cached_backends[backend_id]

    def _call(self, *args: object, **kwargs: object) -> object:
        with self._state:
            if self._compiled is None or self._closing:
                raise RuntimeError("PrecompileSession is not active")
            compiled = self._compiled
            self._active_calls += 1
            caller = threading.get_ident()
            self._active_call_threads[caller] = (
                self._active_call_threads.get(caller, 0) + 1
            )
        try:
            with _capture_config(self._training):
                result = compiled(*args, **kwargs)
        except BaseException as e:
            self._record_capture_error(e)
            raise
        finally:
            with self._state:
                self._active_calls -= 1
                if self._active_call_threads[caller] > 1:
                    self._active_call_threads[caller] -= 1
                else:
                    del self._active_call_threads[caller]
                if self._active_calls == 0:
                    self._state.notify_all()
        return result

    def _drain_then_close(self) -> None:
        """Wait for the calls in flight, then close the session either way.

        The drain has to come first: a call can still be compiling against a
        borrowed cache entry, and collecting the artifacts mutates state that
        compile reads. The close sits in the drain's finally so an interrupt
        raised out of wait() (a KeyboardInterrupt) still closes the session
        instead of leaving cap() callable past the block with the optimize
        context alive to process exit -- at the price of abandoning the calls
        still running, which is why _close then collects nothing for them.
        """
        drained = False
        try:
            with self._state:
                self._closing = True
                while self._active_calls:
                    self._state.wait()
            drained = True
        finally:
            self._close(collect=drained)

    def _close(self, *, collect: bool) -> None:
        """Close the session, and collect its artifacts if the drain completed.

        Publishing the flags is all that is safe with calls still in flight,
        which is what an interrupted drain leaves behind: taking the artifacts
        reads the package's cache entry and _release clears its backends, both of
        which a compile still running is using. So an interrupt abandons those
        calls -- whatever they go on to file is not collected, and the artifact
        this capture renders does not hold their variants -- while the session
        still closes rather than staying open to process exit.
        """
        with self._state:
            # _closing marks a drain in progress, and by here it is over.
            self._closing = False
            self._entered = False
            # The optimize context lives on the compiled callable, so dropping
            # the callable is what releases it.
            self._compiled = None
            self._finished = True
            if collect:
                # The drain is over, so no compile can still land guarded code
                # and a record that is unconfirmed by now never will be. What a
                # later summary() reports is unchanged -- _confirmed_compiles
                # keeps one record per key, so it is idempotent over its own
                # output -- and a bypassed compile stops holding its entry and
                # its facts for the session's life.
                self._compiles = self._confirmed_compiles()
            self._state.notify_all()
        if not collect:
            return
        try:
            self._take_backend_artifacts()
        except BaseException as teardown:
            # Recorded, never raised: a teardown failure must not replace the
            # exception the caller's block is already propagating, and the
            # capture errors are what a render gates on.
            self._record_capture_error(teardown)
        finally:
            self._release()

    def __enter__(self) -> Callable[..., object]:
        # Under _state: the not-finished/not-entered check and the set that
        # follows are one decision, which two concurrent entries must not both
        # pass, and _call and _close read _entered and _compiled under it too.
        with self._state:
            if self._finished:
                raise RuntimeError("PrecompileSession cannot be re-entered")
            if self._entered:
                raise PackageError(
                    "PrecompileSession is already active: a session runs one capture "
                    "block at a time, so serialize concurrent entries."
                )
            # The grad-mode/config patch is per call, in _call, not block-level:
            # user code between calls (optimizer.step, data loading) must run in
            # the ambient mode, not the capture's.
            self._entered = True
        try:
            self._backend_obj = _PrecompileBackend(self._backend, self._keep_graphs)
            optimize_ctx = _optimize_isolated(
                self._backend_obj,
                self._package,
                recompile_limit=self._recompile_limit,
                dynamic=self._dynamic,
                guard_filter_fn=self._guard_filter_fn,
                on_recompile_limit=self._record_recompile_limit,
            )
            compiled = optimize_ctx(self._fn)
            with self._state:
                self._compiled = compiled
        except BaseException as e:
            self._record_capture_error(e)
            # A __enter__ that raises never gets its __exit__, so without this
            # the session is wedged: the block reads as still open. The same
            # drain-then-close __exit__ runs, on the same terms.
            self._drain_then_close()
            raise
        return self._call

    def __exit__(self, *exc: object) -> None:
        if isinstance(exc[1], BaseException):
            self._record_capture_error(exc[1])
        self._drain_then_close()
        if self._invariants_path is None:
            return
        if exc[0] is None:
            try:
                self.write_invariants(self._invariants_path)
            except Exception as error:
                # The teardown policy above, for the same reason: a diagnostic
                # file that could not be written must not turn a capture that
                # succeeded into a raising block. It is reported where the other
                # teardown failures are, so require_complete still sees it.
                self._record_capture_error(error)
        else:
            # A partial capture's report reads exactly like a complete one, so
            # it is not written; say why rather than leaving the user looking
            # for a file that never appeared.
            log.warning(
                "precompile: the capture block raised %s, so no invariants "
                "report was written to %s. Call write_invariants() for the "
                "partial one.",
                getattr(exc[0], "__name__", exc[0]),
                self._invariants_path,
            )

    def _recording_filter(
        self,
        inner: Callable[
            [Sequence[GuardFilterEntry]], tuple[Sequence[bool], Sequence[bool]]
        ],
    ) -> Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]:
        """
        Remember which guard types were discarded. A dropped guard does not
        fail at serving time, it silently widens what a graph is reused for, so
        the set has to be inspectable rather than invisible.
        """

        # Every distinct fact, interned under the lock below so the session
        # grows with facts rather than with variants.
        pool: dict[_GuardFact, _GuardFact] = {}

        def filter_fn(entries: Sequence[GuardFilterEntry]) -> Sequence[bool]:
            # A custom filter composes with the default, so the identity drops
            # the default makes anyway are judged as they always are; only a
            # drop the custom filter ADDED is risky by construction, because
            # nothing here can say what the caller gave up. The composition
            # hands back the default's own verdicts, so it runs once per compile.
            decisions, default_kept = inner(entries)
            global_state_kept = any(
                keep and e.guard_type == "GLOBAL_STATE"
                for keep, e in zip(decisions, entries)
            )
            facts: set[_GuardFact] = set()
            undetermined: set[_GuardFact] = set()
            kept_slots: set[tuple[str, str]] = set()
            dropped_slots: set[tuple[str, str]] = set()
            risky_slots: set[tuple[str, str]] = set()
            # Merged under the one lock acquisition below rather than taken per
            # entry: this runs on the compiling thread for every guard of every
            # compilation.
            dropped_code: dict[tuple[str, str], str] = {}
            for keep, by_default, entry in zip(decisions, default_kept, entries):
                # Normalized where the slot is RECORDED, not where it is read: a
                # Dynamo per-process counter (__builtins_dict___14) otherwise
                # makes one logical slot appear once per compilation, under names
                # that change every run, and the slot lists, the risky subset and
                # the rendered code end up spelling it three different ways.
                # _mask_keys as well because a source name interpolates the
                # data key it reads (cfg['/home/me/w.pt']), and this name is
                # what every slot list and every fact spells the source as.
                slot = (entry.guard_type, _mask_keys(_normalize(entry.name)))
                # A no-op type's marker is not what makes its check: GLOBAL_STATE's
                # leaf is, so the leaf's verdict decides, not the marker's.
                noop = entry.guard_type in _NOOP_GUARD_TYPES
                # A precondition nothing checks however the filter voted:
                # _is_noop_guard_type names the types GuardBuilder emits no check
                # for beyond those markers -- EMPTY_NN_MODULE_HOOKS_DICT under
                # skip_nnmodule_hook_guards, the default -- and FSDP_TRAINING_STATE
                # is state GlobalStateGuard does not snapshot either. Neither
                # dropped nor risky: no filter decision took them away.
                unchecked = entry.guard_type == "FSDP_TRAINING_STATE" or (
                    not noop and _is_noop_guard_type(entry.guard_type)
                )
                enforced = not unchecked and (global_state_kept if noop else keep)
                unmodelled = entry.guard_type in _UNMODELLED_GUARD_TYPES
                # Rendered once per entry: the fact and the dropped-code line
                # want the same rendering, and the report below prints the check
                # of an enforced guard as well as of a dropped one.
                code = _render_code(entry.orig_guard.code_list)
                if enforced:
                    kept_slots.add(slot)
                elif unchecked:
                    # NEITHER list, which is what the two of them mean: the
                    # filter's verdict is what they report, and no verdict took
                    # this slot away -- nothing ever checked it. GuardFact.enforced
                    # below is where the report says so. See PrecompileSummary.
                    pass
                else:
                    dropped_slots.add(slot)
                    rendered = " ; ".join(code)
                    if rendered:
                        # One rendering however many variants dropped the slot, so
                        # a check that embeds its value tells the form of the check
                        # rather than every value the slot took: the first one
                        # here, and summary() takes the first across compiles.
                        dropped_code.setdefault(slot, rendered)
                    # Risky here means a drop the default filter would not have
                    # made, or one whose fact differed between variants, which
                    # summary() decides from the fact sets recorded below.
                    if by_default:
                        risky_slots.add(slot)
                # Never compared, so never claimed to hold: see
                # _UNMODELLED_GUARD_TYPES.
                (undetermined if unmodelled else facts).add(
                    _GuardFact(
                        guard_type=entry.guard_type,
                        source=slot[1],
                        code=code,
                        value=_value_fingerprint(entry),
                        enforced=enforced,
                    )
                )
            # One filter call is one compilation, and only the package knows
            # which frame is being compiled. Without it there is no frame to
            # attribute the facts to, and no entry to tell later whether this
            # compile landed, so they go unrecorded rather than into a made-up
            # frame or into a list that claims an artifact enforces them.
            compiling = self._package._current_entry
            if compiling is None:
                return decisions
            frame_code = compiling.python_code
            key = (
                frame_code.co_name,
                frame_code.co_filename,
                frame_code.co_firstlineno,
            )
            # Recorded under the lock a reader takes, in one step, because this
            # runs on whatever thread is compiling. Recorded per FRAME: entry.name
            # is frame-local, so the same slot name in two frames is two slots,
            # and one fact set for both would read a rebind that never happened.
            # Nothing is merged into the report here -- whether this compile
            # enforces anything is not decided yet, see _RecordedCompile.
            with self._state:
                self._compiles.append(
                    _RecordedCompile(
                        frame=key,
                        entry=compiling,
                        guarded_codes_before=len(compiling.guarded_codes),
                        kept=frozenset(kept_slots),
                        dropped=frozenset(dropped_slots),
                        risky=frozenset(risky_slots),
                        dropped_code=tuple(dropped_code.items()),
                        # One object per distinct fact, so a recompiled frame
                        # repeating nearly all of its guards costs facts rather
                        # than variants.
                        facts=frozenset(pool.setdefault(f, f) for f in facts),
                        undetermined=frozenset(
                            pool.setdefault(f, f) for f in undetermined
                        ),
                    )
                )
            return decisions

        return filter_fn

    def _confirmed_compiles(self) -> list[_RecordedCompile]:
        """The recorded compiles whose guarded code reached the package entry.

        A bypassed compile is dropped whole: its kept guards enforce nothing, its
        dropped guards widened nothing, and its facts are not a variant of the
        artifact. Keyed by (entry, guarded_codes_before) with the last record
        winning, because a bypass and a later compile of the same frame both see
        the same list length and only the later one is evidence that guarded code
        landed. Call under _state.
        """
        latest: dict[tuple[int, int], _RecordedCompile] = {}
        for compiled in self._compiles:
            # The entry is unhashable (a mutable dataclass), and identity is what
            # is wanted anyway: one live object per frame of this capture.
            latest[(id(compiled.entry), compiled.guarded_codes_before)] = compiled
        return [
            compiled
            for compiled in latest.values()
            if len(compiled.entry.guarded_codes) > compiled.guarded_codes_before
        ]

    def _confirmed_facts(self) -> dict[_FrameKey, list[frozenset[_GuardFact]]]:
        """frame -> one fact set per confirmed compile of it.

        What a reader of the recorded guards consumes, the report included: the
        variants the artifact actually carries, in the order they compiled. Call
        under _state.
        """
        confirmed: dict[_FrameKey, list[frozenset[_GuardFact]]] = {}
        for compiled in self._confirmed_compiles():
            confirmed.setdefault(compiled.frame, []).append(compiled.facts)
        return confirmed

    def _confirmed_undetermined(self) -> dict[_FrameKey, set[_GuardFact]]:
        """frame -> the unmodelled facts of its confirmed compiles, merged.

        Merged rather than kept per compile because nothing compares them: the
        report lists them as undetermined, once per frame. Call under _state.
        """
        confirmed: dict[_FrameKey, set[_GuardFact]] = {}
        for compiled in self._confirmed_compiles():
            confirmed.setdefault(compiled.frame, set()).update(compiled.undetermined)
        return confirmed

    def _value_varying_slots(
        self, key: _FrameKey | None = None
    ) -> set[tuple[str, str]]:
        """Dropped slots whose recorded fact differed between one frame's variants.

        Narrower than _varying_guard_slots, deliberately: that one reports what
        tells the variants apart at all, kept guards and present-in-some-variants
        included, and neither is evidence that a DROP widened the artifact. It
        also cannot be read over every recorded fact set: it compares a slot's
        rendered check and value, so it needs sets whose ENFORCED facts carry
        that rendering, and where a producer skipped it -- as the recorder does
        for a guard nothing here reports the check of -- every enforced fact of
        a slot compares equal there and the slot reads invariant. Only the
        variants that dropped the slot are compared, and what is compared is a
        whole variant's set of facts for it, as there, so one variant holding
        several (a HASATTR per attribute name on one parent source) is not
        variation. Over every frame when key is None, which is what summary()
        reports: a slot that varied in the frame that owns it varied. Compares
        the confirmed fact sets, so a compile that bypassed cannot make a slot
        vary. Call under _state.
        """
        varying: set[tuple[str, str]] = set()
        for frame, variants in self._confirmed_facts().items():
            if key not in (None, frame):
                continue
            seen: dict[tuple[str, str], set[frozenset[_SlotCheck]]] = {}
            for facts in variants:
                here: dict[tuple[str, str], set[_SlotCheck]] = {}
                for fact in facts:
                    if fact.enforced:
                        continue
                    slot = (fact.guard_type, fact.source)
                    here.setdefault(slot, set()).add((fact.code, fact.value))
                for slot, rendered in here.items():
                    seen.setdefault(slot, set()).add(frozenset(rendered))
            varying |= {
                slot for slot, per_variant in seen.items() if len(per_variant) > 1
            }
        return varying

    def invariants(self) -> tuple[FrameInvariants, ...]:
        """
        Per frame, the guards that held in EVERY compiled variant of it.

        Intersection is per frame rather than global because guards from
        different frames are not comparable: the entry frame guards its
        arguments, a resume frame guards whatever crossed the graph break, so a
        global intersection would be empty for any model that breaks.

        A frame compiled once reports everything as invariant, which is true but
        uninformative -- exercise more than one variant for the diff to mean
        anything.

        A variant is a compilation whose guards are IN the artifact: a compile
        the serializer bypassed contributes none of them, so it is not counted
        here even though it ran the guard filter.

        GLOBAL_STATE, TORCH_FUNCTION_STATE and FSDP_TRAINING_STATE carry no
        value of their own, so nothing here can say whether two variants agreed
        on, say, autocast. They are listed as UNDETERMINED rather than compared
        -- see _UNMODELLED_GUARD_TYPES, and note that calling them equal is
        precisely how the report would assert a precondition that does not
        hold. GRAD_MODE and DETERMINISTIC_ALGORITHMS are compared like any other
        guard, on the process state GlobalStateGuard snapshots for them.
        """
        # Snapshotted under the lock because a compile on another thread records
        # into these dicts; the facts themselves are immutable, so the rendering
        # below needs no lock.
        with self._state:
            undetermined_facts = self._confirmed_undetermined()
            recorded = [
                (
                    key,
                    list(sets),
                    undetermined_facts.get(key, set()),
                    self._value_varying_slots(key),
                )
                for key, sets in self._confirmed_facts().items()
            ]
        out = []
        for key, sets, undetermined, varying in sorted(recorded, key=lambda i: i[0]):
            name, filename, lineno = key
            shared = frozenset.intersection(*sets) if sets else frozenset()
            # A dropped slot whose VALUE varied did not hold, however its
            # renderings compare: the report masks the id a check embeds, so two
            # distinct objects render one fact. See _value_fingerprint.
            shared -= {f for f in shared if (f.guard_type, f.source) in varying}
            everything: set[_GuardFact] = set()
            for one in sets:
                everything |= one
            out.append(
                FrameInvariants(
                    frame=name,
                    filename=filename,
                    lineno=lineno,
                    variants=len(sets),
                    invariant=tuple(sorted(shared, key=_fact_key)),
                    varying=tuple(sorted(everything - shared, key=_fact_key)),
                    undetermined=tuple(sorted(undetermined, key=_fact_key)),
                )
            )
        return tuple(out)

    def write_invariants(self, path: str, /) -> None:
        """
        Write :meth:`invariants` to ``path`` in human-readable form.

        ``path`` is a FILE, written exactly as given, with parent directories
        created -- ``snapshots/invariants.txt`` is a text file.

        Output is stable across runs of the same capture: object ids and
        Dynamo's per-process counters are normalized away, so the file can be
        committed and diffed to see what a model change did to its guards.
        """
        frames = self.invariants()
        target = getattr(self._fn, "__qualname__", None) or type(self._fn).__qualname__
        lines = [
            f"# precompile invariants for {target}",
            "#",
            "# Conditions that held in EVERY compiled variant of a frame. A call",
            "# violating one cannot be served by any graph in this artifact, so",
            "# these are the preconditions the artifact is only valid under.",
            "# 'varies' lists what differed between variants -- those are what",
            "# distinguish one compiled graph from another, not preconditions.",
            "# 'unknown' lists guards whose check this report cannot model, so it",
            "# cannot say whether they held across variants. Treat them as",
            "# neither: they may or may not be preconditions.",
            "#",
            "# enforced = the guard is serialized and rechecked when the artifact",
            "#            is loaded.",
            "# dropped  = it was not serialized, so it is a precondition",
            "#            NOTHING checks at serving time. See",
            "#            PrecompileSummary.dropped_guards.",
            "#",
            f"# {len(frames)} frame(s), "
            f"{sum(f.variants for f in frames)} compilation(s)",
        ]
        if any(f.variants < 2 for f in frames):
            lines.append(
                "# NOTE: some frames were compiled once, so their invariants are"
                " just every guard. Exercise more variants for a real diff."
            )
        for f in frames:
            where = f"{os.path.basename(f.filename)}:{f.lineno}"
            lines.append("")
            lines.append(
                f"frame {f.frame} ({where})  {f.variants} variant(s), "
                f"{len(f.invariant)} invariant, {len(f.varying)} varying, "
                f"{len(f.undetermined)} undetermined"
            )
            if not f.invariant:
                lines.append("  invariant: (none)")
            for fact in f.invariant:
                lines.append(f"  invariant {_render_fact(fact)}")
            for fact in f.varying:
                lines.append(f"  varies    {_render_fact(fact)}")
            for fact in f.undetermined:
                lines.append(f"  unknown   {_render_fact(fact)}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # UTF-8 explicitly: the report renders user identifiers (module, class and
        # parameter names) and the ambient locale can be ASCII in a container, where
        # a non-ASCII name would raise UnicodeEncodeError and leave a truncated file
        # behind -- from the one call that exists to explain the artifact.
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        log.info(
            "precompile: wrote invariants for %d frame(s) to %s", len(frames), path
        )

    def summary(self) -> PrecompileSummary:
        """The report for the capture so far.

        Callable while the block is open, and it WAITS for the calls in flight:
        reading the package's cache entry needs no compile holding it, which is
        what ``CompilePackage.validate`` refuses, so a concurrent capture call
        would otherwise make this raise ``AssertionError`` on the reader's
        thread. A call from inside the block's own callable cannot be waited for
        -- it would be waiting on itself -- so that raises instead of hanging.

        ``truncated`` is left empty here because the recompile-limit bookkeeping
        that fills it is not part of this build, so ``complete`` cannot see a
        frame that hit the limit: read it as "complete apart from that".
        """
        # Aggregated under the lock because a compile on another thread records
        # into _compiles, and a reader wants one consistent view. Only the
        # compiles whose guarded code landed are counted, so a bypassed compile
        # is in neither list: see _confirmed_compiles. The slots are already
        # normalized, so every list here spells one slot the same way and
        # risky_dropped_guards really is a subset of dropped_guards.
        with self._state:
            if self._active_call_threads.get(threading.get_ident()):
                raise RuntimeError(
                    "PrecompileSession.summary() cannot be called from inside a "
                    "capture call: it would wait for that call to finish. Call "
                    "it from the block, or after it."
                )
            while self._active_calls:
                self._state.wait()
            dropped: set[tuple[str, str]] = set()
            kept: set[tuple[str, str]] = set()
            risky_by_filter: set[tuple[str, str]] = set()
            dropped_code: dict[tuple[str, str], str] = {}
            for compiled in self._confirmed_compiles():
                kept |= compiled.kept
                dropped |= compiled.dropped
                risky_by_filter |= compiled.risky
                for slot, rendered in compiled.dropped_code:
                    # The FIRST rendering, as
                    # PrecompileSummary.dropped_guard_code documents.
                    dropped_code.setdefault(slot, rendered)
            # Two routes, per risky_dropped_guards: a drop the default filter
            # would not have made, and a drop whose value told the variants
            # apart. Merely being absent from one variant is neither -- that
            # flags every ordinary multi-branch capture.
            # Intersected with dropped because a fact of a slot nothing checks
            # varies like any other, and risky_dropped_guards is a subset of
            # dropped_guards.
            risky = risky_by_filter | (self._value_varying_slots() & dropped)
            capture_errors = list(self._capture_errors)
            guard_sets = self._confirmed_facts()
            # Under the lock the drain above left held, so no call can start
            # between the two and reach a compile while the entry is read.
            entry = self._package.cache_entry()
        return _summarize(
            entry,
            dropped=dropped,
            kept=kept,
            # The invariance policy that would prune a kept guard is not part of
            # this build, and neither is the recompile-limit bookkeeping behind
            # truncated.
            policy_dropped=set(),
            risky=risky,
            truncated=frozenset(),
            capture_errors=capture_errors,
            guard_sets=guard_sets,
            dropped_code=dropped_code,
        )

    def _gated_summary(
        self,
        *,
        require_complete: bool,
        require_no_risky_drops: bool,
        require_no_dropped_guards: bool,
    ) -> PrecompileSummary:
        """Run the coverage and guard gates, or raise saying which one failed.

        Callable mid-block, while the compiled region is still live, on
        :meth:`summary`'s terms: it waits for the calls in flight rather than
        reading the package under a compile, and refuses a read from inside the
        block's own callable.
        """
        summary = self.summary()
        if require_complete and summary.capture_errors:
            raise PackageError(
                "Precompilation is incomplete because capture raised: "
                f"{list(summary.capture_errors)}. Re-run every example successfully, "
                "or pass require_complete=False to save the partial artifact."
            )
        if require_no_dropped_guards and summary.dropped_guards:
            raise PackageError(
                f"Precompilation dropped {len(summary.dropped_guards)} guard(s) that "
                f"were not serialized: {list(summary.dropped_guards)}. Rebinding any "
                f"of those sources between capture and load can silently serve a graph "
                f"traced against the old value. Pass require_no_dropped_guards=False "
                f"only to select the relaxed risky-drop policy."
            )
        if summary.risky_dropped_guards and require_no_risky_drops:
            raise PackageError(
                f"Precompilation dropped guard(s) that can affect dispatch on "
                f"{[n for _, n in summary.risky_dropped_guards]}. Each of those names "
                f"either a configuration-dependent identity slot or a guard discarded "
                f"by a custom filter. Nothing checks it at load time, so a different "
                f"value can silently select the wrong graph instead of recompiling. "
                f"Make the value reachable through a serializable guard, pin both "
                f"machines to the same value, or pass "
                f"require_no_risky_drops=False to accept the risk explicitly."
            )
        elif summary.risky_dropped_guards:
            # The caller explicitly accepted the risk.
            _warn_risky_drops(summary.risky_dropped_guards)
        if require_complete:
            if summary.guarded_codes == 0:
                raise PackageError(
                    "Precompilation captured no compiled code. Capture happens by "
                    "execution, so the callable must actually be run inside the "
                    "capture block. A call Dynamo could not turn into guarded code "
                    "is reported separately as an uncovered frame."
                )
            if summary.backend_graphs == 0:
                raise PackageError(
                    "Precompilation compiled no graph: every captured frame was "
                    "empty, so the artifact carries no compiled compute. This is "
                    "what a callable whose whole body sits behind "
                    "torch._dynamo.disable looks like. Pass require_complete=False "
                    "to write the guards-only artifact anyway."
                )
            if summary.uncovered_frames:
                raise PackageError(
                    f"Precompilation exercised frame(s) that produced NO guarded code "
                    f"at all: {list(summary.uncovered_frames)}. Those paths are absent "
                    f"from the artifact, and such a frame is skipped at install and runs "
                    f"eager, so serving() cannot report that gap. This is expected for a "
                    f"frame that only dispatches to covered submodules; it also looks "
                    f"exactly like a frame Dynamo gave up on (check "
                    f"TORCH_LOGS=graph_breaks for gb0124). Pass require_complete=False "
                    f"once you have confirmed which."
                )
            if summary.bypassed:
                raise PackageError(
                    f"Precompilation is incomplete: {len(summary.bypassed)} frame(s) "
                    f"were bypassed and will serve nothing: {list(summary.bypassed)}. "
                    f"This usually means their guards could not be serialized. Pass "
                    f"require_complete=False to accept a partial artifact."
                )
        return summary

    def rendered_backends(
        self, backend_ids: Sequence[str]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Compiled subgraphs as READABLE source, and the reason each of the rest
        stayed pickled, both keyed by backend id.

        The pickled bundle is the fallback, not the goal: a subgraph is Inductor
        output, which has a source form (the make_fx tracer emits exactly this),
        unlike the guard trees and transformed bytecode beside it. Anything that
        fails to render -- an effectful op, a graph with no compute, a training
        shape the composer refuses -- stays pickled, and its reason is warned
        here and written into the artifact header so the fallback is visible.

        Rendering re-runs AOTAutograd + Inductor on the retained graph, so it is
        a second lowering, paid once per subgraph that reaches the artifact.
        """
        from torch._functorch import aot_autograd

        if self._backend_obj is None or self._backend == "eager":
            return {}, {}
        rendered: dict[str, str] = {}
        refused: dict[str, str] = {}
        for backend_id in backend_ids:
            held = self._backend_obj.graphs.get(str(backend_id))
            if held is None:
                continue
            gm, fakes = held
            try:
                # grad_enabled is what makes AOTAutograd emit the joint
                # forward+backward for a training capture; without it the
                # backward is silently absent and the served output loses its
                # grad_fn.
                source, _ = aot_autograd.compile_to_python(gm, fakes)
            except Exception as e:
                reason = " ".join(f"{type(e).__name__}: {e}".split())
                log.warning(
                    "precompile: subgraph %s stays pickled, not rendered as source: %s",
                    backend_id,
                    reason,
                )
                refused[str(backend_id)] = reason
                continue
            rendered[str(backend_id)] = source
        from torch._functorch._aot_autograd.to_standalone_python import (
            namespace_module_names,
        )

        keys = list(rendered)
        namespaced = namespace_module_names([rendered[k] for k in keys])
        return dict(zip(keys, namespaced)), refused

    def snapshot_artifact(
        self,
        *,
        require_complete: bool = True,
        require_no_risky_drops: bool = True,
        require_no_dropped_guards: bool = False,
    ) -> tuple[str, bytes]:
        """Render everything captured SO FAR, leaving this session able to capture more.

        The render reads the package's records without consuming them, which is
        what lets ``save()`` be called repeatedly within one capture block.
        """
        summary = self._gated_summary(
            require_complete=require_complete,
            require_no_risky_drops=require_no_risky_drops,
            require_no_dropped_guards=require_no_dropped_guards,
        )
        from torch._precompile import _build_multigraph_artifact

        backends = self._collect_backends()
        entry = self._package.cache_entry()
        rendered, refused = self.rendered_backends(list(backends))
        return _build_multigraph_artifact(
            entry,
            backends,
            summary,
            self._backend,
            _entry_fn_of(self._fn),
            rendered,
            refused,
        )

    def _collect_backends(self) -> dict[str, Any]:
        """The compiled subgraphs this capture produced, keyed by backend id."""
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        self._take_backend_artifacts()
        collected = dict(self._backend_artifacts)
        if self._backend == "eager":
            # Eager "backends" are fx graphs with no compiled artifact of their
            # own, so they have to be gathered explicitly.
            for backend_id, backend in self._package.cached_backends.items():
                collected[backend_id] = EagerCacheArtifact(
                    key=backend_id, content=backend
                )
        entry = self._package.cache_entry()
        for backend_id in entry.backend_ids:
            if backend_id not in collected:
                artifact = PrecompileContext.take_artifact(backend_id)
                if artifact is not None:
                    collected[backend_id] = artifact
        missing = [b for b in entry.backend_ids if b not in collected]
        if missing:
            raise PackageError(
                _missing_backends_message(
                    len(entry.backend_ids), missing, self._backend
                )
            )
        return {str(b): collected[b] for b in entry.backend_ids}


def precompile_capture(
    fn: Callable[..., object],
    *,
    backend: str = "inductor",
    guard_filter_fn: Callable[[Sequence[GuardFilterEntry]], Sequence[bool]]
    | None = None,
    recompile_limit: int = 256,
    dynamic: bool | None = None,
    training: bool = False,
    keep_graphs: bool = False,
    invariants: str | None = None,
) -> PrecompileSession:
    r"""Begin capturing ``fn`` into a multi-graph artifact.

    ``recompile_limit`` raises Dynamo's usual 8 because a precompile
    deliberately wants one compiled variant per condition, whereas the normal
    limit exists to catch runaway recompilation. Nothing raises the ambient
    ``accumulated_recompile_limit`` (256), which Dynamo checks first and counts
    across every isolated region on the code object, so it is the ceiling
    whatever is passed here, and the default is that ceiling rather than a raise
    above it. Reaching either cap does not refuse the call -- Dynamo runs the
    frame eagerly from then on -- and the capture records the truncation as a
    capture error, so an artifact holding fewer variants than were exercised
    says so.

    The capture is caller-driven: enter the session to get a callable, invoke it
    exactly as you would ``fn`` inside the ``with`` body, and the calls fold into
    the artifact in the ambient grad mode. The compiled region stays alive for
    the whole block, so every call reuses the variants the earlier ones
    produced. ``invariants`` names a file written when the block exits without
    an exception.

    ``guard_filter_fn`` narrows ``default_guard_filter_fn``, and the guards it
    drops leave the live check as well as the serialized copy.
    """
    return PrecompileSession(
        fn,
        backend=backend,
        guard_filter_fn=guard_filter_fn,
        recompile_limit=recompile_limit,
        dynamic=dynamic,
        training=training,
        keep_graphs=keep_graphs,
        invariants=invariants,
    )
