# Copyright (c) 2026, Han Guo, Tri Dao.
"""Identity layer for epilogue and transform mods — a LEAF module (imports
nothing from quack at module level; anyone may import it):

* fail-closed semantic fingerprinting (:func:`function_semantic_key` /
  :func:`semantic_value_key`) — the compile-cache identity of fn-authored
  mods;
* :func:`module_locator` — importable-anchor discovery for digest-keyed
  objects (torch.compile graphs and async workers re-resolve by import);
* :class:`LocalModRegistry` + payload installers — process-local resolution
  for mods with no importable anchor, shipped to async-compile workers as
  cloudpickle side-channel payloads that never touch cache keys;
* ``TORCH_OP_EPI_MODS`` / ``TORCH_OP_TRANSFORM_MODS`` — digest -> mod for
  ``quack::gemm_epi`` resolution (written at mod CONSTRUCTION, never inside
  a traced call: Dynamo buffers global dict mutations as deferred side
  effects, so a trace-time write would be invisible to the fake-tensor pass);
* :class:`GemmClassRef` / :class:`TransformARef` — picklable recipes that
  cross the jit-cache boundary in place of dynamic classes / mod instances.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import importlib
import inspect
import os
import sys
import sysconfig
import threading
import types
from typing import NamedTuple, Optional


def semantic_value_key(value, seen, *, force_source=False):
    """Fail-closed semantic fingerprint of a value an epilogue/transform fn
    depends on.

    Supported: primitives, containers, enums, modules/classes (by qualname —
    their source is covered by the package fingerprint), functions/methods/
    builtins/partials, dataclasses, and anything implementing
    ``__quack_semantic_key__(self) -> object`` (recursed through this same
    keyer). ``force_source`` fingerprints an installed external callback root
    by source too, for caller-authored code outside the QuACK package; its
    dependencies retain the stable installed-function policy. Everything else
    raises: a value we cannot fingerprint must never reach the compile
    cache, because a too-coarse key silently reuses the wrong kernel.
    """
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    qsk = getattr(type(value), "__quack_semantic_key__", None)
    if qsk is not None:
        marker = ("id", id(value))
        if marker in seen:
            return ("qsk_ref", type(value).__module__, type(value).__qualname__)
        seen.add(marker)
        return (
            "qsk",
            type(value).__module__,
            type(value).__qualname__,
            semantic_value_key(qsk(value), seen, force_source=force_source),
        )
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__module__, type(value).__qualname__, value.value)
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(semantic_value_key(v, seen, force_source=force_source) for v in value),
        )
    if isinstance(value, list):
        return (
            "list",
            tuple(semantic_value_key(v, seen, force_source=force_source) for v in value),
        )
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        repr(k),
                        semantic_value_key(v, seen, force_source=force_source),
                    )
                    for k, v in value.items()
                )
            ),
        )
    if isinstance(value, (set, frozenset)):
        return (
            "set",
            tuple(
                sorted(repr(semantic_value_key(v, seen, force_source=force_source)) for v in value)
            ),
        )
    if isinstance(value, types.ModuleType):
        return ("module", value.__name__)
    if inspect.ismethod(value):
        return (
            "method",
            function_semantic_key(value.__func__, seen, force_source=force_source),
            semantic_value_key(value.__self__, seen, force_source=force_source),
        )
    if inspect.isfunction(value):
        return function_semantic_key(value, seen, force_source=force_source)
    wrapped = getattr(value, "__wrapped__", None)
    if callable(value) and wrapped is not None:
        raise TypeError(
            f"cannot fingerprint decorated callable object {value!r}; use a "
            "function/method wrapper or implement __quack_semantic_key__"
        )
    if isinstance(value, (types.BuiltinFunctionType, types.MethodWrapperType)):
        return ("builtin", getattr(value, "__module__", None), value.__qualname__)
    if isinstance(value, functools.partial):
        return (
            "partial",
            semantic_value_key(value.func, seen, force_source=force_source),
            semantic_value_key(value.args, seen, force_source=force_source),
            semantic_value_key(value.keywords, seen, force_source=force_source),
        )
    if inspect.isclass(value):
        return ("class", value.__module__, value.__qualname__)
    if dataclasses.is_dataclass(value):
        marker = ("id", id(value))
        if marker in seen:
            return ("dataclass_ref", type(value).__module__, type(value).__qualname__)
        seen.add(marker)
        return (
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (
                    f.name,
                    semantic_value_key(getattr(value, f.name), seen, force_source=force_source),
                )
                for f in dataclasses.fields(value)
            ),
        )
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return ("torch.dtype", str(value))
    raise TypeError(
        f"epilogue fn depends on {value!r} (type {type(value).__module__}."
        f"{type(value).__qualname__}), which has no fail-closed semantic key. "
        "Supported: primitives, containers, enums, functions, dataclasses, "
        "modules/classes. For anything else, implement "
        "__quack_semantic_key__(self) -> object returning a supported value "
        "that changes whenever the traced math would."
    )


@functools.lru_cache(maxsize=1)
def _stdlib_root() -> str:
    return os.path.abspath(sysconfig.get_paths()["stdlib"]) + os.sep


def _is_extern_function(fn) -> bool:
    """True for functions defined in installed (stdlib / site-packages /
    dist-packages) code outside the quack package. Like classes and modules,
    they fingerprint by qualname only: their source is pinned by the installed
    distribution (the disk cache additionally stamps the cutlass version and
    hashes every quack source file), and recursing into them would pull
    runtime-MUTABLE library globals into the digest — e.g. any fn touching
    cutlass's dsl_user_op machinery reaches cutlass._mlir_helpers.op, which
    lazily materializes _DSL_PACKAGE_ROOT(S) on the first traced op, so the
    digest would depend on whether this process compiled anything yet (async
    workers resolve module-global EpiMods by re-import and reject the ref as
    "changed" on any mismatch)."""
    module = getattr(fn, "__module__", None) or ""
    if module == "torch._vendor.quack" or module.startswith("torch._vendor.quack."):
        return False
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    filename = code.co_filename
    if f"{os.sep}site-packages{os.sep}" in filename or f"{os.sep}dist-packages{os.sep}" in filename:
        return True
    return filename.startswith(_stdlib_root())


def _force_source_dependency(owner, value, force_source):
    """Whether a caller callback dependency belongs to the same source package."""
    if not force_source:
        return False
    if inspect.ismethod(value):
        value = value.__func__
    elif isinstance(value, functools.partial):
        value = value.func
    if not inspect.isfunction(value):
        return False
    owner_package = (getattr(owner, "__module__", None) or "").partition(".")[0]
    value_package = (getattr(value, "__module__", None) or "").partition(".")[0]
    return bool(owner_package and owner_package == value_package and owner_package != "quack")


def _dependency_semantic_key(owner, value, seen, force_source):
    """Fingerprint one dependency under the caller-package source policy."""
    return semantic_value_key(
        value,
        seen,
        force_source=_force_source_dependency(owner, value, force_source),
    )


def _function_source_digest(fn, *, exact: bool = False) -> str:
    """Hash one function body, optionally bypassing ``inspect`` unwrapping."""
    code = getattr(fn, "__code__", None)
    source_target = code if exact and code is not None else fn
    try:
        source = inspect.getsource(source_target).encode()
    except (OSError, TypeError):
        if code is None:
            raise TypeError(f"cannot fingerprint epilogue callable {fn!r}") from None
        source = code.co_code + repr(code.co_consts).encode()
    return hashlib.sha256(source).hexdigest()


def function_semantic_key(fn, seen=None, *, force_source=False):
    """Fingerprint a function root plus stable semantic keys for dependencies."""
    seen = set() if seen is None else seen
    ident = (fn.__module__, fn.__qualname__)
    marker = ("function_id", id(fn))
    if marker in seen:
        return ("function_ref", *ident)
    seen.add(marker)
    if not force_source and _is_extern_function(fn):
        return ("extern_function", *ident)

    wrapped = getattr(fn, "__wrapped__", None)
    if wrapped is not None:
        try:
            closure_vars = inspect.getclosurevars(fn)
            nonlocals = closure_vars.nonlocals
            globals_ = closure_vars.globals
        except TypeError:
            nonlocals = {}
            globals_ = {}
        wrapper_deps = tuple(
            (
                name,
                ("wrapper_ref",)
                if value is fn
                else (
                    ("wrapped_ref",)
                    if value is wrapped
                    else _dependency_semantic_key(fn, value, seen, force_source)
                ),
            )
            for name, value in sorted(nonlocals.items())
            if not name.startswith("__")
        )
        wrapper_globals = tuple(
            (name, _dependency_semantic_key(fn, value, seen, force_source))
            for name, value in sorted(globals_.items())
            if not name.startswith("__")
        )
        return (
            "decorated_function",
            *ident,
            _function_source_digest(fn, exact=True),
            semantic_value_key(fn.__defaults__, seen, force_source=False),
            semantic_value_key(fn.__kwdefaults__, seen, force_source=False),
            wrapper_deps,
            wrapper_globals,
            semantic_value_key(wrapped, seen, force_source=force_source),
        )

    try:
        closure_vars = inspect.getclosurevars(fn)
        referenced = {
            **closure_vars.globals,
            **closure_vars.nonlocals,
        }
    except TypeError:
        referenced = {}
    deps = tuple(
        (name, _dependency_semantic_key(fn, value, seen, force_source))
        for name, value in sorted(referenced.items())
        if not name.startswith("__")
    )
    return (
        "function",
        *ident,
        _function_source_digest(fn),
        semantic_value_key(fn.__defaults__, seen, force_source=False),
        semantic_value_key(fn.__kwdefaults__, seen, force_source=False),
        deps,
    )


def module_locator(obj, fn) -> Optional[tuple]:
    """(module, global_name) if ``obj`` is reachable by import in a fresh
    process — torch.compile graphs and async workers re-resolve digest-keyed
    objects that way. ``fn`` anchors the search (the authored function whose
    ``__module__``/``__name__`` name the natural home); returns None for
    ``__main__`` / notebook objects and objects never bound to a module
    global.

    A binding found here may not exist in a fresh process: a module global
    bound at CALL time (e.g. a test helper's ``global`` trick) satisfies this
    scan but is never replayed by a fresh import. That can't be detected on
    the fn object (a ``global``-declared def even strips the ``<locals>``
    qualname marker), so consumers must tolerate it at resolve time — the
    async pool ships the mod by value as a fallback payload (see
    ``GemmClassRef.__quack_pool_payload__``)."""
    module_name = getattr(fn, "__module__", None)
    if module_name is None or module_name == "__main__":
        return None
    module = sys.modules.get(module_name)
    if module is None:
        return None
    preferred = getattr(fn, "__name__", None)
    if preferred and getattr(module, preferred, None) is obj:
        return module_name, preferred
    names = sorted(name for name, value in vars(module).items() if value is obj)
    if not names:
        return None
    return module_name, names[0]


# digest -> mod, for quack::gemm_epi resolution (quack.gemm_runtime.torch_op).
# Populated at CONSTRUCTION (EpiMod / transform-mod __init__), never inside
# compile_call: Dynamo buffers global dict mutations as deferred side
# effects, so a trace-time write would be invisible to the fake-tensor pass
# that resolves the digest. Same-digest reconstruction overwrites in place.
TORCH_OP_EPI_MODS: dict[str, object] = {}
TORCH_OP_TRANSFORM_MODS: dict[str, object] = {}
_CLOUDPICKLE_BY_VALUE_LOCK = threading.Lock()


class LocalModRegistry:
    """digest -> mod, for refs with no importable module anchor (defined in
    ``__main__``, notebooks, or never bound to a module global). ``consume``
    pops entries on resolve: worker payloads may close over sizeable Python
    state and async workers live for the whole test/autotune session. A
    non-consuming registry holds small format bundles / fn mods that a plan
    may resolve several times."""

    def __init__(self, installer: str, consume: bool):
        self.installer = installer  # module-level installer fn name (PoolPayload target)
        self.consume = consume
        self._mods: dict[str, object] = {}

    def register(self, digest: str, mod) -> None:
        self._mods[digest] = mod

    def resolve(self, digest: str):
        """The registered mod, or None (caller owns the error message)."""
        if self.consume:
            return self._mods.pop(digest, None)
        return self._mods.get(digest)

    def payload(self, digest: str):
        """Worker-side install recipe for a registered mod (cloudpickle side
        channel — never enters the cache key; an unserializable mod makes the
        pool refuse the key and the cold miss compiles in-process)."""
        import cloudpickle

        from torch._vendor.quack.cache.async_compile import PoolPayload

        mod = self._mods[digest]
        fn = getattr(mod, "fn", None)
        module = sys.modules.get(getattr(fn, "__module__", None))
        if module is None:
            data = cloudpickle.dumps(mod)
        else:
            with _CLOUDPICKLE_BY_VALUE_LOCK:
                registered = module.__name__ in cloudpickle.list_registry_pickle_by_value()
                if not registered:
                    cloudpickle.register_pickle_by_value(module)
                try:
                    data = cloudpickle.dumps(mod)
                finally:
                    if not registered:
                        cloudpickle.unregister_pickle_by_value(module)
        return PoolPayload(__name__, self.installer, digest, data)

    def install(self, expected_digest: str, data: bytes) -> None:
        import cloudpickle

        mod = cloudpickle.loads(data)
        if mod.semantic_digest != expected_digest:
            raise ValueError(
                f"local mod payload digest mismatch: expected {expected_digest}, "
                f"got {mod.semantic_digest}"
            )
        self._mods[expected_digest] = mod


LOCAL_EPI_MODS = LocalModRegistry("install_epi_mod_payload", consume=True)
LOCAL_TRANSFORM_MODS = LocalModRegistry("install_transform_mod_payload", consume=False)


def register_local_epi_mod(digest: str, epi_mod) -> None:
    LOCAL_EPI_MODS.register(digest, epi_mod)


def register_local_transform_mod(digest: str, mod) -> None:
    LOCAL_TRANSFORM_MODS.register(digest, mod)


def install_epi_mod_payload(expected_digest: str, data: bytes) -> None:
    """Worker-side installer for ``epi_mod_local`` payloads (see
    ``GemmClassRef.__quack_pool_payload__``)."""
    LOCAL_EPI_MODS.install(expected_digest, data)


def install_transform_mod_payload(expected_digest: str, data: bytes) -> None:
    """Worker-side installer for ``mod_local`` transform payloads (see
    ``TransformARef.__quack_pool_payload__``)."""
    LOCAL_TRANSFORM_MODS.install(expected_digest, data)


class GemmClassRef(NamedTuple):
    """Picklable recipe for resolving a GEMM class in async workers.

    Dynamic epilogue classes must never cross the cache boundary directly:
    their module registration exists only in the creating process. Instead an
    epi_mod reference imports the module-global EpiMod and asks it to mint the
    same class from a semantic digest plus the runtime kind signature.

    ``epi_mod_local`` covers EpiMods with no importable anchor (defined in
    ``__main__`` — scripts, notebooks — or never bound to a module global):
    the semantic digest still keys the disk cache correctly and resolution
    goes through :data:`LOCAL_EPI_MODS`. To reach async workers, the ref
    ships the EpiMod by value as a side-channel payload (cloudpickle, see
    ``__quack_pool_payload__``) — the payload never enters the cache key, so
    shas stay deterministic. If the payload can't be serialized the pool
    refuses the key and the cold miss compiles in-process.
    """

    kind: str  # "static", "epi_mod", or "epi_mod_local"
    module: str
    qualname: str
    mint_key: tuple = ()
    semantic_digest: str = ""

    def __quack_pool_payload__(self):
        """Worker setup payload for the EpiMod.

        ``epi_mod_local``: mandatory (no importable anchor; a serialization
        failure makes the pool refuse the key -> in-process compile).

        ``epi_mod``: best-effort belt-and-braces. Resolvable-by-import at
        submit time does not guarantee resolvable in a worker: a module
        global bound at CALL time (a test helper's ``global`` trick)
        satisfies the locator here but doesn't exist after the worker's
        fresh import — or resolves to a different mod the module binds at
        import time. Ship the mod by value too and let
        :func:`resolve_gemm_class` fall back to the installed payload in
        both cases; if it can't be serialized, keep the by-ref-only behavior
        (those mods resolve fine when the binding is real, exactly as
        before)."""
        if self.kind == "epi_mod_local":
            return LOCAL_EPI_MODS.payload(self.semantic_digest)
        if self.kind != "epi_mod":
            return None
        mod = TORCH_OP_EPI_MODS.get(self.semantic_digest)
        if mod is None:
            return None
        try:
            import cloudpickle

            from torch._vendor.quack.cache.async_compile import PoolPayload

            return PoolPayload(
                __name__, "install_epi_mod_payload", self.semantic_digest, cloudpickle.dumps(mod)
            )
        except Exception:
            return None


def _resolve_qualname(obj, qualname):
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def static_gemm_class_ref(GemmCls):
    return GemmClassRef("static", GemmCls.__module__, GemmCls.__qualname__)


def resolve_gemm_class(ref: GemmClassRef):
    if ref.kind == "epi_mod_local":
        obj = LOCAL_EPI_MODS.resolve(ref.semantic_digest)
        if obj is None:
            raise RuntimeError(
                "process-local epilogue reference is not registered here (created in "
                "another process and its payload was not installed); bind the "
                "@gemm_epilogue object to a module-global name in an importable module "
                "to make it resolvable by import"
            )
        return obj._mint(*ref.mint_key)
    if ref.kind == "static":
        return _resolve_qualname(importlib.import_module(ref.module), ref.qualname)
    if ref.kind != "epi_mod":
        raise ValueError(f"unknown GEMM class reference kind {ref.kind!r}")
    obj = err = None
    try:
        obj = _resolve_qualname(importlib.import_module(ref.module), ref.qualname)
    except AttributeError as e:
        err = e
    if obj is None or obj.semantic_digest != ref.semantic_digest:
        # The submitter saw a module-global binding that a fresh import does
        # not replay (bound at call time, e.g. via a `global` declaration
        # inside a function): the attribute is missing here, or holds a
        # different mod the module binds at import time. Async workers get
        # the mod by value as a side-channel payload for exactly this case
        # (digest-checked at install) — resolve through it.
        installed = LOCAL_EPI_MODS.resolve(ref.semantic_digest)
        if installed is None:
            if err is not None:
                raise err
            raise RuntimeError(
                f"epilogue {ref.module}.{ref.qualname} changed while resolving a compile request"
            )
        obj = installed
    return obj._mint(*ref.mint_key)


class TransformARef(NamedTuple):
    """Picklable recipe for resolving a transform mod in async workers.

    ``w4_name`` re-mints from the W4_FORMATS registry by name; ``mod_local``
    resolves through :data:`LOCAL_TRANSFORM_MODS` (populated by
    ``register_local_transform_mod`` at mod construction/ref time, and by
    :func:`install_transform_mod_payload` in async workers)."""

    kind: str  # "w4_name" | "mod_local"
    name: str = ""
    semantic_digest: str = ""

    def __quack_pool_payload__(self):
        if self.kind != "mod_local":
            return None
        return LOCAL_TRANSFORM_MODS.payload(self.semantic_digest)


def resolve_transform_a(ref: TransformARef):
    if ref.kind == "w4_name":
        # by-name refs re-mint through the memoized normalizer (one mod per
        # format name per process, shared with every other entry surface)
        from torch._vendor.quack.operand_transform.host import as_transform_mod

        return as_transform_mod(ref.name)
    if ref.kind != "mod_local":
        raise ValueError(f"unknown transform reference kind {ref.kind!r}")
    mod = LOCAL_TRANSFORM_MODS.resolve(ref.semantic_digest)
    if mod is None:
        raise RuntimeError(
            "process-local transform reference is not registered here (created in "
            "another process and its payload was not installed)"
        )
    return mod
