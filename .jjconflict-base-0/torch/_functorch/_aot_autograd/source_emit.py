"""Reconstruct live Python values as standalone, auditable source.

The composer (to_standalone_python.py) inlines AOTAutograd's runtime wrappers, which
close over baked metadata objects (view-replay recipes, tensor-subclass specs, ...).
This module turns such a live object into a Python expression that reconstructs an
EQUAL value -- emitted as readable source, never a pickle / base64 blob -- or raises
NotImplementedError if it cannot be expressed. ``emit_value`` is the public entry
point; ``_emit_value`` is the internal recursion (it threads the traversal state);
``_emit_via_reduce`` is the pickle-reduce-as-source fallback and ``_REBUILD_HELPER`` is
the ``_rebuild`` function it references in generated code. Kept as a leaf module so it
is safe to import anywhere in the package.
"""

from __future__ import annotations

import inspect
from typing import Any, NewType, TYPE_CHECKING


# id() returns an object identity used here only as a recursion/alias key, not an
# arbitrary int. Shadowing the builtin under TYPE_CHECKING tags every ``id(obj)`` as
# ``_ObjId`` so ``set[_ObjId]`` propagates without wrapping each call site; the real
# builtin runs at runtime (NewType is identity), so behavior is unchanged.
_ObjId = NewType("_ObjId", int)

if TYPE_CHECKING:

    def id(obj: object, /) -> _ObjId: ...


# The runtime helper for reduce-based metadata reconstruction (see _emit_via_reduce).
# Emitted only when some baked value actually needs it -- many graphs (e.g. a plain
# module with no tensor-subclass metadata) reference none, so it would just be dead
# code. Includes its own trailing blank lines so the splice stays clean.
_REBUILD_HELPER: list[str] = [
    "def _rebuild(obj, state):",
    "    # Apply pickle-style reduce state to a freshly __new__'d object, emitted as",
    "    # readable source instead of an opaque blob (no pickle.loads). Used for the",
    "    # opaque value leaves of baked metadata -- e.g. tensor-subclass placement",
    "    # objects -- mirroring pickle's load_build: prefer __setstate__, else update",
    "    # __dict__ (and slot state for the (dict, slots) 2-tuple form).",
    "    if state is None:",
    "        return obj",
    "    if hasattr(obj, '__setstate__'):",
    "        obj.__setstate__(state)",
    "        return obj",
    "    slotstate = None",
    "    if isinstance(state, tuple) and len(state) == 2:",
    "        state, slotstate = state",
    "    if state:",
    "        obj.__dict__.update(state)",
    "    if slotstate:",
    "        for _k, _v in slotstate.items():",
    "            setattr(obj, _k, _v)",
    "    return obj",
    "",
    "",
]


def _emit_importable(obj: Any, imports: set[str]) -> str:
    """Return an expression referencing ``obj`` via its defining module (recording the
    import). Works for classes, functions, and modules. Raises if ``obj`` is not
    reachable as ``module.qualname`` (e.g. a local or lambda), since then it cannot be
    reproduced in the standalone module without embedding it."""
    import importlib

    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module or not qualname or "<" in qualname:
        raise NotImplementedError(
            f"compile_to_python cannot reference {obj!r} by import: it has no "
            "module/qualname or is a local definition."
        )
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part, None)
    if target is not obj:
        raise NotImplementedError(
            f"compile_to_python cannot reference {qualname} from {module}: it does "
            "not round-trip to the same object."
        )
    imports.add(f"import {module}")
    return f"{module}.{qualname}"


def _reject_shared_mutable(obj: Any, _seen: set[_ObjId]) -> None:
    """Fail loud when a mutable object is reached more than once in a single traversal:
    it would be emitted as two independent source literals, silently dropping the
    shared-identity aliasing (mutating one copy would no longer affect the other), which
    is not expressible as a source literal today. Routed here are the kinds reconstructed
    as fresh MUTABLE copies whose aliasing a flat literal cannot preserve: list / dict /
    set, bytearray, non-frozen dataclasses, and the opaque _emit_via_reduce value objects.
    Kinds treated as immutable value objects are exempt -- scalars, tuple / frozenset, frozen
    dataclasses, and the recipe-like value objects emitted by their own branches
    (functools.partial, ViewMeta, ViewMetaSequence), which are never mutated after
    reconstruction -- so a shared one of those still emits fine. ``_seen`` is threaded in
    place across the whole traversal (unlike the per-level in-progress cycle set), so it
    catches shared siblings, not just self-referential ancestors."""
    if id(obj) in _seen:
        raise NotImplementedError(
            f"compile_to_python cannot bake a shared mutable {type(obj).__qualname__}: "
            "it is reached more than once in the value being baked, so emitting it as "
            "independent source literals would silently drop the shared-identity "
            "aliasing."
        )
    _seen.add(id(obj))


def emit_value(obj: Any, imports: set[str]) -> str:
    """Emit a Python expression (valid in the generated module) that rebuilds ``obj``
    from source -- the pickle-free replacement for embedding a base64 blob.

    Public entry point: recurses through containers and metadata so the artifact stays
    auditable and exec'ing it never runs ``pickle.loads``. Bottoms out for opaque value
    objects (e.g. tensor-subclass placement objects) at the pickle reduce protocol, but
    EMITTED AS SOURCE via ``_rebuild`` (see ``_emit_via_reduce``). Raises
    NotImplementedError at any leaf that cannot be expressed as source (a live tensor, a
    lambda, a still-symbolic Sym*), for a self-referential structure, or for a shared
    mutable object whose aliasing a flat source literal cannot preserve.

    Any import needed by the emitted expression is recorded into ``imports`` (as ``import
    module`` statements)."""
    return _emit_value(obj, imports, set(), set())


def _emit_value(
    obj: Any,
    imports: set[str],
    _in_progress: set[_ObjId],
    _seen: set[_ObjId],
) -> str:
    """Internal recursion behind ``emit_value``; requires the traversal state its guards
    thread. Call ``emit_value`` from outside this module.

    ``_in_progress`` is an identity-keyed set of the objects currently being emitted on
    this recursion PATH (copied per level, like a call stack): revisiting one means the
    metadata is self-referential and cannot be a source literal, so we raise rather than
    recurse forever. ``_seen`` is an identity-keyed set of every MUTABLE object already
    emitted anywhere in this traversal (threaded in place, like ``imports``): revisiting
    one means the same mutable object is shared across positions, whose aliasing a flat
    literal cannot preserve, so we raise (see ``_reject_shared_mutable``)."""
    import dataclasses
    import enum
    import functools
    import math
    import types

    import torch
    from torch._C import _functionalization as _F
    from torch.fx.experimental.symbolic_shapes import SymIntEqByExpr

    from .functional_utils import ViewMetaSequence

    # Live tensors / storages must never be baked: their reduce form is a
    # ``torch.storage._load_from_bytes(b'...')`` pickle blob, which both embeds the
    # raw weight bytes and invokes ``pickle.loads`` at exec time -- violating the
    # module's no-weights-baked / never-pickle.loads / fully-auditable guarantees.
    # Reject them explicitly here, before they can fall through to _emit_via_reduce.
    # A non-Tensor object whose reduce IS ``_load_from_bytes`` (a wrapper that
    # delegates pickling to a storage's bytes) is caught at the reduce callable in
    # _emit_via_reduce, where that callable is actually visible.
    if isinstance(
        obj, (torch.Tensor, torch.storage.TypedStorage, torch.UntypedStorage)
    ):
        raise NotImplementedError(
            f"compile_to_python cannot bake a live {type(obj).__qualname__} into "
            "standalone source: that would embed raw tensor bytes and require "
            "pickle.loads at exec time. The standalone artifact bakes no weights."
        )

    # Primitives first, before any cycle / alias bookkeeping: those guards never matter
    # for immutable scalars, and returning here skips the per-level in-progress copy and
    # the mutable-alias check for the (very common) scalar leaves.
    #
    # Non-finite floats: repr() gives bare ``inf`` / ``-inf`` / ``nan``, which are
    # NameErrors in the generated module (it imports no such names). Emit a
    # self-contained constructor instead. Likewise for a complex with a non-finite
    # component (repr would be e.g. ``(inf+0j)``).
    if isinstance(obj, float) and not math.isfinite(obj):
        if math.isnan(obj):
            return "float('nan')"
        return "float('inf')" if obj > 0 else "float('-inf')"
    if isinstance(obj, complex) and not (
        math.isfinite(obj.real) and math.isfinite(obj.imag)
    ):
        # Use the module-level recursion directly (the in-body shadow is not defined until
        # after the guards below); the components are floats, so they bottom out above.
        real = _emit_value_recurse(obj.real, imports, _in_progress, _seen)
        imag = _emit_value_recurse(obj.imag, imports, _in_progress, _seen)
        return f"complex({real}, {imag})"
    # Plain constants reproduce via repr (but not IntEnum/StrEnum members, whose repr
    # is not valid source -- those fall through to the enum handler below). Match EXACT
    # builtin types only: a subclass of a builtin scalar (e.g. a ``str`` subclass, or an
    # ``int`` subclass with a constructor-style ``__repr__``) would mis-bake under repr
    # -- losing its type or emitting a NameError -- so it must fall through to the
    # importable/reduce path, which reconstructs it faithfully or raises.
    if obj is None or (
        not isinstance(obj, enum.Enum)
        and type(obj) in (bool, int, float, complex, str, bytes)
    ):
        return repr(obj)
    # bytearray is the one mutable builtin whose repr round-trips, so it is emitted as a
    # repr literal like the immutable scalars above -- but being mutable it needs the same
    # shared-alias guard the containers get (reached twice it would become two independent
    # literals, silently dropping the aliasing). It cannot be self-referential, so it needs
    # only the alias check and no cycle guard.
    if type(obj) is bytearray:
        _reject_shared_mutable(obj, _seen)
        return repr(obj)

    # Cycle guard: thread an identity-keyed in-progress set down the recursion so a
    # self-referential metadata object raises NotImplementedError naming the offending
    # type instead of recursing until RecursionError. Primitives already returned above,
    # so no immutable-scalar exemption is needed here.
    if id(obj) in _in_progress:
        raise NotImplementedError(
            f"compile_to_python cannot bake {type(obj).__qualname__}: it is "
            "self-referential, which is not expressible as source."
        )
    # Alias guard: a mutable container reached twice anywhere in the traversal would be
    # emitted as two independent literals, silently dropping the shared-identity aliasing.
    # Match EXACT list/dict/set only -- those are the types the container branches below
    # emit; a subclass falls through to _emit_via_reduce, which tracks it there (matching
    # here too would double-count the same object and falsely reject a single subclass).
    if type(obj) in (list, dict, set):
        _reject_shared_mutable(obj, _seen)

    _child = _in_progress | {id(obj)}

    def _emit_value(obj: Any, imports: set[str]) -> str:
        # Shadow the module-level recursion entry point so every recursive call in
        # this function body automatically threads the current in-progress set (with
        # the parent ``obj`` added) plus the traversal-wide seen set, without touching
        # each call site.
        return _emit_value_recurse(obj, imports, _child, _seen)

    # torch scalar singletons whose repr round-trips, plus device/Size/SymInt.
    if isinstance(obj, (torch.dtype, torch.layout, torch.memory_format)):
        imports.add("import torch")
        return repr(obj)
    if isinstance(obj, torch.device):
        imports.add("import torch")
        return f"torch.device({str(obj)!r})"
    if isinstance(obj, torch.Size):
        imports.add("import torch")
        return f"torch.Size([{', '.join(_emit_value(x, imports) for x in obj)}])"
    if isinstance(obj, torch.SymInt):
        concrete = obj.node.maybe_as_int()
        if concrete is None:
            raise NotImplementedError(
                "compile_to_python cannot bake a symbolic SymInt; precompile "
                "specializes to static shapes."
            )
        return repr(concrete)
    # SymBool / SymFloat get the same treatment as SymInt (bake the concrete value, reject
    # a still-symbolic one with the static-shapes message) so all sym scalar types are
    # handled consistently instead of falling into the reduce path with an opaque error.
    if isinstance(obj, torch.SymBool):
        concrete_bool = obj.node.maybe_as_bool()
        if concrete_bool is None:
            raise NotImplementedError(
                "compile_to_python cannot bake a symbolic SymBool; precompile "
                "specializes to static shapes."
            )
        return repr(concrete_bool)
    if isinstance(obj, torch.SymFloat):
        concrete_float = obj.node.maybe_as_float()
        if concrete_float is None:
            raise NotImplementedError(
                "compile_to_python cannot bake a symbolic SymFloat; precompile "
                "specializes to static shapes."
            )
        return repr(concrete_float)

    # Importable definitions: classes, functions, modules.
    if isinstance(obj, type) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return _emit_importable(obj, imports)
    if isinstance(obj, types.ModuleType):
        imports.add(f"import {obj.__name__}")
        return obj.__name__

    # Python enums and pybind11 enums (e.g. InverseReturnMode) both reference a member
    # by name off the importable type. pybind enum members are not singletons (as_tuple
    # hands back fresh copies), so match by value via the type's __members__ map.
    members = getattr(type(obj), "__members__", None)
    name = getattr(obj, "name", None)
    is_enum_like = isinstance(obj, enum.Enum) or (
        members is not None and name in members and members[name] == obj
    )
    if is_enum_like:
        if name is not None and members is not None and name in members:
            return f"{_emit_importable(type(obj), imports)}.{name}"
        # A combined Flag/IntFlag member (A | B) has no single member name, so emit by
        # value -- ``Type(value)`` reproduces the same combined member. A non-Flag enum
        # with no resolvable name is not source-expressible, so reject it rather than
        # emit ``Type.None`` (a SyntaxError that would break the whole module).
        if isinstance(obj, enum.Flag):
            cls = _emit_importable(type(obj), imports)
            return f"{cls}({_emit_value(obj.value, imports)})"
        # Defensive backstop, effectively unreachable for constructible inputs: a
        # pure-Python non-Flag enum.Enum always resolves a name in __members__ (it rejects
        # unregistered values with ValueError) and so returns by-name above, and a pybind
        # enum-like value only reaches here via the ``name in members`` arm that already
        # returned. Kept so any future name-less enum-like type fails loudly, not silently.
        raise NotImplementedError(
            f"compile_to_python cannot bake enum member {obj!r}: it has no resolvable "
            "member name and is not a Flag."
        )

    if isinstance(obj, functools.partial):
        parts = [_emit_value(obj.func, imports)]
        parts += [_emit_value(a, imports) for a in obj.args]
        parts += [
            f"{k}={_emit_value(v, imports)}" for k, v in (obj.keywords or {}).items()
        ]
        imports.add("import functools")
        return f"functools.partial({', '.join(parts)})"

    # AOT view-replay recipes: ViewMeta C++ objects round-trip through as_tuple();
    # ViewMetaSequence has no public constructor so use its _from_parts factory;
    # SymIntEqByExpr wraps a sympy expr that must be a concrete integer here.
    if isinstance(obj, _F.ViewMeta):
        # as_tuple lives on the ViewMeta subclass bindings (create_binding_with_pickle),
        # not on the base ViewMeta stub, so pyrefly cannot see it on the isinstance type.
        tup = obj.as_tuple()  # pyrefly: ignore[missing-attribute]
        return f"{_emit_importable(type(obj), imports)}({_emit_value(tuple(tup), imports)})"
    if isinstance(obj, ViewMetaSequence):
        cls = _emit_importable(ViewMetaSequence, imports)
        seq = _emit_value(list(obj.sequence), imports)
        return f"{cls}._from_parts({seq}, {_emit_value(obj.metadata, imports)})"
    if isinstance(obj, SymIntEqByExpr):
        if not getattr(obj.val, "is_Integer", False):
            raise NotImplementedError(
                "compile_to_python cannot bake symbolic view metadata; precompile "
                "specializes to static shapes."
            )
        return f"{_emit_importable(SymIntEqByExpr, imports)}({int(obj.val)})"

    # namedtuple before plain tuple (it has a richer, by-name constructor). The plain
    # container branches match EXACT builtin types only (like the scalar branch above): a
    # container SUBCLASS (e.g. a ``list`` subclass carrying extra state) would otherwise be
    # silently downcast to its base type, dropping the subclass and its state. Subclasses
    # fall through to _emit_via_reduce, which reconstructs them faithfully or rejects them
    # (its listitems/dictitems guard fires for list/dict subclasses).
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        cls = _emit_importable(type(obj), imports)
        return f"{cls}({', '.join(_emit_value(x, imports) for x in obj)})"
    if type(obj) is tuple:
        items = [_emit_value(x, imports) for x in obj]
        return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"
    if type(obj) is list:
        return f"[{', '.join(_emit_value(x, imports) for x in obj)}]"
    if type(obj) is set or type(obj) is frozenset:
        ctor = "frozenset" if isinstance(obj, frozenset) else "set"
        # Iteration order of a set is not byte-stable across processes, so emit the
        # elements in a canonical order: sort them by their EMITTED SOURCE EXPRESSION --
        # the exact text that lands in the artifact, so it is the only key guaranteed to
        # be byte-stable across processes. A ``repr``-keyed sort would silently admit a
        # custom-but-nondeterministic ``__repr__`` (e.g. one embedding ``id(self)``),
        # making the emitted source differ run to run, and a numeric sort would raise on a
        # ``nan`` element (``sorted`` does not) and does not exist for mixed types. Each
        # element is emitted exactly once (into the real ``imports``, since every element
        # emitted is one that lands in the output) and then the resulting expressions are
        # sorted -- which forces each element through the same source-expressibility gate,
        # so a non-source-expressible element raises here exactly as at final emit.
        exprs = sorted(_emit_value(x, imports) for x in obj)
        return f"{ctor}([{', '.join(exprs)}])"
    if type(obj) is dict:
        items = [
            f"{_emit_value(k, imports)}: {_emit_value(v, imports)}"
            for k, v in obj.items()
        ]
        return f"{{{', '.join(items)}}}"

    # Dataclasses (e.g. MetadataKey, SubclassViewMetaSequence) reproduce field-by-field
    # through their generated constructor, passing only the init fields. That path is
    # faithful ONLY when the object carries no state outside those init fields: a
    # non-init field (init=False) holding state, or a __post_init__ that derives state
    # the constructor call would not reproduce, would be silently dropped. Guard it the
    # same way the rest of this module guards opaque leaves -- reject rather than emit a
    # subtly-wrong artifact -- by round-tripping through the constructor and requiring
    # the rebuilt instance to compare equal to the original. The in-scope metadata
    # types (e.g. MetadataKey) are plain value dataclasses, so they round-trip and keep
    # working; a type with hidden/derived state does not and raises NotImplementedError
    # naming itself. This strategy needs value equality: a dataclass declared eq=False
    # inherits object.__eq__ (identity), so rebuilt == obj is ALWAYS False even for a
    # pure value object -- detect that and fall through to _emit_via_reduce (which has its
    # own source-expressibility rejects) rather than spuriously rejecting it.
    if (
        dataclasses.is_dataclass(obj)
        and not isinstance(obj, type)
        and type(obj).__eq__ is not object.__eq__
    ):
        # A non-frozen dataclass is mutable, so it gets the same shared-alias guard as the
        # containers: reached twice it would be emitted as two independent constructor
        # calls, silently dropping the shared-identity aliasing. A frozen dataclass is
        # immutable and stays exempt, like a tuple.
        if not obj.__dataclass_params__.frozen:
            _reject_shared_mutable(obj, _seen)
        cls = _emit_importable(type(obj), imports)
        fields = dataclasses.fields(obj)
        init_kwargs = {f.name: getattr(obj, f.name) for f in fields if f.init}
        try:
            # type(obj) is the concrete dataclass at runtime, not the protocol pyrefly
            # infers from the is_dataclass narrowing; the call is correct.
            rebuilt = type(obj)(**init_kwargs)  # pyrefly: ignore[bad-instantiation]
            round_trips = rebuilt == obj
        except Exception:
            round_trips = False
        if not round_trips:
            raise NotImplementedError(
                f"compile_to_python cannot bake dataclass {type(obj).__qualname__}: it "
                "does not round-trip through its constructor from its init fields alone "
                "(it likely has a stateful non-init field or a __post_init__ deriving "
                "state the constructor call would not reproduce), so emitting only the "
                "init fields would silently drop that state."
            )
        kw = [f"{k}={_emit_value(v, imports)}" for k, v in init_kwargs.items()]
        return f"{cls}({', '.join(kw)})"

    return _emit_via_reduce(obj, imports, _child, _seen)


# The recursion entry point used by the in-body shadow of ``_emit_value`` and by
# ``_emit_via_reduce`` to thread the identity-keyed in-progress + seen traversal sets.
_emit_value_recurse = _emit_value


def _emit_via_reduce(
    obj: Any,
    imports: set[str],
    _in_progress: set[_ObjId],
    _seen: set[_ObjId],
) -> str:
    """Last resort for opaque value objects (e.g. DTensor placements, which are C++
    objects with no source-friendly constructor): reconstruct from the pickle reduce
    protocol, but EMITTED AS SOURCE -- ``cls.__new__(cls)`` plus ``_rebuild`` applying
    the reduce state -- so there is no ``pickle.loads`` at exec time and the bytes
    are readable. The reduce state recurses back through ``_emit_value``, so any
    non-source-expressible leaf inside it still raises."""
    import copyreg

    import torch

    # These reduce value objects are mutable, so the same guard the containers get in
    # _emit_value applies: an object reached twice would be emitted as two independent
    # reconstructions, silently dropping the shared-identity aliasing.
    _reject_shared_mutable(obj, _seen)

    def _emit_value(obj: Any, imports: set[str]) -> str:
        # Continue threading the cycle guard + seen set through the reduce-state recursion.
        return _emit_value_recurse(obj, imports, _in_progress, _seen)

    try:
        reduced = obj.__reduce_ex__(2)
    except Exception as e:
        raise NotImplementedError(
            f"compile_to_python cannot make {type(obj).__module__}."
            f"{type(obj).__qualname__} self-contained: it is not source-expressible "
            f"and has no usable reduce ({type(e).__name__})."
        ) from e
    if not isinstance(reduced, tuple) or len(reduced) < 2:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: "
            "unsupported __reduce__ form."
        )
    func, args = reduced[0], reduced[1]
    # The reduce protocol requires the second element to be a tuple of constructor args;
    # validate once here so all three func branches below normalize a malformed reduce
    # into the uniform NotImplementedError contract rather than leaking a bare TypeError.
    if not isinstance(args, tuple):
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: reduce "
            "args field is not a tuple."
        )
    state = reduced[2] if len(reduced) > 2 else None
    listitems = reduced[3] if len(reduced) > 3 else None
    dictitems = reduced[4] if len(reduced) > 4 else None
    # A reduce whose callable is ``torch.storage._load_from_bytes`` would embed raw
    # bytes and require a pickle.loads-equivalent at exec time. This is the ONLY place
    # that callable is visible (it is the reduce result's func, not the object's
    # __reduce_ex__ method), so reject it here to uphold the no-bytes / fully-auditable
    # guarantee for any non-Tensor wrapper that delegates pickling to storage bytes.
    if func is getattr(torch.storage, "_load_from_bytes", None):
        raise NotImplementedError(
            f"compile_to_python cannot bake {type(obj).__qualname__} into standalone "
            "source: its reduce is torch.storage._load_from_bytes, which would embed "
            "raw bytes and require pickle.loads at exec time."
        )
    if listitems is not None or dictitems is not None:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: reduce "
            "produced list/dict items (container subclass)."
        )
    # A non-None 6th element is a protocol-5 ``state_setter``: the object opted out of
    # the default __setstate__/__dict__ install that ``_rebuild`` implements, so applying
    # state via _rebuild would silently use the wrong mechanism. Reject rather than emit
    # a subtly-wrong object.
    state_setter = reduced[5] if len(reduced) > 5 else None
    if state_setter is not None:
        raise NotImplementedError(
            f"compile_to_python cannot reconstruct {type(obj).__qualname__}: reduce "
            "produced a state_setter (protocol-5 form) that _rebuild cannot apply."
        )
    if func is getattr(copyreg, "__newobj__", None):
        if not args:
            raise NotImplementedError(
                f"compile_to_python cannot reconstruct {type(obj).__qualname__}: "
                "__newobj__ reduce produced no class argument."
            )
        cls = _emit_value(args[0], imports)
        extra = ", ".join(_emit_value(a, imports) for a in args[1:])
        base = f"{cls}.__new__({cls}{', ' + extra if extra else ''})"
    elif func is getattr(copyreg, "__newobj_ex__", None):
        if not (
            isinstance(args, tuple)
            and len(args) == 3
            and isinstance(args[1], tuple)
            and isinstance(args[2], dict)
        ):
            raise NotImplementedError(
                f"compile_to_python cannot reconstruct {type(obj).__qualname__}: "
                "__newobj_ex__ reduce did not produce a (cls, args, kwargs) triple."
            )
        cls_obj, new_args, new_kwargs = args
        cls = _emit_value(cls_obj, imports)
        pos = ", ".join(_emit_value(a, imports) for a in new_args)
        kw = ", ".join(f"{k}={_emit_value(v, imports)}" for k, v in new_kwargs.items())
        joined = ", ".join(p for p in (pos, kw) if p)
        base = f"{cls}.__new__({cls}{', ' + joined if joined else ''})"
    else:
        func_expr = _emit_value(func, imports)
        base = f"{func_expr}({', '.join(_emit_value(a, imports) for a in args)})"
    if state is None:
        return base
    return f"_rebuild({base}, {_emit_value(state, imports)})"
