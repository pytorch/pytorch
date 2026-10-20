# Owner(s): ["oncall: pt2"]
import collections
import dataclasses
import enum
import functools
import math

import torch
from torch._functorch._aot_autograd.source_emit import (
    _emit_importable,
    _REBUILD_HELPER,
    emit_value,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymIntEqByExpr
from torch.testing._internal.common_utils import run_tests, TestCase


def _emit(obj):
    """Emit source for ``obj`` and return ``(expr, sorted_imports)``."""
    imports: set[str] = set()
    return emit_value(obj, imports), sorted(imports)


def _roundtrip(obj):
    """Emit ``obj`` as source, then exec the imports + ``_rebuild`` helper and eval the
    expression -- the same way the generated standalone module reconstructs it."""
    expr, imports = _emit(obj)
    ns: dict = {}
    if "_rebuild(" in expr:
        exec("\n".join(_REBUILD_HELPER), ns)
    for stmt in imports:
        exec(stmt, ns)
    return eval(expr, ns)


# Module-level fixtures: _emit_importable rejects any object whose __qualname__ carries a
# "<locals>" component, so enums / dataclasses / namedtuples used in the emission tests
# must be defined at module scope to be source-reconstructible.
class _Color(enum.Enum):
    RED = 1
    BLUE = 2


class _IColor(enum.IntEnum):
    RED = 1
    BLUE = 2


class _Flag(enum.IntFlag):
    A = 1
    B = 2


class _MyList(list):
    # A builtin-container subclass; emitting it as a plain list would silently drop the
    # subclass, so it must be rejected (via _emit_via_reduce's listitems guard).
    pass


_Point = collections.namedtuple("_Point", ["x", "y"])


@dataclasses.dataclass
class _PlainDC:
    a: int
    b: str


@dataclasses.dataclass
class _DerivedDC:
    a: int
    derived: int = dataclasses.field(init=False, default=0)

    def __post_init__(self):
        # Re-derived from an init field, so a constructor call reproduces it: this still
        # round-trips and must NOT be rejected.
        self.derived = self.a * 2


@dataclasses.dataclass
class _StatefulDC:
    a: int
    extra: int = dataclasses.field(init=False, default=0)


@dataclasses.dataclass(frozen=True)
class _FrozenDC:
    # Frozen -> immutable, so a shared instance has no aliasing to lose and is exempt from
    # the shared-mutable guard, like a tuple.
    a: int
    b: str


class _MyInt(int):
    def __repr__(self):
        return f"_MyInt({int(self)})"


class _ReducesToLoadFromBytes:
    # A non-Tensor wrapper whose reduce delegates to storage bytes -- emitting it as
    # source would embed raw bytes and require a pickle.loads-equivalent at exec time.
    def __reduce_ex__(self, protocol):
        return (torch.storage._load_from_bytes, (b"\x00\x01\x02",))


def _ss_new(cls):
    return cls.__new__(cls)


class _StateSetterObj:
    # __reduce_ex__ returns the protocol-5 6-tuple form whose last element is a
    # state_setter; _rebuild cannot apply it, so emission must reject this object.
    def __reduce_ex__(self, protocol):
        return (_ss_new, (type(self),), {"x": 1}, None, None, lambda obj, state: None)


def _make_holder(value):
    return _Holder(value)


class _Holder:
    # Reduces to a top-level factory call (the generic-callable branch of
    # _emit_via_reduce: func is neither copyreg.__newobj__ nor __newobj_ex__).
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return (_make_holder, (self.value,))

    def __eq__(self, other):
        return isinstance(other, _Holder) and self.value == other.value


class _NewObjEx:
    # __getnewargs_ex__ makes __reduce_ex__(2) use copyreg.__newobj_ex__ with non-empty
    # kwargs, exercising that branch of _emit_via_reduce.
    def __new__(cls, a, b):
        obj = object.__new__(cls)
        obj.a = a
        obj.b = b
        return obj

    def __getnewargs_ex__(self):
        return ((self.a,), {"b": self.b})

    def __eq__(self, other):
        return isinstance(other, _NewObjEx) and self.a == other.a and self.b == other.b


class _SlotObj:
    # __slots__ only -> __reduce_ex__(2) yields a (dict_state, slot_state) 2-tuple state,
    # exercising _rebuild's slotstate branch (setattr per slot) -- distinct from the plain
    # __dict__.update branch _NewObjEx hits and the __setstate__ branch a Shard hits.
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, _SlotObj) and self.x == other.x


@dataclasses.dataclass(eq=False)
class _EqFalseDC:
    # eq=False -> inherits object.__eq__ (identity), so the constructor round-trip check
    # (rebuilt == obj) can never pass; _emit_value must fall through to _emit_via_reduce
    # and reconstruct it rather than spuriously rejecting a pure value object.
    a: int
    b: str


class _NewObjNoClass:
    # Malformed protocol-2 reduce: copyreg.__newobj__ with an empty args tuple (no class).
    def __reduce_ex__(self, protocol):
        import copyreg

        return (copyreg.__newobj__, ())


class _NewObjExBadArity:
    # Malformed protocol-2 reduce: copyreg.__newobj_ex__ whose args is not a
    # (cls, args, kwargs) triple.
    def __reduce_ex__(self, protocol):
        import copyreg

        return (copyreg.__newobj_ex__, (type(self),))


class _NewObjExBadTriple:
    # copyreg.__newobj_ex__ with a structurally-3 tuple whose kwargs slot is not a dict;
    # the guard must reject the malformed CONTENTS, not just arity.
    def __reduce_ex__(self, protocol):
        import copyreg

        return (copyreg.__newobj_ex__, (type(self), (), None))


class _ReduceRaises:
    # __reduce_ex__ itself raises -> the "no usable reduce" reject path.
    def __reduce_ex__(self, protocol):
        raise RuntimeError("nope")


class _ShortReduce:
    # __reduce_ex__ returns a 1-tuple (len < 2) -> the "unsupported __reduce__ form" reject.
    def __reduce_ex__(self, protocol):
        return (object,)


class _ReduceNonTupleArgs:
    # __reduce_ex__ returns a non-tuple args field (the generic-callable branch); must
    # reject cleanly rather than leak a bare TypeError on iteration.
    def __reduce_ex__(self, protocol):
        return (list, 5)


class TestSourceEmit(TestCase):
    # Unit coverage of the source-emission helpers: every _emit_value branch round-trips
    # (or raises on a non-source-expressible leaf), so the standalone artifact stays
    # auditable and pickle.loads-free.
    def test_none_and_exact_builtin_scalars(self):
        self.assertEqual(_emit(None), ("None", []))
        self.assertEqual(_emit(True), ("True", []))
        self.assertEqual(_emit(42), ("42", []))
        self.assertEqual(_emit(3.5), ("3.5", []))
        self.assertEqual(_emit("ab"), ("'ab'", []))
        self.assertEqual(_emit(b"xy"), ("b'xy'", []))
        self.assertEqual(_emit(bytearray(b"xy")), ("bytearray(b'xy')", []))
        self.assertEqual(_emit(complex(1, 2)), ("(1+2j)", []))

    def test_non_finite_float_and_complex(self):
        self.assertEqual(_emit(float("inf")), ("float('inf')", []))
        self.assertEqual(_emit(float("-inf")), ("float('-inf')", []))
        self.assertEqual(_emit(float("nan")), ("float('nan')", []))
        self.assertTrue(math.isnan(_roundtrip(float("nan"))))
        self.assertEqual(
            _emit(complex(float("inf"), 0.0))[0], "complex(float('inf'), 0.0)"
        )
        rt = _roundtrip(complex(float("inf"), 0.0))
        self.assertTrue(math.isinf(rt.real) and rt.imag == 0.0)

    def test_int_subclass_falls_through_repr(self):
        # An int subclass with a constructor-style __repr__ must NOT take the exact-type
        # repr branch (which would emit a NameError / lose its type); it round-trips via
        # the reduce path and keeps its type.
        rt = _roundtrip(_MyInt(5))
        self.assertEqual(rt, 5)
        self.assertIs(type(rt), _MyInt)

    def test_torch_scalar_singletons(self):
        self.assertEqual(_emit(torch.float32), ("torch.float32", ["import torch"]))
        self.assertEqual(_emit(torch.strided), ("torch.strided", ["import torch"]))
        self.assertEqual(
            _emit(torch.contiguous_format),
            ("torch.contiguous_format", ["import torch"]),
        )
        self.assertIs(_roundtrip(torch.float32), torch.float32)

    def test_torch_device_size(self):
        self.assertEqual(
            _emit(torch.device("cpu")), ("torch.device('cpu')", ["import torch"])
        )
        self.assertEqual(_roundtrip(torch.device("cpu")), torch.device("cpu"))
        self.assertEqual(
            _emit(torch.Size([2, 3])), ("torch.Size([2, 3])", ["import torch"])
        )
        self.assertEqual(_roundtrip(torch.Size([2, 3])), torch.Size([2, 3]))

    def test_importable_class_function_module(self):
        self.assertIs(_roundtrip(torch.nn.Linear), torch.nn.Linear)
        self.assertIs(_roundtrip(math), math)
        self.assertEqual(_emit(math), ("math", ["import math"]))

    def test_importable_rejects_lambda_and_local(self):
        with self.assertRaisesRegex(NotImplementedError, "local definition"):
            _emit(lambda x: x)

        def _local():
            pass

        with self.assertRaisesRegex(NotImplementedError, "local definition"):
            _emit(_local)

    def test_enums(self):
        self.assertIs(_roundtrip(_Color.RED), _Color.RED)
        self.assertIs(_roundtrip(_Color.BLUE), _Color.BLUE)
        # IntEnum: repr is "<_IColor.RED: 1>" (invalid source), so it must take the enum
        # branch, not the repr branch.
        self.assertIs(_roundtrip(_IColor.RED), _IColor.RED)

    def test_combined_flag_emits_by_value(self):
        # A combined Flag member (A | B) has no single member name; emitting by name would
        # produce "Type.None" (a SyntaxError). It must reconstruct by value instead.
        combined = _Flag.A | _Flag.B
        self.assertEqual(_roundtrip(combined), combined)
        self.assertIs(_roundtrip(_Flag.A), _Flag.A)  # singleton still by-name
        self.assertEqual(_roundtrip(_Flag(0)), _Flag(0))  # empty flag

    def test_functools_partial(self):
        # _roundtrip rebuilds the partial object itself; invoking it then applies the
        # baked func/args/keywords.
        p = _roundtrip(functools.partial(_PlainDC, b="z"))
        self.assertEqual(p(a=1), _PlainDC(1, "z"))
        _expr, imports = _emit(functools.partial(int, "10", base=2))
        self.assertIn("import functools", imports)
        self.assertEqual(_roundtrip(functools.partial(int, "10", base=2))(), 2)

    def test_containers(self):
        self.assertEqual(_emit((1, 2)), ("(1, 2)", []))
        self.assertEqual(_emit((1,)), ("(1,)", []))
        self.assertEqual(_emit(()), ("()", []))
        self.assertEqual(_emit([1, 2]), ("[1, 2]", []))
        self.assertEqual(_emit([]), ("[]", []))
        self.assertEqual(_emit({"a": 1}), ("{'a': 1}", []))
        self.assertEqual(_roundtrip({"k": [1, 2, (3,)]}), {"k": [1, 2, (3,)]})

    def test_namedtuple(self):
        rt = _roundtrip(_Point(1, 2))
        self.assertEqual(rt, _Point(1, 2))
        self.assertIs(type(rt), _Point)

    def test_set_canonical_ordering_is_deterministic(self):
        # Set iteration order is not byte-stable across processes; emission sorts to a
        # canonical order so the artifact is reproducible.
        self.assertEqual(_emit({3, 1, 2})[0], "set([1, 2, 3])")
        self.assertEqual(_emit(frozenset({3, 1, 2}))[0], "frozenset([1, 2, 3])")
        self.assertEqual(_emit({3, 1, 2})[0], _emit({2, 3, 1})[0])
        # Sorting is by EMITTED SOURCE (a lexicographic string sort), not numeric, so a
        # multi-digit int set orders "1" < "10" < "2" -- still deterministic. Lock that
        # order so the canonicalization key can't silently regress to numeric/iteration.
        self.assertEqual(_emit({1, 2, 10})[0], "set([1, 10, 2])")
        # Unorderable elements (int vs str) also sort by emitted source; a numeric sort
        # would raise a TypeError here.
        self.assertEqual(_emit({1, "a"})[0], "set(['a', 1])")
        self.assertEqual(_emit({1, "a", "b"})[0], _emit({"b", 1, "a"})[0])

    def test_plain_dataclass_round_trips(self):
        rt = _roundtrip(_PlainDC(1, "z"))
        self.assertEqual(rt, _PlainDC(1, "z"))

    def test_dataclass_with_derived_post_init_round_trips(self):
        rt = _roundtrip(_DerivedDC(5))
        self.assertEqual(rt, _DerivedDC(5))
        self.assertEqual(rt.derived, 10)

    def test_stateful_dataclass_is_rejected(self):
        # A non-init field mutated to a value the constructor cannot reproduce makes the
        # rebuilt instance compare unequal: emitting only the init fields would silently
        # drop that state, so it must raise.
        obj = _StatefulDC(5)
        obj.extra = 99
        with self.assertRaisesRegex(NotImplementedError, "does not round-trip"):
            _emit(obj)

    def test_live_tensor_and_storage_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "live Tensor"):
            _emit(torch.zeros(2))
        with self.assertRaisesRegex(NotImplementedError, "live UntypedStorage"):
            _emit(torch.zeros(3).untyped_storage())
        # A TypedStorage (the legacy .storage() type) is the third arm of the same
        # live-storage guard: baking it would embed raw bytes and need pickle.loads.
        with self.assertRaisesRegex(NotImplementedError, "live TypedStorage"):
            _emit(torch.zeros(2).storage())

    def test_symbolic_symint_rejected(self):
        # A SymInt with no concrete value cannot be baked: precompile specializes to
        # static shapes, so a symbolic size has no source literal. Build an unbacked
        # symint (maybe_as_int() is None) and confirm the static-shapes reject fires.
        symint = ShapeEnv().create_unbacked_symint()
        self.assertIsNone(symint.node.maybe_as_int())
        static = "specializes to static shapes"
        with self.assertRaisesRegex(NotImplementedError, static):
            emit_value(symint, set())

    def test_symbolic_view_metadata_rejected(self):
        # SymIntEqByExpr wraps a sympy expr that must be a concrete integer to bake; a
        # symbolic one (its .val is non-Integer) hits the same static-shapes reject. The
        # constructor coerces ints to sympy.Integer, so feed it a symbolic SymInt to get
        # a non-Integer .val (sympy.Symbol cannot be passed directly -- it is coerced).
        obj = SymIntEqByExpr(ShapeEnv().create_unbacked_symint())
        self.assertFalse(getattr(obj.val, "is_Integer", False))
        static = "specializes to static shapes"
        with self.assertRaisesRegex(NotImplementedError, static):
            emit_value(obj, set())

    def test_view_meta_sequence_round_trips(self):
        # The view-replay-recipe branches (a ViewMeta via as_tuple, the ViewMetaSequence
        # via its _from_parts factory, and the concrete SymIntEqByExpr bake inside its
        # MetadataKey) are the headline metadata this module reconstructs. Build a
        # ViewMetaSequence straight from parts (no live FunctionalTensor needed) and
        # confirm it emits a _from_parts expression that round-trips to an equal recipe.
        from torch._C import _functionalization as _F
        from torch._functorch._aot_autograd.functional_utils import (
            MetadataKey,
            ViewMetaSequence,
        )

        view_meta = _F.resize__ViewMeta((True, [3, 4, 5]))
        meta = MetadataKey(
            size=(SymIntEqByExpr(2),),
            layout=torch.strided,
            is_sparse=False,
            stride=(SymIntEqByExpr(1),),
            storage_offset=SymIntEqByExpr(0),
            is_conj=False,
            is_neg=False,
        )
        vms = ViewMetaSequence._from_parts([view_meta], meta)
        expr, _imports = _emit(vms)
        self.assertIn("ViewMetaSequence._from_parts(", expr)
        self.assertIn("ViewMeta(", expr)
        rt = _roundtrip(vms)
        self.assertIsInstance(rt, ViewMetaSequence)
        self.assertEqual(rt.metadata, meta)
        self.assertEqual([v.as_tuple() for v in rt.sequence], [view_meta.as_tuple()])

    def test_symbolic_symbool_symfloat_rejected(self):
        # SymBool / SymFloat get the same static-shapes treatment as SymInt: a still-symbolic
        # value cannot be baked and raises the clear static-shapes message (rather than
        # falling into the reduce path with an opaque ValueRanges error).
        shape_env = ShapeEnv()
        static = "specializes to static shapes"
        sym_bool = shape_env.create_unbacked_symbool()
        self.assertIsNone(sym_bool.node.maybe_as_bool())
        with self.assertRaisesRegex(NotImplementedError, static):
            emit_value(sym_bool, set())
        sym_float = shape_env.create_unbacked_symfloat()
        self.assertIsNone(sym_float.node.maybe_as_float())
        with self.assertRaisesRegex(NotImplementedError, static):
            emit_value(sym_float, set())

    def test_concrete_symbool_symfloat_baked(self):
        # A SymBool / SymFloat that folds to a concrete value IS bakeable (mirroring SymInt),
        # emitted as the literal -- previously these fell through and were wrongly rejected.
        s = ShapeEnv().create_unbacked_symint()
        sym_bool = (s - s) == 0
        self.assertIs(sym_bool.node.maybe_as_bool(), True)
        self.assertEqual(_roundtrip(sym_bool), True)
        sym_float = s * 0.0 + 2.5
        self.assertEqual(sym_float.node.maybe_as_float(), 2.5)
        self.assertEqual(_roundtrip(sym_float), 2.5)

    def test_reduce_to_load_from_bytes_rejected(self):
        # Regression: the previous guard compared the __reduce_ex__ METHOD against
        # _load_from_bytes and never fired; the callable is the reduce RESULT's func.
        # A wrapper whose reduce is _load_from_bytes must be rejected, not silently emit
        # raw bytes + a pickle.loads-equivalent.
        with self.assertRaisesRegex(NotImplementedError, "_load_from_bytes"):
            _emit(_ReducesToLoadFromBytes())

    def test_state_setter_reduce_rejected(self):
        # Regression: a protocol-5 6-tuple reduce carries a state_setter _rebuild cannot
        # apply; emitting _rebuild(base, state) would install state via the wrong
        # mechanism, so reject it.
        with self.assertRaisesRegex(NotImplementedError, "state_setter"):
            _emit(_StateSetterObj())

    def test_self_referential_container_rejected(self):
        a = [1, 2]
        a.append(a)
        with self.assertRaisesRegex(NotImplementedError, "self-referential"):
            _emit(a)
        d: dict = {}
        d["self"] = d
        with self.assertRaisesRegex(NotImplementedError, "self-referential"):
            _emit(d)

    def test_shared_mutable_rejected(self):
        # A mutable object reached twice (not on the recursion path -- as siblings) would
        # be emitted as two independent literals, silently dropping the shared-identity
        # aliasing (mutating one reconstructed copy would no longer affect the other). Fail
        # loud instead, consistent with the module's other reject-don't-mis-bake guards.
        a = [1, 2]
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([a, a])
        d = {"k": 1}
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit((d, d))
        s = {1, 2}
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([s, s])
        # A bytearray is emitted as a repr literal but is mutable, so a shared one is
        # rejected too rather than emitted as two independent literals.
        ba = bytearray(b"xy")
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([ba, ba])
        # An opaque reduce value object shared across positions is caught the same way,
        # via _emit_via_reduce's guard rather than the container guard.
        h = _Holder(3)
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([h, h])
        # A shared non-frozen dataclass is mutable too: emitting it as two constructor
        # calls would drop the aliasing, so the dataclass branch also routes it here.
        dc = _PlainDC(1, "z")
        with self.assertRaisesRegex(NotImplementedError, "shared mutable"):
            _emit([dc, dc])

    def test_shared_frozen_dataclass_is_not_a_cycle(self):
        # A frozen dataclass is immutable, so a shared instance has no aliasing to lose and
        # emits fine as two independent literals -- exempt from the shared-mutable guard.
        fdc = _FrozenDC(1, "z")
        expr, _ = _emit([fdc, fdc])
        self.assertIn("_FrozenDC", expr)
        rt = _roundtrip([fdc, fdc])
        self.assertEqual(rt, [_FrozenDC(1, "z"), _FrozenDC(1, "z")])

    def test_repeated_leaf_is_not_a_cycle(self):
        # The cycle guard tracks only ancestors on the recursion path, not siblings, so a
        # leaf repeated across sibling positions is not a false cycle -- for a scalar...
        self.assertEqual(_emit([0, 0, 0]), ("[0, 0, 0]", []))
        # ...and for a shared IMMUTABLE object repeated as siblings (only shared MUTABLE
        # objects are rejected; an immutable tuple has no aliasing to lose).
        inner = (1, 2)
        self.assertEqual(_emit([inner, inner]), ("[(1, 2), (1, 2)]", []))

    def test_container_subclass_rejected(self):
        # A builtin-container subclass must not be silently downcast to its base type; it
        # falls through to _emit_via_reduce, whose listitems guard rejects it.
        with self.assertRaisesRegex(NotImplementedError, "container subclass"):
            _emit(_MyList([1, 2]))

    def test_emit_via_reduce_round_trips_opaque_object(self):
        # An opaque value object with a copyreg.__newobj__ reduce + dict state rebuilds via
        # cls.__new__ + _rebuild, emitted as source (no pickle.loads).
        try:
            from torch.distributed.tensor.placement_types import Shard
        except Exception:
            self.skipTest("DTensor placement_types unavailable")
        rt = _roundtrip(Shard(2))
        self.assertEqual(rt, Shard(2))

    def test_emit_via_reduce_generic_callable(self):
        # An object whose reduce is a plain top-level factory (not copyreg.__newobj__)
        # rebuilds via that factory call, emitted as source.
        rt = _roundtrip(_Holder(7))
        self.assertEqual(rt, _Holder(7))

    def test_emit_via_reduce_newobj_ex(self):
        # __getnewargs_ex__ drives the copyreg.__newobj_ex__ branch (cls.__new__ with
        # positional + keyword args) plus _rebuild for the dict state.
        rt = _roundtrip(_NewObjEx(1, b=2))
        self.assertEqual(rt, _NewObjEx(1, b=2))

    def test_emit_via_reduce_slots_state(self):
        # A __slots__ object reduces to a (dict, slots) 2-tuple state, exercising _rebuild's
        # slotstate branch. _SlotObj.__new__ does NOT set x, so the round-trip is correct
        # only if _rebuild applies the slotstate -- making that branch load-bearing here.
        rt = _roundtrip(_SlotObj(5))
        self.assertEqual(rt, _SlotObj(5))
        self.assertEqual(rt.x, 5)

    def test_eq_false_dataclass_round_trips_via_reduce(self):
        # An eq=False dataclass has identity __eq__, so the constructor round-trip check can
        # never pass; _emit_value must fall through to _emit_via_reduce and reconstruct it
        # (as source) instead of rejecting a pure value object. Assert by fields, since ==
        # is identity for this type.
        obj = _EqFalseDC(7, "z")
        rt = _roundtrip(obj)
        self.assertIs(type(rt), _EqFalseDC)
        self.assertEqual((rt.a, rt.b), (7, "z"))

    def test_newobj_reduce_without_class_rejected(self):
        # copyreg.__newobj__ with an empty args tuple has no class to construct; reject
        # cleanly (NotImplementedError) rather than raising a bare IndexError.
        with self.assertRaisesRegex(NotImplementedError, "no class argument"):
            _emit(_NewObjNoClass())

    def test_newobj_ex_reduce_bad_arity_rejected(self):
        # copyreg.__newobj_ex__ whose args is not a (cls, args, kwargs) triple must reject
        # cleanly rather than raising a bare ValueError on unpack.
        with self.assertRaisesRegex(NotImplementedError, "cls, args, kwargs"):
            _emit(_NewObjExBadArity())

    def test_newobj_ex_reduce_bad_triple_contents_rejected(self):
        # A structurally-3 tuple whose args/kwargs slots are the wrong types must also
        # reject cleanly, not leak a bare AttributeError/TypeError deeper in reconstruction.
        with self.assertRaisesRegex(NotImplementedError, "cls, args, kwargs"):
            _emit(_NewObjExBadTriple())

    def test_reduce_raises_rejected(self):
        # When __reduce_ex__ itself raises, the object has no usable reduce; surface a clean
        # NotImplementedError naming the cause.
        with self.assertRaisesRegex(NotImplementedError, "no usable reduce"):
            _emit(_ReduceRaises())

    def test_short_reduce_form_rejected(self):
        # A reduce tuple of len < 2 is an unsupported form; reject cleanly.
        with self.assertRaisesRegex(NotImplementedError, "unsupported __reduce__ form"):
            _emit(_ShortReduce())

    def test_reduce_non_tuple_args_rejected(self):
        # A reduce whose args field is not a tuple must reject cleanly (uniform with the
        # __newobj__ / __newobj_ex__ guards) rather than leak a bare TypeError.
        with self.assertRaisesRegex(NotImplementedError, "args field is not a tuple"):
            _emit(_ReduceNonTupleArgs())

    def test_emit_importable_rejects_non_round_tripping(self):
        # torch.add is a builtin whose __qualname__ does not round-trip via importlib.
        with self.assertRaisesRegex(NotImplementedError, "does not.*round-trip"):
            _emit_importable(torch.add, set())


if __name__ == "__main__":
    run_tests()
