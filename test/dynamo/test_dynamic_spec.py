# Owner(s): ["module: dynamo"]

import inspect

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.fx.experimental._config as _fx_experimental_config
from torch._dynamo.decorators import mark_static, mark_unbacked, maybe_mark_dynamic
from torch._dynamo.dynamic_spec import (
    DictSpec,
    IntSpec,
    IntSpecType,
    ListSpec,
    ObjectSpec,
    ParamsSpec,
    ShapesSpec,
    TensorSpec,
)
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfTorchDynamo,
)


def _tensor_placeholder_shape(gm):
    """Return the shape of the first tensor-typed placeholder in ``gm``."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ev = node.meta.get("example_value")
            if isinstance(ev, torch.Tensor):
                return ev.shape
    raise AssertionError("no tensor placeholder found")


# Applies per-dim IntSpec to a tensor through ``mark_*`` on each call.
# ``shape_spec`` must be a ``TensorSpec`` (or ``None`` to skip).
def _apply_intspec_to_tensor(tensor, shape_spec):
    if not isinstance(shape_spec, TensorSpec):
        return
    for idx, spec in enumerate(shape_spec):
        if spec is None:
            continue
        if spec._type is IntSpecType.STATIC:
            mark_static(tensor, idx)
        elif spec._type is IntSpecType.BACKED:
            maybe_mark_dynamic(tensor, idx)
        elif spec._type is IntSpecType.UNBACKED:
            mark_unbacked(tensor, idx)


def _compile_with_dynamic_shapes(fn, dynamic_shapes, **compile_kwargs):
    """Compile ``fn`` and apply ``dynamic_shapes`` specs on every call.

    On each call, the wrapper inspects ``dynamic_shapes`` and calls the
    appropriate ``mark_static`` / ``maybe_mark_dynamic`` / ``mark_unbacked``
    on each tensor argument before forwarding to the compiled function.
    """

    compiled = torch.compile(fn, **compile_kwargs)
    sig = inspect.signature(fn)

    @torch._dynamo.disable
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, shape_spec in dynamic_shapes.items():
            if name in bound.arguments:
                arg = bound.arguments[name]
                if isinstance(arg, torch.Tensor):
                    _apply_intspec_to_tensor(arg, shape_spec)
        return compiled(*bound.args, **bound.kwargs)

    return wrapper


class TestIntSpecConstruction(TestCase):
    """Construction via the classmethod factories.

    Field reads use the private slots (``s._name``, ``s._min``, etc.) since
    the fluent setters are write-only.
    """

    def test_static(self):
        s = IntSpec.static("x", value=10)
        self.assertEqual(s._name, "x")
        self.assertEqual(s._type, IntSpecType.STATIC)
        self.assertEqual(s._value, 10)

    def test_static_no_value(self):
        s = IntSpec.static()
        self.assertEqual(s._type, IntSpecType.STATIC)
        self.assertIsNone(s._value)
        # ``name=None`` auto-fills with a stable per-instance handle.
        self.assertTrue(s._name.startswith("_intspec_static_"))

    def test_anonymous_specs_have_distinct_names(self):
        # Two anonymous specs of the same mode get different auto-names so
        # they can be distinguished in error messages / logs.
        a = IntSpec.static()
        b = IntSpec.static()
        self.assertNotEqual(a._name, b._name)

    def test_backed(self):
        s = IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        self.assertEqual(s._name, "batch")
        self.assertEqual(s._type, IntSpecType.BACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)
        self.assertEqual(s._guarding_hint, 32)

    def test_unbacked(self):
        s = IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        self.assertEqual(s._type, IntSpecType.UNBACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 2048)
        self.assertEqual(s._optimization_hint, 512)

    def test_type_required_on_init(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec("x")  # no type kwarg
        # Python-generated message; format pinned for 3.10+.
        self.assertEqual(
            str(cm.exception),
            "IntSpec.__init__() missing 1 required positional argument: 'type'",
        )

    def test_type_not_none(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec("x", type=None)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec type must be an IntSpecType, got None",
        )

    def test_type_as_positional_arg(self):
        s = IntSpec("batch", IntSpecType.BACKED, min=1, max=64)
        self.assertEqual(s._name, "batch")
        self.assertEqual(s._type, IntSpecType.BACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)

    def test_static_with_positional_int_rejected(self):
        # ``IntSpec.static(10)`` would silently bind 10 to ``name``. Must
        # fail with a clear redirect to the kwarg form.
        with self.assertRaises(TypeError) as cm:
            IntSpec.static(10)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.static(value=10))",
        )

    def test_backed_with_positional_int_rejected(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec.backed(5)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.backed(guarding_hint=5))",
        )

    def test_unbacked_with_positional_int_rejected(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec.unbacked(5)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.unbacked(optimization_hint=5))",
        )

    def test_name_wrong_type_on_init(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec(10, IntSpecType.STATIC)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.static(value=10))",
        )

    def test_repr(self):
        # Anonymous: name is auto-generated; check shape via prefix since
        # the trailing id-hex is process-dependent.
        anon = repr(IntSpec.static())
        self.assertTrue(anon.startswith("IntSpec(name='_intspec_static_"))
        self.assertIn("type=STATIC", anon)
        # Named + value.
        self.assertEqual(
            repr(IntSpec.static("x", value=10)),
            "IntSpec(name='x', type=STATIC, value=10)",
        )
        # BACKED with full set of fields.
        self.assertEqual(
            repr(IntSpec.backed("batch", min=1, max=64, guarding_hint=32)),
            "IntSpec(name='batch', type=BACKED, min=1, max=64, guarding_hint=32)",
        )
        # UNBACKED with full set of fields.
        self.assertEqual(
            repr(IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)),
            "IntSpec(name='seq', type=UNBACKED, min=1, max=2048, optimization_hint=512)",
        )


class TestIntSpecTypeImmutable(TestCase):
    """Only ``type`` is pinned. All other fields can be reassigned via the
    fluent setters. This class verifies the type-pin guard, slot discipline
    (no new attrs, cannot delete), and that the mutable fields can in fact
    be updated."""

    def test_type_reassign_via_private_slot_rejected(self):
        # The backing ``_type`` slot is locked: our ``__setattr__``
        # guard catches reassignment. Reads through ``_type`` are fine.
        s = IntSpec.backed("x")
        with self.assertRaises(AttributeError) as cm:
            s._type = IntSpecType.STATIC
        self.assertEqual(
            str(cm.exception),
            "IntSpec type is immutable; cannot reassign",
        )
        self.assertIs(s._type, IntSpecType.BACKED)

    def test_no_fluent_type_reset(self):
        # IntSpec has no instance method that reassigns type. The mode-named
        # factories are classmethods: calling one "on an instance" returns a
        # fresh IntSpec and does not mutate the original.
        s = IntSpec.static("x")
        new = IntSpec.backed("x")
        self.assertIs(s._type, IntSpecType.STATIC)
        self.assertIs(new._type, IntSpecType.BACKED)
        self.assertIsNot(s, new)

    def test_cannot_add_new_attribute(self):
        # __slots__ rejects unknown attributes at the slot layer; message
        # is Python-built and version-dependent, so we only assert the type.
        s = IntSpec.static("x")
        with self.assertRaises(AttributeError):
            s.brand_new_field = 1  # type: ignore[attr-defined]

    def test_cannot_assign_to_method_name(self):
        # ``value`` / ``min`` / ``max`` / ``guarding_hint`` /
        # ``optimization_hint`` are method names backed by ``_value`` /
        # ``_min`` / etc. slots. Direct attribute assignment under those
        # public names has no matching slot and is rejected.
        s = IntSpec.static("x", value=10)
        for attr in ("value", "min", "max", "guarding_hint", "optimization_hint"):
            with self.assertRaises(AttributeError):
                setattr(s, attr, 20)

    def test_cannot_delete_attribute(self):
        s = IntSpec.backed("x", guarding_hint=32)
        with self.assertRaises(AttributeError) as cm:
            del s._guarding_hint
        self.assertEqual(
            str(cm.exception),
            "IntSpec attribute '_guarding_hint' cannot be deleted",
        )


class TestIntSpecFluent(TestCase):
    """Fluent setters (``name`` / ``min`` / ``max`` / ``value`` /
    ``guarding_hint`` / ``optimization_hint``) mutate the receiver in place
    and return ``self`` for chaining. Each chain is equivalent to the
    kwargs-only factory form."""

    def test_setter_returns_self_for_chaining(self):
        base = IntSpec.unbacked("seq")
        chained = base.min(1).max(2048)
        # Mutated in place, same object back.
        self.assertIs(base, chained)
        self.assertEqual(base._min, 1)
        self.assertEqual(base._max, 2048)

    def test_fluent_preserves_existing_fields(self):
        s = IntSpec.backed("batch", min=1, guarding_hint=32).max(64)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)
        self.assertEqual(s._guarding_hint, 32)

    def test_failed_setter_rolls_back_per_mode(self):
        # Per-mode rejection (guarding_hint on STATIC) must roll the slot
        # back so the spec stays in a consistent state.
        s = IntSpec.static("x", value=10)
        with self.assertRaises(ValueError):
            s.guarding_hint(99)
        self.assertIsNone(s._guarding_hint)
        self.assertEqual(s._value, 10)

    def test_failed_setter_rolls_back_min_max(self):
        # Cross-field check (min <= max) also rolls back.
        s = IntSpec.backed("x", min=10, max=20)
        with self.assertRaises(ValueError):
            s.max(5)
        self.assertEqual(s._max, 20)


@instantiate_parametrized_tests
class TestIntSpecRejectionRules(TestCase):
    """Per-mode field-rejection rules, exercised via both entry points.

    Each rule fires inside :meth:`IntSpec._validate`, reached from two
    user-visible paths:

    - ``init``: direct kwargs at the raw constructor, e.g.
      ``IntSpec("x", IntSpecType.STATIC, guarding_hint=10)``.
    - ``fluent``: fluent setter chained off a factory, e.g.
      ``IntSpec.static("x").guarding_hint(10)``.

    Both paths produce the same error message because the constructor
    and the fluent setter both call ``_validate`` (the fluent setter via
    ``_try_set``). Parametrizing the entry-point axis keeps a single
    source of truth per rule.
    """

    @parametrize("entry", ["init", "fluent"])
    @parametrize(
        "IntSpecFactory,mode",
        [
            (IntSpec.backed, IntSpecType.BACKED),
            (IntSpec.unbacked, IntSpecType.UNBACKED),
        ],
    )
    def test_value_rejected_on_non_static(self, IntSpecFactory, mode, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, value=42)  # noqa: E731
        else:
            ctor = lambda: IntSpecFactory("x").value(42)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(str(cm.exception), "value is only valid for STATIC IntSpec")

    @parametrize("entry", ["init", "fluent"])
    @parametrize(
        "IntSpecFactory,mode",
        [
            (IntSpec.static, IntSpecType.STATIC),
            (IntSpec.unbacked, IntSpecType.UNBACKED),
        ],
    )
    def test_guarding_hint_rejected_on_non_backed(self, IntSpecFactory, mode, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, guarding_hint=10)  # noqa: E731
        else:
            ctor = lambda: IntSpecFactory("x").guarding_hint(10)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception), "guarding_hint is only valid for BACKED IntSpec"
        )

    @parametrize("entry", ["init", "fluent"])
    @parametrize(
        "IntSpecFactory,mode",
        [
            (IntSpec.static, IntSpecType.STATIC),
            (IntSpec.backed, IntSpecType.BACKED),
        ],
    )
    def test_optimization_hint_rejected_on_non_unbacked(
        self, IntSpecFactory, mode, entry
    ):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, optimization_hint=10)  # noqa: E731
        else:
            ctor = lambda: IntSpecFactory("x").optimization_hint(10)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            "optimization_hint is only valid for UNBACKED IntSpec",
        )

    @parametrize("entry", ["init", "fluent"])
    @parametrize("field", ["min", "max"])
    def test_min_max_rejected_on_static(self, field, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", IntSpecType.STATIC, **{field: 1})  # noqa: E731
        else:
            ctor = lambda: getattr(IntSpec.static("x"), field)(1)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC",
        )

    @parametrize("IntSpecFactory", [IntSpec.backed, IntSpec.unbacked])
    def test_IntSpec_min_greater_than_max_rejected(self, IntSpecFactory):
        with self.assertRaises(ValueError) as cm:
            IntSpecFactory("x", min=100, max=1)
        self.assertEqual(
            str(cm.exception),
            "min must be <= max, got min=100, max=1",
        )

    # Each case: (field, bad_value, factory_name, expected_type_name).
    # ``factory_name`` is the factory whose mode allows the field — we
    # exercise the type check on the valid mode since mode-rejection is
    # already covered by the four tests above.
    _TYPE_CASES = [
        ("value", "10", "static", "str"),
        ("min", 1.5, "backed", "float"),
        ("max", "64", "backed", "str"),
        ("guarding_hint", True, "backed", "bool"),
        ("optimization_hint", 1.0, "unbacked", "float"),
    ]

    @parametrize("entry", ["init", "fluent"])
    @parametrize("field,bad,factory_name,type_name", _TYPE_CASES)
    def test_field_type_rejected(self, field, bad, factory_name, type_name, entry):
        factory = getattr(IntSpec, factory_name)
        if entry == "init":
            ctor = lambda: factory("x", **{field: bad})  # noqa: E731
        else:
            ctor = lambda: getattr(factory("x"), field)(bad)  # noqa: E731
        with self.assertRaises(TypeError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            f"IntSpec.{field} must be int or None, got {type_name}",
        )


class TestTensorSpecConstruction(TestCase):
    """Construction and list-like interface."""

    def test_basic(self):
        ts = TensorSpec(3)
        self.assertEqual(ts._dim, 3)
        for spec in ts:
            self.assertIsNone(spec)

    def test_zero_dim(self):
        ts = TensorSpec(0)
        self.assertEqual(ts._dim, 0)

    def test_list_construction(self):
        static_spec = IntSpec.static(value=10)
        backed_spec = IntSpec.backed(min=1)
        ts = TensorSpec([static_spec, None, backed_spec])
        self.assertEqual(
            repr(ts),
            f"TensorSpec([{repr(static_spec)}, None, {repr(backed_spec)}])",
        )

    def test_dict_construction(self):
        backed_spec = IntSpec.backed("h")
        static_spec = IntSpec.static()
        ts = TensorSpec({0: backed_spec, 2: static_spec})
        self.assertEqual(
            repr(ts), f"TensorSpec([{repr(backed_spec)}, None, {repr(static_spec)}])"
        )

    def test_unsupported_input_type_rejected(self):
        with self.assertRaisesRegex(TypeError, "expects int / list / tuple / dict"):
            TensorSpec("not a spec")  # type: ignore[arg-type]

    def test_getitem_setitem(self):
        ts = TensorSpec(2)
        spec = IntSpec.backed("batch", min=1)
        ts[0] = spec
        self.assertIs(ts[0], spec)
        self.assertIsNone(ts[1])

    def test_iter(self):
        ts = TensorSpec(2)
        spec = IntSpec.static(value=5)
        ts[0] = spec
        items = list(ts)
        self.assertEqual(len(items), 2)
        self.assertIs(items[0], spec)
        self.assertIsNone(items[1])

    def test_index_out_of_range(self):
        ts = TensorSpec(2)
        with self.assertRaises(IndexError):
            ts[5]

    def test_sparse_set(self):
        ts = TensorSpec(4)
        ts.dim(1, IntSpec.backed("h"))
        ts.dim(3, IntSpec.backed("w"))
        self.assertIsNone(ts[0])
        self.assertIsNotNone(ts[1])
        self.assertIsNone(ts[2])
        self.assertIsNotNone(ts[3])


class TestTensorSpecCompile(TestCase):
    """TensorSpec + compile integration via torch.compile(shapes_spec=...)."""

    def test_tensorspec_list_init_recompile_progression(self):
        """TensorSpec built from a list: dim 0 BACKED absorbs shape changes;
        dim 1 STATIC forces recompile per distinct value.

        Final graph has a backed SymInt at dim 0 and a concrete int at dim 1.
        """
        backend = EagerAndRecordGraphs()
        ts = TensorSpec([IntSpec.backed("batch"), IntSpec.static()])
        fn = torch.compile(
            lambda x: x + 1,
            backend=backend,
            shapes_spec=ShapesSpec(params=ParamsSpec().arg("x", ts)),
        )

        fn(torch.randn(4, 3))
        self.assertEqual(len(backend.graphs), 1)

        # Vary dim 0 (BACKED absorbs it → no new compile).
        fn(torch.randn(8, 3))
        fn(torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        # Vary dim 1 (STATIC pins it → each distinct value recompiles).
        fn(torch.randn(16, 5))
        self.assertEqual(len(backend.graphs), 2)
        fn(torch.randn(16, 7))
        self.assertEqual(len(backend.graphs), 3)

        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)
        self.assertIsInstance(shape[1], int)
        self.assertEqual(shape[1], 7)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_tensorspec_none_entry_inherits_automatic_dynamic(self):
        """A ``None`` entry doesn't mark the dim — it falls through to the
        existing default policy: dynamo's automatic-dynamic
        (``torch._dynamo.config.automatic_dynamic_shapes``, default True).
        That means specialize on first call, promote to backed on first
        variation.
        """
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x + 1,
            backend=backend,
            shapes_spec=ShapesSpec(
                params=ParamsSpec(
                    named_args={"x": TensorSpec([IntSpec.backed("batch"), None])}
                )
            ),
        )

        fn(torch.randn(4, 3))  # specialize dim 1=3; BACKED dim 0
        self.assertEqual(len(backend.graphs), 1)

        fn(torch.randn(8, 3))  # dim 0 absorbed; dim 1 same → cache hit
        self.assertEqual(len(backend.graphs), 1)

        fn(torch.randn(8, 5))  # dim 1 promotes to backed → recompile
        self.assertEqual(len(backend.graphs), 2)

        fn(torch.randn(16, 7))  # dim 0 + dim 1 both backed → cache hit
        self.assertEqual(len(backend.graphs), 2)


class TestIntSpecCompile(TestCase):
    """IntSpec + torch.compile integration via shapes_spec parameter."""

    def test_static_graph_has_concrete_shape(self):
        """STATIC dim appears as a concrete int in the captured graph; each
        distinct shape yields a new graph."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x + 1,
            backend=backend,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.static()}))
            ),
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        fn(torch.randn(12, 3))
        fn(torch.randn(4, 3))  # cache hit

        self.assertEqual(len(backend.graphs), 3)
        for gm in backend.graphs:
            shape = _tensor_placeholder_shape(gm)
            self.assertIsInstance(shape[0], int)

    def test_backed_graph_has_backed_symbol(self):
        """BACKED dim appears as a backed SymInt in the final graph."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.backed("batch")}))
            ),
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))

        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)

    def test_unbacked_graph_has_unbacked_symbol(self):
        """UNBACKED dim appears as an unbacked SymInt; single compile covers all shapes."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.unbacked("batch")}))
            ),
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))

        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertGreater(len(free_unbacked_symbols(shape[0])), 0)

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_raises_dde_on_branching(self):
        """A function that branches on size(0) must raise a data-dependent
        error when that dim is marked UNBACKED."""

        def fn(x):
            if x.size(0) > 5:
                return x + 1
            return x - 1

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.unbacked()}))
            ),
        )
        with self.assertRaises(GuardOnDataDependentSymNode) as cm:
            compiled(torch.randn(10, 3))
        free_syms = cm.exception.cond.free_symbols
        self.assertEqual(len(free_syms), 1)
        (sym,) = free_syms
        self.assertTrue(
            str(sym).startswith("u"),
            msg=f"expected unbacked symbol (u-prefix), got {sym!r}",
        )

    def test_backed_branching_bounded_recompiles(self):
        """BACKED + branching on size(0) > 8 should produce exactly 2 compiles."""
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x.sum(0 if x.size()[0] > 8 else 1),
            backend=cnt,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.backed("batch")}))
            ),
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 2)

    def test_backed_zero_one_specialization(self):
        """BACKED symbols are specialized at 0 and 1 unconditionally."""
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x + 1,
            backend=cnt,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.backed("batch")}))
            ),
        )
        for n in [0, 1, 2, 4, 8]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 3)

    def test_backed_equality_branching(self):
        """BACKED + Python branch on ``==``: point specialization."""
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x + 1 if x.size()[0] == 3 else x - 1,
            backend=cnt,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.backed("batch")}))
            ),
        )
        for n in [3, 4, 5, 6]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 2)

    def test_static_precedence_over_dynamic_true(self):
        """IntSpec.static() must win over compile(dynamic=True)."""
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x + 1,
            backend=cnt,
            dynamic=True,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.static()}))
            ),
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        self.assertEqual(cnt.frame_count, 2)

    def test_backed_precedence_over_dynamic_false(self):
        """IntSpec.backed() must win over compile(dynamic=False)."""
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=cnt,
            dynamic=False,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.backed("batch")}))
            ),
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 1)


@skipIfTorchDynamo()
class TestIntSpecMinMax(TestCase):
    """Tests for min/max constraints via torch._check."""

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_with_min_no_dde(self):
        """UNBACKED dim with min=1: branching on `x.size(0) > 0` should NOT
        raise DDE since torch._check proves it's always True."""
        fn = torch.compile(
            lambda x: x + 1 if x.size(0) > 0 else x - 1,
            backend="eager",
            fullgraph=True,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.unbacked(min=1)}))
            ),
        )
        # Should NOT raise DDE — min=1 proves size(0) > 0
        result = fn(torch.randn(5, 3))
        self.assertEqual(result.shape, (5, 3))

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_with_max_no_dde(self):
        """UNBACKED dim with max=10: branching on `x.size(0) <= 10` should NOT
        raise DDE since torch._check proves it's always True."""
        fn = torch.compile(
            lambda x: x + 1 if x.size(0) <= 10 else x - 1,
            backend="eager",
            fullgraph=True,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.unbacked(max=10)}))
            ),
        )
        # Should NOT raise DDE — max=10 proves size(0) <= 10
        result = fn(torch.randn(5, 3))
        self.assertEqual(result.shape, (5, 3))

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_without_bounds_raises_dde(self):
        """UNBACKED dim WITHOUT min: branching on `x.size(0) > 0` SHOULD raise DDE."""

        fn = torch.compile(
            lambda x: x + 1 if x.size(0) > 0 else x - 1,
            backend="eager",
            fullgraph=True,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("x", TensorSpec({0: IntSpec.unbacked()}))
            ),
        )
        with self.assertRaises(GuardOnDataDependentSymNode):
            fn(torch.randn(5, 3))

    def test_backed_with_min_max_statically_known(self):
        """BACKED dim with min=8, max=64: statically_known_true can prove bounds."""
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        def fn(x):
            size = x.size(0)
            assert statically_known_true(size >= 8)  # noqa: S101
            assert statically_known_true(size <= 64)  # noqa: S101
            return x.sum(0)

        ts = TensorSpec({0: IntSpec.backed(min=8, max=64)})
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec=ShapesSpec(params=ParamsSpec(named_args={"x": ts})),
        )
        result = compiled(torch.randn(16, 3))
        self.assertEqual(result.shape, (3,))

    def test_backed_scalar_int_with_min_max_statically_known(self):
        """BACKED scalar int arg with min=1, max=100: statically_known_true proves bounds."""
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        def fn(x, n):
            assert statically_known_true(n >= 1)  # noqa: S101
            assert statically_known_true(n <= 100)  # noqa: S101
            return x + n

        N = IntSpec.backed(min=1, max=100)
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec=ShapesSpec(params=ParamsSpec().arg("n", N)),
        )
        result = compiled(torch.randn(4), 42)
        self.assertEqual(result.shape, (4,))

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_scalar_int_with_min_no_dde(self):
        """UNBACKED scalar int with min=1: branching on `n > 0` should NOT raise DDE."""

        def fn(x, n):
            if n > 0:
                return x + n
            return x - n

        N = IntSpec.unbacked(min=1)
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            shapes_spec=ShapesSpec(params=ParamsSpec().arg("n", N)),
        )
        result = compiled(torch.randn(4), 5)
        self.assertEqual(result.shape, (4,))

    def test_backed_tensor_dim_hint_override(self):
        """BACKED dim with guarding_hint=32: the symbol's hint in shape_env
        should be 32 regardless of the actual input size."""
        backend = EagerAndRecordGraphs()
        ts = TensorSpec([IntSpec.backed(guarding_hint=32), None])
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            shapes_spec=ShapesSpec(params=ParamsSpec().arg("x", ts)),
        )
        fn(torch.randn(8, 3))
        # The graph's placeholder should have the override in var_to_hint_override
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        expr = shape[0].node.expr
        shape_env = shape[0].node.shape_env
        self.assertIn(expr, shape_env.var_to_hint_override)
        self.assertEqual(shape_env.var_to_hint_override[expr], 32)

    def test_unbacked_tensor_dim_optimization_hint(self):
        """UNBACKED dim with optimization_hint=64: var_to_hint_override should
        record the optimization hint."""
        backend = EagerAndRecordGraphs()
        ts = TensorSpec({0: IntSpec.unbacked(optimization_hint=64)})
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            shapes_spec=ShapesSpec(params=ParamsSpec().arg("x", ts)),
        )
        fn(torch.randn(8, 3))
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        # For unbacked, the hint override should be set
        expr = shape[0].node.expr
        shape_env = shape[0].node.shape_env
        self.assertIn(expr, shape_env.var_to_hint_override)
        self.assertEqual(shape_env.var_to_hint_override[expr], 64)

    def test_backed_scalar_int_hint_override(self):
        """BACKED scalar int with guarding_hint=100: the symbol's hint should be 100."""
        backend = EagerAndRecordGraphs()
        N = IntSpec.backed(guarding_hint=100)

        def fn(x, n):
            return x + n

        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec=ShapesSpec(params=ParamsSpec().arg("n", N)),
        )
        compiled(torch.randn(4), 42)
        # Find the scalar int placeholder in the graph
        for node in backend.graphs[0].graph.nodes:
            if node.op == "placeholder":
                ev = node.meta.get("example_value")
                if isinstance(ev, torch.SymInt):
                    self.assertEqual(ev.node.hint, 100)
                    break

    def test_unbacked_scalar_int_hint_override(self):
        """UNBACKED scalar int with optimization_hint=256: var_to_hint_override should record it."""
        backend = EagerAndRecordGraphs()
        N = IntSpec.unbacked(optimization_hint=256)

        def fn(x, n):
            return x + n

        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec=ShapesSpec(params=ParamsSpec().arg("n", N)),
        )
        compiled(torch.randn(4), 42)
        for node in backend.graphs[0].graph.nodes:
            if node.op == "placeholder":
                ev = node.meta.get("example_value")
                if isinstance(ev, torch.SymInt):
                    expr = ev.node.expr
                    shape_env = ev.node.shape_env
                    self.assertIn(expr, shape_env.var_to_hint_override)
                    self.assertEqual(shape_env.var_to_hint_override[expr], 256)
                    break


@skipIfTorchDynamo()
class TestParameterSpecCompile(TestCase):
    """End-to-end: ``nn.Parameter`` as a top-level arg honors the spec.

    Parameters are normally force-marked static in ``wrap_tensor`` and
    routed to ``register_attr_or_module`` (graph attribute), bypassing
    ``_automatic_dynamic`` where the spec is consulted. The integration
    consults ``lookup_spec_from_dynamo_source`` first and bypasses the
    static-mark / graph-attribute paths when a ``TensorSpec`` is
    provided, letting the Parameter flow through the dynamic-shape
    path.
    """

    def test_parameter_top_level_dim_backed(self):
        def fn(p):
            return p + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg("p", TensorSpec([IntSpec.backed("h"), None]))
            ),
        )

        compiled(torch.nn.Parameter(torch.randn(4, 3)))
        self.assertEqual(len(backend.graphs), 1)

        # Spec drove ``_automatic_dynamic`` — dim 0 is a backed SymInt.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)


class TestObjectSpec(TestCase):
    """Construction, dict-like access, and recursion."""

    def test_empty(self):
        os = ObjectSpec()
        self.assertEqual(len(os), 0)

    def test_dict_construction(self):
        spec = IntSpec.backed("batch")
        os = ObjectSpec({"x": spec, "n": IntSpec.static()})
        self.assertEqual(len(os), 2)
        self.assertIn("x", os)
        self.assertIs(os["x"], spec)

    def test_fluent_field(self):
        spec_x = TensorSpec([IntSpec.backed("batch")])
        spec_n = IntSpec.unbacked("n")
        os = ObjectSpec().field("x", spec_x).field("n", spec_n)
        self.assertEqual(len(os), 2)
        self.assertIs(os["x"], spec_x)
        self.assertIs(os["n"], spec_n)

    def test_setitem(self):
        os = ObjectSpec()
        spec = IntSpec.backed("batch")
        os["x"] = spec
        self.assertIs(os["x"], spec)

    def test_iter_and_items(self):
        spec_x = IntSpec.backed("batch")
        spec_n = IntSpec.static()
        os = ObjectSpec({"x": spec_x, "n": spec_n})
        self.assertEqual(list(os), ["x", "n"])
        self.assertEqual(list(os.items()), [("x", spec_x), ("n", spec_n)])

    def test_repr(self):
        # Single entry — pin a name so the repr is deterministic
        # (anonymous IntSpec auto-generates a process-dependent name).
        os = ObjectSpec({"x": IntSpec.static("x")})
        self.assertEqual(
            repr(os),
            "ObjectSpec({.x: IntSpec(name='x', type=STATIC)})",
        )

    def test_repr_multiple_entries(self):
        # Multiple entries preserve insertion order.
        os = ObjectSpec({"x": IntSpec.static("x"), "n": IntSpec.backed("n")})
        self.assertEqual(
            repr(os),
            "ObjectSpec({"
            ".x: IntSpec(name='x', type=STATIC), "
            ".n: IntSpec(name='n', type=BACKED)"
            "})",
        )

    def test_repr_nested_objectspec(self):
        inner = ObjectSpec().field("weight", TensorSpec([IntSpec.backed("h")]))
        outer = ObjectSpec().field("model", inner).field("x", IntSpec.static("x"))
        self.assertEqual(
            repr(outer),
            "ObjectSpec({"
            ".model: ObjectSpec({"
            ".weight: TensorSpec([IntSpec(name='h', type=BACKED)])}), "
            ".x: IntSpec(name='x', type=STATIC)"
            "})",
        )

    def test_recursive_nesting(self):
        # Recursion in the data class — nested ``ObjectSpec`` is accepted.
        # Initial integration only consumes top-level entries; nested
        # entries remain inert at compile time but the class itself
        # carries them through.
        inner = ObjectSpec({"weight": TensorSpec([IntSpec.backed("h")])})
        outer = ObjectSpec({"model": inner, "n": IntSpec.backed("n")})
        self.assertIs(outer["model"], inner)


class TestObjectSpecLookup(TestCase):
    """``lookup_spec_from_dynamo_source`` walks a dynamo ``Source`` chain
    against an ``ObjectSpec`` and returns the leaf at that path."""

    def _local(self, name):
        from torch._dynamo.source import LocalSource

        return LocalSource(name, is_input=True)

    def _attr(self, base, member):
        from torch._dynamo.source import AttrSource

        return AttrSource(base, member)

    def test_attr_source_descends_into_objectspec(self):
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        leaf = TensorSpec([IntSpec.backed("h")])
        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg("model", ObjectSpec().field("weight", leaf))
        )
        result = lookup_spec_from_dynamo_source(
            self._attr(self._local("model"), "weight"), shapes_spec
        )
        self.assertIs(result, leaf)

    def test_nested_objectspec_walk(self):
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        leaf = TensorSpec([IntSpec.backed("h")])
        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg(
                "model",
                ObjectSpec().field("inner", ObjectSpec().field("weight", leaf)),
            )
        )
        # model.inner.weight
        src = self._attr(self._attr(self._local("model"), "inner"), "weight")
        self.assertIs(lookup_spec_from_dynamo_source(src, shapes_spec), leaf)

    def test_missing_attr_returns_none(self):
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg(
                "model", ObjectSpec().field("weight", TensorSpec(2))
            )
        )
        # model.bias is not in the spec → None
        self.assertIsNone(
            lookup_spec_from_dynamo_source(
                self._attr(self._local("model"), "bias"), shapes_spec
            )
        )

    def test_attr_against_non_objectspec_returns_none(self):
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        # Top-level spec is a TensorSpec but source asks for an attr — mismatch.
        shapes_spec = ShapesSpec(params=ParamsSpec().arg("x", TensorSpec(2)))
        self.assertIsNone(
            lookup_spec_from_dynamo_source(
                self._attr(self._local("x"), "weight"), shapes_spec
            )
        )

    def test_local_source_root_still_works(self):
        # v0 behavior preserved: bare LocalSource → top-level spec.
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        spec_obj = ObjectSpec().field("weight", TensorSpec(2))
        shapes_spec = ShapesSpec(params=ParamsSpec().arg("model", spec_obj))
        self.assertIs(
            lookup_spec_from_dynamo_source(self._local("model"), shapes_spec),
            spec_obj,
        )


@skipIfTorchDynamo()
class TestObjectSpecCompile(TestCase):
    """End-to-end: ``shapes_spec`` routes through ``ObjectSpec`` so a
    ``TensorSpec`` reaches the dynamo builder for an attribute-accessed
    tensor (``obj.w`` where ``obj`` is a function arg)."""

    def test_attr_tensor_dim_backed(self):
        # Plain container — avoids nn.Module wrapping subtleties.
        class Container:
            def __init__(self, w):
                self.w = w

        def fn(obj):
            return obj.w + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg(
                    "obj",
                    ObjectSpec().field("w", TensorSpec([IntSpec.backed("h"), None])),
                )
            ),
        )

        compiled(Container(torch.randn(4, 3)))
        self.assertEqual(len(backend.graphs), 1)

        # Different dim 0 size — backed absorbs it, no recompile.
        compiled(Container(torch.randn(8, 3)))
        self.assertEqual(len(backend.graphs), 1)

        # Captured weight placeholder has a backed SymInt at dim 0.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)


class TestObjectSpecMatch(TestCase):
    """``ObjectSpec.match(obj)`` auto-derives a default spec scaffold
    matching ``obj``'s structure."""

    def test_match_tensor(self):
        x = torch.randn(2, 3, 4)
        spec = ObjectSpec.match(x)
        self.assertIsInstance(spec, TensorSpec)
        self.assertEqual(spec._dim, 3)
        # All dims default-policy.
        for i in range(3):
            self.assertIsNone(spec[i])

    def test_match_int(self):
        spec = ObjectSpec.match(5)
        self.assertIsInstance(spec, IntSpec)
        self.assertIs(spec._type, IntSpecType.STATIC)

    def test_match_bool_rejected(self):
        with self.assertRaisesRegex(TypeError, "bool"):
            ObjectSpec.match(True)

    def test_match_dict(self):
        result = ObjectSpec.match({"x": torch.randn(2, 3), "n": 4})
        self.assertIsInstance(result, DictSpec)
        self.assertIsInstance(result["x"], TensorSpec)
        self.assertEqual(result["x"]._dim, 2)
        self.assertIsInstance(result["n"], IntSpec)

    def test_match_list_and_tuple_preserve_type(self):
        as_list = ObjectSpec.match([torch.randn(2), torch.randn(3, 4)])
        self.assertIsInstance(as_list, list)
        self.assertEqual(as_list[0]._dim, 1)
        self.assertEqual(as_list[1]._dim, 2)

        as_tuple = ObjectSpec.match((torch.randn(2),))
        self.assertIsInstance(as_tuple, tuple)
        self.assertEqual(as_tuple[0]._dim, 1)

    def test_match_nn_module(self):
        # Single-level module: parameters become attr-keyed TensorSpec
        # entries; the result mirrors the module's attribute layout.
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))
                self.bias = torch.nn.Parameter(torch.randn(4))

        spec = ObjectSpec.match(Linear())
        self.assertIsInstance(spec, ObjectSpec)
        self.assertIn("weight", spec)
        self.assertIn("bias", spec)
        self.assertEqual(spec["weight"]._dim, 2)
        self.assertEqual(spec["bias"]._dim, 1)

    def test_match_nested_nn_module(self):
        # Nested module: child modules nest as ObjectSpec under .attr;
        # parameters of the outer module appear alongside.
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.bias = torch.nn.Parameter(torch.randn(4))

        spec = ObjectSpec.match(Outer())
        self.assertIsInstance(spec["inner"], ObjectSpec)
        self.assertIsInstance(spec["inner"]["weight"], TensorSpec)
        self.assertIsInstance(spec["bias"], TensorSpec)

    def test_match_nn_module_with_buffer(self):
        class WithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(2))
                self.register_buffer("running_mean", torch.zeros(2))

        spec = ObjectSpec.match(WithBuffer())
        self.assertIn("weight", spec)
        self.assertIn("running_mean", spec)
        self.assertEqual(spec["running_mean"]._dim, 1)

    def test_match_unknown_type_rejected(self):
        with self.assertRaisesRegex(TypeError, "cannot derive a spec for"):
            ObjectSpec.match("not a spec source")

    def test_match_tensor_repr(self):
        spec = ObjectSpec.match(torch.randn(2, 3))
        self.assertEqual(repr(spec), "TensorSpec([None, None])")

    def test_match_int_repr(self):
        # ``match`` produces an anonymous ``IntSpec.static()`` whose
        # auto-generated ``_name`` is process-dependent; check structure
        # instead of an exact repr.
        spec = ObjectSpec.match(5)
        self.assertIsInstance(spec, IntSpec)
        self.assertIs(spec._type, IntSpecType.STATIC)

    def test_match_dict_repr(self):
        # ``match`` produces a ``DictSpec`` containing an anonymous
        # ``IntSpec`` whose auto-name is process-dependent; structural
        # check on the leaves rather than exact repr.
        result = ObjectSpec.match({"x": torch.randn(2), "n": 4})
        self.assertIsInstance(result, DictSpec)
        self.assertEqual(repr(result["x"]), "TensorSpec([None])")
        self.assertIsInstance(result["n"], IntSpec)
        self.assertIs(result["n"]._type, IntSpecType.STATIC)

    def test_match_list_repr(self):
        result = ObjectSpec.match([torch.randn(2), torch.randn(3, 4)])
        self.assertEqual(
            repr(result),
            "[TensorSpec([None]), TensorSpec([None, None])]",
        )

    def test_match_nn_module_repr(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))
                self.bias = torch.nn.Parameter(torch.randn(4))

        spec = ObjectSpec.match(Linear())
        self.assertEqual(
            repr(spec),
            "ObjectSpec({"
            ".weight: TensorSpec([None, None]), "
            ".bias: TensorSpec([None])"
            "})",
        )

    def test_match_nested_nn_module_repr(self):
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.bias = torch.nn.Parameter(torch.randn(4))

        spec = ObjectSpec.match(Outer())
        self.assertEqual(
            repr(spec),
            "ObjectSpec({"
            ".inner: ObjectSpec({"
            ".weight: TensorSpec([None, None])}), "
            ".bias: TensorSpec([None])"
            "})",
        )


class TestDictSpec(TestCase):
    """``DictSpec`` — dict-keyed entries, MappingKey paths."""

    def test_empty(self):
        ds = DictSpec()
        self.assertEqual(len(ds), 0)

    def test_dict_construction(self):
        spec_x = TensorSpec([IntSpec.backed("batch")])
        spec_n = IntSpec.backed("n")
        ds = DictSpec({"x": spec_x, "n": spec_n})
        self.assertEqual(len(ds), 2)
        self.assertIs(ds["x"], spec_x)
        self.assertIs(ds["n"], spec_n)

    def test_fluent_entry(self):
        spec_x = TensorSpec([IntSpec.backed("batch")])
        ds = DictSpec().entry("x", spec_x)
        self.assertIs(ds["x"], spec_x)

    def test_setitem(self):
        ds = DictSpec()
        spec = IntSpec.backed("n")
        ds["n"] = spec
        self.assertIs(ds["n"], spec)

    def test_iter_and_items(self):
        spec_x = IntSpec.static()
        spec_n = IntSpec.backed("n")
        ds = DictSpec({"x": spec_x, "n": spec_n})
        self.assertEqual(list(ds), ["x", "n"])
        self.assertEqual(list(ds.items()), [("x", spec_x), ("n", spec_n)])

    def test_repr(self):
        ds = DictSpec({"x": IntSpec.static("x")})
        self.assertEqual(
            repr(ds),
            "DictSpec({'x': IntSpec(name='x', type=STATIC)})",
        )


class TestDictSpecLookup(TestCase):
    """``lookup_spec_from_dynamo_source`` walks a ``GetItemSource(str)``
    chain against a ``DictSpec`` and returns the matching leaf."""

    def _local(self, name):
        from torch._dynamo.source import LocalSource

        return LocalSource(name, is_input=True)

    def _item(self, base, index):
        from torch._dynamo.source import GetItemSource

        return GetItemSource(base, index)

    def test_str_key_descends_into_dictspec(self):
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        leaf = TensorSpec([IntSpec.backed("h")])
        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg("kwargs", DictSpec({"x": leaf}))
        )
        result = lookup_spec_from_dynamo_source(
            self._item(self._local("kwargs"), "x"), shapes_spec
        )
        self.assertIs(result, leaf)

    def test_missing_key_returns_none(self):
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg("kwargs", DictSpec({"x": TensorSpec(2)}))
        )
        self.assertIsNone(
            lookup_spec_from_dynamo_source(
                self._item(self._local("kwargs"), "missing"), shapes_spec
            )
        )

    def test_int_key_against_dictspec_returns_none(self):
        # DictSpec only matches str keys; int subscript falls through.
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg("kwargs", DictSpec({"x": TensorSpec(2)}))
        )
        self.assertIsNone(
            lookup_spec_from_dynamo_source(
                self._item(self._local("kwargs"), 0), shapes_spec
            )
        )

    def test_objectspec_attr_then_dictspec_item(self):
        # Mixed walk: obj.config["batch_size"]
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source
        from torch._dynamo.source import AttrSource

        leaf = IntSpec.backed("bs")
        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg(
                "obj",
                ObjectSpec().field("config", DictSpec({"batch_size": leaf})),
            )
        )
        src = self._item(AttrSource(self._local("obj"), "config"), "batch_size")
        self.assertIs(lookup_spec_from_dynamo_source(src, shapes_spec), leaf)


@skipIfTorchDynamo()
class TestDictSpecCompile(TestCase):
    """End-to-end: ``shapes_spec`` routes through ``DictSpec`` so a
    ``TensorSpec`` reaches the dynamo builder for a dict-keyed tensor
    arg."""

    def test_dict_value_dim_backed(self):
        def fn(d):
            return d["x"] + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg(
                    "d",
                    DictSpec({"x": TensorSpec([IntSpec.backed("h"), None])}),
                )
            ),
        )

        compiled({"x": torch.randn(4, 3)})
        self.assertEqual(len(backend.graphs), 1)

        # Different dim 0 — backed absorbs it, no recompile.
        compiled({"x": torch.randn(8, 3)})
        self.assertEqual(len(backend.graphs), 1)

        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)


class TestListSpec(TestCase):
    """``ListSpec`` — positional entries, SequenceKey paths."""

    def test_int_construction(self):
        ls = ListSpec(3)
        self.assertEqual(len(ls), 3)
        for spec in ls:
            self.assertIsNone(spec)

    def test_list_construction(self):
        spec_a = TensorSpec([IntSpec.backed("a")])
        spec_b = IntSpec.static()
        ls = ListSpec([spec_a, None, spec_b])
        self.assertEqual(len(ls), 3)
        self.assertIs(ls[0], spec_a)
        self.assertIsNone(ls[1])
        self.assertIs(ls[2], spec_b)

    def test_tuple_construction(self):
        spec = TensorSpec([IntSpec.backed("a")])
        ls = ListSpec((spec,))
        self.assertEqual(len(ls), 1)
        self.assertIs(ls[0], spec)

    def test_dict_construction(self):
        spec_a = IntSpec.backed("a")
        spec_b = IntSpec.static()
        ls = ListSpec({0: spec_a, 2: spec_b})
        self.assertEqual(len(ls), 3)
        self.assertIs(ls[0], spec_a)
        self.assertIsNone(ls[1])
        self.assertIs(ls[2], spec_b)

    def test_fluent_index(self):
        spec = TensorSpec([IntSpec.backed("a")])
        ls = ListSpec(2).index(0, spec)
        self.assertIs(ls[0], spec)
        self.assertIsNone(ls[1])

    def test_setitem(self):
        ls = ListSpec(2)
        spec = IntSpec.static()
        ls[0] = spec
        self.assertIs(ls[0], spec)

    def test_iter(self):
        spec = IntSpec.static()
        ls = ListSpec([spec, None])
        items = list(ls)
        self.assertEqual(len(items), 2)
        self.assertIs(items[0], spec)
        self.assertIsNone(items[1])

    def test_repr(self):
        ls = ListSpec([IntSpec.static("x"), None])
        self.assertEqual(
            repr(ls),
            "ListSpec([IntSpec(name='x', type=STATIC), None])",
        )

    def test_unsupported_input_type_rejected(self):
        with self.assertRaisesRegex(TypeError, "expects int / list / tuple / dict"):
            ListSpec("not a spec")  # type: ignore[arg-type]


class TestListSpecLookup(TestCase):
    """``lookup_spec_from_dynamo_source`` walks ``GetItemSource(int)``
    against a ``ListSpec`` and returns the leaf at that position."""

    def _local(self, name):
        from torch._dynamo.source import LocalSource

        return LocalSource(name, is_input=True)

    def _item(self, base, index):
        from torch._dynamo.source import GetItemSource

        return GetItemSource(base, index)

    def test_int_index_descends_into_listspec(self):
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        leaf = TensorSpec([IntSpec.backed("h")])
        shapes_spec = ShapesSpec(params=ParamsSpec().arg("xs", ListSpec([leaf, None])))
        result = lookup_spec_from_dynamo_source(
            self._item(self._local("xs"), 0), shapes_spec
        )
        self.assertIs(result, leaf)

    def test_out_of_range_returns_none(self):
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg("xs", ListSpec([TensorSpec(2)]))
        )
        self.assertIsNone(
            lookup_spec_from_dynamo_source(
                self._item(self._local("xs"), 5), shapes_spec
            )
        )

    def test_str_key_against_listspec_returns_none(self):
        # ListSpec only matches int keys; str subscript falls through.
        from torch._dynamo.dynamic_spec import lookup_spec_from_dynamo_source

        shapes_spec = ShapesSpec(
            params=ParamsSpec().arg("xs", ListSpec([TensorSpec(2)]))
        )
        self.assertIsNone(
            lookup_spec_from_dynamo_source(
                self._item(self._local("xs"), "0"), shapes_spec
            )
        )


@skipIfTorchDynamo()
class TestListSpecCompile(TestCase):
    """End-to-end: ``shapes_spec`` routes through ``ListSpec`` so a
    ``TensorSpec`` reaches the dynamo builder for a positionally-indexed
    list arg."""

    def test_list_element_dim_backed(self):
        def fn(xs):
            return xs[0] + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            shapes_spec=ShapesSpec(
                params=ParamsSpec().arg(
                    "xs",
                    ListSpec([TensorSpec([IntSpec.backed("h"), None])]),
                )
            ),
        )

        compiled([torch.randn(4, 3)])
        self.assertEqual(len(backend.graphs), 1)

        # Different dim 0 — backed absorbs it, no recompile.
        compiled([torch.randn(8, 3)])
        self.assertEqual(len(backend.graphs), 1)

        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)


if __name__ == "__main__":
    run_tests()
