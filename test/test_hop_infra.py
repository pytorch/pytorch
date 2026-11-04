# Owner(s): ["module: higher order operators"]
import collections
import contextlib
import copy
import dataclasses
import importlib
import logging
import pkgutil
from collections.abc import Callable, Iterator, Mapping
from typing import Any, cast
from unittest import mock

import torch
from torch._higher_order_ops import user_kernel_hop
from torch._higher_order_ops.reinplace_trigger import ReInplaceTrigger
from torch._higher_order_ops.user_kernel_hop import (
    _snapshot,
    constant_key,
    ConstantRules,
    get_user_kernel_hop_info,
    KernelSideTable,
    register_user_kernel_hop_impls,
    UserKernelHOPInfo,
    UserKernelWrapperHOP,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.hop_db import (
    FIXME_hop_that_doesnt_have_opinfo_test_allowlist,
    hop_db,
)


def do_imports():
    for mod in pkgutil.walk_packages(
        torch._higher_order_ops.__path__, "torch._higher_order_ops."
    ):
        modname = mod.name
        importlib.import_module(modname)


do_imports()


@skipIfTorchDynamo("not applicable")
class TestHOPInfra(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_all_hops_have_opinfo(self):
        """All HOPs should have an OpInfo in torch/testing/_internal/hop_db.py"""
        from torch._ops import _higher_order_ops

        hops_that_have_op_info = {k.name for k in hop_db}
        all_hops = _higher_order_ops.keys()

        missing_ops = set()

        for op in all_hops:
            if (
                op not in hops_that_have_op_info
                and op not in FIXME_hop_that_doesnt_have_opinfo_test_allowlist
            ):
                missing_ops.add(op)

        self.assertTrue(
            len(missing_ops) == 0,
            lambda msg: f"{msg}\nMissing hop_db OpInfo entries for {missing_ops}, please add them to torch/testing/_internal/hop_db.py",
        )

    def test_hop_db_has_no_decorators_or_skips(self):
        """hop_db entries should not decide how tests are skipped or decorated."""
        offending_op_infos = [
            op.full_name for op in hop_db if op.decorators or op.skips
        ]

        self.assertEqual(
            offending_op_infos,
            [],
            msg=(
                "Move hop_db decorators/skips to the tests that need them "
                "instead of storing them on the OpInfo."
            ),
        )

    def test_all_hops_are_imported(self):
        """All HOPs should be listed in torch._higher_order_ops.__all__

        Some constraints (see test_testing.py::TestImports)
        - Sympy must be lazily imported
        - Dynamo must be lazily imported
        """
        imported_hops = torch._higher_order_ops.__all__
        registered_hops = torch._ops._higher_order_ops.keys()

        # Please don't add anything here.
        # We want to ensure that all HOPs are imported at "import torch" time.
        # It is bad if someone tries to access torch.ops.higher_order.cond
        # and it doesn't exist (this may happen if your HOP isn't imported at
        # "import torch" time).
        FIXME_ALLOWLIST = {
            "autograd_function_apply",
            "run_with_rng_state",
            "run_dtensor_rng_op",
            "graphsafe_run_with_rng_state",
            "map_impl",
            "_export_tracepoint",
            "run_and_save_rng_state",
            "map",
            "custom_function_call",
            "trace_wrapped",
            "triton_kernel_wrapper_functional",
            "triton_kernel_wrapper_mutation",
            "wrap",  # Really weird failure -- importing this causes Dynamo to choke on checkpoint
            "control_deps",  # Inductor-internal ordering HOP, not a public torch.ops.higher_order API
        }
        not_imported_hops = registered_hops - imported_hops
        not_imported_hops = not_imported_hops - FIXME_ALLOWLIST
        self.assertEqual(
            not_imported_hops,
            set(),
            msg="All HOPs must be listed under torch/_higher_order_ops/__init__.py's __all__.",
        )

    def test_imports_from_all_work(self):
        """All APIs listed in torch._higher_order_ops.__all__ must be importable"""
        stuff = torch._higher_order_ops.__all__
        for attr in stuff:
            getattr(torch._higher_order_ops, attr)

    def test_hop_deepcopy_preserves_identity(self):
        """HigherOrderOperators should be singletons - deepcopy should return the same object"""
        from torch._ops import _higher_order_ops

        for name, hop in _higher_order_ops.items():
            copied_hop = copy.deepcopy(hop)
            self.assertIs(
                hop,
                copied_hop,
                lambda msg: f"{msg}\ndeepcopy of HigherOrderOperator '{name}' should return the same object (singleton pattern)",
            )
            self.assertEqual(id(hop), id(copied_hop))

    def test_triton_kernel_wrapper_functional_fake_tensor_nested_kwargs(self):
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_functional,
        )
        from torch._subclasses.fake_tensor import (
            FakeTensorMode,
            is_fake_tensor,
            maybe_get_fake_mode,
        )

        mode = FakeTensorMode()
        with mode:
            fake_output = torch.empty_strided((2, 3), (3, 1))

        self.assertTrue(is_fake_tensor(fake_output))

        out = triton_kernel_wrapper_functional(
            kernel_idx=0,
            constant_args_idx=0,
            grid=[(1,)],
            tma_descriptor_metadata={},
            kwargs={"out_ptr": fake_output},
            tensors_to_clone=["out_ptr"],
        )

        self.assertEqual(set(out.keys()), {"out_ptr"})
        self.assertTrue(is_fake_tensor(out["out_ptr"]))
        self.assertIs(maybe_get_fake_mode(out["out_ptr"]), mode)
        self.assertEqual(out["out_ptr"].size(), fake_output.size())
        self.assertEqual(out["out_ptr"].stride(), fake_output.stride())

    def test_triton_kernel_wrapper_mutation_fake_tensor_nested_kwargs(self):
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_mutation,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode, is_fake_tensor

        mode = FakeTensorMode()
        with mode:
            fake_output = torch.empty_strided((2, 3), (3, 1))

        self.assertTrue(is_fake_tensor(fake_output))

        out = triton_kernel_wrapper_mutation(
            kernel_idx=0,
            constant_args_idx=0,
            grid=[(1,)],
            tma_descriptor_metadata={},
            kwargs={"out_ptr": fake_output},
        )

        self.assertIsNone(out)

    def test_kernel_side_table_instances_are_independent(self):
        # Each user-kernel DSL owns a KernelSideTable; their index spaces and
        # constant args must not leak into each other.
        from torch._higher_order_ops.user_kernel_hop import KernelSideTable

        a, b = KernelSideTable(), KernelSideTable()
        k0, k1 = object(), object()
        self.assertEqual(a.add_kernel(k0), 0)
        self.assertEqual(a.add_kernel(k0), 0)
        self.assertEqual(b.add_kernel(k1), 0)
        self.assertIs(a.get_kernel(0), k0)
        self.assertIs(b.get_kernel(0), k1)
        self.assertEqual(a.add_constant_args({"x": 1}), 0)
        self.assertEqual(b.add_constant_args({"y": 2}), 0)
        self.assertEqual(a.get_constant_args(0), {"x": 1})
        b.reset_table()
        self.assertIs(a.get_kernel(0), k0)
        with self.assertRaises(AssertionError):
            b.get_kernel(0)


class _ToyNumeric:
    """A DSL scalar value holding a Python value in ``.value``. Like a cutlass Numeric,
    ``==`` builds a DSL value rather than answering, so constant_key must key it by
    value (see _ToyRules)."""

    def __init__(self, value: object) -> None:
        self.value = value

    def __eq__(self, other: object) -> "_ToyNumeric":  # type: ignore[override]
        return _ToyNumeric(("==", self, other))

    def __hash__(self) -> int:
        return hash((type(self), self.value))


class _ToyInt32(_ToyNumeric):
    pass


class _ToyFloat32(_ToyNumeric):
    pass


class _ToyLayout:
    """A DSL value that is not a Numeric: a plain object, which constant_key keys by
    its attributes unless its DSL's rules claim it."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


# A DSL's types live in its own package: the toydsl package here.
for _cls in (_ToyNumeric, _ToyInt32, _ToyFloat32, _ToyLayout):
    _cls.__module__ = "toydsl.numeric"


class _ToyRules(ConstantRules):
    """The toy DSL's ConstantRules, as CuTeDSL's (D128128704) are for cutlass: the
    types of the toydsl package are its own, and one holding a Python scalar in
    ``.value`` (a Numeric) is keyed by that value."""

    def is_dsl_value(self, value: object) -> bool:
        module = getattr(type(value), "__module__", "") or ""
        return module == "toydsl" or module.startswith("toydsl.")

    def scalar_value(self, value: object) -> bool | int | float | None:
        inner = getattr(value, "value", None) if self.is_dsl_value(value) else None
        return inner if isinstance(inner, (bool, int, float)) else None


_TOY_RULES = _ToyRules()
# The defaults: every value is a plain Python one.
_PLAIN_RULES = ConstantRules()


@dataclasses.dataclass(frozen=True)
class _ToyKernel:
    """A side-table record: the part a JITFunction plays for Triton."""

    fn: Callable[..., None]
    mutates_args: object

    @property
    def name(self) -> str:
        return self.fn.__name__


def _toy_fn() -> None:
    pass


class _ToyKernelWrapper(UserKernelWrapperHOP):
    def __init__(self, name: str) -> None:
        super().__init__(name, cacheable=False)


@contextlib.contextmanager
def _toy_dsl(**options: Any) -> Iterator[UserKernelHOPInfo]:
    """A user-kernel DSL registered with register_user_kernel_hop_impls (``options``
    passed on) while the block runs; its HOPs and its registry entry are dropped after.
    Its side-table records are _ToyKernels: a Python function, called with the
    kernel's arguments, that writes the tensors its mutates_args names (every tensor
    for "unknown")."""
    from torch._ops import _higher_order_ops

    table: KernelSideTable[_ToyKernel] = KernelSideTable(constant_rules=_TOY_RULES)

    def launch(
        *, kernel_idx: int, constant_args_idx: int, kwargs: dict[str, Any]
    ) -> None:
        kernel = table.get_kernel(kernel_idx)
        kernel.fn(**table.kernel_kwargs(constant_args_idx, kwargs))

    def get_mutated_tensors(
        hop_kwargs: Mapping[str, Any], kwargs: dict[str, Any]
    ) -> list[str]:
        mutates_args = table.get_kernel(hop_kwargs["kernel_idx"]).mutates_args
        names = list(kwargs) if mutates_args == "unknown" else mutates_args
        return [name for name in names if isinstance(kwargs.get(name), torch.Tensor)]

    with (
        mock.patch.dict(_higher_order_ops),
        mock.patch.dict(user_kernel_hop._user_kernel_hops),
    ):
        yield register_user_kernel_hop_impls(
            _ToyKernelWrapper("toy_kernel_wrapper_mutation"),
            _ToyKernelWrapper("toy_kernel_wrapper_functional"),
            dsl="toy",
            display_name="ToyDSL",
            side_table=table,
            launch=launch,
            get_mutated_tensors=get_mutated_tensors,
            get_layout_constraint=lambda: "needs_fixed_stride_order",
            reinplace_trigger=ReInplaceTrigger.USER_KERNEL_OPS,
            serializable=False,
            **options,
        )


@skipIfTorchDynamo("not applicable")
class TestUserKernelHOPShared(TestCase):
    """What user_kernel_hop.py shares between the user-kernel DSLs, and the
    triton_kernel_wrap names and patch points code outside torch relies on."""

    hw_classification = HardwareClassification.GENERIC

    def test_side_table_records_and_dedupes_constant_args(self):
        table: KernelSideTable[_ToyKernel] = KernelSideTable(constant_rules=_TOY_RULES)
        idx = table.add_kernel(_ToyKernel(_toy_fn, ("out",)))
        self.assertEqual(idx, table.add_kernel(_ToyKernel(_toy_fn, ("out",))))
        self.assertNotEqual(idx, table.add_kernel(_ToyKernel(_toy_fn, "unknown")))
        self.assertIs(table.get_kernel(idx).fn, _toy_fn)

        def constants(**kwargs: object) -> int:
            return table.add_constant_args(kwargs)

        self.assertEqual(
            constants(a=[1, (2, 3)], b={"k": None}),
            constants(b={"k": None}, a=[1, (2, 3)]),
        )
        self.assertNotEqual(constants(a=[1, 2]), constants(a=(1, 2)))
        self.assertEqual(constants(c=_ToyInt32(3)), constants(c=_ToyInt32(3)))
        self.assertNotEqual(constants(c=_ToyInt32(3)), constants(c=_ToyInt32(4)))
        self.assertNotEqual(constants(c=_ToyInt32(3)), constants(c=_ToyFloat32(3)))
        # An unhashable constant that is no container is refused.
        with self.assertRaisesRegex(TypeError, "type bytearray is unhashable"):
            constants(o=bytearray(b"x"))
        # Without constant_rules (Triton's table) every call gets an index.
        plain: KernelSideTable[_ToyKernel] = KernelSideTable()
        self.assertNotEqual(
            plain.add_constant_args({"a": 1}), plain.add_constant_args({"a": 1})
        )

    def test_side_table_dedupes_by_content_and_snapshots(self):
        # The side-table checks of the CuTeDSL stack's test_side_table_dedupes_constant_args
        # (D128128704), on a table of its own: interchangeable constants share one
        # entry (see constant_key), which holds a snapshot of the constants it was
        # added with.
        Tile = collections.namedtuple("Tile", ["m", "n"])

        @dataclasses.dataclass(frozen=True)
        class FrozenCfg:
            alpha: float

        class Cfg:  # a plain config object: compared and hashed by identity
            def __init__(self, alpha: float) -> None:
                self.alpha = alpha

        table: KernelSideTable[_ToyKernel] = KernelSideTable(constant_rules=_TOY_RULES)
        idx = table.add_constant_args({"alpha": 2, "k": None})
        self.assertEqual(table.add_constant_args({"k": None, "alpha": 2}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": 2.0, "k": None}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": True, "k": None}), idx)

        def add(c: object) -> int:
            return table.add_constant_args({"c": c})

        # By type and content all the way down: inside containers and dataclasses
        # too, equal values of other types, and floats that differ only in sign, are
        # other constants.
        for a, b in (
            ([1, 2], (1, 2)),
            (Tile(1, 2), Tile(1.0, 2)),
            (Tile(1, 2), Tile(True, 2)),
            (Tile(0.0, 1), Tile(-0.0, 1)),
            (FrozenCfg(0.0), FrozenCfg(-0.0)),
        ):
            self.assertNotEqual(add(a), add(b))
        # Mutable values too (a namedtuple holding a list, a plain object), stored as
        # they were when added: mutating the caller's value changes neither the entry
        # nor the entry its old contents map to.
        tile, cfg = Tile([128], 64), Cfg(1.0)
        tiles, cfgs = add(tile), add(cfg)
        self.assertEqual((add(Tile([128], 64)), add(Cfg(1.0))), (tiles, cfgs))
        tile.m.append(32)
        cfg.alpha = 2.0
        self.assertEqual(table.get_constant_args(tiles)["c"], Tile([128], 64))
        self.assertEqual(table.get_constant_args(cfgs)["c"].alpha, 1.0)
        self.assertEqual((add(Tile([128], 64)), add(Cfg(1.0))), (tiles, cfgs))
        self.assertNotEqual(add(tile), tiles)
        self.assertNotEqual(add(cfg), cfgs)
        # A DSL Numeric is keyed by its value, so the entry holds a copy too.
        numeric = _ToyFloat32(2.0)
        numerics = add(numeric)
        numeric.value = 5.0
        self.assertEqual(add(_ToyFloat32(2.0)), numerics)
        self.assertEqual(table.get_constant_args(numerics)["c"].value, 2.0)

        # An object with __slots__ attributes, even one with a __dict__ too, is keyed
        # as itself: keyed by its __dict__, objects differing only in a slot would
        # share one entry.
        class SlotBase:
            __slots__ = ("alpha",)

        class SlotCfg(SlotBase):
            def __init__(self, alpha: float) -> None:
                self.alpha = alpha

        slot_cfgs = [table.get_constant_args(add(SlotCfg(a)))["c"] for a in (1.0, 2.0)]
        self.assertEqual([c.alpha for c in slot_cfgs], [1.0, 2.0])
        with self.assertRaisesRegex(TypeError, "type set is unhashable"):
            add({1, 2})

    def test_constant_key_refuses_cycles(self):
        # The walk checks of the CuTeDSL stack's test_cyclic_constants_are_refused
        # (D128128704): constant_key and _snapshot walk a constant's contents, so a
        # constant that refers to itself is refused with the cycle named, instead of
        # recursing until a RecursionError. An object, dict or list reached twice on
        # different paths is no cycle.
        class Cfg:
            def __init__(self, alpha: float, **attrs: object) -> None:
                self.alpha = alpha
                vars(self).update(attrs)

        @dataclasses.dataclass
        class Node:
            next: object = None

        cfg = Cfg(2.0, parent=Cfg(1.0))
        cfg.parent.child = cfg  # type: ignore[attr-defined]
        node = Node()
        node.next = Node(node)
        items: list[object] = [1.0]
        items.append(items)
        table: dict[str, object] = {}
        table["self"] = table
        keyed = Cfg(1.0)
        keyed.lookup = {keyed: 1}  # type: ignore[attr-defined]
        # A logger of its own manager: its manager's loggerDict holds it.
        log = logging.Manager(logging.RootLogger(logging.WARNING)).getLogger("cfg")
        cycles = (
            (cfg, "Cfg", r"\.parent\.child"),
            (node, "Node", r"\.next\.next"),
            ((0, Cfg(1.0, items=items)), "list", r"\[1\]"),
            (table, "dict", r"\['self'\]"),
            (keyed, "Cfg", r"\.lookup\.keys\(\)\[0\]"),
            (Cfg(1.0, log=log), "Logger", r"\.manager\.loggerDict\['cfg'\]"),
        )
        side_table: KernelSideTable[_ToyKernel] = KernelSideTable(
            constant_rules=_PLAIN_RULES
        )
        for value, name, path in cycles:
            msg = f"^a constant kernel argument of type {name} refers to itself "
            msg += f"through {path}$"
            for walk in (constant_key, _snapshot):
                with self.assertRaisesRegex(TypeError, msg):
                    walk(value, _PLAIN_RULES)
            with self.assertRaisesRegex(TypeError, msg):
                side_table.add_constant_args({"cfg": value})

        shared = Cfg(1.0, lookup={"k": [0.5]})
        twice = Cfg(2.0, a=shared, b=[shared], c=shared.lookup)  # type: ignore[attr-defined]
        snapshot = cast(Any, _snapshot(twice, _PLAIN_RULES))
        self.assertEqual(
            constant_key(snapshot, _PLAIN_RULES), constant_key(twice, _PLAIN_RULES)
        )
        self.assertTrue(all(c is not shared for c in (snapshot.a, snapshot.b[0])))
        self.assertIsNot(snapshot.c, shared.lookup)  # type: ignore[attr-defined]

    def test_constant_key_refuses_tensors(self):
        # Keyed by identity, a tensor inside a constant would stay alive with its
        # side-table entry: constant_key refuses one as a constant or in a list,
        # tuple, dict, dataclass field or plain-object attribute, naming where, and
        # so does a deduping side table (D128128704's
        # test_constants_holding_tensors_are_refused, without a DSL).
        @dataclasses.dataclass
        class Cfg:
            alpha: float
            weight: object = None

        class Holder:
            def __init__(self, w: object) -> None:
                self.w = w

        t = torch.ones(4)
        table: KernelSideTable[object] = KernelSideTable(constant_rules=_PLAIN_RULES)
        msg = "^a constant kernel argument holds a tensor{}: pass each tensor as its "
        msg += "own argument$"
        for value, where in (
            ([t], r" at \[0\]"),
            ((1.0, {"w": t}), r" at \[1\]\['w'\]"),
            ({t: 1.0}, r" at \.keys\(\)\[0\]"),
            (Cfg(2.0, weight=t), r" at \.weight"),
            (Holder(t), r" at \.w"),
            (t, ""),
        ):
            with self.assertRaisesRegex(TypeError, msg.format(where)):
                constant_key(value, _PLAIN_RULES)
            with self.assertRaisesRegex(TypeError, msg.format(where)):
                table.add_constant_args({"cfg": value})

    def test_constant_rules(self):
        # With the default rules, a DSL Numeric is keyed by the object, whose ==
        # answers with a DSL value (truthy): two different values would share a key.
        # A DSL's other values are plain objects, keyed by their attributes.
        self.assertTrue(
            constant_key(_ToyInt32(3), _PLAIN_RULES)
            == constant_key(_ToyInt32(4), _PLAIN_RULES)
        )
        layout = _ToyLayout((4, 8))
        self.assertEqual(
            constant_key(layout, _PLAIN_RULES),
            constant_key(_ToyLayout((4, 8)), _PLAIN_RULES),
        )

        # The DSL's rules key its Numerics by type and value, at any depth, and copy
        # them in a snapshot (an owner can still change a Numeric's value).
        def key(value: object) -> object:
            return constant_key(value, _TOY_RULES)

        self.assertEqual(key(_ToyInt32(3)), key(_ToyInt32(3)))
        self.assertNotEqual(key(_ToyInt32(3)), key(_ToyInt32(4)))
        self.assertNotEqual(key(_ToyInt32(3)), key(_ToyFloat32(3)))
        self.assertNotEqual(key([_ToyFloat32(0.5)]), key([0.5]))
        numeric = _ToyFloat32(0.5)
        copied = cast(Any, _snapshot(numeric, _TOY_RULES))
        self.assertIsNot(copied, numeric)
        self.assertEqual(copied.value, 0.5)
        # Its values that hold no Python scalar are keyed as themselves, never by
        # their attributes, and not copied.
        value = _ToyNumeric("s")
        self.assertIs(cast(tuple, key(value))[1], value)
        self.assertNotEqual(key(layout), key(_ToyLayout((4, 8))))
        self.assertIs(_snapshot(layout, _TOY_RULES), layout)

        # Only the DSL package's own types are its values: a class from a module
        # merely named like it is a plain object, keyed by its attributes, even
        # with a numeric .value.
        class Cfg:
            __module__ = "toydsl_configs"

            def __init__(self, value: object) -> None:
                self.value = value

        cfg = Cfg(1)
        self.assertEqual(key(cfg), key(Cfg(1)))
        self.assertNotEqual(key(cfg), key(Cfg(2)))
        copied = cast(Any, _snapshot(cfg, _TOY_RULES))
        self.assertIsNot(copied, cfg)
        self.assertEqual(copied.value, 1)

    def test_constant_key_floats_by_bits(self):
        # 0.0 == -0.0 (with equal hashes), but a kernel compiled for one does not
        # compute the other's results: floats are keyed by their bits, at any depth
        # and inside a DSL Numeric. Every NaN is one key.
        def key(value: object) -> object:
            return constant_key(value, _TOY_RULES)

        for a, b in (
            (0.0, -0.0),
            ((0.0,), (-0.0,)),
            ([1, {"k": 0.0}], [1, {"k": -0.0}]),
            (_ToyFloat32(0.0), _ToyFloat32(-0.0)),
        ):
            with self.subTest(a=a):
                self.assertNotEqual(key(a), key(b))
        self.assertEqual(key(float("nan")), key(float("nan")))
        self.assertEqual(key((0.5, -0.0)), key((0.5, -0.0)))
        self.assertNotEqual(key(1.0), key(1))
        table: KernelSideTable[_ToyKernel] = KernelSideTable(constant_rules=_TOY_RULES)
        self.assertNotEqual(
            table.add_constant_args({"a": 0.0}), table.add_constant_args({"a": -0.0})
        )
        nan_idx = table.add_constant_args({"a": float("nan")})
        self.assertEqual(nan_idx, table.add_constant_args({"a": float("nan")}))

    def test_constant_rules_leaf_key(self):
        # A DSL keys the values it compares its own way (e.g. a lambda it compiles
        # in, by its code) with its rules' leaf_key, wherever constant_key meets one.
        class LowerRules(ConstantRules):
            def leaf_key(self, value: object) -> object:
                return ("str", value.lower()) if isinstance(value, str) else None

        rules = LowerRules()
        self.assertEqual(constant_key(["A"], rules), constant_key(["a"], rules))
        self.assertNotEqual(
            constant_key(["A"], _PLAIN_RULES), constant_key(["a"], _PLAIN_RULES)
        )
        table: KernelSideTable[_ToyKernel] = KernelSideTable(constant_rules=rules)
        idx = table.add_constant_args({"a": {"k": ("A", 1)}})
        self.assertEqual(table.add_constant_args({"a": {"K": ("a", 1)}}), idx)
        self.assertNotEqual(table.add_constant_args({"a": {"k": ("a", 2)}}), idx)
        with self.assertRaisesRegex(TypeError, "type set is unhashable"):
            table.add_constant_args({"a": {1}})

        # leaf_key comes after the tensor refusal and ahead of the other rules: a
        # tensor it claims is refused all the same, while a Numeric (a scalar to
        # _ToyRules) and a list (a container) it claims are keyed by its key.
        class ClaimingRules(_ToyRules):
            def leaf_key(self, value: object) -> object:
                claimed = (torch.Tensor, _ToyNumeric, list)
                return ("claimed",) if isinstance(value, claimed) else None

        claiming = ClaimingRules()
        with self.assertRaisesRegex(TypeError, "holds a tensor"):
            constant_key(torch.ones(1), claiming)
        self.assertEqual(constant_key(_ToyInt32(3), claiming), ("claimed",))
        self.assertEqual(constant_key([1.0], claiming), ("claimed",))

    def test_kernel_kwargs(self):
        table: KernelSideTable[_ToyKernel] = KernelSideTable()
        idx = table.add_constant_args({"BLOCK": 128, "alpha": 0.5})
        graph_args = {"x": "tensor", "n": 4}
        joined = table.kernel_kwargs(idx, graph_args)
        self.assertEqual(joined, {"x": "tensor", "n": 4, "BLOCK": 128, "alpha": 0.5})
        joined["BLOCK"] = 0
        self.assertEqual(graph_args, {"x": "tensor", "n": 4})
        self.assertEqual(table.get_constant_args(idx), {"BLOCK": 128, "alpha": 0.5})

    def test_registered_infos(self):
        from torch._higher_order_ops import triton_kernel_wrap as tkw

        info = tkw.triton_kernel_hop_info
        hops = (
            tkw.triton_kernel_wrapper_mutation,
            tkw.triton_kernel_wrapper_functional,
        )
        self.assertEqual((info.dsl, info.display_name), ("triton", "Triton"))
        self.assertIs(info.side_table, tkw.kernel_side_table)
        self.assertEqual((info.mutation, info.functional), hops)
        for hop in hops:
            self.assertIs(get_user_kernel_hop_info(hop), info)
        dense = tkw.triton_kernel_wrapper_functional.py_kernels[
            torch._C.DispatchKey.CompositeExplicitAutograd
        ]
        self.assertIs(tkw.triton_kernel_wrapper_functional_dense, dense)
        self.assertIs(info.get_kernel_name, tkw.get_triton_kernel_name)
        self.assertIs(info.get_layout_constraint, tkw.get_triton_layout_constraint)
        self.assertEqual(
            (info.reinplace_trigger, info.serializable),
            (ReInplaceTrigger.TRITON_OPS, True),
        )
        with _toy_dsl() as toy:
            self.assertEqual((toy.dsl, toy.display_name), ("toy", "ToyDSL"))
            for hop in (toy.mutation, toy.functional):
                self.assertIs(get_user_kernel_hop_info(hop), toy)
            # The default kernel name: the side-table record's.
            idx = toy.get_side_table().add_kernel(_ToyKernel(_toy_fn, ()))
            self.assertEqual(toy.get_kernel_name({"kernel_idx": idx}), "_toy_fn")
            self.assertEqual(toy.get_layout_constraint(), "needs_fixed_stride_order")
            # The functional HOP's dense impl, registered before the entry (the
            # registry requires one): the mutation HOP on clones.
            self.assertIn(
                torch._C.DispatchKey.CompositeExplicitAutograd,
                toy.functional.py_kernels,
            )
        self.assertIsNone(get_user_kernel_hop_info(toy.mutation))

    def test_info_defaults(self):
        # A DSL that builds its entry itself (D128128702's scale DSL in
        # test_triton_kernels, say) may leave out display_name, which defaults to
        # dsl, and side_table, which only the shared capture code reads.
        from torch._ops import _higher_order_ops

        with mock.patch.dict(_higher_order_ops):
            info = UserKernelHOPInfo(
                dsl="toy",
                mutation=_ToyKernelWrapper("toy_kernel_wrapper_mutation"),
                functional=_ToyKernelWrapper("toy_kernel_wrapper_functional"),
                get_mutated_tensors=lambda hop_kwargs, kwargs: [],
                get_kernel_name=lambda hop_kwargs: "k",
                get_layout_constraint=lambda: "needs_fixed_stride_order",
                reinplace_trigger=ReInplaceTrigger.USER_KERNEL_OPS,
                serializable=False,
            )
        self.assertEqual(info.display_name, "toy")
        self.assertIsNone(info.side_table)
        with self.assertRaisesRegex(AssertionError, "the toy DSL's entry has no side"):
            info.get_side_table()
        table: KernelSideTable[_ToyKernel] = KernelSideTable()
        info = dataclasses.replace(info, display_name="ToyDSL", side_table=table)
        self.assertEqual(info.display_name, "ToyDSL")
        self.assertIs(info.get_side_table(), table)

    def test_registered_impls(self):
        # The impls register_user_kernel_hop_impls registers, on a toy DSL: the eager
        # launch, the fake impl, tracing, and functionalization into the functional
        # HOP, whose dense impl launches the kernel on clones of the written tensors.
        from torch._subclasses.fake_tensor import FakeTensorMode

        def scale(x: torch.Tensor, out: torch.Tensor, alpha: float) -> None:
            out.copy_(x * alpha)

        with _toy_dsl() as toy:
            kidx = toy.get_side_table().add_kernel(_ToyKernel(scale, ("out",)))
            cidx = toy.get_side_table().add_constant_args({"alpha": 2.0})

            def f(x: torch.Tensor) -> torch.Tensor:
                out = torch.zeros_like(x)
                toy.mutation(
                    kernel_idx=kidx, constant_args_idx=cidx, kwargs={"x": x, "out": out}
                )
                return out

            x = torch.arange(4.0)
            self.assertEqual(f(x), x * 2)
            untyped_hop: Any = toy.mutation
            with self.assertRaises(TypeError):
                untyped_hop(kidx, cidx, {"x": x, "out": x})
            with FakeTensorMode():
                fake = torch.empty(4)
                self.assertIsNone(
                    toy.mutation(
                        kernel_idx=kidx,
                        constant_args_idx=cidx,
                        kwargs={"x": fake, "out": fake},
                    )
                )
            gm = make_fx(torch.func.functionalize(f), tracing_mode="fake")(x)
            (node,) = gm.graph.find_nodes(op="call_function", target=toy.functional)
            # The node keeps the caller's keyword order; the empty tensor_bases is left
            # out.
            self.assertEqual(
                list(node.kwargs),
                ["kernel_idx", "constant_args_idx", "kwargs", "tensors_to_clone"],
            )
            self.assertEqual(node.kwargs["tensors_to_clone"], ["out"])
            self.assertEqual(gm(x), x * 2)

    def test_get_node_mutated_tensors(self):
        from torch._higher_order_ops import triton_kernel_wrap as tkw

        graph = torch.fx.Graph()
        x, out = graph.placeholder("x"), graph.placeholder("out")
        x.meta["val"], out.meta["val"] = torch.ones(2), torch.zeros(2)
        hop_kwargs = {"kernel_idx": 0, "constant_args_idx": 0, "grid": [(1, 1, 1)]}
        target = tkw.triton_kernel_wrapper_mutation
        kernel_args = {"x": x, "out": out, "n": 2}
        node = graph.call_function(target, (), {**hop_kwargs, "kwargs": kernel_args})
        calls = []

        def get_mutated_tensors(hop_kw: Any, kwargs: dict[str, Any]) -> list[str]:
            calls.append((hop_kw, kwargs))
            return ["out"]

        info = dataclasses.replace(
            tkw.triton_kernel_hop_info, get_mutated_tensors=get_mutated_tensors
        )
        self.assertEqual(info.get_node_mutated_tensors(node), ["out"])
        hop_kw, kwargs = calls.pop()
        self.assertIs(hop_kw, node.kwargs)
        self.assertEqual(list(kwargs), ["x", "out", "n"])
        self.assertIs(kwargs["x"], x.meta["val"])
        self.assertIs(kwargs["out"], out.meta["val"])
        self.assertEqual(kwargs["n"], 2)
        lowered = {"kernel_idx": 0}
        info.get_node_mutated_tensors(node, lowered)
        self.assertIs(calls.pop()[0], lowered)
        bad = graph.call_function(target, (), {**hop_kwargs, "kwargs": x})
        with self.assertRaisesRegex(AssertionError, "Expected dict"):
            info.get_node_mutated_tensors(bad)

    def test_triton_get_mutated_tensors_patch_point(self):
        # Code outside torch patches triton_kernel_wrap.get_mutated_tensors: the
        # Triton info and the shared functionalization rule look it up on each call.
        from torch._higher_order_ops import triton_kernel_wrap as tkw

        calls = []

        def patched(kernel_idx, constant_args_idx, kwargs, tma_descriptor_metadata):
            calls.append((kernel_idx, constant_args_idx, sorted(kwargs)))
            return ["out"]

        def f(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            tkw.triton_kernel_wrapper_mutation(
                kernel_idx=0,
                constant_args_idx=0,
                grid=[(1, 1, 1)],
                tma_descriptor_metadata={},
                kwargs={"x": x, "out": out},
            )
            return out

        info = tkw.triton_kernel_hop_info
        hop_kwargs = {"kernel_idx": 3, "constant_args_idx": 4}
        with mock.patch.object(tkw, "get_mutated_tensors", patched):
            self.assertEqual(
                info.get_mutated_tensors(
                    {**hop_kwargs, "tma_descriptor_metadata": {}}, {"out": 1}
                ),
                ["out"],
            )
            self.assertEqual(calls.pop(), (3, 4, ["out"]))
            gm = make_fx(torch.func.functionalize(f), tracing_mode="fake")(
                torch.ones(4), torch.zeros(4)
            )
        self.assertEqual(calls, [(0, 0, ["out", "x"])])
        (node,) = gm.graph.find_nodes(
            op="call_function", target=tkw.triton_kernel_wrapper_functional
        )
        self.assertEqual(node.kwargs["tensors_to_clone"], ["out"])

    def test_triton_kernel_wrap_public_names(self):
        # Imported from triton_kernel_wrap by code outside torch (MTIA dispatch modes
        # and compiler, executorch, AOTI post-grad passes, ads_mkl, glow, vllm, APS)
        # or by other torch modules.
        from torch._higher_order_ops import triton_kernel_wrap as tkw

        names = (
            "analyze_kernel_access",
            "clone_preserve_strides",
            "create_tma_experimental_metadata",
            "create_tma_stable_metadata",
            "generate_ttir",
            "get_kernel",
            "get_mutated_tensors",
            "get_tma_stores",
            "get_triton_kernel_name",
            "identify_accessed_tensors",
            "identify_triton_stores",
            "identify_triton_stores_from_ast",
            "kernel_side_table",
            "register_kernel_access_op",
            "TMADescriptorMetadata",
            "TraceableTritonKernelWrapper",
            "tracing_triton_hopifier_singleton",
            "TracingTritonHOPifier",
            "triton_kernel_wrapper_functional",
            "triton_kernel_wrapper_functional_dense",
            "triton_kernel_wrapper_mutation",
            "triton_kernel_wrapper_mutation_dense",
            "TritonHOPifier",
            "TritonKernelWrapperFunctional",
            "TritonKernelWrapperMutation",
            "ttir_to_functions",
        )
        self.assertEqual([n for n in names if not hasattr(tkw, n)], [])

    def test_overlap_check_on_symbolic_sizes(self):
        # Views of one element size are compared by compute_overlapping_tensors,
        # which decides by the size hints and guards on the decision: x[:n] and x[m:]
        # (n = 10, m = 20) are disjoint while n <= m. Views of different element sizes
        # are compared by their byte ranges, without guards: x[:n] ends where the
        # float16 view from element 2n starts, and may overlap the one from 2m. An
        # unbacked size, which cannot be guarded on, counts as an overlap, even in
        # x[:u] and x[u:].
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        may_overlap = user_kernel_hop._may_overlap
        results = []

        def f(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            n, m = y.shape[0], z.shape[0]
            halves = x.view(torch.float16)
            head, tail = x[:n], x[m:]
            halves_n, halves_m = halves[2 * n :], halves[2 * m :]
            shape_env = cast(torch.SymInt, n).node.shape_env
            guards = len(shape_env.guards)
            results.extend(may_overlap(head, h) for h in (halves_n, halves_m))
            self.assertEqual(len(shape_env.guards), guards)
            results.append(may_overlap(head, tail))
            return x

        gm = make_fx(f, tracing_mode="symbolic")(
            torch.zeros(100), torch.zeros(10), torch.zeros(20)
        )
        self.assertEqual(results, [False, True, False])
        inputs = [node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]
        holds = gm.shape_env.evaluate_guards_for_args
        sizes_ok = [torch.zeros(100), torch.zeros(10), torch.zeros(20)]
        self.assertTrue(holds(inputs, sizes_ok))
        sizes_swapped = [torch.zeros(100), torch.zeros(20), torch.zeros(10)]
        self.assertFalse(holds(inputs, sizes_swapped))

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            u = shape_env.create_unbacked_symint()
            x = torch.empty(100)
            self.assertTrue(may_overlap(x[:u], x[u:]))

    def test_overlap_check_rechecks_2d_rows(self):
        # TODO(zoranzhao): remove with _rows_disjoint once
        # https://github.com/pytorch/pytorch/pull/202848 lands. Without that fix,
        # compute_overlapping_tensors answers "disjoint" for base[:4, :3] and
        # base[1:5, 2:5]: with that answer forced, _may_overlap still finds them
        # overlapping, while an empty view, views whose ranges do not meet and the
        # q, k and v columns of a fused buffer stay disjoint, under symbolic sizes
        # too.
        may_overlap = user_kernel_hop._may_overlap
        results = []

        def f(base: torch.Tensor) -> torch.Tensor:
            n, d = base.shape
            qkv = base.new_empty(n, 3 * d)
            q, k, v = (qkv.narrow(1, i * d, d) for i in range(3))
            pairs = [
                (base[:4, :3], base[1:5, 2:5]),
                (base[:4, :3], base[1:3, 0:0]),
                (base[:2, :5], base[2:4, :5]),
                (q, k),
                (k, v),
                (q, v),
            ]
            results.extend(may_overlap(a, b) for a, b in pairs)
            return base

        with mock.patch.object(
            torch._C._dynamo.guards, "compute_overlapping_tensors", return_value=set()
        ):
            f(torch.zeros(10, 25))
            make_fx(f, tracing_mode="symbolic")(torch.zeros(10, 25))
        self.assertEqual(results, ([True] + [False] * 5) * 2)

    def test_functionalize_refuses_aliased_writes(self):
        # register_user_kernel_hop_impls(refuse_aliased_writes=True):
        # functionalization refuses two written arguments that may overlap (one write
        # would be lost): the same tensor, or views that overlap whatever their
        # shapes, offsets and dtypes. Views of one base proven disjoint (rows of a 2-D
        # base, the q, k and v columns of a fused 2-D buffer) are cloned and written
        # back each on its own, unless functionalization copies views
        # (remove="mutations_and_views"): the copies do not show where the views sit,
        # so any two views are refused. Without it (Triton) such calls are traced.

        # The check itself, whatever the DSL.
        check = user_kernel_hop.check_written_args_disjoint
        base = torch.zeros(10, 100)
        flat = base.view(-1)
        overlapping = (
            (flat, flat),
            (base[0], base[0]),
            (flat[1:], flat[:-1]),
            (flat, base),
            (flat[500:], flat.view(torch.float16)[1000:]),
            # Each row of the second starts inside a row of the first.
            (base[:4, :3], base[1:5, 2:5]),
        )
        for a, b in overlapping:
            kwargs = {"a": a, "b": b}
            with self.assertRaises(RuntimeError) as cm:
                check("ToyDSL", "k", ["a", "b"], kwargs, kwargs, mutates_args=())
            self.assertEqual(
                str(cm.exception),
                "ToyDSL kernel k: 'a' and 'b' are the same tensor or views that may "
                "overlap, and both count as written (mutates_args=()), which PT2 "
                "cannot functionalize: one write would be lost. Pass tensors that do "
                "not overlap, or give mutates_args naming only the parameters the "
                "kernel writes.",
            )
        qkv = torch.zeros(40, 75)
        q, k, v = (qkv.narrow(1, i * 25, 25) for i in range(3))
        written = ["a", "b"]
        for a, b in (
            (base[0], base[1]),
            (q, k),
            (k, v),
            (flat, flat[500:500]),
            (flat, torch.zeros(4)),
        ):
            kwargs = {"a": a, "b": b, "n": 4}
            self.assertIs(
                check("ToyDSL", "k", written, kwargs, kwargs, mutates_args=()), written
            )
        # Functional tensors that alias, unwrapped to values that do not (copies of
        # the views), count as overlapping.
        wrapped = {"a": base[0], "b": base[1]}
        copies = {"a": base[0].clone(), "b": base[1].clone()}
        with self.assertRaisesRegex(RuntimeError, "^ToyDSL kernel k: 'a' and 'b' "):
            check("ToyDSL", "k", written, copies, wrapped, mutates_args=())

        for refuse in (True, False):
            with (
                self.subTest(refuse=refuse),
                _toy_dsl(refuse_aliased_writes=refuse) as toy,
            ):
                kidx = toy.get_side_table().add_kernel(_ToyKernel(_toy_fn, "unknown"))
                cidx = toy.get_side_table().add_constant_args({})

                def launch(a: torch.Tensor, b: torch.Tensor) -> None:
                    toy.mutation(
                        kernel_idx=kidx, constant_args_idx=cidx, kwargs={"a": a, "b": b}
                    )

                def in_place(x: torch.Tensor) -> torch.Tensor:
                    launch(x, x)
                    return x

                def shifted(x: torch.Tensor) -> torch.Tensor:
                    launch(x[1:], x[:-1])
                    return x

                def views(base: torch.Tensor) -> torch.Tensor:
                    launch(base[0], base[1])
                    return base

                for f in (in_place, shifted):
                    trace = make_fx(torch.func.functionalize(f), tracing_mode="fake")
                    if refuse:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"^ToyDSL kernel _toy_fn: 'a' and 'b' are the same tensor or "
                            r"views that may overlap, and both count as written "
                            r"\(mutates_args='unknown'\)",
                        ):
                            trace(torch.zeros(4))
                    else:
                        trace(torch.zeros(4))
                copied = torch.func.functionalize(views, remove="mutations_and_views")
                trace = make_fx(copied, tracing_mode="fake")
                if refuse:
                    with self.assertRaisesRegex(
                        RuntimeError, "^ToyDSL kernel _toy_fn: 'a' and 'b' "
                    ):
                        trace(torch.zeros(2, 4))
                else:
                    trace(torch.zeros(2, 4))
                gm = make_fx(torch.func.functionalize(views), tracing_mode="fake")(
                    torch.zeros(2, 4)
                )
                (node,) = gm.graph.find_nodes(op="call_function", target=toy.functional)
                self.assertEqual(node.kwargs["tensors_to_clone"], ["a", "b"])


if __name__ == "__main__":
    run_tests()
