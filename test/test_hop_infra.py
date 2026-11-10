# Owner(s): ["module: higher order operators"]
import collections
import copy
import dataclasses
import enum
import importlib
import logging
import pkgutil
import types
from unittest import mock

import torch
from torch._higher_order_ops.cutedsl_kernel_wrap import (
    CUTEDSL_CONSTANT_RULES,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLHostFnInfo,
    python_scalar,
)
from torch._higher_order_ops.user_kernel_hop import (
    _snapshot,
    constant_key,
    KernelSideTable,
)
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
            "cutedsl_kernel_wrapper_functional",  # Imports sympy, as Triton's does
            "cutedsl_kernel_wrapper_mutation",  # Imports sympy, as Triton's does
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


@skipIfTorchDynamo("not applicable")
class TestCuteDSLHOPHelpers(TestCase):
    """The parts of the CuTeDSL user-kernel HOPs that need neither CUDA nor
    nvidia-cutlass-dsl: how the side table keys and stores constants, how a call is
    split, and who owns their reinplacing trigger. The tests that compile and launch
    host functions are in test/inductor/test_cutedsl_kernels.py."""

    hw_classification = HardwareClassification.GENERIC

    def test_side_table_dedupes_constant_args(self):
        # The CuTeDSL table is never reset and every eager launch through the HOP adds
        # its constants, so interchangeable constants share one entry (see
        # constant_key), which holds a snapshot of the constants it was added with.
        Tile = collections.namedtuple("Tile", ["m", "n"])

        @dataclasses.dataclass(frozen=True)
        class FrozenCfg:
            alpha: float

        class Cfg:  # a plain config object: compared and hashed by identity
            def __init__(self, alpha):
                self.alpha = alpha

        class Numeric:  # keyed like a cutlass Numeric, such as cutlass.Float32(2.0)
            __module__ = "cutlass.base_dsl.typing"

            def __init__(self, value):
                self.value = value

        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)
        idx = table.add_constant_args({"alpha": 2, "k": None})
        self.assertEqual(table.add_constant_args({"k": None, "alpha": 2}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": 2.0, "k": None}), idx)
        self.assertNotEqual(table.add_constant_args({"alpha": True, "k": None}), idx)

        def add(c):
            return table.add_constant_args({"c": c})

        # By type and content all the way down: inside containers and dataclasses
        # too, equal values of other types, and floats that differ only in sign, are
        # other constants. Every NaN is the same one.
        for a, b in (
            ([1, 2], (1, 2)),
            (Tile(1, 2), Tile(1.0, 2)),
            (Tile(1, 2), Tile(True, 2)),
            (Tile(0.0, 1), Tile(-0.0, 1)),
            (FrozenCfg(0.0), FrozenCfg(-0.0)),
            (Numeric(0.0), Numeric(-0.0)),
        ):
            self.assertNotEqual(add(a), add(b))
        self.assertEqual(add(float("nan")), add(float("nan")))
        # Mutable values too (a namedtuple holding a list, a plain object, a DSL
        # scalar), stored as they were when added: mutating the caller's value
        # changes neither the entry nor the entry its old contents map to.
        tile, cfg, numeric = Tile([128], 64), Cfg(1.0), Numeric(2.0)
        indices = add(tile), add(cfg), add(numeric)

        def add_equal_values():
            return add(Tile([128], 64)), add(Cfg(1.0)), add(Numeric(2.0))

        self.assertEqual(add_equal_values(), indices)
        tile.m.append(32)
        cfg.alpha = 2.0
        numeric.value = 5.0
        t, c, n = (table.get_constant_args(i)["c"] for i in indices)
        self.assertEqual((t, c.alpha, n.value), (Tile([128], 64), 1.0, 2.0))
        self.assertEqual(add_equal_values(), indices)
        for changed, idx in zip((tile, cfg, numeric), indices):
            self.assertNotEqual(add(changed), idx)

        # An object with __slots__ attributes, even one with a __dict__ too, is keyed
        # as itself (keyed by its __dict__, objects differing only in a slot would
        # share one entry), so an unhashable one is refused.
        class SlotBase:
            __slots__ = ("alpha",)

        class SlotCfg(SlotBase):
            def __init__(self, alpha):
                self.alpha = alpha

        class EqSlotCfg(SlotCfg):  # compared by value, so unhashable
            def __eq__(self, other):
                return self.alpha == other.alpha

        slot_cfgs = [table.get_constant_args(add(SlotCfg(a)))["c"] for a in (1.0, 2.0)]
        self.assertEqual([c.alpha for c in slot_cfgs], [1.0, 2.0])
        msg = "^a constant kernel argument of type {} is unhashable: pass a hashable"
        for value in ({1, 2}, EqSlotCfg(1.0)):
            with self.assertRaisesRegex(TypeError, msg.format(type(value).__name__)):
                add(value)

    def test_numpy_float_constants_are_keyed_by_their_bits(self):
        # An np.float16 or np.float32 scalar is keyed as a float is, by type and
        # hex(): signed zeros apart, equal values of other types apart, and every NaN
        # of one type the same constant, in the side table too.
        import numpy as np

        def distinct_keys(*values):
            return len({constant_key(v, CUTEDSL_CONSTANT_RULES) for v in values})

        for a, b in (
            (np.float32(-0.0), np.float32(0.0)),
            (np.float16(-0.0), np.float16(0.0)),
            (np.float32(1.0), np.float16(1.0)),
            (np.float32(1.0), 1.0),
        ):
            self.assertEqual(distinct_keys(a, b), 2, f"{a!r} and {b!r}")
        nan = float("nan")
        for make in (np.float32, np.float16):
            self.assertEqual(distinct_keys(make(nan), make(nan)), 1, make.__name__)
        self.assertEqual(distinct_keys(-np.float32(nan), np.float32(nan)), 1)
        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)
        values = (np.float32(-0.0), np.float32(0.0), np.float32(nan), np.float32(nan))
        indices = [table.add_constant_args({"alpha": v}) for v in values]
        self.assertEqual(indices, [0, 1, 2, 2])

    def test_side_table_reset_clears_the_dedupe_index(self):
        # Without the index, a reset table would hand out the index of an entry that
        # no longer exists. The table also keeps each entry's keys, which the eager
        # launch reuses.
        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)
        table.add_constant_args({"a": 1})
        self.assertEqual(table.add_constant_args({"a": 2.0}), 1)
        self.assertEqual(table.get_constant_args_keys(1), {"a": (float, (2.0).hex())})
        table.reset_table()
        self.assertEqual(table.add_constant_args({"a": 2.0}), 0)
        self.assertEqual(table.get_constant_args(0), {"a": 2.0})
        plain = KernelSideTable()
        self.assertEqual(plain.get_constant_args_keys(plain.add_constant_args({})), {})

    def test_constants_from_cutlass_named_modules_are_keyed_by_content(self):
        # Only the cutlass package's own values are the DSL's: an object of a class
        # from a module merely named like it (cutlass_configs, cutlass_library), even
        # one with a numeric .value like a cutlass Numeric's, is keyed by its
        # attributes and stored as a snapshot.
        class Cfg:
            __module__ = "cutlass_configs"

            def __init__(self, **attrs):
                vars(self).update(attrs)

        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)

        def add(cfg):
            return table.add_constant_args({"cfg": cfg})

        cfg = Cfg(value=1, alpha=2.0)
        idx = add(cfg)
        self.assertEqual(add(Cfg(value=1, alpha=2.0)), idx)
        self.assertNotEqual(add(Cfg(value=1, alpha=3.0)), idx)
        self.assertEqual(add(Cfg(alpha=2.0)), add(Cfg(alpha=2.0)))
        cfg.alpha = 3.0
        snapshot = table.get_constant_args(idx)["cfg"]
        self.assertEqual(vars(snapshot), {"value": 1, "alpha": 2.0})

    def test_constants_from_the_cutlass_package_are_keyed_as_themselves(self):
        # A value of a class from the cutlass package is the DSL's own: keyed by its
        # own equality, never by its attributes, and stored as itself. So equal DslCfg
        # objects get their own entries, and Basis objects (like cute.ScaledBasis) are
        # keyed by value and mode, not by their numeric .value as a Numeric is.
        class DslCfg:
            __module__ = "cutlass.cute.fake"

            def __init__(self, alpha):
                self.alpha = alpha

        @dataclasses.dataclass(frozen=True)
        class Basis:
            __module__ = "cutlass.cute.fake"
            value: int
            mode: int

        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)
        cfg = DslCfg(2.0)
        idx = table.add_constant_args({"cfg": cfg})
        self.assertNotEqual(table.add_constant_args({"cfg": DslCfg(2.0)}), idx)
        self.assertIs(table.get_constant_args(idx)["cfg"], cfg)
        basis = Basis(2, 0)
        idx = table.add_constant_args({"cfg": basis})
        self.assertNotEqual(table.add_constant_args({"cfg": Basis(2, 1)}), idx)
        self.assertEqual(table.add_constant_args({"cfg": Basis(2, 0)}), idx)
        self.assertIs(table.get_constant_args(idx)["cfg"], basis)

    def test_dataclass_constants_are_keyed_by_all_their_attributes(self):
        # A dataclass instance is keyed and stored as a snapshot by its fields and the
        # other attributes in its __dict__ (set in __post_init__ or ad hoc), frozen or
        # not. One with __slots__ attributes that are not fields is keyed as itself:
        # by identity (eq=False), by its own equality, which compares its fields alone
        # (frozen), or not at all (eq=True and not frozen, so unhashable).
        @dataclasses.dataclass
        class Cfg:
            alpha: float

            def __post_init__(self):
                self.tiles = [self.alpha]

        @dataclasses.dataclass(frozen=True)
        class FrozenCfg:
            alpha: float

            def __post_init__(self):
                object.__setattr__(self, "tiles", [self.alpha])

        @dataclasses.dataclass(slots=True)
        class SlotsCfg:
            alpha: float

        class SlotBase:
            __slots__ = ("beta",)

        @dataclasses.dataclass(eq=False)
        class SlotBaseCfg(SlotBase):
            alpha: float

        @dataclasses.dataclass(frozen=True)
        class FrozenSlotBaseCfg(SlotBase):
            alpha: float

        @dataclasses.dataclass
        class EqSlotBaseCfg(SlotBase):
            alpha: float

        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)

        def add(c):
            return table.add_constant_args({"c": c})

        for cls in (Cfg, FrozenCfg):
            a, b, c = cls(1.0), cls(1.0), cls(1.0)
            b.tiles.append(2.0)
            object.__setattr__(c, "mode", "fast")
            self.assertEqual(len({add(a), add(b), add(c)}), 3)
            self.assertEqual(add(cls(1.0)), add(a))
            stored = table.get_constant_args(add(c))["c"]
            c.tiles.append(3.0)
            object.__setattr__(c, "mode", "slow")
            self.assertEqual((stored.tiles, stored.mode), ([1.0], "fast"))
            self.assertEqual(table.get_constant_args(add(cls(1.0)))["c"].tiles, [1.0])
        slots = SlotsCfg(1.0)
        self.assertEqual(add(slots), add(SlotsCfg(1.0)))
        self.assertIsNot(table.get_constant_args(add(slots))["c"], slots)
        s1, s2 = SlotBaseCfg(1.0), SlotBaseCfg(1.0)
        s1.beta, s2.beta = 1, 2
        self.assertNotEqual(add(s1), add(s2))
        self.assertIs(table.get_constant_args(add(s1))["c"], s1)
        f1, f2 = FrozenSlotBaseCfg(1.0), FrozenSlotBaseCfg(1.0)
        object.__setattr__(f1, "beta", 1)
        object.__setattr__(f2, "beta", 2)
        self.assertEqual(add(f1), add(f2))
        self.assertIs(table.get_constant_args(add(f2))["c"], f1)
        msg = "^a constant kernel argument of type EqSlotBaseCfg is unhashable"
        with self.assertRaisesRegex(TypeError, msg):
            add(EqSlotBaseCfg(1.0))

    def test_constants_with_builtin_state_are_keyed_as_themselves(self):
        # A set subclass or an exception keeps state outside its __dict__ (its items,
        # its args), so it is keyed as itself, not by its __dict__ (or, for a
        # dataclass such as NamedTags, by its fields): refused if it is unhashable, by
        # identity otherwise. Keyed by their equal __dict__s, Tags({1}) and Tags({2}),
        # or ValueError("a") and ValueError("b"), would share an entry.
        rules = CUTEDSL_CONSTANT_RULES

        class Tags(set):
            pass

        @dataclasses.dataclass
        class NamedTags(set):
            name: str

        with self.assertRaisesRegex(TypeError, "of type Tags is unhashable"):
            constant_key(Tags({1}), rules)
        with self.assertRaisesRegex(TypeError, "of type NamedTags is unhashable"):
            constant_key(NamedTags("a"), rules)
        table = KernelSideTable(constant_rules=rules)
        idx = table.add_constant_args({"e": ValueError("a")})
        self.assertNotEqual(table.add_constant_args({"e": ValueError("b")}), idx)

    def test_cyclic_constants_are_refused(self):
        # constant_key and _snapshot walk a constant's contents, so a constant that
        # refers to itself is refused with a ValueError naming the cycle, instead of
        # recursing until a RecursionError. An object, dict or list reached twice on
        # different paths is no cycle.
        class Cfg:
            def __init__(self, alpha, **attrs):
                self.alpha = alpha
                vars(self).update(attrs)

        @dataclasses.dataclass
        class Node:
            next: object = None

        cfg = Cfg(2.0, parent=Cfg(1.0))
        cfg.parent.child = cfg
        node = Node()
        node.next = Node(node)
        items = [1.0]
        items.append(items)
        table = {}
        table["self"] = table
        keyed = Cfg(1.0)
        keyed.lookup = {keyed: 1}
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
        rules = CUTEDSL_CONSTANT_RULES
        side_table = KernelSideTable(constant_rules=rules)
        root_handlers = logging.root.handlers
        for value, name, path in cycles:
            msg = f"^a constant kernel argument of type {name} refers to itself "
            msg += f"through {path}$"
            for walk in (constant_key, _snapshot):
                with self.assertRaisesRegex(ValueError, msg):
                    walk(value, rules)
            with self.assertRaisesRegex(ValueError, msg):
                side_table.add_constant_args({"cfg": value})
        # _snapshot copied the private RootLogger without writing onto the process's.
        self.assertIs(logging.root.handlers, root_handlers)

        shared = Cfg(1.0, lookup={"k": [0.5]})
        twice = Cfg(2.0, a=shared, b=[shared], c=shared.lookup)
        snapshot = _snapshot(twice, rules)
        self.assertEqual(constant_key(snapshot, rules), constant_key(twice, rules))
        self.assertTrue(all(c is not shared for c in (snapshot.a, snapshot.b[0])))
        self.assertIsNot(snapshot.c, shared.lookup)

    def test_snapshot_of_a_value_whose_copy_is_itself(self):
        # copy.copy returns whatever a class's __copy__ or __reduce__ returns, even
        # the object itself, and a stored constant that is the caller's object changes
        # with it. _snapshot makes a plain object with object.__new__, as copy.copy
        # does by default, stores a DSL value that is not a Numeric (an enum member) as
        # itself, and refuses any other value whose copy is itself.
        class SelfCopy:
            def __copy__(self):
                return self

        class Cfg(SelfCopy):
            def __init__(self, alpha):
                self.alpha = alpha

        class NewCfg(Cfg):  # defines __new__, so copied by copy.copy
            def __new__(cls, alpha):
                return super().__new__(cls)

        class Numeric(SelfCopy):  # keyed like a cutlass Numeric
            __module__ = "cutlass.base_dsl.typing"

            def __init__(self, value):
                self.value = value

        class Group(enum.Enum):  # keyed like a cutlass enum, such as CtaGroup.ONE
            __module__ = "cutlass.fake"
            ONE = 1

        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)

        def add(c):
            return table.add_constant_args({"c": c})

        cfg = Cfg(1.0)
        stored = table.get_constant_args(add(cfg))["c"]
        self.assertIsNot(stored, cfg)
        self.assertEqual((type(stored), vars(stored)), (Cfg, {"alpha": 1.0}))
        self.assertIs(table.get_constant_args(add(Group.ONE))["c"], Group.ONE)
        msg = "^a constant kernel argument of type {} cannot be copied: copy.copy "
        msg += "returns the object itself$"
        for value in (NewCfg(1.0), Numeric(2.0)):
            with self.assertRaisesRegex(TypeError, msg.format(type(value).__name__)):
                add(value)

    def test_constants_holding_tensors_are_refused(self):
        # Keyed by identity, a tensor inside a constant would stay alive with its
        # side-table entry, and it would reach the DSL without from_dlpack:
        # constant_key refuses one as a constant or in a list, tuple, dict, dataclass
        # attribute or plain-object attribute, naming where.
        @dataclasses.dataclass
        class Cfg:
            alpha: float
            weight: object = None

        t = torch.randn(4)
        with_extra = Cfg(2.0)
        with_extra.extra = t
        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)
        msg = "^a constant kernel argument holds a tensor{}: pass each tensor as its "
        msg += "own argument$"
        for value, where in (
            ([t], r" at \[0\]"),
            ((1.0, {"w": t}), r" at \[1\]\['w'\]"),
            ({t: 1.0}, r" at \.keys\(\)\[0\]"),
            (Cfg(2.0, weight=t), r" at \.weight"),
            (with_extra, r" at \.extra"),
            (types.SimpleNamespace(w=t), r" at \.w"),
            (t, ""),
        ):
            with self.assertRaisesRegex(TypeError, msg.format(where)):
                table.add_constant_args({"cfg": value})

    def test_container_subclasses_are_refused(self):
        # Only an exact list, tuple or dict, an OrderedDict (rebuilt in its order), a
        # namedtuple and a torch.Size are keyed and copied by their items. Any other
        # list, tuple or dict subclass is refused, in a constant or inside one: its
        # items can leave out its state (a defaultdict's default_factory, attributes)
        # or not rebuild it (a Counter's update() counts the (key, value) pairs).
        # So is a namedtuple whose __iter__ is not tuple's (_snapshot rebuilds from it).
        class Scales(list):
            pass

        class Swapped(collections.namedtuple("Swapped", ["m", "n"])):
            __slots__ = ()

            def __iter__(self):
                return iter((self.n, self.m))

        rules = CUTEDSL_CONSTANT_RULES
        msg = "^a constant kernel argument of type {} is a list, tuple or dict subclass"
        for value, name in (
            (Scales([1.0]), "Scales"),
            ((1, collections.Counter(a=3)), "Counter"),
            ({"tiles": collections.defaultdict(None, a=[1])}, "defaultdict"),
            (Swapped([1], 2), "Swapped"),
        ):
            with self.assertRaisesRegex(TypeError, msg.format(name)):
                constant_key(value, rules)
        ordered = collections.OrderedDict([("b", [1]), ("a", 2)])
        snapshot = _snapshot(ordered, rules)
        self.assertIs(type(snapshot), collections.OrderedDict)
        self.assertEqual(list(snapshot.items()), [("b", [1]), ("a", 2)])
        self.assertIsNot(snapshot["b"], ordered["b"])
        size = torch.Size([4, 2])
        self.assertIs(_snapshot(size, rules), size)
        self.assertNotEqual(constant_key(size, rules), constant_key((4, 2), rules))

    def test_host_fn_info_checks_and_splits_a_call(self):
        # check_call and split_args need only a parsed signature, written out here: a
        # tensor x, a cutlass Numeric n (int stands for its type), the stream, an
        # unannotated k with a default, a Constexpr alpha and an unannotated cfg that
        # defaults to None.
        import numpy as np

        info = CuteDSLHostFnInfo(
            params=("x", "n", "stream", "k", "alpha", "cfg"),
            stream_param="stream",
            constexpr_params=frozenset({"alpha"}),
            numeric_params={"n": int},
            defaults={"k": 0.5, "cfg": None},
        )
        for names, msg in (
            (["x", "alpha"], r"missing arguments \['n'\]"),
            (["x", "n", "alpha", "kk"], r"got an unexpected argument 'kk'; its param"),
            (["x", "n", "alpha", "stream"], "got a value for its stream parameter"),
        ):
            with self.assertRaisesRegex(TypeError, f"^CuTeDSL kernel f: {msg}"):
                info.check_call("f", names)
        info.check_call("f", ["x", "n", "alpha"])

        x = torch.zeros(4)
        runtime, constants = info.split_args("f", {"x": x, "n": 3, "alpha": 2})
        self.assertEqual(list(runtime), ["x", "n", "k"])
        self.assertEqual(constants, {"alpha": 2, "cfg": None})
        # None is never a runtime scalar, not even for a cutlass Numeric parameter.
        _, constants = info.split_args("f", {"x": x, "n": None, "alpha": 2})
        self.assertEqual(constants, {"n": None, "alpha": 2, "cfg": None})
        # numpy scalars of bool, integer and floating types for an unannotated
        # parameter are runtime scalars, a complex one is not.
        for k in (np.float32(1.5), np.float16(1.5), np.int64(3), np.bool_(True)):
            runtime, _ = info.split_args("f", {"x": x, "n": 3, "alpha": 2, "k": k})
            self.assertIs(runtime["k"], k)
        _, constants = info.split_args("f", {"x": x, "n": 3, "alpha": 2, "k": 1j})
        self.assertEqual(constants["k"], 1j)
        _, constants = info.split_args(
            "f", {"x": x, "n": 3, "alpha": 2, "k": np.complex64(1j)}
        )
        self.assertIn("k", constants)
        tensor_for_scalar = {"n": {"n": x, "alpha": 2}, "alpha": {"n": 3, "alpha": x}}
        for param, args in tensor_for_scalar.items():
            msg = f"^CuTeDSL kernel f: parameter '{param}' takes a scalar .*, not a "
            with self.assertRaisesRegex(TypeError, msg + "tensor$"):
                info.split_args("f", {"x": x, **args})
        info = dataclasses.replace(info, defaults={"x": x, "k": 0.5, "cfg": None})
        msg = "^CuTeDSL kernel f: parameter 'x' defaults to a tensor; pass the tensor$"
        with self.assertRaisesRegex(ValueError, msg):
            info.split_args("f", {"n": 3, "alpha": 2})

    def test_python_scalar_converts_numpy_scalars(self):
        # The DSL takes only an exact bool, int or float for a runtime scalar of an
        # unannotated parameter.
        import numpy as np

        for value, expected in (
            (np.float32(1.5), 1.5),
            (np.float16(1.5), 1.5),
            (np.float64(1.5), 1.5),
            (np.int64(-3), -3),
            (np.uint64(2**64 - 1), 2**64 - 1),
            (np.bool_(True), True),
        ):
            converted = python_scalar(value)
            self.assertIs(type(converted), type(expected))
            self.assertEqual(converted, expected)

    def test_cutedsl_trigger_refused_to_other_dsls(self):
        # ReInplaceTrigger.CUTEDSL_OPS belongs to the cutedsl DSL (TRIGGER_DSLS)
        # whether or not CuTeDSL has registered yet (it registers when
        # cutedsl_kernel_wrap is first imported), so another DSL registering first
        # cannot take it and make CuTeDSL's own registration fail.
        from torch._higher_order_ops.user_kernel_hop import (
            _user_kernel_hops,
            get_user_kernel_hop_info,
            register_user_kernel_hop,
        )

        info = get_user_kernel_hop_info(cutedsl_kernel_wrapper_mutation)
        msg = (
            "^ReInplaceTrigger.CUTEDSL_OPS belongs to the cutedsl DSL; register the "
            "toy DSL with ReInplaceTrigger.USER_KERNEL_OPS or a member of its own$"
        )
        with mock.patch.dict(_user_kernel_hops, clear=True):
            with self.assertRaisesRegex(ValueError, msg):
                register_user_kernel_hop(dataclasses.replace(info, dsl="toy"))
            self.assertEqual(_user_kernel_hops, {})

    def test_cutedsl_layout_constraint_ignores_triton_config(self):
        # Inductor keeps the traced strides of a CuTeDSL kernel's tensors, whatever
        # Triton's layout config says.
        from torch._higher_order_ops.user_kernel_hop import get_user_kernel_hop_info
        from torch._inductor import config

        info = get_user_kernel_hop_info(cutedsl_kernel_wrapper_mutation)
        for layout in ("needs_fixed_stride_order", "flexible_layout"):
            with config.patch(triton_kernel_default_layout_constraint=layout):
                constraint = info.get_layout_constraint()
            self.assertEqual(constraint, "needs_fixed_stride_order")


if __name__ == "__main__":
    run_tests()
