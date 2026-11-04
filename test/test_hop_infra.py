# Owner(s): ["module: higher order operators"]
import collections
import copy
import dataclasses
import importlib
import logging
import pkgutil
import re
import types
from unittest import mock

import torch
from torch._higher_order_ops.cutedsl_kernel_wrap import (
    _may_overlap,
    CUTEDSL_CONSTANT_RULES,
    cutedsl_kernel_side_table,
    cutedsl_kernel_wrapper_mutation,
    CuteDSLHostFnInfo,
    CuteDSLKernel,
    python_scalar,
)
from torch._higher_order_ops.user_kernel_hop import (
    _snapshot,
    constant_key,
    KernelSideTable,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
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
    split, and the overlap checks of functionalization. The tests that compile and
    launch host functions are in test/inductor/test_cutedsl_kernels.py."""

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
            __module__ = "cutlass.fake"

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
        # share one entry), so an unhashable one is refused with that reason.
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
        msg = "^a constant kernel argument of type {} is unhashable: {}"
        with self.assertRaisesRegex(TypeError, msg.format("set", "pass a hashable")):
            add({1, 2})
        with self.assertRaisesRegex(TypeError, msg.format("EqSlotCfg", "it has state")):
            add(EqSlotCfg(1.0))

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
        # A value of a class from the cutlass package is the DSL's own, never keyed
        # or copied by its attributes: equal ones get their own entries, which hold
        # the objects themselves.
        class DslCfg:
            __module__ = "cutlass.cute.fake"

            def __init__(self, alpha):
                self.alpha = alpha

        table = KernelSideTable(constant_rules=CUTEDSL_CONSTANT_RULES)
        cfg = DslCfg(2.0)
        idx = table.add_constant_args({"cfg": cfg})
        self.assertNotEqual(table.add_constant_args({"cfg": DslCfg(2.0)}), idx)
        self.assertIs(table.get_constant_args(idx)["cfg"], cfg)

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
        msg = (
            "^a constant kernel argument of type EqSlotBaseCfg is unhashable: it has "
            "state in __slots__ attributes that are not dataclass fields, so it is "
            "keyed as itself, not by content; make it hashable, or keep that state in "
            "dataclass fields or its __dict__$"
        )
        with self.assertRaisesRegex(TypeError, msg):
            add(EqSlotBaseCfg(1.0))

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
        for value, name, path in cycles:
            msg = f"^a constant kernel argument of type {name} refers to itself "
            msg += f"through {path}$"
            for walk in (constant_key, _snapshot):
                with self.assertRaisesRegex(ValueError, msg):
                    walk(value, rules)
            with self.assertRaisesRegex(ValueError, msg):
                side_table.add_constant_args({"cfg": value})
        # A logger of the process: which cycle its walk meets first depends on the
        # process.
        msg = r"^a constant kernel argument of type \w+ refers to itself through .+$"
        with self.assertRaisesRegex(ValueError, msg):
            constant_key(Cfg(1.0, log=logging.getLogger(__name__)), rules)

        shared = Cfg(1.0, lookup={"k": [0.5]})
        twice = Cfg(2.0, a=shared, b=[shared], c=shared.lookup)
        snapshot = _snapshot(twice, rules)
        self.assertEqual(constant_key(snapshot, rules), constant_key(twice, rules))
        self.assertTrue(all(c is not shared for c in (snapshot.a, snapshot.b[0])))
        self.assertIsNot(snapshot.c, shared.lookup)

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

        results = []

        def f(x, y, z):
            n, m = y.shape[0], z.shape[0]
            halves = x.view(torch.float16)
            head, tail = x[:n], x[m:]
            halves_n, halves_m = halves[2 * n :], halves[2 * m :]
            shape_env = n.node.shape_env
            guards = len(shape_env.guards)
            results.extend(_may_overlap(head, h) for h in (halves_n, halves_m))
            self.assertEqual(len(shape_env.guards), guards)
            results.append(_may_overlap(head, tail))
            return x

        sizes = (100, 10, 20)
        gm = make_fx(f, tracing_mode="symbolic")(*map(torch.randn, sizes))
        self.assertEqual(results, [False, True, False])
        inputs = [node.meta["val"] for node in gm.graph.find_nodes(op="placeholder")]
        holds = gm.shape_env.evaluate_guards_for_args
        self.assertTrue(holds(inputs, list(map(torch.randn, sizes))))
        self.assertFalse(holds(inputs, list(map(torch.randn, (100, 20, 10)))))

        shape_env = ShapeEnv()
        with FakeTensorMode(shape_env=shape_env):
            u = shape_env.create_unbacked_symint()
            x = torch.empty(100)
            self.assertTrue(_may_overlap(x[:u], x[u:]))

    def test_overlap_check_rechecks_2d_rows(self):
        # TODO(zoranzhao): remove with _rows_disjoint once
        # https://github.com/pytorch/pytorch/pull/202848 lands. Without that fix,
        # compute_overlapping_tensors answers "disjoint" for 2-D views whose rows
        # start inside the other's rows other than the first, such as base[:4, :3]
        # and base[1:5, 2:5]. With its answer forced to "disjoint" for every pair,
        # _may_overlap still finds exactly the views that overlap, in either order,
        # with real and symbolic sizes. The pairs marked * overlap in a way the real
        # function finds itself, so only the forced answer has _rows_disjoint decide.
        def pairs(base):
            s, (n, d) = base.stride(0), base.shape
            qkv = base.new_empty(n, 3 * d)
            q, k, v = (qkv.narrow(1, i * d, d) for i in range(3))
            # Rows that start past the other's rows but run into their next row.
            narrow, wide = (base.as_strided((4, w), (s, 1), s - 2) for w in (3, 5))
            row = base[0]
            return (
                ("starts in a later row", base[:4, :3], base[1:5, 2:5], True),
                ("starts in a later row, swapped", base[1:5, 2:5], base[:4, :3], True),
                ("runs into the next row*", narrow, base[:4, :3], True),
                ("expanded rows*", row[:3].expand(4, 3), row[2:5].expand(4, 3), True),
                ("wider first*", base[:4, :5], base[:4, 3:5], True),
                ("wider second*", base[:4, :1], wide, True),
                ("unequal widths", base[:4, :3], base[:4, 3:8], False),
                ("empty", base[:4, :3], base[1:3, 0:0], False),
                ("ranges do not meet", base[:2, :5], base[2:4, :5], False),
                ("ranges do not meet, swapped", base[2:4, :5], base[:2, :5], False),
                ("q and k", q, k, False),
                ("k and v", k, v, False),
                ("q and v", q, v, False),
            )

        results = []

        def f(base):
            for name, a, b, overlap in pairs(base):
                results.append((name, _may_overlap(a, b), overlap))
            return base

        with mock.patch.object(
            torch._C._dynamo.guards, "compute_overlapping_tensors", return_value=set()
        ) as overlapping:
            f(torch.randn(10, 25))
            make_fx(f, tracing_mode="symbolic")(torch.randn(10, 25))
        self.assertEqual(overlapping.call_count, len(results))
        self.assertEqual(len(results), 26)
        for name, overlaps, expected in results:
            self.assertEqual(overlaps, expected, name)

    @parametrize("api", ["python", "cpp", "cpp_view_copies"])
    def test_functionalization_refuses_writes_that_may_overlap(self, api):
        # Each written argument is cloned and written back on its own, so two written
        # arguments that may overlap would lose a write: functionalization refuses the
        # same tensor twice, and views that overlap whatever their shapes, offsets and
        # dtypes. Views of one base that do not overlap are accepted, unless
        # functionalization copies views (remove="mutations_and_views"): the copies
        # do not show where the views sit, so any two views are refused, with a
        # message of their own. With mutates_args="unknown" the record needs no
        # signature, and with fake tensors the kernel never runs, so a plain function
        # stands for a host function. test/inductor/test_cutedsl_kernels.py runs the
        # accepted calls and the refusals under an explicit mutates_args.
        def add_kernel(mX, mY, mOut, n, stream):
            raise AssertionError("never launched")

        table = cutedsl_kernel_side_table
        kidx = table.add_kernel(CuteDSLKernel(add_kernel, "unknown"))
        cidx = table.add_constant_args({})

        def launch(x, y, out):
            kwargs = {"mX": x, "mY": y, "mOut": out, "n": out.numel()}
            cutedsl_kernel_wrapper_mutation(
                kernel_idx=kidx, constant_args_idx=cidx, kwargs=kwargs
            )

        def trace(f, *args):
            if api == "python":
                from torch._subclasses.functional_tensor import (
                    FunctionalTensorMode,
                    PythonFunctionalizeAPI,
                )

                with FunctionalTensorMode():
                    f_func = PythonFunctionalizeAPI().functionalize(f)
                    return make_fx(f_func, tracing_mode="fake")(*args)
            remove = "mutations_and_views" if api == "cpp_view_copies" else "mutations"
            f_func = torch.func.functionalize(f, remove=remove)
            return make_fx(f_func, tracing_mode="fake")(*args)

        def in_place(x, y):
            launch(x, y, x)
            return x

        def reshaped(x, y):
            launch(x, y, x.view(10, 100))
            return x

        def shifted(x, y):
            launch(x[1:], y[1:], x[:-1])
            return x

        def retyped(x, y):
            launch(x, y, x.view(torch.float16)[1000:])  # the bytes of x[500:]
            return x

        def overlapping_columns(x, y):
            # Rows 1 to 3 of column 2 are in both: each row of out starts inside a
            # row of the other, though not inside the first one.
            base = x.view(10, 100)
            launch(base[:4, :3], y[:12].view(4, 3), base[1:5, 2:5])
            return x

        def views(base, y):
            launch(base[0], y, base[1])
            return base

        def columns(qkv):
            d = qkv.shape[1] // 3
            launch(*(qkv.narrow(1, i * d, d) for i in range(3)))
            return qkv

        def empty(x, y):
            launch(x, y, x[500:500])  # an empty view overlaps nothing
            return x

        written = (
            ", and both count as written (mutates_args='unknown'). PT2 clones each "
            "written argument and writes it back on its own, so one write could be "
            "lost. "
        )
        narrow = " Or give mutates_args naming only the parameters it writes."
        overlap = (
            "CuTeDSL kernel add_kernel: 'mX' and 'mOut' are the same tensor or views "
            f"that may overlap{written}Pass tensors that do not overlap.{narrow}"
        )
        copied = (
            "CuTeDSL kernel add_kernel: 'mX' and '{}' are views of one tensor, which "
            'functionalization with remove="mutations_and_views" copied, so PT2 '
            f"cannot tell whether they overlap{written}Functionalize with "
            'remove="mutations" (the default), which keeps views, or pass tensors '
            f"that are not views of one tensor.{narrow}"
        )
        x, y = torch.randn(1000), torch.randn(1000)
        with self.assertRaisesRegex(RuntimeError, re.escape(overlap)):
            trace(in_place, x.clone(), y)
        view_copies = api == "cpp_view_copies"
        msg = copied.format("mOut") if view_copies else overlap
        for f in (reshaped, shifted, retyped, overlapping_columns):
            with self.assertRaisesRegex(RuntimeError, re.escape(msg)):
                trace(f, x.clone(), y)
        base, qkv = torch.stack([x, x]), torch.randn(40, 75)
        if view_copies:
            for f, args, name in ((views, (base, y), "mOut"), (columns, (qkv,), "mY")):
                msg = re.escape(copied.format(name))
                with self.assertRaisesRegex(RuntimeError, msg):
                    trace(f, *args)
        else:
            trace(views, base, y)
            trace(columns, qkv)
            trace(empty, x, y)


instantiate_parametrized_tests(TestCuteDSLHOPHelpers)


if __name__ == "__main__":
    run_tests()
