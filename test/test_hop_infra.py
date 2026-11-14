# Owner(s): ["module: higher order operators"]
import contextlib
import copy
import importlib
import pkgutil
from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import (
    fake_tensor_modes,
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
@instantiate_parametrized_tests
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

    @fake_tensor_modes
    def test_triton_kernel_wrapper_functional_fake_tensor_nested_kwargs(self, mode_cls):
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_functional,
        )
        from torch._subclasses.fake_tensor import is_fake_tensor, maybe_get_fake_mode

        mode = mode_cls()
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

    @fake_tensor_modes
    def test_triton_kernel_wrapper_mutation_fake_tensor_nested_kwargs(self, mode_cls):
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_mutation,
        )
        from torch._subclasses.fake_tensor import is_fake_tensor

        mode = mode_cls()
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

    @fake_tensor_modes
    @parametrize("active", [False, True])
    @parametrize("raises", [False, True])
    def test_fake_handler_mode_state(self, mode_cls, active, raises):
        from torch._higher_order_ops.utils import _fake_mode_handler
        from torch._ops import HigherOrderOperator
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.utils._python_dispatch import is_in_torch_dispatch_mode

        class TestHOP(HigherOrderOperator):
            def __init__(self):
                super().__init__("_test_fake_handler_mode_state")

            def __call__(self, x):
                with torch._C._AutoDispatchBelowAutograd():
                    return super().__call__(x)

        def state():
            return (
                torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE),
                is_in_torch_dispatch_mode(),
                torch._C._only_lift_cpu_tensors(),
            )

        seen = []
        mode = mode_cls()
        fake = mode.from_tensor(torch.empty(2))
        with patch.dict(torch._ops._higher_order_ops):
            hop = TestHOP()

            @hop.py_impl(FakeTensorMode)
            def fake_handler(fake_mode, x):
                self.assertIs(fake_mode, mode)
                seen.append(state())
                if raises:
                    raise RuntimeError("fake handler failure")
                return x

            hop.py_impl(torch._C.DispatchKey.Fake)(_fake_mode_handler(fake_handler))
            with mode if active else contextlib.nullcontext():
                before = state()
                if raises:
                    with self.assertRaisesRegex(RuntimeError, "fake handler failure"):
                        hop(fake)
                else:
                    self.assertIs(hop(fake), fake)
                self.assertEqual(seen, [(None, *before[1:])])
                self.assertEqual(state(), before)

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


if __name__ == "__main__":
    run_tests()
