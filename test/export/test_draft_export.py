# Owner(s): ["oncall: export"]
import copy
import tempfile
import unittest

import torch
from torch.export import Dim, export
from torch.export._draft_export import draft_export, FailureType
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)
from torch.utils._pytree import tree_leaves


class TestDraftExport(TestCase):
    def setUp(self):
        init_torchbind_implementations()

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, queue):
                self.queue = queue

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                return cls(**dict(flattened_ctx))

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                return self.queue.pop(0)

            def size(self):
                return len(self.queue)

            def is_empty(self):
                return len(self.queue) == 0

            def float_size(self):
                return float(len(self.queue))

        self.torch_bind_ops = [
            torch.ops._TorchScriptTesting.queue_pop,
            torch.ops._TorchScriptTesting.queue_push,
            torch.ops._TorchScriptTesting.queue_size,
        ]

    def tearDown(self):
        torch._library.fake_class_registry.deregister_fake_class(
            "_TorchScriptTesting::_TensorQueue"
        )

    def test_missing_meta_kernel_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo2", mutates_args={})
            def foo2_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return a + b

            class M(torch.nn.Module):
                def forward(self, a, b):
                    res = torch.ops.mylib.foo2(a, b)
                    return res

            inp = (torch.ones(3, 3), torch.ones(3, 3))

            ep, report = draft_export(M(), inp)

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )

            inp = (torch.randn(3, 3), torch.randn(3, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))

    def test_missing_meta_kernel_impl(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            def foo_impl(a, b):
                return a + b

            class M(torch.nn.Module):
                def forward(self, a, b):
                    res = torch.ops.mylib.foo(a, b)
                    return res

            inp = (torch.ones(3, 3), torch.ones(3, 3))

            ep, report = draft_export(M(), inp)

            self.assertEqual(len(report.failures), 1)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )

            inp = (torch.randn(3, 3), torch.randn(3, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))

    @unittest.skipIf(not torch.cuda.is_available(), "Requires cuda")
    def test_missing_meta_kernel_guard(self):
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo4", mutates_args={})
            def foo4_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return a + b

            class M(torch.nn.Module):
                def forward(self, a, b):
                    res1 = torch.ops.mylib.foo4(a, b)
                    return res1

            inp = (
                torch.ones(3, 4),
                torch.ones(3, 4),
            )

            ep, report = draft_export(
                M(),
                inp,
                dynamic_shapes={
                    "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                    "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                },
            )

            inp = (torch.randn(2, 3), torch.randn(2, 3))
            self.assertEqual(ep.module()(*inp), M()(*inp))
            m = ep.module()
            with self.assertRaisesRegex(RuntimeError, "Tensor device mismatch!"):
                bad_device_inps = (
                    torch.randn(2, 3, device=torch.device("cuda")),
                    torch.randn(2, 3, device=torch.device("cuda")),
                )
                m(*bad_device_inps)

            with self.assertRaisesRegex(RuntimeError, "Tensor dtype mismatch!"):
                bad_dtype_inps = (
                    torch.randn(2, 3, dtype=torch.float16),
                    torch.randn(2, 3, dtype=torch.float16),
                )
                m(*bad_dtype_inps)

    def test_fake_infer_dense_in_memory_check(self):
        with torch.library._scoped_library("mylib", "FRAGMENT"):

            @torch.library.custom_op("mylib::foo5", mutates_args={})
            def foo5_impl(a: torch.Tensor) -> torch.Tensor:
                return a * 2

            @torch.library.custom_op("mylib::foo6", mutates_args={})
            def foo6_impl(a: torch.Tensor) -> torch.Tensor:
                return (a * 2)[:, :-1, :-1]  # not dense in memory

            @torch.library.custom_op("mylib::foo7", mutates_args={})
            def foo7_impl(a: torch.Tensor) -> torch.Tensor:
                return (a * 2)[:, 1:-1, :]  # non-zero storage offset

            class Foo(torch.nn.Module):
                def forward(self, x, opt):
                    if opt == 0:
                        return torch.ops.mylib.foo5(x)
                    elif opt == 1:
                        return torch.ops.mylib.foo6(x)
                    else:
                        return torch.ops.mylib.foo7(x)

            draft_export(Foo(), (torch.randn(80, 4, 4), 0))
            draft_export(Foo(), (torch.randn(80, 1, 4), 0))
            draft_export(Foo(), (torch.randn(1, 4, 1, 1, 4, 1, 4), 0))
            with self.assertRaisesRegex(
                RuntimeError,
                "a return was not dense in memory",
            ):
                draft_export(Foo(), (torch.randn(4, 6, 8), 1))
            with self.assertRaisesRegex(
                RuntimeError,
                "a return has a non-zero storage offset",
            ):
                draft_export(Foo(), (torch.randn(4, 6, 8), 2))

    def test_data_dependent_failure(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo1",
                "(Tensor a, Tensor b) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo1", "cpu", lib=lib)
            def foo_impl(a, b):
                return a + b

            class M(torch.nn.Module):
                def forward(self, a, b, c):
                    res = torch.ops.mylib.foo1(a, b)

                    c_item = c.item()
                    return res[:c_item]

            inp = (torch.ones(3, 3), torch.ones(3, 3), torch.tensor(3))

            ep, report = draft_export(M(), inp)
            self.assertTrue(len(report.failures) > 0)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.MISSING_FAKE_KERNEL
            )
            self.assertEqual(
                report.failures[1].failure_type, FailureType.DATA_DEPENDENT_ERROR
            )

            inp = (torch.randn(3, 3), torch.randn(3, 3), torch.tensor(2))
            self.assertEqual(ep.module()(*inp), M()(*inp))

    def test_unbacked_div_mod_replacement(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x = torch.zeros(x.item())
                x = x.unsqueeze(0).repeat(10, 2)
                return x.view(-1, 2, 2345)

        _, report = draft_export(M(), (torch.tensor([938]),))
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.DATA_DEPENDENT_ERROR
        )
        self.assertEqual(report.failures[0].data["expr"], "Eq(2*u1, 10)")

    def test_dedup_data_dependent_failure(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                res = 0
                for v in [x, y]:
                    if v.item() > 10:
                        res += v * v
                    else:
                        res += v + v

                return z * res

        inp = (torch.tensor(5), torch.tensor(3), torch.tensor(2))

        ep, report = draft_export(M(), inp)
        self.assertTrue(len(report.failures) > 0)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.DATA_DEPENDENT_ERROR
        )

        inp = (torch.tensor(4), torch.tensor(2), torch.tensor(6))
        self.assertEqual(ep.module()(*inp), M()(*inp))

    def test_complex_data_dependent_expr(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                a = x.item()
                a = -a
                a = a // 3
                a = a + 5

                z = torch.cat([y, y])

                torch._check_is_size(a)
                torch._check(a < z.shape[0])

                return z[:a]

        _, report = draft_export(
            M(),
            (torch.tensor(6), torch.randn(5)),
            dynamic_shapes={"x": None, "y": {0: Dim.DYNAMIC}},
        )
        self.assertTrue(len(report.failures) > 0)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.DATA_DEPENDENT_ERROR
        )

    def test_offsets(self):
        class M(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                if a == 0:
                    raise RuntimeError("bad")
                return x * a

        inp = (torch.tensor(3),)
        ep, report = draft_export(M(), inp)

    def test_shape_failure(self):
        class M(torch.nn.Module):
            def forward(self, a):
                assert a.shape[0] == 3
                return a * a

        inp = (torch.ones(3, 3),)

        ep, report = draft_export(M(), inp, dynamic_shapes={"a": {0: Dim("a0")}})

        self.assertEqual(len(report.failures), 1)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.CONSTRAINT_VIOLATION_ERROR
        )

        inp = (torch.randn(3, 3),)
        self.assertEqual(ep.module()(*inp), M()(*inp))

        inp = (torch.randn(4, 3),)
        with self.assertRaises(RuntimeError):
            ep.module()(*inp)

    def test_side_effect1(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.tensor(2))

            def forward(self, b):
                a_item = self.a.item()
                if a_item == 2:
                    res = a_item * b
                else:
                    res = (a_item + 1) * b

                self.a.add_(1)
                a_item = self.a.item()

                if a_item == 3:
                    res = a_item * res
                else:
                    res = (a_item + 1) * res
                return res

        inp = (torch.ones(3, 3),)
        mod = M()
        ep, report = draft_export(mod, inp)
        self.assertEqual(mod.a, torch.tensor(2))
        FileCheck().check_count("torch.ops.aten.add.default", 0, exactly=True).run(
            ep.graph_module.code
        )

    def test_side_effect_inps(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x.sin_()
                return x

        inp = (torch.ones(3, 3),)
        ep, report = draft_export(M(), inp)
        self.assertTrue(report.successful())
        self.assertEqual(inp[0], torch.ones(3, 3))

    def test_torchbind(self):
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, tq, x):
                x_cos = tq.pop() + tq.float_size() + self.linear(x)
                if tq.is_empty():
                    x_sin = self.linear(tq.pop()) - tq.size() + x
                else:
                    x_sin = tq.pop() + tq.size() + x
                return x_sin, x_cos, tq

        mod = Model()
        tq = _empty_tensor_queue()
        tq2 = copy.deepcopy(tq)
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        tq.push(a)
        tq.push(b)
        tq3 = copy.deepcopy(tq)
        inp = (tq, torch.randn(2, 2))
        ep, report = draft_export(mod, inp)
        self.assertTrue(report.successful())
        self.assertEqual(tq2.size(), 0)
        self.assertEqual(tq3.size(), 2)
        self.assertEqual(tq.size(), 2)

    def test_override_size_and_dtype_mismatched_fake_kernels(self):
        class M(torch.nn.Module):
            def forward(self, a):
                return torch.ops.mylib.foo(a)

        @torch.library.custom_op("mylib::foo", mutates_args={})
        def foo(a: torch.Tensor) -> list[torch.Tensor]:
            x = a * 2
            y = a.repeat(2, 2)
            z = a.to(torch.bfloat16)
            return [x, y, z]

        @foo.register_fake
        def foo_fake_impl(a):
            x = torch.empty_like(a)  # good
            y = torch.empty_like(a)  # size mismatch
            z = torch.empty_like(a)  # dtype mismatch
            return [x, y, z]

        mod = M()
        inputs = (torch.randn(3, 3),)
        with self.assertRaises(RuntimeError):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                export(mod, inputs, strict=True)

        ep, report = draft_export(mod, inputs)
        for ep_out, eager_out in zip(ep.module()(*inputs), mod(*inputs)):
            self.assertTrue(torch.allclose(ep_out, eager_out))
            self.assertEqual(ep_out.dtype, eager_out.dtype)

        self.assertEqual(len(report.failures), 2)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
        )
        self.assertEqual(
            report.failures[1].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
        )
        self.assertEqual(
            sorted([f.data["reason"] for f in report.failures]),
            [
                "Dtypes torch.bfloat16 and torch.float32 are not equal!",
                "mismatch between fake value 3 and real value 6 ",
            ],
        )

    def test_override_incorrectly_aliasing_kernel(self):
        class M(torch.nn.Module):
            def forward(self, a):
                return torch.ops.mylib.foo(a)

        @torch.library.custom_op("mylib::foo", mutates_args={})
        def foo(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return a * 2, a + 2

        @foo.register_fake
        def foo_fake_impl(a):
            return a, torch.empty_like(a)  # incorrectly aliasing

        mod = M()
        inputs = (torch.randn(3, 3),)
        with self.assertRaisesRegex(
            RuntimeError,
            "Real tensor propagation found an aliasing mismatch",
        ):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                export(mod, inputs, strict=True)

        ep, report = draft_export(mod, inputs)
        for ep_out, eager_out in zip(
            tree_leaves(ep.module()(*inputs)), tree_leaves(mod(*inputs))
        ):
            self.assertTrue(torch.allclose(ep_out, eager_out))
            self.assertEqual(ep_out.dtype, eager_out.dtype)

        self.assertEqual(len(report.failures), 1)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
        )
        self.assertTrue(
            "Mismatched aliasing spec between fake kernel and real kernel"
            in report.failures[0].data["reason"]
        )

    def test_override_mismatched_fake_kernel_with_unbacked_symbols(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.mylib.foo(a, b)

        @torch.library.custom_op("mylib::foo", mutates_args={})
        def foo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a[b.item()].to(torch.bfloat16)

        @foo.register_fake
        def foo_fake_impl(a, b):
            ctx = torch.library.get_ctx()
            u = ctx.new_dynamic_size()
            return torch.empty(u, a.shape[1], dtype=a.dtype)

        mod = M()
        inputs = (torch.randn(100, 4), torch.tensor(10))

        ep, report = draft_export(mod, inputs)
        for ep_out, eager_out in zip(ep.module()(*inputs), mod(*inputs)):
            self.assertTrue(torch.allclose(ep_out, eager_out))
            self.assertEqual(ep_out.dtype, eager_out.dtype)

        self.assertEqual(len(report.failures), 1)
        self.assertEqual(
            report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL
        )
        self.assertEqual(
            report.failures[0].data["reason"],
            "Dtypes torch.bfloat16 and torch.float32 are not equal!",
        )

    # https://github.com/pytorch/pytorch/issues/140625
    @unittest.skipIf(IS_WINDOWS, "aoti_compile_and_package not supported on Windows")
    def test_constantify_unbacked_symbol(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                xt = torch.tensor(x.shape)
                u0 = xt[0].item()
                return y * torch.arange(u0)

        mod = M()
        example_inputs = (torch.randn(3, 5), torch.randn(3))
        draft_ep, _ = draft_export(mod, example_inputs)
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            torch._inductor.aoti_compile_and_package(
                draft_ep,
                package_path=f.name,
            )


if __name__ == "__main__":
    run_tests()
