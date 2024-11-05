# Owner(s): ["oncall: export"]
import copy
from typing import List

import torch
from torch.export import Dim, export
from torch.export._draft_export import draft_export, FailureType
from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase
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
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:

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

            @torch.library.register_fake("mylib::foo1", lib=lib)
            def mylib_foo_default_fake(*args, **kwargs):
                ctx = torch.library.get_ctx()
                fake_shape = [ctx.new_dynamic_size() for _ in range(2)]
                return torch.empty(fake_shape, dtype=torch.float32, device="cpu")

            class M(torch.nn.Module):
                def forward(self, a, b, c):
                    res = torch.ops.mylib.foo1(a, b)

                    c_item = c.item()
                    return res[:c_item]

            inp = (torch.ones(3, 3), torch.ones(3, 3), torch.tensor(3))

            ep, report = draft_export(M(), inp)
            self.assertTrue(len(report.failures) > 0)
            self.assertEqual(
                report.failures[0].failure_type, FailureType.DATA_DEPENDENT_ERROR
            )

            inp = (torch.randn(3, 3), torch.randn(3, 3), torch.tensor(2))
            self.assertEqual(ep.module()(*inp), M()(*inp))

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
        def foo(a: torch.Tensor) -> List[torch.Tensor]:
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
                export(mod, inputs)

        ep, report = draft_export(mod, inputs)
        for ep_out, eager_out in zip(ep.module()(*inputs), mod(*inputs)):
            self.assertTrue(torch.allclose(ep_out, eager_out))
            self.assertEqual(ep_out.dtype, eager_out.dtype)

        self.assertEqual(len(report.failures), 2)
        self.assertEqual(report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL)
        self.assertEqual(report.failures[1].failure_type, FailureType.MISMATCHED_FAKE_KERNEL)
        self.assertEqual(
            sorted([f.data["reason"] for f in report.failures]),
            [
                "Dtypes torch.float32 and torch.bfloat16 are not equal!",
                "mismatch between fake value 3 and real value 6 ",
            ],
        )

    def test_override_pytree_mismatched_fake_kernels(self):
        class M(torch.nn.Module):
            def forward(self, a):
                return torch.ops.mylib.foo(a)

        @torch.library.custom_op("mylib::foo", mutates_args={})
        def foo(a: torch.Tensor) -> List[torch.Tensor]:
            x = a * 2
            y = a * 2
            z = a * 2
            return [x, y, z]  # type: ignore[return-value]

        @foo.register_fake
        def foo_fake_impl(a):
            x = torch.empty_like(a)
            y = torch.empty_like(a)
            return [x, y]  # mismatch on num outputs

        mod = M()
        inputs = (torch.randn(3, 3),)
        with self.assertRaises(RuntimeError):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                export(mod, inputs)

        ep, report = draft_export(mod, inputs)
        for ep_out, eager_out in zip(
            tree_leaves(ep.module()(*inputs)), tree_leaves(mod(*inputs))
        ):
            self.assertTrue(torch.allclose(ep_out, eager_out))
            self.assertEqual(ep_out.dtype, eager_out.dtype)

        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].failure_type, FailureType.MISMATCHED_FAKE_KERNEL)
        self.assertTrue(
            "Mismatched output structure between fake kernel with output TreeSpec:" in report.failures[0].data["reason"]
        )

if __name__ == "__main__":
    run_tests()
