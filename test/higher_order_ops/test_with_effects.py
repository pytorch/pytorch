# Owner(s): ["module: functorch"]
import unittest
from collections import deque
from functools import partial
from typing import List

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch._functorch.aot_autograd import aot_export_module
from torch._higher_order_ops.effects import with_effects
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import _get_torch_cuda_version, SM80OrLater
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    TEST_CUDA,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations
from torch.utils.hooks import RemovableHandle  # noqa: TCH001


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestWithEffects(TestCase):
    def setUp(self):
        init_torchbind_implementations()

    def test_print(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._print("moo")
                res = x + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # Without functionalization, print should just appear in the graph directly
        gm = make_fx(M())(*inputs)
        FileCheck().check_count("torch.ops.aten._print.default", 2, exactly=True).run(
            gm.code
        )

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

        with torch._functorch.config.patch(unlift_effect_tokens=True):
            gm, gs = aot_export_module(M(), inputs, trace_joint=False)
            self.assertExpectedInline(
                str(gm.code).strip(),
                """\
def forward(self, arg1_1):
    _make_token_default = torch.ops.prims._make_token.default()
    with_effects = torch._higher_order_ops.effects.with_effects(_make_token_default, torch.ops.aten._print.default, 'moo');  _make_token_default = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default((getitem_2,));  getitem_2 = _sink_tokens_default = None
    return (add,)""",  # noqa: B950
            )

    def test_torchbind_custom_op(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return (x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x),)

        with enable_torchbind_tracing():
            gm, gs = aot_export_module(M(), (torch.ones(2, 3),), trace_joint=False)

        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _torchbind_obj0 = self._torchbind_obj0
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo.default, _torchbind_obj0, arg1_1);  arg0_1 = _torchbind_obj0 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, getitem_1);  arg1_1 = getitem_1 = None
    return (getitem, add)""",  # noqa: B950
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

    def test_print_with_buffer_mutations(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(3))

            def forward(self, x):
                torch.ops.aten._print("moo")
                res = x + x
                self.buf.add_(res)
                res = self.buf + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg2_1, arg2_1)
    add_1 = torch.ops.aten.add.Tensor(arg1_1, add);  arg1_1 = add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, arg2_1);  arg2_1 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add_1, add_2)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.buffers_to_mutate), 1)

    def test_print_with_input_mutations(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                torch.ops.aten._print("moo")
                res = x + x
                x.add_(res)
                res = x + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.user_inputs_to_mutate), 1)

    def test_alias_op(self):
        def f(token, x):
            token, out = with_effects(token, torch.ops.aten.absolute_.default, x)
            return token, out

        with self.assertRaisesRegex(
            AssertionError, r"Ops with aliasing is not supported"
        ):
            make_fx(f)(torch.tensor([]), torch.tensor(4))

    def test_compile_aot_eager(self):
        def f(x):
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3),)

        res = torch.compile(f, backend="aot_eager")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "triton")
    def test_compile_inductor(self):
        def f(x):
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3),)

        res = torch.compile(f, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @skipIfNoDynamoSupport
    def test_compile_inductor_external_op_return_none(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::inplace_add",
                "(Tensor input, Tensor(a!) output) -> ()",
                lib=lib,
            )

            def inplace_add(input: torch.Tensor, output: torch.Tensor) -> None:
                assert input.device == output.device
                output.add_(input)

            lib.impl("inplace_add", inplace_add, "CompositeExplicitAutograd")

            def f(x):
                out = torch.empty(3)
                out = torch.zeros_like(out)
                torch.ops.mylib.inplace_add(x, out)
                return out

            inputs = (torch.randn(3),)

            res = torch.compile(f, backend="inductor")(*inputs)
            self.assertTrue(torch.allclose(res, f(*inputs)))

    def test_compile_aot_eager_requires_grad(self):
        def f(x):
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3, requires_grad=True),)

        res = torch.compile(f, backend="aot_eager")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

        res.sum().backward()

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(TEST_WITH_ROCM, "triton")
    @unittest.skipIf(not SM80OrLater, "triton")
    @unittest.skipIf(_get_torch_cuda_version() >= (11, 7), "triton")
    @unittest.skipIf(not TEST_CUDA, "triton")
    @skipIfNoDynamoSupport
    def test_register_effectful_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            torch.library.define(
                "mylib::record_scalar_tensor",
                "(Tensor x, str prefix) -> ()",
                lib=lib,
            )

            # global variable to store the recorded tensor and prefix.
            recorded_dict = {}

            # Pytorch custorm op implementation
            @torch.library.impl(
                "mylib::record_scalar_tensor",
                "CompositeExplicitAutograd",
                lib=lib,
            )
            def record_scalar_tensor(x, prefix):
                recorded_dict[prefix] = x.clone()
                return

            # Meta function of the custom op
            @torch.library.impl_abstract(
                "mylib::record_scalar_tensor",
                lib=lib,
            )
            def record_scalar_tensor_meta(x, prefix):
                return

            from torch._higher_order_ops.effects import (
                _EffectType,
                _register_effectful_op,
            )

            _register_effectful_op(
                torch.ops.mylib.record_scalar_tensor.default, _EffectType.ORDERED
            )

            my_config = {}
            my_config["MockModule"] = "mean"
            my_config["MockModule.linear"] = "mean"
            my_config["MockModule.relu"] = "mean"

            class MyLinear(torch.nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.weight = torch.nn.Parameter(
                        torch.randn(out_features, in_features), requires_grad=True
                    )
                    self.bias = torch.nn.Parameter(
                        torch.randn(out_features), requires_grad=True
                    )

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.weight, self.bias)

            class MockModule(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = MyLinear(10, 10)
                    self.register_buffer(
                        "buf0", torch.randn(10, 10, requires_grad=True)
                    )

                def forward(self, x):
                    return torch.nn.functional.relu(self.linear(x) + self.buf0)

            def forward_hook(
                module: torch.nn.Module,
                inputs: torch.Tensor,
                output: torch.Tensor,
                prefix: str,
                aggregate_method: str,
            ) -> torch.Tensor:
                if aggregate_method == "mean":
                    torch.ops.mylib.record_scalar_tensor(output.mean(), prefix)
                elif aggregate_method == "max":
                    torch.ops.mylib.record_scalar_tensor(output.max(), prefix)
                else:
                    # demo purpose, using "min"
                    torch.ops.mylib.record_scalar_tensor(output.sum(), prefix)
                return output

            def add_hooks(module, config):
                handles: List[RemovableHandle] = []
                q = deque([(module.__class__.__name__, module)])
                while q:
                    name, m = q.pop()
                    children = [(name + "." + n, y) for (n, y) in m.named_children()]
                    q.extend(children)
                    aggregate_method = config.get(name, "mean")
                    prefix = name + ":" + aggregate_method
                    handle = m.register_forward_hook(
                        partial(
                            forward_hook,
                            prefix=prefix,
                            aggregate_method=aggregate_method,
                        )
                    )
                    if handle:
                        handles.append(handle)
                return handles

            x = torch.randn(10, 10, device="cuda")
            mod = MockModule().to("cuda")

            add_hooks(mod, my_config)

            opt_mod = torch.compile(backend="inductor")(mod)
            y = opt_mod(x)

            self.assertTrue(torch.allclose(y, mod(x)))
            # Ensure it works well with backward
            y.sum().backward()
            # Ensure the grad is existing
            self.assertTrue(isinstance(opt_mod.linear.weight.grad, torch.Tensor))

            self.assertEqual(len(recorded_dict), 2)
            self.assertTrue("MockModule.linear:mean" in recorded_dict)
            self.assertTrue("MockModule:mean" in recorded_dict)


if __name__ == "__main__":
    run_tests()
