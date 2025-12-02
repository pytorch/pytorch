# Owner(s): ["module: dynamo"]

import torch
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch.testing._internal.common_utils import run_tests, TestCase


class TestForwardLossBackward(TestCase):
    def test_autograd_grad_basic(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)

        def fn(x):
            res = mod(x)
            loss = res.sum()
            params = tuple(mod.parameters())
            grads = torch.autograd.grad(loss, params)
            return loss.detach(), grads[0].sum(), grads[1].sum()

        cnt = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        for e, c in zip(eager_result, compiled_result):
            self.assertEqual(e, c)
        self.assertEqual(cnt.frame_count, 1)

        gm = cnt.graphs[0]
        actual = normalize_gm(gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, """\
class GraphModule(torch.nn.Module):
    def forward(self, L_mod_parameters_weight_: "f32[4, 4]", L_mod_parameters_bias_: "f32[4]", L_x_: "f32[2, 4]"):
        l_mod_parameters_weight_ = L_mod_parameters_weight_
        l_mod_parameters_bias_ = L_mod_parameters_bias_
        l_x_ = L_x_

        res: "f32[2, 4]" = torch._C._nn.linear(l_x_, l_mod_parameters_weight_, l_mod_parameters_bias_);  l_x_ = None

        loss: "f32[]" = res.sum();  res = None

        grad = torch.autograd.grad(loss, (l_mod_parameters_weight_, l_mod_parameters_bias_));  l_mod_parameters_weight_ = l_mod_parameters_bias_ = None
        getitem: "f32[4, 4]" = grad[0]
        getitem_1: "f32[4]" = grad[1];  grad = None

        detach: "f32[]" = loss.detach();  loss = None
        sum_2: "f32[]" = getitem.sum();  getitem = None
        sum_3: "f32[]" = getitem_1.sum();  getitem_1 = None
        return (detach, sum_2, sum_3)
""")

    def test_autograd_grad_rejects_non_leaf(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4, requires_grad=True)

        def fn(x):
            res = mod(x)
            loss = res.sum()
            intermediate = res * 2
            grads = torch.autograd.grad(loss, (intermediate,))
            return loss.detach()

        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "torch.compile does not currently support torch.autograd.grad with non-leaf"
        ):
            compiled_fn(x)

    def test_autograd_grad_with_kwargs(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)

        def fn(x):
            res = mod(x)
            loss = res.sum()
            params = tuple(mod.parameters())
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=params,
                retain_graph=False,
                create_graph=False,
            )
            return loss.detach()

        cnt = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(cnt.frame_count, 1)

        gm = cnt.graphs[0]
        actual = normalize_gm(gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, """\
class GraphModule(torch.nn.Module):
    def forward(self, L_mod_parameters_weight_: "f32[4, 4]", L_mod_parameters_bias_: "f32[4]", L_x_: "f32[2, 4]"):
        l_mod_parameters_weight_ = L_mod_parameters_weight_
        l_mod_parameters_bias_ = L_mod_parameters_bias_
        l_x_ = L_x_

        res: "f32[2, 4]" = torch._C._nn.linear(l_x_, l_mod_parameters_weight_, l_mod_parameters_bias_);  l_x_ = None

        loss: "f32[]" = res.sum();  res = None

        grad = torch.autograd.grad(outputs = loss, inputs = (l_mod_parameters_weight_, l_mod_parameters_bias_), retain_graph = False, create_graph = False);  l_mod_parameters_weight_ = l_mod_parameters_bias_ = grad = None

        detach: "f32[]" = loss.detach();  loss = None
        return (detach,)
""")

    def test_autograd_grad_single_tensor(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)

        def fn(x):
            res = mod(x)
            loss = res.sum()
            grad = torch.autograd.grad(loss, mod.weight)
            return loss.detach(), grad[0].sum()

        cnt = CompileCounterWithBackend("aot_eager")
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        for e, c in zip(eager_result, compiled_result):
            self.assertEqual(e, c)
        self.assertEqual(cnt.frame_count, 1)

        gm = cnt.graphs[0]
        actual = normalize_gm(gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, """\
class GraphModule(torch.nn.Module):
    def forward(self, L_mod_parameters_weight_: "f32[4, 4]", L_mod_parameters_bias_: "f32[4]", L_x_: "f32[2, 4]"):
        l_mod_parameters_weight_ = L_mod_parameters_weight_
        l_mod_parameters_bias_ = L_mod_parameters_bias_
        l_x_ = L_x_

        res: "f32[2, 4]" = torch._C._nn.linear(l_x_, l_mod_parameters_weight_, l_mod_parameters_bias_);  l_x_ = l_mod_parameters_bias_ = None

        loss: "f32[]" = res.sum();  res = None

        grad = torch.autograd.grad(loss, l_mod_parameters_weight_);  l_mod_parameters_weight_ = None
        getitem: "f32[4, 4]" = grad[0];  grad = None

        detach: "f32[]" = loss.detach();  loss = None
        sum_2: "f32[]" = getitem.sum();  getitem = None
        return (detach, sum_2)
""")

    def test_autograd_grad_rejects_graph_input_with_grad_fn(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4, requires_grad=True)
        external_computation = x * 2

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(external_input):
            res = mod(external_input)
            loss = res.sum()
            grads = torch.autograd.grad(loss, external_input)
            return loss.detach()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "torch.compile does not currently support torch.autograd.grad with non-leaf"
        ):
            fn(external_computation)


if __name__ == "__main__":
    run_tests()
