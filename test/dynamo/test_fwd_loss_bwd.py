# Owner(s): ["module: dynamo"]

import torch
from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_utils import run_tests, skipIfCrossRef, TestCase


class TestForwardLossBackward(TestCase):
    @skipIfCrossRef
    def test_autograd_grad_basic(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)

        def fn(x):
            res = mod(x)
            loss = res.sum()
            params = tuple(mod.parameters())
            grads = torch.autograd.grad(loss, params)
            return loss.detach(), grads[0].sum(), grads[1].sum()

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        for e, c in zip(eager_result, compiled_result):
            self.assertEqual(e, c)
        self.assertEqual(len(backend.graphs), 1)

        gm = backend.graphs[0]
        actual = normalize_gm(gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
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
""",  # noqa: B950
        )

        self.assertEqual(len(backend.fw_graphs), 1)
        fw_graph = backend.fw_graphs[0]
        fw_actual = normalize_gm(fw_graph.print_readable(print_output=False))
        self.assertExpectedInline(
            fw_actual,
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 4]", arg1_1: "f32[4]", arg2_1: "f32[2, 4]"):
        t: "f32[4, 4]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
        addmm: "f32[2, 4]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, t);  arg1_1 = t = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(addmm);  addmm = None

        ones_like: "f32[]" = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format)
        expand: "f32[2, 4]" = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
        t_1: "f32[4, 2]" = torch.ops.aten.t.default(expand)
        mm: "f32[4, 4]" = torch.ops.aten.mm.default(t_1, arg2_1);  t_1 = arg2_1 = None
        t_2: "f32[4, 4]" = torch.ops.aten.t.default(mm);  mm = None
        sum_2: "f32[1, 4]" = torch.ops.aten.sum.dim_IntList(expand, [0], True);  expand = None
        view: "f32[4]" = torch.ops.aten.view.default(sum_2, [4]);  sum_2 = None
        t_3: "f32[4, 4]" = torch.ops.aten.t.default(t_2);  t_2 = None

        detach: "f32[]" = torch.ops.aten.detach.default(sum_1);  sum_1 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(t_3);  t_3 = None
        sum_4: "f32[]" = torch.ops.aten.sum.default(view);  view = None
        return (detach, sum_3, sum_4)
""",  # noqa: B950
        )

    @skipIfCrossRef
    def test_autograd_grad_with_kwargs(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)

        def fn(x):
            res = mod(x)
            loss = res.sum()
            params = tuple(mod.parameters())
            _ = torch.autograd.grad(
                outputs=loss,
                inputs=params,
                retain_graph=False,
                create_graph=False,
            )
            return loss.detach()

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(len(backend.graphs), 1)

        gm = backend.graphs[0]
        actual = normalize_gm(gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
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
""",  # noqa: B950
        )

        self.assertEqual(len(backend.fw_graphs), 1)
        fw_graph = backend.fw_graphs[0]
        fw_actual = normalize_gm(fw_graph.print_readable(print_output=False))
        self.assertExpectedInline(
            fw_actual,
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 4]", arg1_1: "f32[4]", arg2_1: "f32[2, 4]"):
        t: "f32[4, 4]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
        addmm: "f32[2, 4]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, t);  arg1_1 = arg2_1 = t = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(addmm);  addmm = None

        detach: "f32[]" = torch.ops.aten.detach.default(sum_1);  sum_1 = None
        return (detach,)
""",  # noqa: B950
        )

    @skipIfCrossRef
    def test_autograd_grad_single_tensor(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)

        def fn(x):
            res = mod(x)
            loss = res.sum()
            grad = torch.autograd.grad(loss, mod.weight)
            return loss.detach(), grad[0].sum()

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        for e, c in zip(eager_result, compiled_result):
            self.assertEqual(e, c)
        self.assertEqual(len(backend.graphs), 1)

        gm = backend.graphs[0]
        actual = normalize_gm(gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
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
""",  # noqa: B950
        )

        self.assertEqual(len(backend.fw_graphs), 1)
        fw_graph = backend.fw_graphs[0]
        fw_actual = normalize_gm(fw_graph.print_readable(print_output=False))
        self.assertExpectedInline(
            fw_actual,
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 4]", arg1_1: "f32[4]", arg2_1: "f32[2, 4]"):
        t: "f32[4, 4]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
        addmm: "f32[2, 4]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, t);  arg1_1 = t = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(addmm);  addmm = None

        ones_like: "f32[]" = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format)
        expand: "f32[2, 4]" = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
        t_1: "f32[4, 2]" = torch.ops.aten.t.default(expand);  expand = None
        mm: "f32[4, 4]" = torch.ops.aten.mm.default(t_1, arg2_1);  t_1 = arg2_1 = None
        t_2: "f32[4, 4]" = torch.ops.aten.t.default(mm);  mm = None
        t_3: "f32[4, 4]" = torch.ops.aten.t.default(t_2);  t_2 = None

        detach: "f32[]" = torch.ops.aten.detach.default(sum_1);  sum_1 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(t_3);  t_3 = None
        return (detach, sum_2)
""",  # noqa: B950
        )

    def test_autograd_grad_rejects_graph_input_with_grad_fn(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4, requires_grad=True)
        external_computation = x * 2

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(external_input):
            res = mod(external_input)
            loss = res.sum()
            _ = torch.autograd.grad(loss, external_input)
            return loss.detach()

        with self.assertRaisesRegex(
            RuntimeError,
            "Compiled function receives an input with external grad_fn",
        ):
            fn(external_computation)

    def test_autograd_grad_manual_update_matches_eager(self):
        mod_eager = torch.nn.Linear(4, 4)
        mod_compiled = torch.nn.Linear(4, 4)

        with torch.no_grad():
            mod_compiled.weight.copy_(mod_eager.weight)
            mod_compiled.bias.copy_(mod_eager.bias)

        x = torch.randn(2, 4)

        def step_fn(mod):
            res = mod(x)
            loss = res.sum()
            params = tuple(mod.parameters())
            param_grads = torch.autograd.grad(
                loss, params, materialize_grads=False, allow_unused=True
            )
            for p, g_p in zip(params, param_grads):
                if p.grad is None:
                    p.grad = g_p
                elif g_p is not None:
                    p.grad.add_(g_p)
            return loss.detach()

        eager_loss = step_fn(mod_eager)

        compiled_step_fn = torch.compile(
            lambda: step_fn(mod_compiled), backend="aot_eager", fullgraph=True
        )
        compiled_loss = compiled_step_fn()

        self.assertEqual(eager_loss, compiled_loss)

        self.assertIsNotNone(mod_eager.weight.grad)
        self.assertIsNotNone(mod_compiled.weight.grad)
        self.assertEqual(mod_eager.weight.grad, mod_compiled.weight.grad)

        self.assertIsNotNone(mod_eager.bias.grad)
        self.assertIsNotNone(mod_compiled.bias.grad)
        self.assertEqual(mod_eager.bias.grad, mod_compiled.bias.grad)

    def test_autograd_grad_missing_detach_errors_like_eager(self):
        mod_eager = torch.nn.Linear(4, 4)
        x_eager = torch.randn(2, 4)

        def step_eager():
            res = mod_eager(x_eager)
            loss = res.sum()
            params = tuple(mod_eager.parameters())
            torch.autograd.grad(loss, params)
            return loss

        loss_eager = step_eager()
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to backward through the graph a second time",
        ):
            loss_eager.backward()

        torch._dynamo.reset()
        mod_compiled = torch.nn.Linear(4, 4)
        x_compiled = torch.randn(2, 4)

        @torch.compile(fullgraph=True, backend="aot_eager")
        def step_compiled():
            res = mod_compiled(x_compiled)
            loss = res.sum()
            params = tuple(mod_compiled.parameters())
            torch.autograd.grad(loss, params)
            return loss

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Trying to backward through the graph a second time",
        ):
            step_compiled()


if __name__ == "__main__":
    run_tests()
