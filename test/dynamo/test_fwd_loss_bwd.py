# Owner(s): ["module: dynamo"]

import copy
import re
import textwrap

import torch
import torch._dynamo
from torch._dynamo.testing import AotEagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    TestCase,
)


@torch._dynamo.config.patch(trace_autograd_ops=True)
@skipIfTorchDynamo()
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
            return loss.detach(), grads[0], grads[1]

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        for e, c in zip(eager_result, compiled_result):
            self.assertEqual(e, c)
        self.assertEqual(len(backend.graphs), 1)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
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
        return (detach, getitem, getitem_1)
""",  # noqa: B950
        )

        self.assertEqual(len(backend.fw_graphs), 1)
        fw_actual = normalize_gm(
            backend.fw_graphs[0].print_readable(print_output=False)
        )
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
        return (detach, t_3, view)
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

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
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
        fw_actual = normalize_gm(
            backend.fw_graphs[0].print_readable(print_output=False)
        )
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
            return loss.detach(), grad[0]

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        for e, c in zip(eager_result, compiled_result):
            self.assertEqual(e, c)
        self.assertEqual(len(backend.graphs), 1)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
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
        return (detach, getitem)
""",  # noqa: B950
        )

        self.assertEqual(len(backend.fw_graphs), 1)
        fw_actual = normalize_gm(
            backend.fw_graphs[0].print_readable(print_output=False)
        )
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
        return (detach, t_3)
""",  # noqa: B950
        )

    @skipIfCrossRef
    def test_autograd_grad_double_backward_outside_compile(self):
        def fn(x, weight):
            y = x @ weight
            loss = y.sum()
            # Compute gradient w.r.t weight using autograd.grad
            # Use retain_graph=True to retain the graph for double backward
            (grad_weight,) = torch.autograd.grad(loss, weight, retain_graph=True)
            # Return loss - with retain_graph=True, loss can still be backwarded
            return loss, grad_weight

        x = torch.randn(2, 4, requires_grad=True)
        weight = torch.randn(4, 3, requires_grad=True)

        x_eager = x.clone().detach().requires_grad_(True)
        weight_eager = weight.clone().detach().requires_grad_(True)
        loss_eager, grad_weight_eager = fn(x_eager, weight_eager)
        loss_eager.backward()

        x_compile = x.clone().detach().requires_grad_(True)
        weight_compile = weight.clone().detach().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        loss_compile, grad_weight_compile = compiled_fn(x_compile, weight_compile)

        self.assertEqual(loss_eager, loss_compile)
        self.assertEqual(grad_weight_eager, grad_weight_compile)

        # Backward through the returned loss from compile should work
        loss_compile.backward()

        # Gradients should match
        self.assertEqual(x_eager.grad, x_compile.grad)
        self.assertEqual(weight_eager.grad, weight_compile.grad)

    @skipIfCrossRef
    def test_autograd_grad_with_external_grad_fn_as_grad_target(self):
        mod_eager = torch.nn.Linear(4, 4)
        x_eager = torch.randn(2, 4, requires_grad=True)
        external_computation_eager = x_eager * 2

        mod_compile = copy.deepcopy(mod_eager)
        x_compile = copy.deepcopy(x_eager).requires_grad_(True)
        external_computation_compile = x_compile * 2

        def fn(mod, external_input):
            res = mod(external_input)
            loss = res.sum()
            grads = torch.autograd.grad(loss, external_input)
            return loss.detach(), grads

        # This should work because we're computing grad w.r.t. external_input itself
        # The gradient computation stops at external_input, never traversing its grad_fn
        loss_eager, grads_eager = fn(mod_eager, external_computation_eager)
        loss_compile, grads_compile = torch.compile(
            fn, fullgraph=True, backend="aot_eager"
        )(mod_compile, external_computation_compile)
        self.assertEqual(loss_eager, loss_compile)
        self.assertEqual(grads_eager, grads_compile)

    @skipIfCrossRef
    def test_autograd_grad_rejects_external_grad_fn_on_output(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4, requires_grad=True)
        external_computation = x * 2

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(external_input):
            res = mod(external_input)
            loss = res.sum()
            # Computing grad w.r.t. model parameters when loss depends on
            # external_input which has an external grad_fn - this should fail
            # because we'd need to traverse external_input's grad_fn to reach params
            grads = torch.autograd.grad(loss, mod.weight)
            return loss.detach(), grads

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            re.escape(
                """\
autograd.grad with external grad_fn
  Explanation: torch.autograd.grad() cannot trace through the autograd graph because it's output depends on a tensor that was created outside the compiled region and has a grad_fn attached. The autograd graph extends beyond the compiled region boundary, which Dynamo cannot trace.
  Hint: If you don't need gradients to flow back to the original tensor outside the compiled region, detach the input: `tensor.detach().requires_grad_(True)`.
  Hint: Otherwise, move the autograd.grad() call outside the compiled region.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: inputs with external grad_fn: ["L['external_input']"]"""  # noqa: B950
            ),
        ):
            fn(external_computation)

    @skipIfCrossRef
    def test_autograd_grad_manual_update_matches_eager(self):
        mod_eager = torch.nn.Linear(4, 4)
        mod_compiled = copy.deepcopy(mod_eager)
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
        self.assertEqual(mod_eager.weight.grad, mod_compiled.weight.grad)
        self.assertEqual(mod_eager.bias.grad, mod_compiled.bias.grad)

    def test_autograd_grad_rejects_external_grad_fn_from_outer_scope(self):
        x = torch.randn(2, 4, requires_grad=True)
        external = x * 2

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(ext):
            y = (ext.sin()).sum()
            # x is from outer scope - this requires traversing ext's grad_fn
            gx = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)[0]
            return gx

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            re.escape(
                """\
autograd.grad with external grad_fn
  Explanation: torch.autograd.grad() cannot trace through the autograd graph because it's output depends on a tensor that was created outside the compiled region and has a grad_fn attached. The autograd graph extends beyond the compiled region boundary, which Dynamo cannot trace.
  Hint: If you don't need gradients to flow back to the original tensor outside the compiled region, detach the input: `tensor.detach().requires_grad_(True)`.
  Hint: Otherwise, move the autograd.grad() call outside the compiled region.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: inputs with external grad_fn: ["L['ext']"]"""  # noqa: B950
            ),
        ):
            fn(external)

    @skipIfCrossRef
    def test_autograd_grad_with_unrelated_requires_grad_output(self):
        mod = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4, requires_grad=True)

        def step_fn(x):
            res = mod(x)
            loss = res.sum()
            params = tuple(mod.parameters())
            (weight_grad, bias_grad) = torch.autograd.grad(
                loss, params, materialize_grads=False, allow_unused=True
            )
            grad_norm = weight_grad.sum() + bias_grad.sum()
            # return unrelated output that requires_grad
            return grad_norm.detach(), x.sin()

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(step_fn, backend=backend, fullgraph=True)
        result = compiled_fn(x)
        self.assertTrue(result[1].requires_grad)

        # Verify dynamo graph captures autograd.grad and unrelated output
        self.assertEqual(len(backend.graphs), 1)
        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_mod_parameters_weight_: "f32[4, 4]", L_mod_parameters_bias_: "f32[4]", L_x_: "f32[2, 4]"):
        l_mod_parameters_weight_ = L_mod_parameters_weight_
        l_mod_parameters_bias_ = L_mod_parameters_bias_
        l_x_ = L_x_

        res: "f32[2, 4]" = torch._C._nn.linear(l_x_, l_mod_parameters_weight_, l_mod_parameters_bias_)

        loss: "f32[]" = res.sum();  res = None

        grad = torch.autograd.grad(loss, (l_mod_parameters_weight_, l_mod_parameters_bias_), materialize_grads = False, allow_unused = True);  loss = l_mod_parameters_weight_ = l_mod_parameters_bias_ = None
        weight_grad: "f32[4, 4]" = grad[0]
        bias_grad: "f32[4]" = grad[1];  grad = None

        sum_2: "f32[]" = weight_grad.sum();  weight_grad = None
        sum_3: "f32[]" = bias_grad.sum();  bias_grad = None
        grad_norm: "f32[]" = sum_2 + sum_3;  sum_2 = sum_3 = None

        detach: "f32[]" = grad_norm.detach();  grad_norm = None
        sin: "f32[2, 4]" = l_x_.sin();  l_x_ = None
        return (detach, sin)
""",  # noqa: B950
        )

        self.assertEqual(len(backend.fw_graphs), 1)
        fw_actual = normalize_gm(
            backend.fw_graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            fw_actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 4]", primals_2: "f32[4]", primals_3: "f32[2, 4]"):
        t: "f32[4, 4]" = torch.ops.aten.t.default(primals_1);  primals_1 = None
        addmm: "f32[2, 4]" = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = t = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(addmm);  addmm = None

        ones_like: "f32[]" = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
        expand: "f32[2, 4]" = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
        t_1: "f32[4, 2]" = torch.ops.aten.t.default(expand)
        mm: "f32[4, 4]" = torch.ops.aten.mm.default(t_1, primals_3);  t_1 = None
        t_2: "f32[4, 4]" = torch.ops.aten.t.default(mm);  mm = None
        sum_2: "f32[1, 4]" = torch.ops.aten.sum.dim_IntList(expand, [0], True);  expand = None
        view: "f32[4]" = torch.ops.aten.view.default(sum_2, [4]);  sum_2 = None
        t_3: "f32[4, 4]" = torch.ops.aten.t.default(t_2);  t_2 = None

        sum_3: "f32[]" = torch.ops.aten.sum.default(t_3);  t_3 = None
        sum_4: "f32[]" = torch.ops.aten.sum.default(view);  view = None
        add: "f32[]" = torch.ops.aten.add.Tensor(sum_3, sum_4);  sum_3 = sum_4 = None

        detach: "f32[]" = torch.ops.aten.detach.default(add);  add = None
        sin: "f32[2, 4]" = torch.ops.aten.sin.default(primals_3)
        return (detach, sin, primals_3)
""",  # noqa: B950
        )

        # Trigger backward to compile the backward graph
        result[1].sum().backward()

        self.assertEqual(len(backend.bw_graphs), 1)

        bw_actual = normalize_gm(
            backend.bw_graphs[0].print_readable(print_output=False)
        )
        self.assertExpectedInline(
            bw_actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[2, 4]", tangents_1: "f32[2, 4]"):
        cos: "f32[2, 4]" = torch.ops.aten.cos.default(primals_3);  primals_3 = None
        mul: "f32[2, 4]" = torch.ops.aten.mul.Tensor(tangents_1, cos);  tangents_1 = cos = None
        return (None, None, mul)
""",  # noqa: B950
        )

    @skipIfCrossRef
    def test_autograd_grad_missing_detach_errors_like_eager(self):
        def step(mod, x):
            res = mod(x)
            loss = res.sum()
            params = tuple(mod.parameters())
            grads = torch.autograd.grad(loss, params)
            # Compute something with the gradients after autograd.grad
            # This allows us to test that code after graph break is compiled
            grad_sum = grads[0].sum() + grads[1].sum()
            return loss, grad_sum

        loss_eager, grad_sum_eager = step(torch.nn.Linear(4, 4), torch.randn(2, 4))
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to backward through the graph a second time",
        ):
            loss_eager.backward()

        torch._dynamo.reset()
        # With fullgraph=True, we get an Unsupported error at compile time
        step_compiled_fullgraph = torch.compile(
            step, fullgraph=True, backend="aot_eager"
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            re.escape(
                """\
autograd.grad consumed returned tensor's grad_fn
  Explanation: torch.autograd.grad() consumes grad_fns that are needed by tensors returned from this compiled function. This would cause 'backward through graph a second time' errors.
  Hint: If you don't need to backward through the returned tensor, call .detach() before returning: `return loss.detach()`
  Hint: If you need to backward through the returned tensor, use retain_graph=True in autograd.grad()."""  # noqa: B950
            ),
        ):
            step_compiled_fullgraph(torch.nn.Linear(4, 4), torch.randn(2, 4))

        torch._dynamo.reset()

        # With fullgraph=False, code before and after autograd.grad is compiled,
        # but autograd.grad itself runs in eager.
        cnt = torch._dynamo.testing.CompileCounter()
        step_compiled_graph_break = torch.compile(step, fullgraph=False, backend=cnt)

        loss_compiled, grad_sum_compiled = step_compiled_graph_break(
            torch.nn.Linear(4, 4), torch.randn(2, 4)
        )
        self.assertEqual(cnt.frame_count, 2)

        # The returned loss still has the eager behavior issue
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to backward through the graph a second time",
        ):
            loss_compiled.backward()

    def test_autograd_grad_consumed_intermediate_tensor(self):
        torch._dynamo.reset()

        def fn(x):
            y = x * 2  # y.grad_fn = MulBackward
            z = y + 1  # z.grad_fn = AddBackward -> MulBackward
            grad = torch.autograd.grad(
                z.sum(), x
            )  # consumes AddBackward and MulBackward
            return y, grad  # y's grad_fn was consumed!

        # Verify eager fails
        x_eager = torch.randn(4, requires_grad=True)
        y_eager, grad_eager = fn(x_eager)
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to backward through the graph a second time",
        ):
            y_eager.sum().backward()

        # Compiled should detect this and raise Unsupported
        torch._dynamo.reset()
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")

        msg = textwrap.dedent(
            """\
autograd.grad consumed returned tensor's grad_fn
  Explanation: torch.autograd.grad() consumes grad_fns that are needed by tensors returned from this compiled function. This would cause 'backward through graph a second time' errors.
  Hint: If you don't need to backward through the returned tensor, call .detach() before returning: `return loss.detach()`
  Hint: If you need to backward through the returned tensor, use retain_graph=True in autograd.grad()."""  # noqa: B950
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            re.escape(msg) + r"[\s\S]*",
        ):
            compiled_fn(torch.randn(4, requires_grad=True))

    @skipIfCrossRef
    def test_autograd_grad_external_grad_fn_detached(self):
        torch._dynamo.reset()

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(y_ext, x_internal):
            # y_ext has external grad_fn but we detach it
            y_detached = y_ext.detach()
            z = y_detached + x_internal * 3
            return torch.autograd.grad(z.sum(), x_internal)

        x_external = torch.randn(3, requires_grad=True)
        y_external = x_external * 2  # has grad_fn
        x_internal = torch.randn(3, requires_grad=True)

        # This should work because y_external's grad_fn is detached
        result = fn(y_external, x_internal)
        self.assertEqual(result[0], torch.full((3,), 3.0))

    @skipIfCrossRef
    def test_autograd_grad_rejects_external_gradient_edge(self):
        from torch.autograd.graph import GradientEdge

        torch._dynamo.reset()

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(edge, x):
            return torch.autograd.grad(edge, x, grad_outputs=torch.ones(3))

        x = torch.randn(3, requires_grad=True)
        y = x * 2
        edge = GradientEdge(y.grad_fn, 0)

        msg = textwrap.dedent(
            """\
autograd.grad with external GradientEdge
  Explanation: torch.autograd.grad() cannot be used with GradientEdge inputs passed from outside the compiled region. The GradientEdge contains a reference to an autograd node that was created before torch.compile started tracing, so Dynamo cannot trace through its computation.
  Hint: Move the autograd.grad() call outside the torch.compile region.
  Hint: Or use tensor inputs directly instead of GradientEdge objects.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: GradientEdge in outputs: L['edge']"""  # noqa: B950
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            re.escape(msg),
        ):
            fn(edge, x)

    @skipIfCrossRef
    def test_autograd_grad_rejects_tuple_of_external_gradient_edges(self):
        from torch.autograd.graph import GradientEdge

        torch._dynamo.reset()

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(edges, x):
            return torch.autograd.grad(
                edges, x, grad_outputs=[torch.ones(3), torch.ones(3)]
            )

        x = torch.randn(3, requires_grad=True)
        y = x * 2
        z = x * 3
        edges = (GradientEdge(y.grad_fn, 0), GradientEdge(z.grad_fn, 0))

        msg = textwrap.dedent(
            """\
autograd.grad with external GradientEdge
  Explanation: torch.autograd.grad() cannot be used with GradientEdge inputs passed from outside the compiled region. The GradientEdge contains a reference to an autograd node that was created before torch.compile started tracing, so Dynamo cannot trace through its computation.
  Hint: Move the autograd.grad() call outside the torch.compile region.
  Hint: Or use tensor inputs directly instead of GradientEdge objects.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: GradientEdge in outputs[0]: L['edges'][0]"""  # noqa: B950
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            re.escape(msg),
        ):
            fn(edges, x)

    @skipIfCrossRef
    def test_autograd_grad_returning_input_tensor(self):
        torch._dynamo.reset()

        def fn(x):
            y = x * 2  # y.grad_fn = MulBackward
            z = y * 3  # z.grad_fn = MulBackward2 -> MulBackward
            # autograd.grad stops AT y, so y.grad_fn should NOT be consumed
            grad = torch.autograd.grad(z.sum(), y)
            return y, grad[0]  # Returning y should be SAFE

        x_eager = torch.randn(4, requires_grad=True)
        y_eager, grad_eager = fn(x_eager)
        y_eager.sum().backward()
        self.assertEqual(x_eager.grad, torch.full((4,), 2.0))

        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        x_compiled = torch.randn(4, requires_grad=True)

        y_compiled, grad_compiled = compiled_fn(x_compiled)

    @skipIfCrossRef
    def test_autograd_grad_gradient_penalty(self):
        torch._dynamo.reset()

        discriminator = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
        )

        def compute_gradient_penalty_and_grads(discriminator, interpolates):
            params = tuple(discriminator.parameters())

            # Forward pass
            d_interpolates = discriminator(interpolates)

            # First autograd.grad: compute gradients w.r.t. interpolated inputs
            # create_graph=True is essential - it allows backprop through the gradient computation
            (gradients,) = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones_like(d_interpolates),
                create_graph=True,
            )
            gradient_norm = gradients.norm(2, dim=1)
            gradient_penalty = ((gradient_norm - 1) ** 2).mean()

            # Second autograd.grad: compute gradients of penalty w.r.t. parameters
            param_grads = torch.autograd.grad(
                gradient_penalty, params, allow_unused=True
            )

            return gradient_penalty.detach(), param_grads

        real_samples = torch.randn(4, 4)
        fake_samples = torch.randn(4, 4)
        alpha = torch.rand(4, 1)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).detach()
        interpolates.requires_grad_(True)

        # Eager execution
        discriminator_eager = copy.deepcopy(discriminator)
        gp_eager, grads_eager = compute_gradient_penalty_and_grads(
            discriminator_eager, interpolates
        )

        interpolates_compiled = (
            alpha * real_samples + (1 - alpha) * fake_samples
        ).detach()
        interpolates_compiled.requires_grad_(True)

        discriminator_compiled = copy.deepcopy(discriminator)
        compiled_fn = torch.compile(
            compute_gradient_penalty_and_grads, backend="aot_eager", fullgraph=True
        )
        gp_compiled, grads_compiled = compiled_fn(
            discriminator_compiled, interpolates_compiled
        )

        # Gradient penalty values should match
        self.assertEqual(gp_eager, gp_compiled)

        # Parameter gradients should match
        for g_eager, g_compiled in zip(grads_eager, grads_compiled):
            self.assertEqual(g_eager, g_compiled)

    @skipIfCrossRef
    def test_autograd_grad_with_tensor_subclass(self):
        torch._dynamo.reset()

        from torch.testing._internal.two_tensor import TwoTensor

        inner_a = torch.randn(4, requires_grad=True)
        inner_b = torch.randn(4, requires_grad=True)
        two_tensor = TwoTensor(inner_a, inner_b)

        def fn(tt, param):
            result = tt.a * param + tt.b
            loss = result.sum()
            grad = torch.autograd.grad(loss, param)
            return grad[0]

        param = torch.randn(4, requires_grad=True)

        # Verify eager behavior
        result_eager = fn(two_tensor, param)
        self.assertEqual(result_eager, two_tensor.a)

        # Verify compiled matches eager
        compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
        result_compiled = compiled_fn(two_tensor, param)
        self.assertEqual(result_compiled, two_tensor.a)

    @skipIfCrossRef
    def test_autograd_grad_double_consumption_detected(self):
        torch._dynamo.reset()

        def fn(x):
            y = x * 2
            z = y * 3
            # First autograd.grad consumes MulBackward from z and y
            grad1 = torch.autograd.grad(z.sum(), x)
            # Second autograd.grad tries to consume the same grad_fns - error!
            grad2 = torch.autograd.grad(z.sum(), x)
            return grad1[0] + grad2[0]

        # Verify eager fails with "backward through graph a second time"
        x_eager = torch.randn(4, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to backward through the graph a second time",
        ):
            fn(x_eager)

        # Compiled should detect this at compile time
        x = torch.randn(4, requires_grad=True)

        msg = textwrap.dedent(
            """\
            autograd.grad with already consumed grad_fn
              Explanation: torch.autograd.grad() is trying to consume grad_fns that were already consumed by a previous autograd.grad() call. This would cause 'backward through graph a second time' errors at runtime.
              Hint: Use retain_graph=True in the first autograd.grad() call if you need to compute gradients through the same graph multiple times."""  # noqa: B950
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            re.escape(msg),
        ):
            compiled_fn = torch.compile(fn, fullgraph=True, backend="aot_eager")
            compiled_fn(x)


if __name__ == "__main__":
    run_tests()
