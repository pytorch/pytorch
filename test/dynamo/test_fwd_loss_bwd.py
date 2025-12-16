# Owner(s): ["module: dynamo"]

import re

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

    @skipIfCrossRef
    def test_autograd_grad_output_not_connected_to_grad(self):
        x = torch.randn(4, requires_grad=True)

        def fn(x):
            y = x.sin()
            (grad,) = torch.autograd.grad(y.sum(), x)
            # z is independent - not connected to autograd.grad
            z = x.cos()
            # Return both: grad (from autograd.grad) and z (independent, requires_grad)
            return grad.sum(), z

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)

        eager_result = fn(x)
        compiled_result = compiled_fn(x)

        self.assertEqual(eager_result[0], compiled_result[0])
        self.assertEqual(eager_result[1], compiled_result[1])
        self.assertEqual(len(backend.graphs), 1)

        gm = backend.graphs[0]
        actual = normalize_gm(gm.print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4]"):
        l_x_ = L_x_

        y: "f32[4]" = l_x_.sin()

        sum_1: "f32[]" = y.sum();  y = None
        grad = torch.autograd.grad(sum_1, l_x_);  sum_1 = None
        grad_1: "f32[4]" = grad[0];  grad = None

        z: "f32[4]" = l_x_.cos();  l_x_ = None

        sum_2: "f32[]" = grad_1.sum();  grad_1 = None
        return (sum_2, z)
""",  # noqa: B950
        )

        # z requires_grad=True, so we should have a joint graph
        # Call backward to trigger backward compilation
        compiled_result[1].sum().backward()

        # Verify backward graph was compiled (joint partitioning happened)
        self.assertEqual(len(backend.bw_graphs), 1)
        bw_graph = backend.bw_graphs[0]
        bw_actual = normalize_gm(bw_graph.print_readable(print_output=False))
        self.assertExpectedInline(
            bw_actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4]", tangents_1: "f32[4]"):
        sin: "f32[4]" = torch.ops.aten.sin.default(primals_1);  primals_1 = None

        neg: "f32[4]" = torch.ops.aten.neg.default(sin);  sin = None
        mul_1: "f32[4]" = torch.ops.aten.mul.Tensor(tangents_1, neg);  tangents_1 = neg = None
        return (mul_1,)
""",  # noqa: B950
        )

    @skipIfCrossRef
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
            r"autograd.grad with external grad_fn",
        ):
            fn(external_computation)

    @skipIfCrossRef
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

    def test_autograd_grad_with_external_grad_fn_training_outside(self):
        x = torch.randn(2, 4, requires_grad=True)
        external = x * 2

        @torch.compile(fullgraph=True, backend="aot_eager")
        def fn(ext):
            y = (ext.sin()).sum()
            gx = torch.autograd.grad(y, x, retain_graph=True, allow_unused=True)[0]
            return gx

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("Developer debug context: input with external grad_fn: L['ext']"),
        ):
            _ = fn(external)

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
            unrelated_output = x.sin()
            return grad_norm.detach(), unrelated_output

        backend = AotEagerAndRecordGraphs()
        compiled_fn = torch.compile(step_fn, backend=backend, fullgraph=True)
        result = compiled_fn(x)
        self.assertTrue(result[1].requires_grad)

        # Verify dynamo graph captures autograd.grad and unrelated output
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

        res: "f32[2, 4]" = torch._C._nn.linear(l_x_, l_mod_parameters_weight_, l_mod_parameters_bias_)

        loss: "f32[]" = res.sum();  res = None

        grad = torch.autograd.grad(loss, (l_mod_parameters_weight_, l_mod_parameters_bias_), materialize_grads = False, allow_unused = True);  loss = l_mod_parameters_weight_ = l_mod_parameters_bias_ = None
        weight_grad: "f32[4, 4]" = grad[0]
        bias_grad: "f32[4]" = grad[1];  grad = None

        sum_2: "f32[]" = weight_grad.sum();  weight_grad = None
        sum_3: "f32[]" = bias_grad.sum();  bias_grad = None
        grad_norm: "f32[]" = sum_2 + sum_3;  sum_2 = sum_3 = None

        unrelated_output: "f32[2, 4]" = l_x_.sin();  l_x_ = None

        detach: "f32[]" = grad_norm.detach();  grad_norm = None
        return (detach, unrelated_output)
""",  # noqa: B950
        )

        self.assertEqual(len(backend.fw_graphs), 1)
        fw_graph = backend.fw_graphs[0]
        fw_actual = normalize_gm(fw_graph.print_readable(print_output=False))
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

        sin: "f32[2, 4]" = torch.ops.aten.sin.default(primals_3)

        detach: "f32[]" = torch.ops.aten.detach.default(add);  add = None
        return (detach, sin, primals_3)
""",  # noqa: B950
        )

        # Trigger backward to compile the backward graph
        result[1].sum().backward()

        self.assertEqual(len(backend.bw_graphs), 1)
        bw_graph = backend.bw_graphs[0]
        bw_actual = normalize_gm(bw_graph.print_readable(print_output=False))
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

        # With fullgraph=True, we get an Unsupported error at compile time
        @torch.compile(fullgraph=True, backend="aot_eager")
        def step_compiled_fullgraph():
            res = mod_compiled(x_compiled)
            loss = res.sum()
            params = tuple(mod_compiled.parameters())
            torch.autograd.grad(loss, params)
            return loss

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "autograd.grad with output that requires grad",
        ):
            step_compiled_fullgraph()

        torch._dynamo.reset()
        mod_compiled2 = torch.nn.Linear(4, 4)
        x_compiled2 = torch.randn(2, 4)

        # With fullgraph=False, graph breaks and runs in eager mode
        # So we get the same error as eager when calling backward()
        @torch.compile(fullgraph=False, backend="aot_eager")
        def step_compiled_no_fullgraph():
            res = mod_compiled2(x_compiled2)
            loss = res.sum()
            params = tuple(mod_compiled2.parameters())
            torch.autograd.grad(loss, params)
            return loss

        loss_compiled = step_compiled_no_fullgraph()
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to backward through the graph a second time",
        ):
            loss_compiled.backward()

    @skipIfCrossRef
    def test_trace_autograd_ops_config(self):
        """Test that trace_autograd_ops config controls whether autograd.grad is traced."""

        def fn(x):
            y = x.sin()
            (gx,) = torch.autograd.grad(y, x)
            return gx.cos()  # Add op after autograd.grad to detect graph break

        x = torch.tensor(1.0, requires_grad=True)
        eager_result = fn(x)

        # With trace_autograd_ops=False, should graph break
        torch._dynamo.reset()
        with torch._dynamo.config.patch(trace_autograd_ops=False):
            cnt = torch._dynamo.testing.CompileCounter()
            compiled_fn = torch.compile(backend=cnt)(fn)
            compiled_result = compiled_fn(torch.tensor(1.0, requires_grad=True))
            self.assertEqual(eager_result, compiled_result)
            # Should have graph break due to autograd.grad being skipped
            self.assertEqual(cnt.frame_count, 2)

        # With trace_autograd_ops=True, should trace in single graph
        torch._dynamo.reset()
        with torch._dynamo.config.patch(trace_autograd_ops=True):
            cnt = torch._dynamo.testing.CompileCounter()
            compiled_fn = torch.compile(backend=cnt)(fn)
            compiled_result = compiled_fn(torch.tensor(1.0, requires_grad=True))
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(cnt.frame_count, 1)

    @skipIfCrossRef
    def test_autograd_grad_external_grad_fn_not_reachable(self):
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

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"autograd.grad with external GradientEdge",
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

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"autograd.grad with external GradientEdge",
        ):
            fn(edges, x)


if __name__ == "__main__":
    run_tests()
