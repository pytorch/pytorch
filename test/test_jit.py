import torch
import torch.jit
import torch.nn as nn
import unittest
from itertools import product
from torch.autograd import Variable, Function
from common import TestCase, run_tests
import io

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


class TestJit(TestCase):
    maxDiff = None

    def test_simple(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace = torch._C._tracer_enter((x, y), 0)
        z = torch.sigmoid(torch.tanh(x * (x + y)))
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)

        self.assertExpected(str(trace))

    def test_lstm(self):
        # Careful: don't use fused backend (enabled with CUDA)
        # Pasted from test_LSTM_cell
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        trace, _ = torch.jit.record_trace(
            nn.LSTMCell(10, 20), input, (hx, cx))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    def test_function_as_argument(self):
        # Careful: don't use fused backend (enabled with CUDA)
        # Pasted from test_LSTM_cell
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        lstm = nn.LSTMCell(10, 20)

        def a_function(a, b):
            return lstm(a, b)
        trace, _ = torch.jit.record_trace(
            a_function, input, (hx, cx), parameters=lstm.parameters())
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    def test_verify(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(
            doit, enabled=True, verify=True, time=True, optimize=False)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_traced_function(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(doit)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_disabled_traced_function(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced = torch.jit.traced(doit, enabled=False)
        z = traced(x, y)
        z2 = traced(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_traced_module(self):
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))
        lstm = nn.LSTMCell(10, 20)
        lstm = torch.jit.traced(lstm, verify=True)

        out = lstm(input, (hx, cx))
        out2 = lstm(input, (hx, cx))
        self.assertEqual(out, out2)

    def test_autograd_closure(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace = torch._C._tracer_enter((x, y), 1)

        z = torch.sigmoid(x * (x + y))
        w = torch.abs(x * x * x + y) + Variable(torch.ones(1))

        torch._C._tracer_exit((z, w))
        torch._C._jit_pass_lint(trace)

        (z * w).backward()
        torch._C._jit_pass_dce(trace)
        torch._C._jit_pass_lint(trace)

        x_grad = x.grad.data.clone()
        x.grad.data.zero_()

        function = torch._C._jit_createAutogradClosure(trace)
        torch._C._jit_pass_lint(trace)
        z2, w2 = function()(x, y)
        (z2 * w2).backward()
        self.assertEqual(z, z2)
        self.assertEqual(w, w2)
        self.assertEqual(x.grad.data, x_grad)

    def test_constant(self):
        x = Variable(torch.randn(2, 2), requires_grad=True)

        trace = torch._C._tracer_enter((x,), 0)

        y = Variable(torch.diag(torch.Tensor([2, 2])))
        z = x.matmul(y)

        torch._C._tracer_exit((z,))
        function = torch._C._jit_createAutogradClosure(trace)

        z2 = function()(x)
        self.assertEqual(z, z2)

        y.data.fill_(1000)  # make sure the data has been cloned

        x2 = Variable(torch.ones(2, 2) * 2, requires_grad=True)
        z3 = function()(x2)
        self.assertEqual(z3.data, torch.ones(2, 2) * 4)

    def test_c_function(self):
        x = Variable(torch.randn(1, 3, 10, 10))
        m = nn.Conv2d(3, 8, 3, 1)

        trace = torch._C._tracer_enter((x,) + tuple(m.parameters()), 0)
        y = m(x)
        torch._C._tracer_exit((y,))
        self.assertExpected(str(trace))

    def test_legacy_fail(self):

        class Legacy(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace = torch._C._tracer_enter((x,), 0)
        self.assertRaises(RuntimeError, lambda: Legacy()(x))
        torch._C._tracer_exit((x,))

    def test_inplace_transplant(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace = torch._C._tracer_enter((x,), 0)
        y = x.clone()
        y.add_(2)
        y.add_(3)
        torch._C._tracer_exit((y,))
        self.assertExpected(str(trace))

    def test_backward(self):
        a = Variable(torch.randn(2, 2), requires_grad=True)
        b = Variable(torch.randn(2, 2), requires_grad=True)

        x = a
        y = a * b

        trace = torch._C._tracer_enter((x, y), 2)
        z = y * 2 * x
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)

        # Run first backward
        grad, = torch.autograd.grad(z, x, Variable(torch.ones(2, 2), requires_grad=True), create_graph=True)
        torch._C._jit_pass_lint(trace)

        # Run second backward
        grad.sum().backward(create_graph=True)
        torch._C._jit_pass_lint(trace)

        # Run dead code elimination to remove unused trace nodes
        torch._C._jit_pass_dce(trace)
        self.assertExpected(str(trace))

    def test_trace_expire(self):
        x = Variable(torch.randn(2, 2), requires_grad=True)
        y = Variable(torch.randn(2, 2), requires_grad=True)

        def record_trace(num_backwards):
            trace = torch._C._tracer_enter((x, y), num_backwards)
            z = y * 2 * x
            torch._C._tracer_exit((z,))
            return z, trace

        def check(expired, complete):
            self.assertEqual(trace.is_expired, expired)
            self.assertEqual(trace.is_complete, complete)

        z, trace = record_trace(0)
        check(False, True)
        del z
        check(False, True)

        z, trace = record_trace(1)
        check(False, False)
        del z
        check(True, False)

        z, trace = record_trace(1)
        check(False, False)
        z.sum().backward()
        check(False, True)
        del z
        check(False, True)

    def test_multiuse_fn(self):
        x = Variable(torch.randn(2, 2), requires_grad=True)
        w = Variable(torch.randn(2, 2), requires_grad=True)

        @torch.jit.trace(parameters=[w])
        def cell(x):
            return x * w + 2

        out = cell(cell(cell(x)))
        self.assertFalse(cell.has_trace_for(x))

        out.sum().backward()
        self.assertTrue(cell.has_trace_for(x))

    def test_flags(self):
        x = Variable(torch.randn(2, 2))
        y = Variable(torch.randn(2, 2))

        @torch.jit.traced
        def fn(x, y):
            return (x * x + y * y + x * y).sum()

        grads = {}
        for rx, ry in product((True, False), repeat=2):
            x.requires_grad = rx
            y.requires_grad = ry

            self.assertFalse(fn.has_trace_for(x, y))
            out = fn(x, y)

            self.assertFalse(fn.has_trace_for(x, y))
            for v, name, compute in [(x, 'x', rx), (y, 'y', ry)]:
                if not compute:
                    continue
                grad_v, = torch.autograd.grad(out, v, retain_graph=True)
                expected_grad = grads.setdefault(name, grad_v)
                self.assertEqual(grad_v, expected_grad)
            self.assertEqual(fn.has_trace_for(x, y), rx or ry)

    def test_volatile_fallback(self):
        """Check that Traceable falls back to num_backwards=0 if given volatile inputs"""
        x = Variable(torch.randn(2, 2))
        y = Variable(torch.randn(2, 2), requires_grad=True)

        @torch.jit.traced
        def fn(x, y):
            return x * x + x * y

        out = fn(x, y)
        self.assertFalse(fn.has_trace_for(x, y))

        x.volatile = True
        self.assertFalse(fn.has_trace_for(x, y))
        out = fn(x, y)
        self.assertTrue(fn.has_trace_for(x, y))

    def test_backward_flag_checks(self):
        x = Variable(torch.randn(1), requires_grad=True)

        @torch.jit.trace(num_derivatives=2)
        def fn(x):
            return x * x

        grad_x, = torch.autograd.grad(fn(x), (x,), create_graph=True)
        self.assertFalse(fn.has_trace_for(x))
        grad_x.backward()
        self.assertTrue(fn.has_trace_for(x))

        with self.assertRaisesRegex(RuntimeError, 'different flags'):
            fn(x).backward(Variable(torch.ones(1), requires_grad=True))
        # TODO: enable once AutogradClosure registers prev stage inputs correctly
        # grad_x, = torch.autograd.grad(fn(x), (x,), create_graph=True)
        # grad_x.backward()

    def test_python_ir(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced, _ = torch.jit.record_trace(doit, x, y)
        g = torch._C._jit_get_graph(traced)
        g2 = torch._C.Graph()
        g_to_g2 = {}
        for node in g.inputs():
            g_to_g2[node] = g2.addInput()
        for node in g.nodes():
            if node.kind() == "PythonOp":
                n_ = g2.create(node.pyname(),
                               [g_to_g2[i] for i in node.inputs()]) \
                    .setType(node.typeOption()) \
                    .s_("note", "from_pyop") \
                    .i_("some_value", len(node.scalar_args()))
                assert(n_.i("some_value") == len(node.scalar_args()))
            else:
                n_ = g2.createClone(node, lambda x: g_to_g2[x])
                assert(n_.kindOf("Offset") == "i")

            g_to_g2[node] = g2.appendNode(n_)

        for node in g.outputs():
            g2.registerOutput(g_to_g2[node])

        t_node = g2.create("TensorTest").t_("a", torch.ones([2, 2]))
        assert(t_node.attributeNames() == ["a"])
        g2.appendNode(t_node)
        assert(torch.equal(torch.ones([2, 2]), t_node.t("a")))
        self.assertExpected(str(g2))

    def test_cpp(self):
        torch._C._jit_run_cpp_tests()

    def test_batchnorm(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(nn.BatchNorm2d(2), x)
        self.assertExpected(str(trace))

    def test_batchnorm_verify(self):
        bn = torch.jit.traced(nn.BatchNorm2d(1), enabled=True, verify=True)
        x = Variable(torch.randn(5, 1))
        z = bn(x)
        z2 = bn(x)
        self.assertEqual(z, z2)

    def test_conv(self):
        x = Variable(torch.randn(20, 16, 50, 40).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(nn.Conv2d(16, 13, 3, bias=False), x)
        self.assertExpected(str(trace))

    @skipIfNoTorchVision
    def test_alexnet(self):
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(torchvision.models.AlexNet(), x)
        self.assertExpected(str(trace))
        # NB: Purposely NOT testing protobuf export here

if __name__ == '__main__':
    run_tests()
