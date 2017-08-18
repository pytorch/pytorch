import torch
import torch.jit
import torch.nn as nn
import unittest
from torch.autograd import Variable, Function
from common import TestCase, run_tests

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import toffee
    import google.protobuf.text_format
    HAS_TOFFEE = True
except ImportError:
    HAS_TOFFEE = False

toffee_only = unittest.skipIf(not HAS_TOFFEE, "no toffee support library")
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


class TestJit(TestCase):
    maxDiff = None

    def test_simple(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace = torch._C._tracer_enter((x, y))
        z = torch.sigmoid(torch.tanh(x * (x + y)))
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)

        self.assertExpected(str(trace))

    def assertToffeeExpected(self, binary_pb, subname=None):
        graph_def = toffee.GraphProto.FromString(binary_pb)
        self.assertExpected(google.protobuf.text_format.MessageToString(graph_def, float_format='.15g'), subname)

    @toffee_only
    def test_export(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        trace = torch._C._tracer_enter((x, y))
        z = -torch.sigmoid(torch.tanh(x * (x + y)))
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        self.assertToffeeExpected(torch._C._jit_pass_export(trace))

    @toffee_only
    def test_export_view(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, _ = torch.jit.record_trace(lambda x: x.view(1, 1), x)
        self.assertToffeeExpected(torch._C._jit_pass_export(trace))

    @toffee_only
    def test_export_data(self):
        x = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        y = Variable(torch.Tensor([[1, 2], [3, 4]]), requires_grad=True)
        trace, _ = torch.jit.record_trace(lambda x, y: -torch.sigmoid(torch.tanh(x * (x + y))), x, y)
        initializers = [x.data]
        self.assertToffeeExpected(torch._C._jit_pass_export(trace, initializers))

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

        trace = torch._C._tracer_enter((x, y))

        z, _ = torch.max(x * (x + y), 0)
        w = torch.abs(x * x * x + y)

        torch._C._tracer_exit((z, w))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_init(trace)
        torch._C._jit_pass_lint(trace)
        closure = torch._C._jit_createAutogradClosure(trace)
        z2, w2 = Variable._execution_engine.run_forward(closure, (x, y))
        self.assertEqual(z, z2)
        self.assertEqual(w, w2)

    def test_constant(self):
        x = Variable(torch.randn(2, 2), requires_grad=True)

        trace = torch._C._tracer_enter((x,))

        y = Variable(torch.diag(torch.Tensor([2, 2])))
        z = x.matmul(y)

        torch._C._tracer_exit((z,))
        closure = torch._C._jit_createAutogradClosure(trace)

        z2, = Variable._execution_engine.run_forward(closure, (x,))
        self.assertEqual(z, z2)

        y.data.fill_(1000)  # make sure the data has been cloned

        x2 = Variable(torch.ones(2, 2) * 2, requires_grad=True)
        z3, = Variable._execution_engine.run_forward(closure, (x2,))
        self.assertEqual(z3.data, torch.ones(2, 2) * 4)

    def test_c_function(self):
        x = Variable(torch.randn(1, 3, 10, 10))
        m = nn.Conv2d(3, 8, 3, 1)

        trace = torch._C._tracer_enter((x,) + tuple(m.parameters()))
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
        trace = torch._C._tracer_enter((x,))
        self.assertRaises(RuntimeError, lambda: Legacy()(x))
        torch._C._tracer_exit((x,))

    def test_inplace_transplant(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace = torch._C._tracer_enter((x,))
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

        trace = torch._C._tracer_enter((x, y))
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
        torch._C._jit_pass_dco(trace)
        self.assertExpected(str(trace))

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

    @toffee_only
    def test_batchnorm_export(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(nn.BatchNorm2d(2), x)
        self.assertToffeeExpected(torch._C._jit_pass_export(trace))

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

    @toffee_only
    def test_conv_export(self):
        x = Variable(torch.randn(20, 16, 50, 40).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(nn.Conv2d(16, 13, 3, bias=False), x)
        self.assertToffeeExpected(torch._C._jit_pass_export(trace))

    @skipIfNoTorchVision
    def test_alexnet(self):
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.record_trace(torchvision.models.AlexNet(), x)
        self.assertExpected(str(trace))
        # NB: Purposely NOT testing protobuf export here

    @skipIfNoTorchVision
    def test_densenet(self):
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        dense121 = torchvision.models.DenseNet(num_init_features=64, growth_rate=32,
                                               block_config=(6, 12, 24, 16))
        trace, _ = torch.jit.record_trace(dense121, x)
        # Densenet trace is pretty large, so we don't assert on it

if __name__ == '__main__':
    run_tests()
