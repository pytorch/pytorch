import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import unittest
from contextlib import contextmanager
from itertools import product
from torch.autograd import Variable, Function
from torch.autograd.function import traceable
from common import TestCase, run_tests
import io

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)
    return hy, cy


def LSTMCellC(*args, **kwargs):
    hy, cy = LSTMCell(*args, **kwargs)
    return torch.cat((hy, cy))


class TestJit(TestCase):
    maxDiff = None

    @contextmanager
    def assertCompiled(self, fn):
        compiled_fn = fn.compiled_fn
        self.assertIsInstance(compiled_fn, torch._C.CompiledFunction)
        hits, misses = compiled_fn.hits, compiled_fn.misses
        yield
        self.assertLess(hits, compiled_fn.hits)
        self.assertEqual(misses, compiled_fn.misses)

    def test_simple(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def f(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        trace, z = torch.jit.trace(f, (x, y), nderivs=0)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    @unittest.skipIf(not torch.cuda.is_available(), "fuser requires CUDA")
    def test_lstm_fusion(self):
        input = Variable(torch.randn(3, 10).float().cuda())
        hx = Variable(torch.randn(3, 20).float().cuda())
        cx = Variable(torch.randn(3, 20).float().cuda())
        module = nn.LSTMCell(10, 20).float().cuda()  # Just to allocate weights with correct sizes

        trace, _ = torch.jit.trace(LSTMCell, (input, (hx, cx)) + tuple(module.parameters()))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    @unittest.skipIf(not torch.cuda.is_available(), "fuser requires CUDA")
    def test_run_lstm_fusion(self):
        input = Variable(torch.randn(3, 10).float().cuda())
        hx = Variable(torch.randn(3, 20).float().cuda())
        cx = Variable(torch.randn(3, 20).float().cuda())
        module = nn.LSTMCell(10, 20).float().cuda()  # Just to allocate weights with correct sizes

        CompiledLSTMCell = torch.jit.compile(nderivs=0)(LSTMCell)

        z = CompiledLSTMCell(input, (hx, cx), *module.parameters())
        with self.assertCompiled(CompiledLSTMCell):
            z2 = CompiledLSTMCell(input, (hx, cx), *module.parameters())
        self.assertEqual(z, z2)

    @unittest.skipIf(not torch.cuda.is_available(), "fuser requires CUDA")
    def test_run_lstm_fusion_concat(self):
        input = Variable(torch.randn(3, 10).float().cuda())
        hx = Variable(torch.randn(3, 20).float().cuda())
        cx = Variable(torch.randn(3, 20).float().cuda())
        module = nn.LSTMCell(10, 20).float().cuda()  # Just to allocate weights with correct sizes

        CompiledLSTMCell = torch.jit.compile(nderivs=0)(LSTMCellC)

        z = CompiledLSTMCell(input, (hx, cx), *module.parameters())
        with self.assertCompiled(CompiledLSTMCell):
            z2 = CompiledLSTMCell(input, (hx, cx), *module.parameters())
        self.assertEqual(z, z2)

    @unittest.skipIf(not torch.cuda.is_available(), "fuser requires CUDA")
    def test_concat_fusion(self):
        hx = Variable(torch.randn(3, 20).float().cuda())
        cx = Variable(torch.randn(3, 20).float().cuda())

        def Foo(hx, cx):
            return torch.cat((hx + cx, hx * cx))

        trace, _ = torch.jit.trace(Foo, (hx, cx))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    @unittest.skipIf(not torch.cuda.is_available(), "fuser requires CUDA")
    def test_fusion_distribute(self):
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2
        x = Variable(torch.randn(4, 4).float().cuda())
        y = Variable(torch.randn(4, 4).float().cuda())
        trace, _ = torch.jit.trace(f, (x, y), nderivs=0)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace), 'raw')
        torch._C._jit_pass_fuse(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace))

    def test_cse(self):
        x = Variable(torch.Tensor([0.4, 0.3]), requires_grad=True)
        y = Variable(torch.Tensor([0.7, 0.5]), requires_grad=True)

        trace = torch._C._tracer_enter((x, y), 0)
        w = (x + y) * (x + y) * (x + y)
        t = torch.tanh(w) + torch.tanh(w)
        z = (x + y) * (x + y) * (x + y) + t
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_cse(trace)

        self.assertExpected(str(trace))

    def test_compile_run_twice(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        @torch.jit.compile(nderivs=0, optimize=False)
        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        z = doit(x, y)
        with self.assertCompiled(doit):
            z2 = doit(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    @unittest.skipIf(not torch.cuda.is_available(), "fuser requires CUDA")
    def test_compile_addc(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True).float().cuda()
        y = Variable(torch.Tensor([0.7]), requires_grad=True).float().cuda()

        @torch.jit.compile(nderivs=0)
        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y) + 1))

        z = doit(x, y)
        with self.assertCompiled(doit):
            z2 = doit(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y) + 1)))
        self.assertEqual(z, z2)

    def test_traced_function(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        @torch.jit.compile(nderivs=0)
        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        z = doit(x, y)
        with self.assertCompiled(doit):
            z2 = doit(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_disabled_traced_function(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        @torch.jit.compile(enabled=False)
        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        z = doit(x, y)
        z2 = doit(x, y)
        self.assertEqual(z, torch.sigmoid(torch.tanh(x * (x + y))))
        self.assertEqual(z, z2)

    def test_assign_traces(self):
        """Check that output Variables are assigned traces before they are saved."""
        @traceable
        class MyFn(Function):
            @staticmethod
            def forward(ctx, a):
                out = a * 2
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad_a):
                a, = ctx.saved_variables
                return a * grad_a

        x = Variable(torch.randn(10, 10), requires_grad=True)
        trace, out = torch.jit.trace(MyFn.apply, x, nderivs=1)
        out.sum().backward()
        torch._C._jit_pass_dce(trace)
        self.assertExpected(str(trace))

    def test_traced_module(self):
        input = Variable(torch.randn(3, 10))
        hx = Variable(torch.randn(3, 20))
        cx = Variable(torch.randn(3, 20))

        @torch.jit.compile(nderivs=0)
        class MyLSTMCell(nn.LSTMCell):
            pass

        lstm = MyLSTMCell(10, 20)

        out = lstm(input, (hx, cx))
        with self.assertCompiled(lstm):
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

    def test_verify(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        @torch.jit.compile
        def f(x, y):
            z = torch.sigmoid(x * (x + y))
            w = torch.abs(x * x * x + y) + Variable(torch.ones(1))
            return z, w

        torch.jit.verify(f, (x, y), loss_fn=lambda z, w: z * w, devices=[])

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

        class MyLegacyFn(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace = torch._C._tracer_enter((x,), 0)
        self.assertRaisesRegex(RuntimeError, "MyLegacyFn", lambda: MyLegacyFn()(x))
        torch._C._tracer_exit((x,))

    def test_inplace_transplant(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace = torch._C._tracer_enter((x,), 0)
        y = x.clone()
        y.add_(2)
        y.add_(3)
        torch._C._tracer_exit((y,))
        self.assertExpected(str(trace))

    def test_inplace_flags(self):
        class InplaceFn(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                return x.add_(1)

            @staticmethod
            def backward(ctx, go):
                return go

        class RegularFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x.add(1)

            @staticmethod
            def backward(ctx, go):
                return go

        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace = torch._C._tracer_enter((x,), 0)
        y = RegularFn.apply(x)
        y = InplaceFn.apply(y)
        y = InplaceFn.apply(y)
        y = RegularFn.apply(y)
        torch._C._tracer_exit((y,))
        ops = [n for n in trace.graph().nodes() if n.kind() != 'Select']
        for op in ops:
            self.assertTrue(op.hasAttribute('inplace'))
        inplace_flags = [False, True, True, False]
        for op, is_inplace in zip(ops, inplace_flags):
            self.assertEqual(op.i('inplace'), is_inplace)

    def test_inplace_check(self):
        class MyInplaceFn(Function):
            @staticmethod
            def forward(self, x):
                x.add_(1)
                self.mark_dirty(x)
                return x

            @staticmethod
            def backward(self, grad):
                return grad

        @torch.jit.compile(nderivs=0)
        def fn(x):
            return MyInplaceFn.apply(x)
        x = Variable(torch.randn(5, 5))
        fn(x)  # trace
        with self.assertRaisesRegex(RuntimeError, 'inplace MyInplaceFn'):
            fn(x)

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

    def test_backward_opaque(self):
        x = Variable(torch.randn(3, 3), requires_grad=True)
        y = Variable(torch.randn(3, 3), requires_grad=True)

        trace = torch._C._tracer_enter((x, y), 2)
        z = x.cross(y)
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)

        # Run first backward
        grad, = torch.autograd.grad(z, x, Variable(torch.ones(3, 3), requires_grad=True), create_graph=True)
        torch._C._jit_pass_lint(trace)

        # Run dead code elimination to remove unused trace nodes
        torch._C._jit_pass_dce(trace)
        self.assertExpected(str(trace))

    def test_backward_closure(self):
        """Check that autograd closures handle multiple stages correctly."""
        x = Variable(torch.randn(1), requires_grad=True)

        @torch.jit.compile(nderivs=2)
        def fn(x):
            return x * x

        # Generate trace
        grad_x, = torch.autograd.grad(fn(x), (x,), create_graph=True)
        self.assertFalse(fn.has_trace_for(x))
        grad_x.backward()
        self.assertTrue(fn.has_trace_for(x))

        x_grad = x.grad.data.clone()
        x.grad.data.zero_()

        # Run the trace
        with self.assertCompiled(fn):
            output = fn(x)
        grad_x, = torch.autograd.grad(output, (x,), create_graph=True)
        grad_x.backward()

        self.assertEqual(x.grad.data, x_grad)

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

        @torch.jit.compile
        def cell(x, w):
            return x * w + 2

        out = cell(cell(cell(x, w), w), w)
        self.assertFalse(cell.has_trace_for(x, w))

        out.sum().backward()
        self.assertTrue(cell.has_trace_for(x, w))

        torch.jit.verify(cell, (x, w), devices=[])

    def test_output_unflatten(self):
        """Check that outputs of traced functions retain the original structure and nesting"""
        x = Variable(torch.randn(2, 2), requires_grad=True)

        def fn(x):
            return (x * 2, (x ** 2, x + 4, (x + 2,), ), x * 4)

        expected_out = fn(x)
        fn = torch.jit.compile(fn)

        def recursive_sum(obj):
            if isinstance(obj, Variable):
                return obj.sum()
            else:
                return sum(recursive_sum(o) for o in obj)

        recursive_sum(fn(x)).backward()
        self.assertTrue(fn.has_trace_for(x))
        with self.assertCompiled(fn):
            self.assertEqual(fn(x), expected_out)

    def test_input_flatten(self):
        """Check that inputs to traced functions are flattened"""
        def make_var():
            return Variable(torch.randn(1), requires_grad=True)
        x = (make_var(), (make_var(), make_var()))

        def fn(x, t):
            y, z = t
            return x * y * z

        expected_out = fn(*x)
        fn = torch.jit.compile(fn)
        fn(*x).backward()
        self.assertTrue(fn.has_trace_for(*x))
        with self.assertCompiled(fn):
            self.assertEqual(fn(*x), expected_out)

    def test_flags(self):
        x = Variable(torch.randn(2, 2))
        y = Variable(torch.randn(2, 2))

        @torch.jit.compile
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

        @torch.jit.compile
        def fn(x, y):
            return x * x + x * y

        out = fn(x, y)
        self.assertFalse(fn.has_trace_for(x, y))

        x.volatile = True
        self.assertFalse(fn.has_trace_for(x, y))
        out = fn(x, y)
        self.assertTrue(fn.has_trace_for(x, y))
        with self.assertCompiled(fn):
            out2 = fn(x, y)
        self.assertEqual(out, out2)

    def test_backward_flag_checks(self):
        x = Variable(torch.randn(1), requires_grad=True)

        @torch.jit.compile(nderivs=2)
        def fn(x):
            return x * x

        grad_x, = torch.autograd.grad(fn(x), (x,), create_graph=True)
        self.assertFalse(fn.has_trace_for(x))
        grad_x.backward()
        self.assertTrue(fn.has_trace_for(x))

        with self.assertRaisesRegex(RuntimeError, 'different flags'):
            fn(x).backward(Variable(torch.ones(1), requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, 'different flags'):
            grad_x, = torch.autograd.grad(fn(x), (x,), create_graph=True)
            grad_x.backward(Variable(torch.ones(1), requires_grad=True))

        # TODO: Test executing this

    def test_python_ir(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def doit(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        traced, _ = torch.jit.trace(doit, (x, y))
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

            g_to_g2[node] = g2.appendNode(n_)

        for node in g.outputs():
            g2.registerOutput(g_to_g2[node])

        t_node = g2.create("TensorTest").t_("a", torch.ones([2, 2]))
        assert(t_node.attributeNames() == ["a"])
        g2.appendNode(t_node)
        assert(torch.equal(torch.ones([2, 2]), t_node.t("a")))
        self.assertExpected(str(g2))

    @unittest.skipIf(not torch.cuda.is_available(), "cpp tests require CUDA")
    def test_cpp(self):
        torch._C._jit_run_cpp_tests()

    @unittest.skip("Broken")
    def test_batchnorm(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.trace(nn.BatchNorm2d(2), x)
        self.assertExpected(str(trace))

    def test_dropout(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.trace(nn.Dropout(0.6), x)
        self.assertExpected(str(trace))

    @unittest.skip("unrecognized NodeKind: SpatialBN")
    def test_batchnorm_run_twice(self):
        @torch.jit.compile(nderivs=0)
        class MyBatchNorm2d(nn.BatchNorm2d):
            pass

        bn = MyBatchNorm2d(1)
        x = Variable(torch.randn(5, 1))
        z = bn(x)
        with self.assertCompiled(bn):
            z2 = bn(x)
        self.assertEqual(z, z2)

    def test_non_decorator_use_fails(self):
        MyLSTM = torch.jit.compile(nn.LSTM)
        self.assertRaisesRegex(TypeError, "class decorator", lambda: MyLSTM(2, 2))

    def test_conv(self):
        x = Variable(torch.randn(20, 16, 50, 40).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.trace(nn.Conv2d(16, 13, 3, bias=False), x)
        self.assertExpected(str(trace))

    def test_reuse_function(self):
        @torch.jit.compile(nderivs=0)
        def clinear(*args):
            return F.linear(*args)

        def cast(x):
            return x

        input = Variable(cast(torch.randn(1, 1)))
        weights = Variable(cast(torch.randn(1, 1)))
        bias = Variable(cast(torch.randn(1, 1)))

        # linear AKA addmm without bias is of particular interest
        # because we allocate a zero-filled new variable when we execute,
        # and then *fill* it with the result

        r1_ = clinear(input, weights)
        with self.assertCompiled(clinear):
            r1 = clinear(r1_, weights)
        r2 = F.linear(F.linear(input, weights), weights)

        self.assertEqual(r1, r2)

    @unittest.skip("Broken. Enable once new JIT interpreter is merged")
    def test_mini_wlm(self):
        """Exercise null-edge pruning in the tracer."""

        @torch.jit.compile
        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.encoder = nn.Embedding(2, 2)

            def forward(self, input, hidden):
                emb = self.encoder(input)
                hidden = hidden.clone()  # simulate some RNN operation
                return emb, hidden

        model = MyModel()

        x = Variable(torch.LongTensor([[0, 1], [1, 0]]))
        y = Variable(torch.FloatTensor([0]))

        z, _ = model(x, y)
        z.sum().backward()

        with self.assertCompiled(model):
            z, _ = model(x, y)
        z.sum().backward()

    @skipIfNoTorchVision
    def test_alexnet(self):
        return
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.trace(torchvision.models.AlexNet(), x)
        self.assertExpected(str(trace))
        # NB: Purposely NOT testing protobuf export here

if __name__ == '__main__':
    run_tests()
