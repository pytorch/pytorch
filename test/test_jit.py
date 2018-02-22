import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import unittest
from contextlib import contextmanager
from itertools import product
import torch.jit.frontend
from torch.autograd import Variable, Function
from torch.autograd.function import traceable
from common import TestCase, run_tests, IS_WINDOWS
import io
import sys

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

RUN_CUDA = torch.cuda.is_available()
if torch.cuda.is_available():
    CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    for d in range(torch.cuda.device_count()):
        major = torch.cuda.get_device_capability(d)[0]
        if (CUDA_VERSION < 8000 and major >= 6) or (CUDA_VERSION < 9000 and major >= 7):
            RUN_CUDA = False

RUN_CUDA_MULTI_GPU = RUN_CUDA and torch.cuda.device_count() > 1

PY2 = sys.version_info[0] == 2


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
    def assertCompiled(self, compiled_fn):
        self.assertIsInstance(compiled_fn, torch._C.CompiledFunction)
        hits, misses = compiled_fn.hits, compiled_fn.misses
        yield
        self.assertLess(hits, compiled_fn.hits)
        self.assertEqual(misses, compiled_fn.misses)

    def assertExpectedTrace(self, trace, *args, **kwargs):
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_dce(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_canonicalize(trace)
        torch._C._jit_pass_lint(trace)
        self.assertExpected(str(trace), *args, **kwargs)

    def test_simple(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def f(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        trace, z = torch.jit.trace(f, (x, y), nderivs=0)
        self.assertExpectedTrace(trace)

    # matmul is currently implemented as a native function, which
    # exercises different codepaths in the JIT.  The following two
    # tests ensure that (1) matmul indeed traces into an atomic,
    # native operation, and (2) the JIT knows how to run it

    def test_matmul_native(self):
        x = Variable(torch.Tensor([[0.4]]), requires_grad=True)
        y = Variable(torch.Tensor([[0.7]]), requires_grad=True)

        trace, z = torch.jit.trace(lambda x, y: x.matmul(y), (x, y), nderivs=0)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_dce(trace)
        self.assertExpectedTrace(trace)

    def test_matmul_native_run(self):
        x = Variable(torch.Tensor([[0.4]]), requires_grad=True)
        y = Variable(torch.Tensor([[0.7]]), requires_grad=True)

        @torch.jit.compile(nderivs=0)
        def fn(x, y):
            return x.matmul(y)

        z = fn(x, y)
        with self.assertCompiled(fn):
            z2 = fn(x, y)
        self.assertEqual(z, z2)

    # index-2 is not implemented in interpreter
    @unittest.expectedFailure
    def test_index(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.LongTensor([0]), requires_grad=True)

        @torch.jit.compile(nderivs=0)
        def fn(x, y):
            return x[y]

        z = fn(x, y)
        with self.assertCompiled(fn):
            z2 = fn(x, y)
        self.assertEqual(z, z2)

    # Backwards tracing was broken for indexing by a constant,
    # because it's internally implemented using as_strided,
    # and we attempted to trace its derivative (which is not
    # currently supported.)  It currently works because
    # slice() is now not marked as traceable.
    def test_index_constant(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)

        @torch.jit.compile(nderivs=1)
        def fn(x):
            return x[0]

        z = fn(x)
        z.backward()
        grad = x.grad.clone()
        x.grad.zero_()
        with self.assertCompiled(fn):
            z2 = fn(x)
            z2.backward()
            grad2 = x.grad.clone()
        self.assertEqual(z, z2)
        self.assertEqual(grad, grad2)

    def test_scopes(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def f(x, y):
            out = x + y
            with torch.jit.scope('Foo', out):
                out = x * out
                with torch.jit.scope('Bar', out):
                    out = torch.tanh(out)
                out = torch.sigmoid(out)
            return out

        trace, z = torch.jit.trace(f, (x, y), nderivs=0)
        self.assertExpectedTrace(trace)

    def test_scopes_intermediate_node(self):

        class Net(nn.Module):
            def forward(self, x):
                return F.log_softmax(x, dim=0)

        net = Net()
        t = Variable(torch.ones(2), requires_grad=True)
        trace, _ = torch.jit.trace(net, (t, ))
        torch.onnx._optimize_trace(trace, False)

        self.assertExpectedTrace(trace)

    def test_scopes_identity_node(self):

        class Net(nn.Module):

            def __init__(self):
                super(Net, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )

            def forward(self, x):
                x = self.features(x)
                return x

        model = Net()

        t = Variable(torch.ones(1, 3, 227, 227), requires_grad=True)

        with torch.onnx.set_training(model, False):
            trace, _ = torch.jit.trace(model, (t, ))

        torch.onnx._optimize_trace(trace, False)

        self.assertExpectedTrace(trace)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lstm_fusion(self):
        input = Variable(torch.randn(3, 10).float().cuda())
        hx = Variable(torch.randn(3, 20).float().cuda())
        cx = Variable(torch.randn(3, 20).float().cuda())
        module = nn.LSTMCell(10, 20).float().cuda()  # Just to allocate weights with correct sizes

        trace, _ = torch.jit.trace(LSTMCell, (input, (hx, cx)) + tuple(module.parameters()))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_dce(trace)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        self.assertExpectedTrace(trace)

    def run_lstm_fusion(self, use_cuda):
        def to_type(x):
            x = x.float()
            if use_cuda:
                x = x.cuda()
            return x

        def rand_v(a, b):
            return Variable(to_type(torch.randn(a, b)))

        input = rand_v(3, 10)
        hx = rand_v(3, 20)
        cx = rand_v(3, 20)
        module = to_type(nn.LSTMCell(10, 20))  # Just to allocate weights with correct sizes

        CompiledLSTMCell = torch.jit.compile(nderivs=0)(LSTMCell)

        z = CompiledLSTMCell(input, (hx, cx), *module.parameters())
        with self.assertCompiled(CompiledLSTMCell):
            z2 = CompiledLSTMCell(input, (hx, cx), *module.parameters())
        self.assertEqual(z, z2)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_run_lstm_fusion_cuda(self):
        self.run_lstm_fusion(True)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    def test_run_lstm_fusion_cpu(self):
        self.run_lstm_fusion(False)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
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

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_concat_fusion(self):
        hx = Variable(torch.randn(3, 20).float().cuda())
        cx = Variable(torch.randn(3, 20).float().cuda())

        def Foo(hx, cx):
            return torch.cat((hx + cx, hx * cx))

        trace, _ = torch.jit.trace(Foo, (hx, cx))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_fuse(trace)
        self.assertExpectedTrace(trace)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_fusion_distribute(self):
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2
        x = Variable(torch.randn(4, 4).float().cuda())
        y = Variable(torch.randn(4, 4).float().cuda())
        trace, _ = torch.jit.trace(f, (x, y), nderivs=0)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_dce(trace)
        self.assertExpectedTrace(trace, 'raw')
        torch._C._jit_pass_fuse(trace)
        self.assertExpectedTrace(trace)

    def test_arg_configurations(self):
        """Different arg configurations should trigger different traces"""
        x = Variable(torch.FloatTensor(4, 4).uniform_())
        x_double = Variable(x.data.double())
        x_grad = Variable(x.data.clone(), requires_grad=True)
        y = Variable(torch.randn(4))

        configurations = [
            (x,),
            (x_double,),
            (x_grad,),
            (y,),
            ([x, x],),
            ([x, y],),
        ]
        if torch.cuda.is_available():
            x_cuda = Variable(x.data.cuda())
            configurations += [
                (x_cuda,),
                ([x, x_cuda],),
                ([x_cuda, x],),
                ([[x_cuda, x]],),
            ]
            if torch.cuda.device_count() > 1:
                x_cuda_1 = Variable(x.data.cuda(1))
                configurations += [
                    (x_cuda_1,),
                    ([x_cuda, x_cuda_1],),
                ]

        @torch.jit.compile(nderivs=0)
        def fn(*args):
            in_vars, _ = torch._C._jit_flatten(args)
            return in_vars[0] + 1

        for i, config in enumerate(configurations):
            self.assertFalse(fn.has_trace_for(*config))
            fn(*config)
            self.assertTrue(fn.has_trace_for(*config))
            for unk_config in configurations[i + 1:]:
                self.assertFalse(fn.has_trace_for(*unk_config))
        self.assertEqual(fn.hits, 0)

    def test_cse(self):
        x = Variable(torch.Tensor([0.4, 0.3]), requires_grad=True)
        y = Variable(torch.Tensor([0.7, 0.5]), requires_grad=True)

        trace, inputs = torch._C._tracer_enter((x, y), 0)

        def fn(x, y):
            w = (x + y) * (x + y) * (x + y)
            t = torch.tanh(w) + torch.tanh(w)
            z = (x + y) * (x + y) * (x + y) + t
            return z
        z = fn(*inputs)
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_cse(trace)

        self.assertExpectedTrace(trace)

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

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
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

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "needs non-zero device")
    def test_compile_fuse_last_device(self):
        max_device = torch.cuda.device_count() - 1
        x = Variable(torch.Tensor([0.4]), requires_grad=True).float().cuda(max_device)
        y = Variable(torch.Tensor([0.7]), requires_grad=True).float().cuda(max_device)

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
        self.assertExpectedTrace(trace)

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

        trace, inputs = torch._C._tracer_enter((x, y), 1)

        def fn(x, y):
            z = torch.sigmoid(x * (x + y))
            w = torch.abs(x * x * x + y) + Variable(torch.ones(1))
            return z, w
        z, w = fn(*inputs)

        torch._C._tracer_exit((z, w))
        torch._C._jit_pass_lint(trace)

        (z * w).backward()
        torch._C._jit_pass_dce(trace)
        torch._C._jit_pass_lint(trace)

        x_grad = x.grad.data.clone()
        x.grad.data.zero_()

        function = torch._C._jit_createInterpreterFactory(trace)
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

        trace, (tx,) = torch._C._tracer_enter((x,), 0)

        y = Variable(torch.diag(torch.Tensor([2, 2])))
        z = tx.matmul(y)

        torch._C._tracer_exit((z,))
        function = torch._C._jit_createInterpreterFactory(trace)

        z2 = function()(x)
        self.assertEqual(z, z2)

        y.data.fill_(1000)  # make sure the data has been cloned

        x2 = Variable(torch.ones(2, 2) * 2, requires_grad=True)
        z3 = function()(x2)
        self.assertEqual(z3.data, torch.ones(2, 2) * 4)

    def test_c_function(self):
        x = Variable(torch.randn(1, 3, 10, 10))
        m = nn.Conv2d(3, 8, 3, 1)

        trace, inputs = torch._C._tracer_enter((x,) + tuple(m.parameters()), 0)
        y = m(inputs[0])
        torch._C._tracer_exit((y,))
        self.assertExpectedTrace(trace)

    def test_legacy_fail(self):

        class MyLegacyFn(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, inputs = torch._C._tracer_enter((x,), 0)
        self.assertRaisesRegex(RuntimeError, "MyLegacyFn", lambda: MyLegacyFn()(*inputs))
        torch._C._tracer_exit(inputs)

    def test_inplace_transplant(self):
        x = Variable(torch.Tensor([0]), requires_grad=True)
        trace, inputs = torch._C._tracer_enter((x,), 0)

        def fn(x):
            y = x.clone()
            y.add_(2)
            y.add_(3)
            return y
        y = fn(*inputs)
        torch._C._tracer_exit((y,))
        self.assertExpectedTrace(trace)

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
        trace, inputs = torch._C._tracer_enter((x,), 0)

        def fn(x):
            y = RegularFn.apply(x)
            y = InplaceFn.apply(y)
            y = InplaceFn.apply(y)
            y = RegularFn.apply(y)
            return y
        y = fn(*inputs)
        torch._C._tracer_exit((y,))
        torch._C._jit_pass_dce(trace)
        ops = [n for n in trace.graph().nodes()]
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

        trace, inputs = torch._C._tracer_enter((x, y), 2)

        def fn(x, y):
            return y * 2 * x
        z = fn(*inputs)
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
        # This is nondeterministic, see:
        #   https://github.com/ezyang/pytorch/issues/227
        # self.assertExpectedTrace(trace)
        self.skipTest("output is nondeterministic on Travis/Python 3.5")

    def test_backward_opaque(self):
        x = Variable(torch.randn(3, 3), requires_grad=True)
        y = Variable(torch.randn(3, 3), requires_grad=True)

        trace, inputs = torch._C._tracer_enter((x, y), 2)

        def fn(x, y):
            return x.cross(y)
        z = fn(*inputs)
        torch._C._tracer_exit((z,))
        torch._C._jit_pass_lint(trace)

        # Run first backward
        grad, = torch.autograd.grad(z, x, Variable(torch.ones(3, 3), requires_grad=True), create_graph=True)
        torch._C._jit_pass_lint(trace)

        # Run dead code elimination to remove unused trace nodes
        torch._C._jit_pass_dce(trace)
        # This is nondeterministic, see:
        #   https://github.com/ezyang/pytorch/issues/227
        # self.assertExpectedTrace(trace)
        self.skipTest("output is nondeterministic on Travis/Python 3.5")

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
            trace, inputs = torch._C._tracer_enter((x, y), num_backwards)

            def fn(x, y):
                return y * 2 * x
            z = fn(*inputs)
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

    def test_no_grad_fallback(self):
        """Check that Traceable falls back to num_backwards=0 if in no-backprop mode"""
        x = Variable(torch.randn(2, 2))
        y = Variable(torch.randn(2, 2), requires_grad=True)

        @torch.jit.compile
        def fn(x, y):
            return x * x + x * y

        out = fn(x, y)
        self.assertFalse(fn.has_trace_for(x, y))
        with torch.no_grad():
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

        with self.assertRaisesRegex(RuntimeError, 'was compiled with'):
            fn(x).backward(Variable(torch.ones(1), requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, 'was compiled with'):
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
            n_ = g2.createClone(node, lambda x: g_to_g2[x])
            g2.appendNode(n_)
            for o, no in zip(node.outputs(), n_.outputs()):
                g_to_g2[o] = no

        for node in g.outputs():
            g2.registerOutput(g_to_g2[node])

        t_node = g2.create("TensorTest").t_("a", torch.ones([2, 2]))
        assert(t_node.attributeNames() == ["a"])
        g2.appendNode(t_node)
        assert(torch.equal(torch.ones([2, 2]), t_node.t("a")))
        self.assertExpected(str(g2))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "cpp tests require CUDA")
    def test_cpp(self):
        # rather than rebuild assertExpected in cpp,
        # just glob all the cpp outputs into one file for now
        self.assertExpected(torch._C._jit_run_cpp_tests())

    def test_batchnorm(self):
        x = Variable(torch.randn(2, 2, 2, 2).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.trace(nn.BatchNorm2d(2), x)
        self.assertExpectedTrace(trace)

    def test_dropout(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.trace(nn.Dropout(0.6), x)
        self.assertExpectedTrace(trace)

    def test_batchnorm_run_twice(self):
        @torch.jit.compile(nderivs=0)
        class MyBatchNorm2d(nn.BatchNorm2d):
            pass

        bn = MyBatchNorm2d(1)
        x = Variable(torch.randn(5, 1, 2, 1))
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
        self.assertExpectedTrace(trace)

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

    def test_unused_input(self):
            @torch.jit.compile(nderivs=1)
            def fn(a, b, c):
                return a + b

            a, b, c = [Variable(torch.randn(2, 2), requires_grad=True) for _ in range(3)]
            fn(a, b, c).sum().backward()
            with self.assertCompiled(fn):
                fn(a, b, c).sum().backward()

    def test_repeated_input(self):
        @torch.jit.compile(nderivs=1)
        def fn(a, b):
            return a + b

        a, b = [Variable(torch.randn(2, 2), requires_grad=True) for _ in range(2)]
        fn(a, a).sum().backward()
        with self.assertCompiled(fn):
            fn(a, a).sum().backward()
        with self.assertCompiled(fn):
            fn(a, b).sum().backward()
        self.assertExpected(str(fn.graph_for(a, a)))

    def test_repeated_output(self):
        @torch.jit.compile(nderivs=1)
        def fn(a, b):
            z = a + b
            return z, z

        a, b = [Variable(torch.randn(2, 2), requires_grad=True) for _ in range(2)]
        sum(fn(a, b)).sum().backward()
        with self.assertCompiled(fn):
            sum(fn(a, b)).sum().backward()
        self.assertExpected(str(fn.graph_for(a, b)))

    def test_re_enter(self):
            @torch.jit.compile(nderivs=1)
            def fn(a, b):
                return a + b

            @torch.jit.compile(nderivs=1)
            def fn2(a, b, c):
                    return fn(a, b) + c

            a, b, c = [Variable(torch.randn(2, 2), requires_grad=True) for _ in range(3)]

            fn(a, b).sum().backward()
            with self.assertCompiled(fn):
                fn(a, b).sum().backward()

            fn2(a, b, c).sum().backward()
            with self.assertCompiled(fn2):
                fn2(a, b, c).sum().backward()

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
        self.assertTrue(model.has_trace_for(x, y))

        with self.assertCompiled(model):
            z, _ = model(x, y)
        z.sum().backward()

    def test_module_cast(self):
        """Compiled modules can be casted to other data types"""
        @torch.jit.compile(nderivs=0)
        class Adder(nn.Module):
            def __init__(self):
                super(Adder, self).__init__()
                self.y = nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                return x + self.y

        x = Variable(torch.randn(2, 2).float())
        # Wrap it in a sequential to make sure it works for submodules
        a = nn.Sequential(Adder()).float()

        def check_type(caster):
            caster(a)
            a(caster(x))
            with self.assertCompiled(a[0]):
                a(caster(x))

        check_type(lambda x: x)
        check_type(lambda x: x.double())
        if torch.cuda.is_available():
            check_type(lambda x: x.float().cuda())
            check_type(lambda x: x.double().cuda())
        self.assertEqual(a[0].hits, 4 if torch.cuda.is_available() else 2)

    # Tracer fails when it receives the same grad variable as multiple input to
    # traced region. The problem is that it's not immediately obvious how to
    # assign multiple inputs to this Variable. It might be possible to solve
    # this using the view mechanism, but this requires some thought.
    # In general, it should be supported, because the user has no control
    # over this (and it's quite common, e.g. the sum call below will pass the same
    # grad variable as both inputs to grad of fn).
    @unittest.skip("Broken - repeated grads trigger an assertion failure.")
    def test_repeated_grad(self):
        @torch.jit.compile
        def fn(x):
            return x * x, x + x

        x = Variable(torch.randn(5, 5), requires_grad=True)
        # This shouldn't raise!
        sum(fn(x)).sum().backward()

    def test_input_pruning(self):
        """Check that stage 1 will return only one value"""
        # One of the inputs doesn't require grad, so it should be pruned
        @torch.jit.compile
        def fn(x, y):
            return x * y, x + y

        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = Variable(torch.randn(5, 5))

        out = fn(x, y)
        (out[0] * out[1]).sum().backward()
        with self.assertCompiled(fn):
            fn(x, y)
        self.assertExpected(str(fn.graph_for(x, y)))

    def test_output_pruning(self):
        """Check that stage 1 will take one value as an argument"""
        # One of the outputs doesn't require grad, so it should be pruned
        @torch.jit.compile
        def fn(x, y):
            return x * y, y + y

        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = Variable(torch.randn(5, 5))

        out = fn(x, y)
        (out[0] * out[1]).sum().backward()
        with self.assertCompiled(fn):
            fn(x, y)
        self.assertExpected(str(fn.graph_for(x, y)))

    @skipIfNoTorchVision
    def test_alexnet(self):
        return
        x = Variable(torch.randn(10, 3, 224, 224).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.trace(torchvision.models.AlexNet(), x)
        self.assertExpectedTrace(trace)
        # NB: Purposely NOT testing protobuf export here

    def test_debug_info(self):
        """Check that debug info doesn't crash and has some reasonable info"""

        @torch.jit.compile(nderivs=1)
        def fn(x, y):
            return x * y + x + y

        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = Variable(torch.randn(5, 5), requires_grad=True)

        out = fn(x, y)

        out.sum().backward()

        for _ in range(0, 100):
            out = fn(x, y)
        info_str = fn.jit_debug_info()
        self.assertTrue("hits: 100" in info_str)
        self.assertTrue("stage 1" in info_str)

    # Inplace copies don't work with tracer yet.
    # This is actually somewhat important to support correctly
    # as all backwards functions of views are implemented
    # as a zero filled tensor with a gradient fill on the
    # viewed portion.
    @unittest.expectedFailure
    def test_inplace_copy(self):
        x = Variable(torch.randn(4, 4), requires_grad=True)

        def f(x):
            out = Variable(torch.zeros(x.size()))
            out.copy_(x)
            return out

        trace, z = torch.jit.trace(f, (x, ), nderivs=0)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_dce(trace)
        self.assertExpectedTrace(trace)

    def test_index_trace(self):
        x = Variable(torch.randn(4, 4), requires_grad=True)
        trace, z = torch.jit.trace(lambda x: x[0], (x, ), nderivs=1)
        z.sum().backward()
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_dce(trace)
        self.assertExpectedTrace(trace)

    def test_saved_output(self):
        x = Variable(torch.randn(4, 4), requires_grad=True)

        @torch.jit.compile(nderivs=1)
        def fn(x):
            return x.sigmoid()

        fn(x).sum().backward()
        self.assertExpected(str(fn.graph_for(x)))

    def test_shared_param(self):

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.b = self.a = nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                return x * self.a + self.b

        m = MyModule()
        trace, _ = torch.jit.trace(m, (Variable(torch.randn(2, 2)),), nderivs=0)
        self.assertEqual(len(list(trace.graph().inputs())), 2)
        self.assertExpected(str(trace))

    def test_nested_inplace(self):
        x = Variable(torch.randn(2, 2))
        trace, _ = torch.jit.trace(lambda x: F.threshold(x, 0, 0, inplace=True), (x,), nderivs=0)
        self.assertExpectedTrace(trace)

    def checkGraphExecutor(self, func, reference_tensors, input_tensors=None, optimize=True, drop=None):
        def allSum(vs):
            # drop allows us to remove some values from ever being used
            # to test unused outputs
            if drop is not None:
                vs = vs[:-drop]
            # we don't want all the grad for all the outputs to be the same
            # so we multiply each by a constant
            return sum([(i + 1) * v.sum() for i, v in enumerate(vs) if v is not None])
        if input_tensors is None:
            input_tensors = reference_tensors

        nograd_inputs = [Variable(t) for t in reference_tensors]
        recording_inputs = [Variable(t, requires_grad=True)
                            for t in reference_tensors]

        ge = torch._C.GraphExecutor(func, [Variable(t) for t in input_tensors], optimize)

        # test no gradients case

        outputs = func(*nograd_inputs)
        outputs_ge = ge(*nograd_inputs)
        self.assertEqual(outputs, outputs_ge)

        # test single grad case

        outputs = func(*recording_inputs)
        grads = torch.autograd.grad(allSum(outputs), recording_inputs)

        outputs_ge = ge(*recording_inputs)
        grads_ge = torch.autograd.grad(allSum(outputs_ge), recording_inputs)
        self.assertEqual(outputs, outputs_ge)
        self.assertEqual(grads, grads_ge)

        # test the grad grad case

        outputs = func(*recording_inputs)
        l1 = allSum(outputs)
        grads = torch.autograd.grad(l1, recording_inputs, create_graph=True)
        l2 = (allSum(grads) * l1)
        grads2 = torch.autograd.grad(l2, recording_inputs)

        recording_inputs = [Variable(t, requires_grad=True)
                            for t in reference_tensors]

        outputs_ge = ge(*recording_inputs)
        l1_ge = allSum(outputs_ge)
        grads_ge = torch.autograd.grad(
            l1_ge, recording_inputs, create_graph=True)
        l2_ge = (allSum(grads_ge) * l1_ge)
        grads2_ge = torch.autograd.grad(l2_ge, recording_inputs)

        self.assertEqual(outputs, outputs_ge)
        self.assertEqual(grads, grads_ge)
        self.assertEqual(grads2, grads2_ge)

    def run_ge_tests(self, optimize, use_cuda):
        def rand(*args):
            t = torch.rand(*args).float()
            if use_cuda:
                t = t.cuda()
            return t
        self.checkGraphExecutor(lambda a, b: a * b + b,
                                [rand(1), rand(1)], [rand(2, 3), rand(2, 3)],
                                optimize=optimize)
        # trivial identity
        self.checkGraphExecutor(lambda a, b: (
            b, a), [rand(1), rand(1)], optimize=optimize)

        def foo(a):
            t = a * a
            return t * t, 4 * t
        self.checkGraphExecutor(foo, [rand(1)], optimize=optimize)
        # unused input
        self.checkGraphExecutor(
            lambda a, b: a * a, [rand(1), rand(1)], optimize=optimize)
        # test outputs that do not get used in grad
        self.checkGraphExecutor(foo, [rand(1)], drop=1, optimize=optimize)
        # test autograd fallback
        self.checkGraphExecutor(lambda a, b: a * b /
                                (a - 2 * b) + b, [rand(1), rand(1)],
                                optimize=optimize)

    def test_ge_unoptimized(self):
        self.run_ge_tests(False, False)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    def test_ge_optimized(self):
        self.run_ge_tests(True, False)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_ge_cuda(self):
        self.run_ge_tests(True, True)

    # more manual test of graph executor that can be used as a scratchpad
    def test_ge(self):
        def foo(a, b):
            return a * b / (a - b) + b
        V = Variable
        a, b = V(torch.rand(1)), V(torch.rand(1))
        ge = torch._C.GraphExecutor(foo, (a, b))
        a, b = V(torch.rand(1), requires_grad=True), V(
            torch.rand(1), requires_grad=True)
        r, = ge(a, b)
        da, db = torch.autograd.grad(r + 3, [a, b], create_graph=True)

        l2 = (da * db + db * db)
        g2result = torch.autograd.grad(l2, [da, db])

        r = foo(a, b)
        da2, db2 = torch.autograd.grad(r + 3, [a, b], create_graph=True)
        self.assertEqual(da, da2)
        self.assertEqual(db, db2)
        l3 = (da2 * db2 + db2 * db2)
        g2result2 = torch.autograd.grad(l3, [da2, db2])
        self.assertEqual(g2result, g2result2)

    def checkScript(self, script, inputs, outputs, optimize, name='func'):
        cu = torch.jit._jit_script_compile(script)
        graph = cu.get_graph(name)
        ge = torch._C.GraphExecutor(graph, optimize)
        outputs_ge = ge(*inputs)
        self.assertEqual(outputs, outputs_ge)

    def test_script_add(self):
        script = '''
        def func(a, b) -> (c):
            c = a + b
            c += a
        '''

        a = Variable(torch.rand(1), requires_grad=True)
        b = Variable(torch.rand(1), requires_grad=True)
        outputs = a + b + a
        self.checkScript(script, [a, b], [outputs], False)

    def test_script_mul(self):
        script = '''
        def func(a, b) -> (c):
            c = a * b
        '''

        a = Variable(torch.rand(1), requires_grad=True)
        b = Variable(torch.rand(1), requires_grad=True)
        outputs = a * b
        self.checkScript(script, [a, b], [outputs], False)

    @unittest.skip("RuntimeError: Expected object of type CPUFloatType "
                   "but found type Variable[CPUFloatType] for argument #2 'other'")
    def test_script_triple(self):
        script = '''
        def func(x) -> (y):
            y = 3f * x
        '''
        x = Variable(torch.rand(1).float(), requires_grad=True)
        outputs = 3 * x
        self.checkScript(script, [x], [outputs], False)

    def test_script_slice(self):
        script = '''
        def func(x) -> (head):
            head = x[:5]
        '''
        x = Variable(torch.rand(10).float(), requires_grad=True)
        outputs = x[:5]
        self.checkScript(script, [x], [outputs], False)

    def test_script_gather(self):
        script = '''
        def func(x) -> (y):
            y = x[0]
        '''
        x = Variable(torch.rand(10).float(), requires_grad=True)
        outputs = x[0]
        self.checkScript(script, [x], [outputs], False)

    def test_script_func_call(self):
        script = '''
        def add(a, b) -> (c):
            c = a + b

        def mul(a, x) -> (y):
            y = a * x

        def func(alpha, beta, x, y) -> (z):
            z = add(mul(alpha, x), mul(beta, y))
        '''
        alpha = Variable(torch.rand(1).float(), requires_grad=True)
        beta = Variable(torch.rand(1).float(), requires_grad=True)
        x = Variable(torch.rand(3).float(), requires_grad=True)
        y = Variable(torch.rand(3).float(), requires_grad=True)
        outputs = alpha * x + beta * y
        self.checkScript(script, [alpha, beta, x, y], [outputs], False)

    @unittest.skip("RuntimeError: VariableType::ID() not implemented")
    def test_script_cast(self):
        script = '''
        def to_int(x) -> (y):
            y = int(x)
        '''
        x = Variable(torch.FloatTensor([1.1, 2.3]), requires_grad=True)
        outputs = Variable(torch.IntTensor([1, 2]), requires_grad=True)
        self.checkScript(script, 'to_int', [x], [outputs], False)

    def test_python_frontend(self):
        def fn(x, y, z):
            q = x + y - z
            w = -z
            if not x and not y and z:
                m = x if not z else y
            while x < y > z:
                q = x
            return x

        ast = torch.jit.frontend.get_jit_ast(fn)
        self.assertExpected(str(ast))

    def test_script_while(self):
        cu = torch.jit._jit_script_compile('''
        def test_while(a, b) -> (c):
            while a < 10:
                a = a + 1
                b = b + 1
            c = a + b
        ''')
        self.assertExpected(str(cu.get_graph('test_while')))

    def test_script_fibb(self):
        cu = torch.jit._jit_script_compile('''
        def test_while(lim) -> (third):
            first = 1
            second = 1
            i = 1
            somenum = 5
            dontmutateme = 3
            third = 0 # TODO: python lexical scoping
            while i < lim:
                third = first + second
                first = second
                second = third
                j = 0
                while j < 10:
                    somenum = somenum * 2
                    j = j + 1
                i = i + j
                i = i + dontmutateme

            st = second + third
            fs = first + second

        ''')
        self.assertExpected(str(cu.get_graph('test_while')))

    def test_script_if(self):
        cu = torch.jit._jit_script_compile('''
        def test_if(a, b) -> (c):
            d = 3
            if a > 10:
                a = 3 + d
            else:
                b = 3 + d
                d = 4
            c = a + b
        ''')
        self.assertExpected(str(cu.get_graph('test_if')))

    def test_script_if_noelse(self):
        cu = torch.jit._jit_script_compile('''
        def test_if_noelse(a, b) -> (c):
            if a > 10:
                a = 3 + b
            c = a + b
        ''')
        self.assertExpected(str(cu.get_graph('test_if_noelse')))

    def test_script_while_nonexistant_value(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value x"):
            cu = torch.jit._jit_script_compile('''
            def test_while(a, b) -> (c):
                while a < 10:
                    a = a + x
                    b = b + 1
                c = a + b
            ''')

    def test_script_while_nonexistant_cond_value(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value x"):
            cu = torch.jit._jit_script_compile('''
            def test_while(a, b) -> (c):
                while a < x:
                    a = a + 1
                    b = b + 1
                c = a + b
            ''')

    def test_script_while_write_outer_then_read(self):
        cu = torch.jit._jit_script_compile('''
        def test_while(a, b) -> (c):
            while a < 10:
                a = a + 1
                b = a + 1
            c = a + b
        ''')
        self.assertExpected(str(cu.get_graph('test_while')))

    def test_script_while_nest_if(self):
        cu = torch.jit._jit_script_compile('''
        def test_while_if(a, b) -> (c):
            c = 0
            while a < 10:
                a = a + 1
                b = b + 1
                if a > b:
                    c = -a
                else:
                    c = -b
            c = c + 1
        ''')
        self.assertExpected(str(cu.get_graph('test_while_if')))

    def test_script_if_nest_while(self):
        cu = torch.jit._jit_script_compile('''
        def test_if_while(a, b) -> (c):
            c = 0
            if a > b:
                while a > b:
                    b = b + 1
                    c = -b
        ''')
        self.assertExpected(str(cu.get_graph('test_if_while')))

    def test_script_ternary(self):
        cu = torch.jit._jit_script_compile('''
        def test_ternary_control(a, b) -> (c):
            c = 3
            if a > 3:
                c = a + b
            else:
                c = b
        ''')
        cu2 = torch.jit._jit_script_compile('''
        def test_ternary(a, b) -> (c):
            c = 3
            c = a + b if a > 3 else b
        ''')
        self.assertEqual(
            str(cu.get_graph('test_ternary_control')),
            str(cu2.get_graph('test_ternary')),
        )

if __name__ == '__main__':
    run_tests()
