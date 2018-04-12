import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from itertools import product, chain
import torch.jit.frontend
from torch.autograd import Variable, Function
from torch.autograd.function import traceable
from common import TestCase, run_tests, IS_WINDOWS
import io
import sys
import unittest
import inspect
import textwrap
import numpy as np
import tempfile
import shutil

from torch.jit.frontend import NotSupportedError

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
WINDOWS = sys.platform == 'win32'


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
        torch._C._jit_pass_lint(trace.graph())
        torch._C._jit_pass_dce(trace.graph())
        torch._C._jit_pass_lint(trace.graph())
        trace.set_graph(torch._C._jit_pass_canonicalize(trace.graph()))
        torch._C._jit_pass_lint(trace.graph())
        self.assertExpected(str(trace), *args, **kwargs)

    def assertExportImport(self, trace, inputs):
        initializers = []
        graph1 = trace.graph()

        ge1 = torch._C.GraphExecutor(graph1, False)
        out1 = ge1(*inputs)
        proto, _ = trace.graph().export(initializers, onnx_opset_version=0,
                                        defer_weight_export=False, export_raw_ir=True)

        graph2, initializers = torch._C._jit_import_graph(proto)
        ge2 = torch._C.GraphExecutor(graph2, False)
        out2 = ge2(*inputs)

        self.assertEqual(out1, out2)

    def test_simple(self):
        x = Variable(torch.Tensor([0.4]), requires_grad=True)
        y = Variable(torch.Tensor([0.7]), requires_grad=True)

        def f(x, y):
            return torch.sigmoid(torch.tanh(x * (x + y)))

        trace, z = torch.jit.get_trace_graph(f, (x, y), nderivs=0)
        self.assertExpectedTrace(trace)
        self.assertExportImport(trace, (x, y))

    # matmul is currently implemented as a native function, which
    # exercises different codepaths in the JIT.  The following two
    # tests ensure that (1) matmul indeed traces into an atomic,
    # native operation, and (2) the JIT knows how to run it

    def test_matmul_native(self):
        x = Variable(torch.Tensor([[0.4]]), requires_grad=True)
        y = Variable(torch.Tensor([[0.7]]), requires_grad=True)

        trace, z = torch.jit.get_trace_graph(lambda x, y: x.matmul(y), (x, y), nderivs=0)
        torch._C._jit_pass_lint(trace.graph())
        torch._C._jit_pass_dce(trace.graph())
        self.assertExpectedTrace(trace)
        self.assertExportImport(trace, (x, y))

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

        trace, z = torch.jit.get_trace_graph(f, (x, y), nderivs=0)
        self.assertExpectedTrace(trace)
        self.assertExportImport(trace, (x, y))

    def test_scopes_intermediate_node(self):

        class Net(nn.Module):
            def forward(self, x):
                return F.log_softmax(x, dim=0)

        net = Net()
        t = Variable(torch.ones(2), requires_grad=True)
        trace, _ = torch.jit.get_trace_graph(net, (t, ))
        self.assertExportImport(trace, (t, ))
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
            trace, _ = torch.jit.get_trace_graph(model, (t, ))

        self.assertExportImport(trace, (t, ) + tuple(model.parameters()))

        torch.onnx._optimize_trace(trace, False)

        self.assertExpectedTrace(trace)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_lstm_fusion(self):
        input = Variable(torch.randn(3, 10).float().cuda())
        hx = Variable(torch.randn(3, 20).float().cuda())
        cx = Variable(torch.randn(3, 20).float().cuda())
        module = nn.LSTMCell(10, 20).float().cuda()  # Just to allocate weights with correct sizes

        trace, _ = torch.jit.get_trace_graph(LSTMCell, (input, (hx, cx)) + tuple(module.parameters()))
        torch._C._jit_pass_lint(trace.graph())
        torch._C._jit_pass_dce(trace.graph())
        torch._C._jit_pass_lint(trace.graph())
        self.assertExportImport(trace, (input, hx, cx) + tuple(module.parameters()))
        torch._C._jit_pass_fuse(trace.graph())
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

        trace, _ = torch.jit.get_trace_graph(Foo, (hx, cx))
        self.assertExportImport(trace, (hx, cx))
        torch._C._jit_pass_lint(trace.graph())
        torch._C._jit_pass_fuse(trace.graph())
        self.assertExpectedTrace(trace)

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "fuser requires CUDA")
    def test_fusion_distribute(self):
        def f(x, y):
            z1, z2 = (x + y).chunk(2, dim=1)
            return z1 * z2
        x = Variable(torch.randn(4, 4).float().cuda())
        y = Variable(torch.randn(4, 4).float().cuda())
        trace, _ = torch.jit.get_trace_graph(f, (x, y), nderivs=0)
        torch._C._jit_pass_lint(trace.graph())
        torch._C._jit_pass_dce(trace.graph())
        self.assertExpectedTrace(trace, 'raw')
        self.assertExportImport(trace, (x, y))
        torch._C._jit_pass_fuse(trace.graph())
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
        torch._C._jit_pass_lint(trace.graph())
        torch._C._jit_pass_cse(trace.graph())

        self.assertExpectedTrace(trace)
        self.assertExportImport(trace, (x, y))

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
                a, = ctx.saved_tensors
                return a * grad_a

        x = Variable(torch.randn(10, 10), requires_grad=True)
        trace, out = torch.jit.get_trace_graph(MyFn.apply, x, nderivs=1)
        out.sum().backward()
        torch._C._jit_pass_dce(trace.graph())
        self.assertExpectedTrace(trace)

    def test_legacy_traced_module(self):
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
        torch._C._jit_pass_lint(trace.graph())

        (z * w).backward()
        torch._C._jit_pass_dce(trace.graph())
        torch._C._jit_pass_lint(trace.graph())

        x_grad = x.grad.data.clone()
        x.grad.data.zero_()

        function = torch._C._jit_createInterpreterFactory(trace)
        torch._C._jit_pass_lint(trace.graph())
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
        self.assertExportImport(trace, inputs)

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
        self.assertExportImport(trace, inputs)

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
        torch._C._jit_pass_dce(trace.graph())
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
        torch._C._jit_pass_lint(trace.graph())

        # Run first backward
        grad, = torch.autograd.grad(z, x, Variable(torch.ones(2, 2), requires_grad=True), create_graph=True)
        torch._C._jit_pass_lint(trace.graph())

        # Run second backward
        grad.sum().backward(create_graph=True)
        torch._C._jit_pass_lint(trace.graph())

        # Run dead code elimination to remove unused trace nodes
        torch._C._jit_pass_dce(trace.graph())
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
        torch._C._jit_pass_lint(trace.graph())

        # Run first backward
        grad, = torch.autograd.grad(z, x, Variable(torch.ones(3, 3), requires_grad=True), create_graph=True)
        torch._C._jit_pass_lint(trace.graph())

        # Run dead code elimination to remove unused trace nodes
        torch._C._jit_pass_dce(trace.graph())
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

    def test_trace_size(self):
        def fn(x):
            return x.view(x.shape[1] * 2, x.size(0), 2)

        x = torch.randn(5, 2, 4)
        y = torch.randn(4, 8, 4)

        # Check that it behaves as expected
        traced_fn = torch.jit.trace(x)(fn)
        self.assertEqual(traced_fn(y), fn(y))
        self.assertEqual(traced_fn(x), fn(x))

        # Check that the trace looks ok
        trace, _ = torch.jit.get_trace_graph(fn, (x,))
        self.assertExpectedTrace(trace)

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

        traced, _ = torch.jit.get_trace_graph(doit, (x, y))
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

        t_node = g2.create("prim::TensorTest").t_("a", torch.ones([2, 2]))
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
        trace, _ = torch.jit.get_trace_graph(nn.BatchNorm2d(2), x)
        self.assertExpectedTrace(trace)

    def test_dropout(self):
        x = Variable(torch.randn(2, 2).fill_(1.0), requires_grad=True)
        trace, _ = torch.jit.get_trace_graph(nn.Dropout(0.6), x)
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
        trace, _ = torch.jit.get_trace_graph(nn.Conv2d(16, 13, 3, bias=False), x)
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
        trace, _ = torch.jit.get_trace_graph(torchvision.models.AlexNet(), x)
        self.assertExpectedTrace(trace)
        self.assertExportImport(trace, (x, ))
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

        trace, z = torch.jit.get_trace_graph(f, (x, ), nderivs=0)
        torch._C._jit_pass_lint(trace.graph())
        torch._C._jit_pass_dce(trace.graph())
        self.assertExpectedTrace(trace)
        self.assertExportImport(trace, (x, ))

    def test_index_trace(self):
        x = Variable(torch.randn(4, 4), requires_grad=True)
        trace, z = torch.jit.get_trace_graph(lambda x: x[0], (x, ), nderivs=1)
        z.sum().backward()
        torch._C._jit_pass_lint(trace.graph())
        torch._C._jit_pass_dce(trace.graph())
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
        trace, _ = torch.jit.get_trace_graph(m, (Variable(torch.randn(2, 2)),), nderivs=0)
        self.assertEqual(len(list(trace.graph().inputs())), 2)
        self.assertExpected(str(trace))

    def test_nested_inplace(self):
        x = Variable(torch.randn(2, 2))
        trace, _ = torch.jit.get_trace_graph(lambda x: F.threshold(x, 0, 0, inplace=True), (x,), nderivs=0)
        self.assertExpectedTrace(trace)
        self.assertExportImport(trace, (x, ))

    def checkGraphExecutor(self, func, reference_tensors, input_tensors=None,
                           optimize=True, drop=None, allow_unused=False):
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
        grads = torch.autograd.grad(allSum(outputs), recording_inputs,
                                    allow_unused=allow_unused)

        outputs_ge = ge(*recording_inputs)
        grads_ge = torch.autograd.grad(allSum(outputs_ge), recording_inputs,
                                       allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_ge)
        self.assertEqual(grads, grads_ge)

        # test the grad grad case

        outputs = func(*recording_inputs)
        l1 = allSum(outputs)
        grads = torch.autograd.grad(l1, recording_inputs, create_graph=True,
                                    allow_unused=allow_unused)
        l2 = (allSum(grads) * l1)
        grads2 = torch.autograd.grad(l2, recording_inputs, allow_unused=allow_unused)

        recording_inputs = [Variable(t, requires_grad=True)
                            for t in reference_tensors]

        outputs_ge = ge(*recording_inputs)
        l1_ge = allSum(outputs_ge)
        grads_ge = torch.autograd.grad(
            l1_ge, recording_inputs, create_graph=True, allow_unused=allow_unused)
        l2_ge = (allSum(grads_ge) * l1_ge)
        grads2_ge = torch.autograd.grad(l2_ge, recording_inputs, allow_unused=allow_unused)

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
            lambda a, b: a * a, [rand(1), rand(1)], optimize=optimize,
            allow_unused=True)
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

    def test_trace_annotation(self):
        @torch.jit.trace(Variable(torch.rand(1)))
        def foo(a):
            return a + a + a
        s = Variable(torch.rand(2))

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    @unittest.skipIf(not RUN_CUDA, "calls .cuda()")
    def test_traced_module(self):
        class Model(nn.Module):
            def __init__(self, num_features, num_layers):
                super(Model, self).__init__()
                self.num_layers = num_layers
                layers = [[nn.Linear(num_features, num_features), nn.Sigmoid()]
                          for _ in range(num_layers)]
                self.submodule = nn.Sequential(*chain(*layers))

            def forward(self, x):
                for i in range(self.num_layers):
                    x = self.submodule[i](x) + x
                return x

        model = Model(5, 3)
        x = torch.randn(2, 5)
        traced_model = torch.jit.trace(x)(model)

        # We're missing some attributes these modules had initially. Make sure we can
        # still get the __repr__()
        model.__repr__()

        # XXX: indexing sequentials is broken
        linear_submodule = next(iter(traced_model.submodule._modules.values()))

        # All attributes that aren't parameters should raise
        with self.assertRaises(AttributeError):
            linear_submodule.in_features
        linear_submodule.weight
        with self.assertRaises(RuntimeError):
            traced_model.asdf = 4
        linear_submodule.weight = nn.Parameter(torch.randn(linear_submodule.weight.shape))
        with self.assertRaises(RuntimeError):
            del linear_submodule.weight

        # Submodules can't be called
        with self.assertRaises(RuntimeError):
            linear_submodule(x)

        # Type casts
        linear_submodule.cuda()
        traced_model.float().cuda()
        cuda_out = traced_model(x.float().cuda())
        traced_model.cpu()
        cpu_out = traced_model(x.float())
        self.assertEqual(cpu_out, cuda_out)
        traced_model.double()

        # state_dict + load_state_dict
        state = {k: v.clone() for k, v in traced_model.state_dict().items()}
        new_state = {k: v.clone().fill_(1) for k, v in state.items()}
        out = traced_model(x)
        traced_model.load_state_dict(new_state)
        out_ones = traced_model(x)
        traced_model.load_state_dict(state)
        out_state = traced_model(x)
        self.assertEqual(out, out_state)
        self.assertNotEqual(out, out_ones)

    def test_shape_prop_mismatch_output(self):
        with self.assertRaises(RuntimeError):
            cu = torch.jit.CompilationUnit('''
            def test_shape_prop_mismatch_output(a):
                b = slice(a, dim=0, end=-2, start=2, step=1)
                b = topk(a, dim=0, k=2, largest=True, sorted=True)
                return b
            ''')
            inputs = [torch.zeros(10)]
            outputs = [torch.zeros(2), torch.from_numpy(np.array([1, 5])).long()]

            real_outs = cu.test_shape_prop_mismatch_output(*inputs)
            self.assertEqual(real_outs, outputs)

    def test_view_shape_prop(self):
        cu = torch.jit.CompilationUnit('''
        def test_view_shape_prop(a):
            return view(a, size=[-1])
        ''')
        inputs = [torch.zeros(10, 10)]
        outputs = torch.zeros(100)

        real_outs = cu.test_view_shape_prop(*inputs)
        self.assertEqual(real_outs, outputs)

    def test_integral_shape_inference(self):
        cu = torch.jit.CompilationUnit('''
        def test_integral_shape_inference(a):
            return a / a
        ''')
        inputs = [torch.ones(10, 10).type(torch.LongTensor)]
        outputs = torch.ones(10, 10)

        self.assertEqual(cu.test_integral_shape_inference(*inputs), outputs)

    def test_shape_analysis_broadcast(self):
        def broadcast(a, b):
            return a + b

        x = torch.randn(3, 1, 5, requires_grad=True)
        y = torch.randn(4, 1, 8, 5, requires_grad=True)

        graph = torch.jit._script_graph(broadcast)
        torch._C._jit_pass_shape_analysis(graph, (x, y), False)
        self.assertExpected(str(graph))

    def test_fuser_multiple_blocks(self):
        cu = torch.jit.CompilationUnit('''
        def test_fuser_multiple_blocks(this, that, theother, meme):
            i = 0
            while i < 20:
                this = cat([this, meme], dim=0)
                that = cat([that, meme], dim=0)
                theother = cat([theother, meme], dim=0)
                i = i + 1
            return this, that, theother
        ''')

        inputs = [torch.ones(0, 10, 10)] * 3
        inputs += [torch.ones(1, 10, 10)]
        outputs = [torch.ones(20, 10, 10)] * 3

        self.assertEqual(cu.test_fuser_multiple_blocks(*inputs), outputs)

    def test_dropout_script(self):

        eg = torch.zeros(1, 2, 3, requires_grad=True)

        @torch.jit.trace(eg)
        def foo(x):
            x = torch.neg(x)
            return F.dropout(x)

        class MyDrop(nn.Module):
            def forward(self, x):
                return foo(x)

        f = io.BytesIO()
        torch.onnx.export(MyDrop(), (eg,), f, verbose=False)

    def test_python_function(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x + 1

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        @torch.jit.trace(torch.zeros(2))
        def fn(x):
            return MyFn.apply(x + 2) + 3

        x = torch.tensor([1., 2., 3.])
        y = torch.randn(2, 2, requires_grad=True)
        fn(x)
        fn(y)


class TestScript(TestCase):

    @contextmanager
    def capture_stdout(self):
        # No idea how to capture stdout from C++ on Windows
        if WINDOWS:
            yield ['']
            return
        import os
        import fcntl
        import errno
        sys.stdout.flush()
        stdout_fd = os.dup(1)
        r, w = os.pipe()
        try:
            # Override stdout with r - dup is guaranteed to return the lowest free fd
            os.close(1)
            os.dup(w)

            captured_stdout = ['']
            yield captured_stdout
            sys.stdout.flush()  # Make sure that Python hasn't buffered anything

            # Do the ugly dance to read all the data that was written into the pipe
            fcntl.fcntl(r, fcntl.F_SETFL, os.O_NONBLOCK)
            total_stdout = ''
            while True:
                try:
                    total_stdout += os.read(r, 1000).decode('ascii')
                except OSError as e:
                    if e.errno != errno.EAGAIN:
                        raise
                    break
            captured_stdout[0] = total_stdout
        finally:
            # Revert the change, and clean up all fds
            os.close(1)
            os.dup(stdout_fd)
            os.close(stdout_fd)
            os.close(r)
            os.close(w)

    def checkScript(self, script, inputs, optimize, outputs=None, name='func', capture_output=False):
        if isinstance(script, str):
            cu = torch.jit.CompilationUnit(script, optimize)
            ge = getattr(cu, name)
        else:
            if capture_output:
                with self.capture_stdout() as captured:
                    outputs = script(*inputs)
            else:
                outputs = script(*inputs)
            # Check the string frontend first
            source = textwrap.dedent(inspect.getsource(script))
            self.checkScript(source, inputs, optimize, outputs, script.__name__, capture_output)
            # Continue checking the Python frontend
            ge = torch.jit.script(script)

        if capture_output:
            with self.capture_stdout() as captured:
                outputs_ge = ge(*inputs)
            if not WINDOWS:
                self.assertExpected(captured[0], subname='stdout')
        else:
            outputs_ge = ge(*inputs)
        self.assertEqual(outputs, outputs_ge)

    def test_script_cu(self):
        cu = torch.jit.CompilationUnit('''
            def foo(a):
                b = a
                return b
        ''')
        a = Variable(torch.rand(1))
        self.assertEqual(a, cu.foo(a))

    def test_script_annotation(self):
        @torch.jit.script
        def foo(a):
            return a + a + a
        s = Variable(torch.rand(2))
        self.assertEqual(s + s + s, foo(s))

    def test_add(self):
        def func(a, b):
            c = a + b
            c += a
            return c

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)

    def test_mul(self):
        def func(a, b):
            return a * b

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)

    def test_triple(self):
        def func(x):
            return 3. * x

        x = torch.rand(1, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_slice(self):
        def func(x):
            return x[:5]

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_gather(self):
        def func(x):
            return x[0]

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_keyword(self):
        @torch.jit.script
        def func(x):
            return torch.sum(x, dim=0, keepdim=True)

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        y = func(x)
        y2 = torch.sum(x, dim=0, keepdim=True)
        self.assertEqual(y, y2)

    def test_literal(self):
        def func(a, b):
            c = [a, b]
            d, e = c
            return d + e

        def func2(a, b):
            c = a, b
            d, e = c
            return d + e

        def func3(a, b):
            c = a, (a, b)
            d, e = c
            f, g = e
            return d + f + g

        def func4(a, b):
            c = 0, (0, 0)
            x = True
            while x:
                x = False
                c = a, (a, b)
            d, e = c
            f, g = e
            return d + f + g

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)
        self.checkScript(func2, (a, b), optimize=True)
        self.checkScript(func3, (a, b), optimize=True)
        self.checkScript(func4, (a, b), optimize=True)

    def test_cat(self):
        @torch.jit.script
        def func(x):
            return torch.cat((x, x), dim=0)

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.assertEqual(func(x), torch.cat((x, x), dim=0))

        with self.assertRaisesRegex(RuntimeError, "expected 1 input"):
            @torch.jit.script
            def func(x):
                return torch.cat((x, x), x, dim=0)

    def test_func_call(self):
        script = '''
        def add(a, b):
            return a + b

        def mul(a, x):
            return a * x

        def func(alpha, beta, x, y):
            return add(mul(alpha, x), mul(beta, y))
        '''
        alpha = torch.rand(1, dtype=torch.float, requires_grad=True)
        beta = torch.rand(1, dtype=torch.float, requires_grad=True)
        x = torch.rand(3, dtype=torch.float, requires_grad=True)
        y = torch.rand(3, dtype=torch.float, requires_grad=True)
        outputs = alpha * x + beta * y
        # NOTE: cannot optimize yet because broadcasts are not inserted before the fuser runs
        self.checkScript(script, [alpha, beta, x, y], optimize=False, outputs=outputs)

    @unittest.skip("RuntimeError: VariableType::ID() not implemented")
    def test_cast(self):
        script = '''
        def to_int(x):
            return int(x)
        '''
        x = Variable(torch.FloatTensor([1.1, 2.3]), requires_grad=True)
        out = Variable(torch.IntTensor([1, 2]), requires_grad=True)
        self.checkScript(script, [x], optimize=True, outputs=[out], func='to_int')

    def test_python_frontend(self):
        def fn(x, y, z):
            q = x + y - z.sigmoid()
            print(q)
            w = -z
            if not x and not y and z:
                m = x if not z else y
            while x < y > z:
                q = x
            return x

        ast = torch.jit.frontend.get_jit_ast(fn)
        self.assertExpected(str(ast))

    def _make_scalar_vars(self, arr, dtype):
        return [torch.tensor(val, dtype=dtype) for val in arr]

    def test_while(self):
        def func(a, b, max):
            while a < max:
                a = a + 1
                b = b + 1
            c = a + b
            return c

        inputs = self._make_scalar_vars([1, 1, 10], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_fibb(self):
        def func(lim):
            first = 1
            second = 1
            i = 1
            somenum = 5
            dontmutateme = 3
            third = 0
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
            return third, st, fs

        inputs = self._make_scalar_vars([10], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if(self):
        def func(a, b):
            d = 3
            if a > 10:
                a = 3 + d
            else:
                b = 3 + d
                d = 4
            c = a + b
            return c

        inputs = self._make_scalar_vars([1, -1], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if_noelse(self):
        def func(a, b):
            if a > 10:
                a = 3 + b
            c = a + b
            return c

        inputs = self._make_scalar_vars([-1, 1], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_while_nonexistent_value(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value x"):
            torch.jit.CompilationUnit('''
            def test_while(a, b):
                while a < 10:
                    a = a + x
                    b = b + 1
                return a + b
            ''')

    def test_while_nonexistent_cond_value(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value x"):
            torch.jit.CompilationUnit('''
            def test_while(a, b):
                while a < x:
                    a = a + 1
                    b = b + 1
                return a + b
            ''')

    def test_while_write_outer_then_read(self):
        def func(a, b):
            while a < 10:
                a = a + 1
                b = a + 1
            return a + b

        inputs = self._make_scalar_vars([42, 1337], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_while_nest_if(self):
        def func(a, b):
            c = 0
            while a < 10:
                a = a + 1
                b = b + 1
                if a > b:
                    c = -a
                else:
                    c = -b
            return c + 1

        inputs = self._make_scalar_vars([-1234, 4321], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if_nest_while(self):
        def func(a, b):
            c = 0
            if a > b:
                while a > b:
                    b = b + 1
                    c = -b
            return c

        inputs = self._make_scalar_vars([4321, 1234], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_script_for_in_range(self):
        script = '''
        def test_for_in_range():
            c = 0
            for i in range(100):
                c += i
            return c
        '''
        self.checkScript(script, [], outputs=[4950], optimize=True, name='test_for_in_range')

    def test_script_for_in_range_dynamic(self):
        script = '''
        def test_script_for_in_range_dynamic():
            c = 0
            for i in range(100):
                acc = 0
                for j in range(i):
                    acc += j
                c += acc
            return c
        '''
        self.checkScript(script, [], outputs=[161700], optimize=True, name='test_script_for_in_range_dynamic')

    def test_script_for_in_range_ast(self):
        @torch.jit.script
        def test_script_for_in_range_ast(zero):
            c = zero
            for i in range(100):
                acc = zero
                for j in range(i):
                    acc += j
                c += acc
            return c

        inputs = self._make_scalar_vars([0], torch.int64)

        self.assertEqual(test_script_for_in_range_ast(*inputs), 161700)

    def test_script_bool_constant(self):
        script = '''
        def test_script_bool_constant():
            a = True
            return a
        '''
        outputs = [1]
        self.checkScript(script, [], outputs[0], True, 'test_script_bool_constant')

    def test_ternary(self):
        def func(a, b):
            c = 3
            c = a + b if a > 3 else b
            return c

        inputs_true = self._make_scalar_vars([5, 2], torch.int64)
        inputs_false = self._make_scalar_vars([1, 0], torch.int64)
        self.checkScript(func, inputs_true, optimize=True)
        self.checkScript(func, inputs_false, optimize=True)

    def test_print(self):
        def func(x, y):
            q = (x + y).sigmoid()
            print(q)
            w = -q
            return w * w

        x = torch.arange(4., requires_grad=True)
        y = torch.arange(0., 8, 2, requires_grad=True)
        self.checkScript(func, [x, y], optimize=True, capture_output=True)

    def test_multiple_assignment(self):
        def outer_func(x):
            return x * 2, x + 2

        @torch.jit.script
        def func(x):
            y, z = outer_func(x)
            return y + z

        x = torch.arange(4)
        self.assertEqual(func(x), x * 2 + x + 2)

    def test_literals(self):
        def func(a):
            return a.view(size=[1, 2, 3])

        a = torch.randn(6)
        self.checkScript(func, [a], optimize=True)

    def test_return(self):
        def no_return(a):
            a + 1

        def void_return(a):
            return

        def one_return(a):
            return a + 1.

        def multiple_returns(a):
            return a * 1., a * 2., a * 3.

        a = torch.randn(1, dtype=torch.float)
        self.checkScript(no_return, [a], optimize=True)
        self.checkScript(void_return, [a], optimize=True)
        self.checkScript(one_return, [a], optimize=True)
        self.checkScript(multiple_returns, [a], optimize=True)

    def test_error(self):
        @torch.jit.script
        def foo(a):
            return a.t()
        s = Variable(torch.rand(10))
        # XXX: this should stay quiet in stay propagation and only fail in the interpreter
        with self.assertRaisesRegex(RuntimeError, "failed in interpreter"):
            foo(s)

        @torch.jit.script
        def bar(c, b):
            return c / b

        with self.assertRaisesRegex(RuntimeError, "failed in interpreter"):
            bar(Variable(torch.rand(10), requires_grad=True), Variable(torch.rand(9), requires_grad=True))

    def test_binop_unsupported_error(self):
        with self.assertRaisesRegex(NotSupportedError, "unsupported binary operator:"):
            @torch.jit.script
            def binop(x, y):
                # Replace this with another unsupported op when/if it gets supported
                return x ** y

    def test_python_call(self):
        def pyfunc(a):
            return a * 3.0

        cu = torch.jit.CompilationUnit('''
        def other_func(a):
            return a + a

        def test_call_python(a):
            b = pyfunc(a)
            b = other_func(b)
            i = 0
            step = 1
            while i < 10:
                b = pyfunc(b)
                if b > 3.0:
                    b = pyfunc(b)
                i = 11
            return b
        ''')
        inputs = self._make_scalar_vars([1], torch.float)
        outputs = self._make_scalar_vars([54], torch.float)

        self.assertEqual(cu.test_call_python(*inputs), outputs[0])

    def test_python_call_failure(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value pyfunc2"):
            def pyfunc(a):
                return a * 3.0

            cu = torch.jit.CompilationUnit('''
            def other_func(a):
                return a + a

            def test_call_python(a):
                b = pyfunc(a)
                b = other_func(b)
                i = 0
                step = 1
                while i < 10:
                    b = pyfunc2(b)
                    if b > 3.0:
                        b = pyfunc(b)
                    i = 11
                return b
            ''')
            inputs = self._make_scalar_vars([1], torch.float)
            outputs = self._make_scalar_vars([54], torch.float)

            self.assertEqual(cu.test_call_python(*inputs), outputs)

    def test_python_call_annotation(self):
        def pyfunc(a):
            return a * 3.0

        @torch.jit.script
        def foo(a):
            return pyfunc(a) + pyfunc(a)

        inputs = self._make_scalar_vars([1], torch.float)
        outputs = self._make_scalar_vars([6], torch.float)
        self.assertEqual(foo(*inputs), outputs[0])

    def test_python_call_annoytation_failure(self):
        with self.assertRaisesRegex(RuntimeError, "undefined value pyfunc2"):
            def pyfunc(a):
                return a * 3.0

            @torch.jit.script
            def foo(a):
                return pyfunc2(a) + pyfunc(a)

            inputs = self._make_scalar_vars([1], torch.float)
            outputs = self._make_scalar_vars([6], torch.float)

            self.assertEqual(foo(*inputs), outputs[0])

    def test_desugar_module(self):
        import torch.nn.functional as F

        def fn(x, slope):
            a = torch.abs(x)
            b = torch.nn.functional.prelu(x, slope)
            c = F.prelu(x, slope)
            return a, b, c

        x = torch.arange(-3., 4)
        slope = torch.tensor([0.5])
        self.checkScript(fn, [x, slope], optimize=True)

    def test_script_module(self):
        class M1(torch.jit.ScriptModule):
            def __init__(self):
                super(M1, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class PModule(nn.Module):
            def __init__(self):
                super(PModule, self).__init__()
                self.a = nn.Parameter(torch.randn(2, 3))

            def forward(self, a):
                return self.a.mm(a)

        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(False)
                # test submodule
                self.sub = M1()
                self.sub2 = PModule()
                # test parameters
                self.weight = nn.Parameter(torch.randn(2, 3))
                self.bias = nn.Parameter(torch.randn(2))
                # test defining a method from a string
                self.define("""
                    def hi(self, a):
                        return self.weight.mm(a)
                """)
            # test script methods

            @torch.jit.script_method
            def doit(self, input):
                # test use of parameter
                return self.weight.mm(input)

            @torch.jit.script_method
            def doit2(self, input):
                return self.weight.mm(input)

            @torch.jit.script_method
            def forward(self, input):
                a = self.doit(input)
                b = self.doit2(input)
                c = self.hi(input)
                d = self.sub2(input)
                return a + b + self.bias + self.sub(a) + c + d
        m2 = M2()
        input = torch.randn(3, 2)
        a = m2.weight.mm(input)
        b = m2.weight.mm(input)
        c = m2.weight.mm(input)
        d = m2.sub2.a.mm(input)
        ref = a + b + m2.bias + m2.sub.weight + a + c + d
        self.assertEqual(ref, m2.forward(input))
        m2.weight = nn.Parameter(torch.zeros_like(m2.weight))
        m2.bias = nn.Parameter(torch.zeros_like(m2.bias))
        m2.sub.weight = nn.Parameter(torch.zeros_like(m2.sub.weight))
        m2.sub2.a.data.zero_()
        self.assertEqual(torch.zeros(2, 2), m2.forward(torch.randn(3, 2)))

    def test_script_module_call_noscript(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)
                self.value = 1

            def foo(self):
                return torch.ones(2, 2) + self.value

            @torch.jit.script_method
            def forward(self, input):
                return input + self.foo()

        m = M()
        input = torch.randn(2, 2)
        o = m(input)
        self.assertEqual(o, input + torch.ones(2, 2) + 1)
        # check that we can change python attributes
        # and that those changes are picked up in script methods
        m.value = 2
        o = m(input)
        self.assertEqual(o, input + torch.ones(2, 2) + 2)

    def test_script_module_nochange_submodule(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)
                self.sub = nn.Linear(5, 5)

            @torch.jit.script_method
            def forward(self, input):
                return self.sub(input)

        m = M()
        input = torch.randn(1, 5, 5)
        o = m(input)
        self.assertEqual(o, m.sub(input))
        with self.assertRaisesRegex(RuntimeError, "cannot re-assign"):
            m.sub = nn.Linear(5, 5)

    def test_script_inline_trace_multiple_args(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)

            def forward(self, input, input2):
                return input + input2

        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(False)
                self.m = torch.jit.trace(torch.zeros(4, 3), torch.zeros(4, 3))(M())

            @torch.jit.script_method
            def forward(self, inp):
                return self.m(inp, inp)

        m2 = M2()
        m2(torch.zeros(4, 3))

    def test_script_module_const(self):
        class M(torch.jit.ScriptModule):

            __constants__ = ['b', 'i', 'c']

            def __init__(self):
                super(M, self).__init__(False)
                self.b = False
                self.i = 1
                self.c = 3.5

            @torch.jit.script_method
            def forward(self):
                return self.b, self.i, self.c

        m = M()
        o0, o1, o2 = m()
        self.assertEqual(o0, 0)
        self.assertEqual(o1, 1)
        self.assertEqual(o2, 3.5)

    def test_script_module_fail_const(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)
                self.b = False

            @torch.jit.script_method
            def forward(self):
                return self.b
        with self.assertRaisesRegex(RuntimeError, "is not usable in a script method"):
            M()

    def test_script_module_valid_consts(self):
        class Foo(torch.jit.ScriptModule):
            __constants__ = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

            def __init__(self):
                super(Foo, self).__init__(False)
                self.a = 1
                self.b = 1.2
                self.c = False
                self.d = [nn.Linear(3, 4)]
                self.e = lambda x: x
                self.f = [3, 4, 5]
                self.assertTrue(type(self.f) is tuple)
                self.g = [3, (3, 4), 5]
                with self.assertRaisesRegex(TypeError, "is not a valid constant"):
                    self.h = type(1)
                with self.assertRaisesRegex(TypeError, "is not a valid constant"):
                    self.i = (3, 4, {})

    def test_script_module_for(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ['b']

            def __init__(self):
                super(M, self).__init__(False)
                self.b = [1, 2, 3, 4]

            @torch.jit.script_method
            def forward(self):
                sum = 0
                for i in self.b:
                    sum += i
                return sum

        m = M()
        self.assertEqual(m(), 10)

    def test_script_module_for2(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                super(M, self).__init__(False)
                self.mods = nn.ModuleList([Sub() for i in range(10)])

            @torch.jit.script_method
            def forward(self, v):
                for m in self.mods:
                    v = m(v)
                return v

        i = torch.Tensor(2)
        m = M()
        o = m(i)
        v = i
        for sub in m.mods:
            v = sub(v)
        self.assertEqual(o, v)

    def test_script_module_const_submodule_fail(self):
        class Sub(torch.jit.ScriptModule):
            def __init__(self):
                super(Sub, self).__init__(False)
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                return self.weight + thing

        class M(torch.jit.ScriptModule):
            def __init__(self):
                super(M, self).__init__(False)
                self.mods = [Sub() for _ in range(10)]

            @torch.jit.script_method
            def forward(self):
                for _ in self.mods:
                    print(1)
                return 4

        with self.assertRaisesRegex(RuntimeError, "did you forget to add it __constants__"):
            M()

    def test_script_module_not_tuple(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                super(M, self).__init__(False)
                self.mods = 1

            @torch.jit.script_method
            def forward(self, v):
                for m in self.mods:
                    print(m)
                return v
        with self.assertRaisesRegex(RuntimeError, "cannot be used as a tuple"):
            M()

    def test_constant_as_attr(self):
        class M(torch.jit.ScriptModule):
            __constants__ = ['dim']

            def __init__(self):
                super(M, self).__init__(False)
                self.dim = 1

            @torch.jit.script_method
            def forward(self, v):
                return torch.cat([v, v, v], dim=self.dim)
        v = torch.zeros(1, 1)
        self.assertEqual(torch.cat([v, v, v], dim=1), M()(v))

    class StarTestSumStarred(torch.nn.Module):
        def __init__(self):
            super(TestScript.StarTestSumStarred, self).__init__()

        def forward(self, *inputs):
            output = inputs[0]
            for i in range(1, len(inputs)):
                output += inputs[i]
            return output

    class StarTestReturnThree(torch.nn.Module):
        def __init__(self):
            super(TestScript.StarTestReturnThree, self).__init__()

        def forward(self, rep):
            return rep, rep, rep

    def test_script_star_expr(self):

        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.m = torch.jit.trace(
                    torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3))(TestScript.StarTestSumStarred())
                self.g = torch.jit.trace(torch.ones(4, 3))(TestScript.StarTestReturnThree())

            @torch.jit.script_method
            def forward(self, rep):
                tup = self.g(rep)
                return self.m(*tup)

        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    def test_script_star_expr_string(self):
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.m = torch.jit.trace(
                    torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3))(TestScript.StarTestSumStarred())
                self.g = torch.jit.trace(torch.ones(4, 3))(TestScript.StarTestReturnThree())

                self.define('''
            def forward(self, rep):
                tup = self.g(rep)
                return self.m(*tup)
                ''')

        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    class StarTestSumAndReturnThree(torch.nn.Module):
        def __init__(self):
            super(TestScript.StarTestSumAndReturnThree, self).__init__()

        def forward(self, *inputs):
            output = inputs[0]
            for i in range(1, len(inputs)):
                output += inputs[i]
            return output, output, output

    def test_script_star_assign(self):
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.g = torch.jit.trace(torch.ones(4, 3))(TestScript.StarTestSumAndReturnThree())
                self.define('''
            def forward(self, rep):
                head, *tail = self.g(rep)
                return head
                ''')

        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    def test_script_module_star_assign2(self):
        class M2(torch.jit.ScriptModule):
            def __init__(self):
                super(M2, self).__init__(True)
                self.g = torch.jit.trace(
                    torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)
                )(
                    TestScript.StarTestSumAndReturnThree()
                )
                self.define('''
            def forward(self, rep):
                *head, tail = self.g(rep, rep, rep)
                return tail
                ''')

        m = M2()
        self.assertEqual(m(torch.ones(4, 3)), 3 * torch.ones(4, 3))

    def test_script_module_star_assign_fail_pythonop(self):

        with self.assertRaisesRegex(RuntimeError, "value cannot be used as a tuple"):
            class M2(torch.jit.ScriptModule):
                def __init__(self):
                    super(M2, self).__init__(True)

                    def myfunc():
                        return torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)

                    self.define('''
                def forward(self, rep):
                    a, *b = myfunc()
                    return a
                    ''')

            m = M2()
            m(torch.zeros(4, 3))

    def test_script_module_star_assign_fail_builtin(self):
        with self.assertRaisesRegex(RuntimeError, "value cannot be used as a tuple"):
            class M2(torch.jit.ScriptModule):
                def __init__(self):
                    super(M2, self).__init__(True)

                    self.define('''
                def forward(self, rep):
                    a, *b = torch.neg(rep)
                    return a
                    ''')

            m = M2()
            m(torch.zeros(4, 3))

    def test_pack_padded_pad_packed_trace(self):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        T, B, C = 3, 5, 7

        class PadPackedWrapper(torch.nn.Module):
            def __init__(self):
                super(PadPackedWrapper, self).__init__()

            def forward(self, x, seq_lens):
                x = pack_padded_sequence(x, seq_lens)
                x, _ = pad_packed_sequence(x)
                return x

        x = np.ones((T, B, C))
        seq_lens = np.array([3, 3, 2, 2, 1], dtype=np.int32)
        # set padding value so we can test equivalence
        for b in range(B):
            if seq_lens[b] < T:
                x[seq_lens[b]:, b, :] = 0
        seq_lens = torch.from_numpy(seq_lens)
        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=True)

        m = PadPackedWrapper()
        m_traced = torch.jit.trace(x, seq_lens)(m)

        y = m(x, seq_lens)
        loss = torch.sum(y)
        loss.backward()
        grad = x.grad.clone()
        x.grad.zero_()

        y_traced = m_traced(x, seq_lens)
        loss_traced = torch.sum(y_traced)
        loss_traced.backward()
        grad_traced = x.grad.clone()

        self.assertEqual(y_traced, x)
        self.assertEqual(y_traced, y)
        self.assertEqual(grad, grad_traced)

        f = io.BytesIO()
        torch.onnx._export(m, (x, seq_lens), f, verbose=False)

    def test_script_outputs(self):
        with self.assertRaisesRegex(RuntimeError, "value cannot be used as a tuple"):
            @torch.jit.script
            def foo(a):
                c, d = a + a
                return c + d

        @torch.jit.script
        def return3():
            return 1, 2, 3

        with self.assertRaisesRegex(RuntimeError, "too many values to unpack"):
            @torch.jit.script
            def bind2():
                a, b = return3()
                print(a)
                print(b)

    def test_script_chunk(self):
        @torch.jit.script
        def foo(a):
            b, c = torch.chunk(a, dim=0, chunks=2)
            return b
        v = torch.rand(10, 3)
        self.assertEqual(torch.chunk(v, dim=0, chunks=2)[0], foo(v))

        with self.assertRaisesRegex(RuntimeError, "too many values to unpack"):
            @torch.jit.script
            def foo(a):
                b, c = torch.chunk(a, dim=0, chunks=3)
                return b

    def test_rnn_trace_override(self):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        num_layers = 3
        T, B, C = 11, 5, 7

        class RNNTraceWrapper(torch.nn.Module):
            def __init__(self, cell_type):
                super(RNNTraceWrapper, self).__init__()
                if cell_type == 'RNN':
                    self.rnn = torch.nn.RNN(input_size=C, hidden_size=C, num_layers=num_layers)
                elif cell_type == 'LSTM':
                    self.rnn = torch.nn.LSTM(input_size=C, hidden_size=C, num_layers=num_layers)
                elif cell_type == 'GRU':
                    self.rnn = torch.nn.GRU(input_size=C, hidden_size=C, num_layers=num_layers)

            def forward(self, x, seq_lens):
                x = pack_padded_sequence(x, seq_lens)
                x, _ = self.rnn(x)
                x, _ = pad_packed_sequence(x)
                return x

        for cell_type in ['RNN', 'LSTM', 'GRU']:
            x = torch.ones(T, B, C, requires_grad=True)
            seq_lens = torch.from_numpy(np.array([11, 3, 2, 2, 1], dtype=np.int32))

            m = RNNTraceWrapper(cell_type)
            m_traced = torch.jit.trace(x, seq_lens)(m)

            y = m(x, seq_lens)
            loss = torch.sum(y)
            loss.backward()
            grad = x.grad.clone()
            x.grad.zero_()

            y_traced = m_traced(x, seq_lens)
            loss_traced = torch.sum(y_traced)
            loss_traced.backward()
            grad_traced = x.grad.clone()

            self.assertEqual(y_traced, y)
            self.assertEqual(grad, grad_traced)

            f = io.BytesIO()
            torch.onnx._export(m, (x, seq_lens), f, verbose=False)

    def test_tuples(self):
        @torch.jit.script
        def foo(i):
            a = torch.chunk(i, dim=0, chunks=2)
            c = a
            # some nonsense with if-statements and loops to check
            # that tuple lowering doesn't fail
            if True:
                c = torch.chunk(i, dim=0, chunks=2)
            t0, t1 = c
            while False:
                t0, t1 = c
                c = torch.chunk(i, dim=0, chunks=2)
            return t0

        v = torch.rand(10, 3)
        self.assertEqual(torch.chunk(v, dim=0, chunks=2)[0], foo(v))

        with self.assertRaisesRegex(RuntimeError, r"variable 'a' previously has type \(Tensor, Tensor\)"):
            @torch.jit.script
            def mixtypes():
                a = torch.chunk(1, dim=0, chunks=2)
                if True:
                    a = 4

    def test_script_define_order(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                pass

            @torch.jit.script_method
            def call_foo(self, input):
                return self.foo(input)

            @torch.jit.script_method
            def foo(self, input):
                return input + 1
        m = M()
        self.assertEqual(2, m.call_foo(torch.ones((), dtype=torch.int64)))

    def test_script_define_order_recursive_fail(self):
        class M(torch.jit.ScriptModule):
            def __init__(self):
                pass

            @torch.jit.script_method
            def call_foo(self, input):
                return self.foo(input)

            @torch.jit.script_method
            def foo(self, input):
                self.call_foo(input)

        with self.assertRaisesRegex(RuntimeError, 'called recursively involving'):
            M()

    @unittest.skipIf(IS_WINDOWS, "NYI: fuser support for Windows")
    def test_trace_of_script(self):
        @torch.jit.script
        def foo(a, c):
            b = 0.0
            if a == 0.0:
                b = 1.0
            return b + c

        a = torch.ones(1, dtype=torch.float)

        @torch.jit.trace(torch.zeros(1, dtype=torch.float))
        def use(b):
            return foo(b - 1.0, a) + 1.0

        # test we propagated shapes through the function
        self.assertTrue("Dynamic" not in str(use.graph))

        self.assertEqual(3, use(torch.ones(1, dtype=torch.float)))
        self.assertEqual(2, use(torch.zeros(1, dtype=torch.float)))

    def test_if_define(self):
        @torch.jit.script
        def foo(a):
            if a == 0:
                b = 1
            else:
                b = 0
            return b + 1

        @torch.jit.script
        def foo2(a):
            b = 0
            if a == 0:
                b = 1

            return b + 1

        @torch.jit.script
        def foo3(a):
            b = 1
            if a == 0:
                c = 4
            else:
                b = 0

            return b + 1

        a = torch.ones(1, dtype=torch.long)
        b = torch.zeros(1, dtype=torch.long)
        self.assertEqual(1, foo(a))
        self.assertEqual(2, foo(b))
        self.assertEqual(1, foo2(a))
        self.assertEqual(2, foo2(b))
        self.assertEqual(1, foo3(a))
        self.assertEqual(2, foo3(b))

    def test_onnx_export_script_module(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                y = x - x
                return x + x

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx._export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_python_fail(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()

            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = ModuleToInline()

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        f = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "Couldn't export Python operator"):
            torch.onnx._export(mte, (torch.zeros(1, 2, 3),), f, verbose=False,
                               example_outputs=outputs)

    def test_onnx_export_script_inline_trace(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()

            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = torch.jit.trace(torch.zeros(1, 2, 3))(ModuleToInline())

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx._export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_inline_script(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = ModuleToInline()

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx._export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_module_loop(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                for _ in range(100):
                    x = x + x
                return x

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3))
        self.assertExpected(torch.onnx._export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_module_if(self):
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()

            @torch.jit.script_method
            def forward(self, x):
                if torch.sum(x) > 0:
                    x = torch.neg(x)
                return x

        mte = ModuleToExport()
        outputs = mte(torch.zeros(1, 2, 3, dtype=torch.long))
        self.assertExpected(torch.onnx._export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3),), None, verbose=False,
            example_outputs=outputs))

    def test_onnx_export_script_inline_params(self):
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToInline, self).__init__()
                self.m = torch.nn.Parameter(torch.ones(3, 3))
                self.unused = torch.nn.Parameter(torch.ones(1, 2, 3))

            @torch.jit.script_method
            def forward(self, x):
                return torch.mm(x, self.m)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super(ModuleToExport, self).__init__()
                self.mod = ModuleToInline()
                self.param = torch.nn.Parameter(torch.ones(3, 4))

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return torch.mm(y, self.param)

        mte = ModuleToExport()
        result = mte(torch.zeros(2, 3))
        reference = torch.mm(torch.mm(torch.zeros(2, 3), torch.ones(3, 3)), torch.ones(3, 4))
        self.assertEqual(result, reference)
        self.assertExpected(torch.onnx._export_to_pretty_string(
            mte, (torch.ones(2, 3),), None, verbose=False,
            example_outputs=result, propagate=True))

    def test_trace_with_size(self):
        @torch.jit.trace(torch.zeros(1, 1))
        def foo(x):
            return x + 1

        @torch.jit.script
        def bar(x):
            y = foo(x)
            if True:
                y = 7
            return y + 1

        self.assertEqual(8, bar(torch.ones(1, 1)))

    def test_index_select_shape_prop(self):

        @torch.jit.script
        def foo(x, y):
            return torch.index_select(x, y, dim=1)

        a = torch.zeros(2, 2)
        b = torch.zeros(4, dtype=torch.long)
        foo.graph.propagate_shapes((a, b), False)
        self.assertExpected(str(torch._C._jit_pass_canonicalize(foo.graph)))

    def test_onnx_export_speculate(self):

        class Foo(torch.jit.ScriptModule):
            def __init__(self, m):
                super(Foo, self).__init__()
                self.m = m

            @torch.jit.script_method
            def forward(self, x):
                x += x
                if True:
                    if True:
                        y = self.m(x)
                    else:
                        y = self.m(x)
                else:
                    y = self.m(x)
                return y

        linear = torch.jit.trace(torch.zeros(1, 10, dtype=torch.float))(nn.Linear(10, 20).float())

        @torch.jit.script
        def transpose(x):
            return x.t()

        f1 = Foo(transpose)
        outputs_f1 = f1(torch.ones(1, 10, dtype=torch.float))
        f2 = Foo(linear)
        outputs_f2 = f2(torch.ones(1, 10, dtype=torch.float))

        onnx_ish = torch.onnx._export_to_pretty_string(
            f1,
            (torch.ones(1, 10, dtype=torch.float), ),
            None, verbose=False, example_outputs=outputs_f1)
        self.assertExpected(onnx_ish, subname='f1')
        onnx_ish = torch.onnx._export_to_pretty_string(
            f2,
            (torch.ones(1, 10, dtype=torch.float), ),
            None, verbose=False, example_outputs=outputs_f2)
        self.assertExpected(onnx_ish, subname='f2')

    def test_onnx_export_shape_reshape(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                import torch.onnx.operators
                x = x.repeat(5, 1, 1)
                shape = torch.onnx.operators.shape_as_tensor(x)
                reshaped = torch.onnx.operators.reshape_from_tensor_shape(x, shape)
                return reshaped

        foo = torch.jit.trace(torch.zeros(1, 2, 3))(Foo())
        outputs = foo(torch.zeros(1, 2, 3))
        f = io.BytesIO()
        s = torch.onnx._export_to_pretty_string(foo, (torch.zeros(1, 2, 3)), f,
                                                example_outputs=outputs)
        self.assertExpected(s)

    def test_shape_analysis_loop(self):
        def foo(a, b, x):
            c = a
            # on the first iteration of the loop it appears that
            # c should have a expand to the size of b
            # but on the second+ iterations, there is no broadcast and the
            # sizes are different.
            # previously this would cause the compiler to (1) enter an infinite
            # loop trying to compute the shape, and (2) insert invalid
            # broadcasts.
            # this test ensure we don't regress on these issues
            for _ in range(2):
                a = c + b
                c = x
                b = x
            return a

        self.checkScript(foo, (torch.zeros(1), torch.zeros(4), torch.zeros(5)), False)

    def test_intlist_args(self):
        def func_1(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, 1)

        def func_2(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, output_size=1)

        def func_3(x):
            return torch.nn.functional.adaptive_avg_pool1d(x, output_size=[1])

        x = torch.randn(8, 8, 8)
        self.checkScript(func_1, [x], optimize=True)
        self.checkScript(func_2, [x], optimize=True)
        self.checkScript(func_3, [x], optimize=True)


# Smoke tests for export methods
class TestPytorchExportModes(unittest.TestCase):
    class MyModel(nn.Module):
        def __init__(self):
            super(TestPytorchExportModes.MyModel, self).__init__()

        def forward(self, x):
            return x.t()

    def test_protobuf(self):
        torch_model = TestPytorchExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.PROTOBUF_FILE)

    def test_zipfile(self):
        torch_model = TestPytorchExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.ZIP_ARCHIVE)

    def test_compressed_zipfile(self):
        torch_model = TestPytorchExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(torch_model, (fake_input), f, verbose=False,
                           export_type=torch.onnx.ExportTypes.COMPRESSED_ZIP_ARCHIVE)

    def test_directory(self):
        torch_model = TestPytorchExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        d = tempfile.mkdtemp()
        torch.onnx._export(torch_model, (fake_input), d, verbose=False,
                           export_type=torch.onnx.ExportTypes.DIRECTORY)
        shutil.rmtree(d)


if __name__ == '__main__':
    run_tests()
