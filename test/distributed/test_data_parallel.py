# Owner(s): ["oncall: distributed"]

import contextlib
import io
from copy import deepcopy
from collections import OrderedDict
from itertools import product
import functools

import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn.parallel as dp
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, onlyCUDA, skipMeta
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_utils import skip_but_pass_in_sandcastle_if
import torch.nn.functional as F

torch.set_default_dtype(torch.double)

NO_NCCL = not hasattr(torch.distributed, "ProcessGroupNCCL")

# batched grad doesn't support data parallel
gradcheck = functools.partial(gradcheck, check_batched_grad=False)
_assertGradAndGradgradChecks = functools.partial(_assertGradAndGradgradChecks, check_batched_grad=False)

class TestDataParallel(TestCase):

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_buffers_requiring_grad(self):
        class TestModule(nn.Module):
            def __init__(self, t):
                super().__init__()
                self.register_buffer('t_rg', t)
                self.register_buffer('t_not_rg', t.clone().detach())

            def forward(self, x):
                return x * self.t_rg + self.t_not_rg

        m = TestModule(torch.randn(100, device='cuda', requires_grad=True))
        self.assertTrue(m.t_rg.requires_grad)

        dpm = nn.DataParallel(m, [0, 1])
        inp = torch.randn(2, 100, device='cuda')

        def fn(t):
            return dpm(inp)

        gradcheck(fn, (m.t_rg,))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_rnn(self):

        class TestModule(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(300, 1024, 1, batch_first=True, bidirectional=True)

            def forward(self, x):
                self.rnn.flatten_parameters()
                return self.rnn(x)

        def step(model):
            opt = torch.optim.SGD(model.parameters(), lr=10)
            input = torch.ones(4, 4, 300).to(0)
            output = model(input)
            loss = F.mse_loss(output[0], torch.zeros_like(output[0]))
            loss.backward()
            opt.step()

        with torch.no_grad():
            model = TestModule().to(0)
            model_dp = torch.nn.DataParallel(deepcopy(model))

            # make sure DP does not crash when grad is disabled.
            # See #21108
            model_dp(torch.rand(2, 4, 300).to(0))

        step(model)
        step(model_dp)

        for p1, p2 in zip(model.parameters(), model_dp.parameters()):
            self.assertTrue(p1.allclose(p2))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_lazy_linear(self):

        with self.assertRaisesRegex(RuntimeError, 'Modules with uninitialized parameters'):
            model_dp = torch.nn.DataParallel(torch.nn.LazyLinear(10).to(0))
            model_dp(torch.rand(10, 10).to(0))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parallel_apply(self):
        l1 = nn.Linear(10, 5).to("cuda:0", torch.float)
        l2 = nn.Linear(10, 5).to("cuda:1", torch.float)
        i1 = torch.randn(2, 10, device="cuda:0", dtype=torch.float)
        i2 = torch.randn(2, 10, device="cuda:1", dtype=torch.float)
        expected1 = l1(i1)
        expected2 = l2(i2)
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)

        # each input can be either a collection of positional arguments
        #                       or an object representing the single argument
        for inputs in [((i1,), (i2,)), (i1, i2)]:
            outputs = dp.parallel_apply(modules, inputs, None)
            for out, expected in zip(outputs, expected_outputs):
                self.assertEqual(out, expected)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parallel_apply_autocast(self):
        l1 = nn.Linear(10, 5).to("cuda:0", torch.float)
        l2 = nn.Linear(10, 5).to("cuda:1", torch.float)
        i1 = torch.randn(2, 10, device="cuda:0", dtype=torch.float)
        i2 = torch.randn(2, 10, device="cuda:1", dtype=torch.float)
        with autocast():
            expected1 = l1(i1)
            expected2 = l2(i2)
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)

        # each input can be either a collection of positional arguments
        #                       or an object representing the single argument
        for inputs in [((i1,), (i2,)), (i1, i2)]:
            with autocast():
                outputs = dp.parallel_apply(modules, inputs, None)
            for out, expected in zip(outputs, expected_outputs):
                self.assertEqual(out, expected)

    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "CUDA unavailable")
    def test_parallel_apply_passes_exception(self):
        # we define and instantiate a module that will throw a KeyError
        class TestModule(nn.Module):

            def forward(self, *args):
                return {}['wonderful']

        l1 = TestModule().to("cuda", torch.float)
        # and check that parallel_apply passes on the exception
        # (we can use a single device twice for this test)
        with self.assertRaisesRegex(KeyError,
                                    'Caught KeyError in replica \\d '
                                    'on device 0.\nOriginal Traceback'
                                    '[\\s\\S]+wonderful'):
            dp.parallel_apply(modules=(l1, l1), inputs=(None, None))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_multiple_input(self):
        class TestModule(nn.Module):

            def forward(self, var1, var2, float1, var3=None):
                if var3 is None:
                    return float1 * (var1 * var2)
                else:
                    return float1 * (var1 * var2 + var3)

        m = TestModule()
        var1 = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
        var2 = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
        var3 = torch.randn(5, 5, dtype=torch.float, requires_grad=False)

        float1 = torch.randn(1).item()

        expected = m(var1, var2, float1)
        loss = expected.sum()
        loss.backward()
        gvar1_exp = var1.grad.clone()
        gvar2_exp = var2.grad.clone()

        def local_test(out):
            with torch.no_grad():
                var1.grad.fill_(0.0)
                var2.grad.fill_(0.0)
            loss = out.sum()
            loss.backward()
            self.assertEqual(out, expected)
            self.assertEqual(gvar1_exp, var1.grad)
            self.assertEqual(gvar2_exp, var2.grad)

        out = dp.data_parallel(m, (var1, var2, float1), (0, 1))
        local_test(out)

        out = dp.data_parallel(m, (var1, var2, float1), (1, 0))
        local_test(out)

        out = dp.data_parallel(m, (var1, var2, float1), (0,))
        local_test(out)

        with torch.no_grad():
            var1.grad.fill_(0.0)
            var2.grad.fill_(0.0)
        expected = m(var1, var2, float1, var3=var3)
        loss = expected.sum()
        loss.backward()
        gvar1_exp = var1.grad.clone()
        gvar2_exp = var2.grad.clone()

        dpm = nn.DataParallel(TestModule())
        out = dpm(var1, var2, float1, var3=var3)
        local_test(out)

        dpm = nn.DataParallel(TestModule(), device_ids=[0])
        out = dpm(var1, var2, float1, var3=var3)
        local_test(out)

        kwarg_wrap = {'var3': var3}
        out = dp.data_parallel(
            m, (var1, var2, float1), (0, 1), module_kwargs=kwarg_wrap)
        local_test(out)

        out = dp.data_parallel(
            m, (var1, var2, float1), (0,), module_kwargs=kwarg_wrap)
        local_test(out)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_small_back(self):
        l = nn.Linear(10, 5).float().cuda()
        i = torch.randn(20, 10, dtype=torch.float, device="cuda")
        out = dp.data_parallel(l, i, (0, 1))
        self.assertEqual(out, l(i))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_model_device(self):
        r"""Test device[0] check at forward time.
        """
        l = nn.Linear(2, 2)
        inp = torch.randn(2, 2)
        inp_cuda0 = inp.cuda(0)
        inp_cuda1 = inp.cuda(1)

        error_msg = "module must have its parameters and buffers on device {}"

        @contextlib.contextmanager
        def dummy_ctx_manager():
            yield

        def test(inner_m, dp_device, inp, device_ids, should_fail):
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))

            if isinstance(device_ids[0], torch.device):
                expect_device = device_ids[0]
            else:
                expect_device = torch.device("cuda:{}".format(device_ids[0]))

            if should_fail:
                def assert_correct():
                    return self.assertRaisesRegex(RuntimeError, error_msg.format(expect_device))
            else:
                assert_correct = dummy_ctx_manager

            # test DataParallel module
            dpm = nn.DataParallel(inner_m, device_ids)
            if dp_device is not None:
                dpm = dpm.to(dp_device)

            with assert_correct():
                dpm(inp)

            # test functional
            with assert_correct():
                nn.parallel.data_parallel(inner_m.to(dp_device), inp, device_ids)

        test(l.to('cpu'), None, inp, None, should_fail=True)
        test(l.cuda(1), None, inp_cuda0, None, should_fail=True)
        test(l.cuda(), None, inp_cuda0, [1, 0], should_fail=True)

        test(l.cuda(), None, inp_cuda0, None, should_fail=False)
        test(l.cpu(), 'cuda', inp_cuda0, None, should_fail=False)
        test(l.cuda(1), None, inp_cuda1, [1, 0], should_fail=False)
        test(l.cpu(), 'cuda:1', inp_cuda1, [1, 0], should_fail=False)

        s = nn.Sequential(l.cpu())
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(deepcopy(l).cpu(), l.cuda())
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(l.cuda(), deepcopy(l).cuda(1))
        test(s, None, inp, None, should_fail=True)
        test(s, None, inp, [0, 1], should_fail=True)
        test(s, None, inp, [1, 0], should_fail=True)

        s = nn.Sequential(l.cuda(), deepcopy(l).cuda())
        test(s, None, inp, None, should_fail=False)
        test(s, None, inp, [0, 1], should_fail=False)
        test(s, None, inp, [1, 0], should_fail=True)
        test(s.cpu(), None, inp, [1, 0], should_fail=True)
        test(s.cuda(1), None, inp, [1, 0], should_fail=False)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_model_no_refcycles(self):
        # Python 2.7 will create reference cycles with the following
        # Module on multiple GPUs, but Python 3 shouldn't unless
        # there are refcycles on the PyTorch side (or the defined module)
        import gc

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        gc.collect()
        model = nn.DataParallel(Model().cuda())
        data = torch.randn(1, device="cuda")
        model(data)

        refcycles = gc.collect()
        self.assertEqual(refcycles, 0)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_no_grad(self):
        test = self

        class Layer(nn.Module):
            def forward(self, x):
                test.assertFalse(torch.is_grad_enabled())
                return x

        l = Layer()
        i = torch.randn(20, 10, dtype=torch.float, device="cuda")
        with torch.no_grad():
            dp.data_parallel(l, i, (0, 1))
        self.assertRaises(AssertionError, lambda: dp.data_parallel(l, i, (0, 1)))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel(self):
        l = nn.Linear(10, 5).float().cuda()
        i = torch.randn(20, 10, dtype=torch.float, device="cuda:1")
        l.cuda(1)
        expected_out = l(i)
        loss = expected_out.sum()
        loss.backward()
        expected_grads = []
        for param in l.parameters():
            expected_grads.append(param.grad.clone())
        dev_ids_list = [(0, 1), (1, 0)]
        for dev_id in dev_ids_list:
            with torch.cuda.device(dev_id[0]):
                l.cuda()
                l.zero_grad()
                out = dp.data_parallel(l, i, dev_id)
                loss = out.sum()
                loss.backward()
                self.assertEqual(out.get_device(), dev_id[0])
                self.assertEqual(out, expected_out)
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad, expected)

        # Check for None device_ids
        l = l.cuda()
        out = dp.data_parallel(l, i)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_sparse(self):
        l = nn.Embedding(10, 5, sparse=True).to("cuda:1")
        i = torch.randint(10, (20, 5), device="cuda:1", dtype=torch.long)
        expected_out = l(i)
        loss = expected_out.sum()
        loss.backward()
        expected_grads = []
        for param in l.parameters():
            expected_grads.append(param.grad.clone())
        dev_ids_list = [(0, 1), (1, 0)]
        for dev_id in dev_ids_list:
            with torch.cuda.device(dev_id[0]):
                l.cuda()
                l.zero_grad()
                out = dp.data_parallel(l, i, dev_id)
                loss = out.sum()
                loss.backward()
                self.assertEqual(out.get_device(), dev_id[0])
                self.assertEqual(out, expected_out)
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad.coalesce(), expected.coalesce())

        # Check for None device_ids
        l = l.cuda()
        out = dp.data_parallel(l, i)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_output(self):
        def fn(input):
            return [
                input, (input.sin(), input.cos(), [input.add(1)]), input,
                OrderedDict(a=input, b=[input.sin()])
            ]

        class Net(nn.Module):
            def forward(self, input):
                return fn(input)

        i = torch.randn(2, 2).float().cuda(1)
        gpus = range(torch.cuda.device_count())
        output = dp.data_parallel(Net(), i, gpus)
        self.assertEqual(output, fn(i))
        self.assertIsInstance(output[0], torch.Tensor)
        self.assertIsInstance(output[1], tuple)
        self.assertIsInstance(output[1][0], torch.Tensor)
        self.assertIsInstance(output[1][1], torch.Tensor)
        self.assertIsInstance(output[1][2], list)
        self.assertIsInstance(output[1][2][0], torch.Tensor)
        self.assertIsInstance(output[2], torch.Tensor)
        self.assertIsInstance(output[3], dict)
        self.assertEqual(len(output[3]), 2)
        self.assertIn('a', output[3])
        self.assertIn('b', output[3])
        self.assertIsInstance(output[3]['a'], torch.Tensor)
        self.assertIsInstance(output[3]['b'], list)
        self.assertIsInstance(output[3]['b'][0], torch.Tensor)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_nested_input(self):
        def fn(input):
            return input[1][0]

        class Net(nn.Module):
            def forward(self, *input):
                return fn(input)

        i = torch.randn(20, 3, dtype=torch.float, device="cuda:1")
        input = (i.cos(), (i.sin(), i), i.sin())
        gpus = range(torch.cuda.device_count())
        output = dp.data_parallel(Net(), input, gpus)
        self.assertEqual(output, fn(input))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_module_zero_inputs(self):
        class TestModule(nn.Module):
            def forward(self):
                t = torch.eye(2, 3, device='cuda:0')
                return t + (1 - t)

        def test_helper(output, expected):
            self.assertEqual(output.get_device(), 0)
            self.assertEqual(output, expected)

        expected = torch.ones(2, 3, device='cuda:0')
        model = TestModule()

        test_helper(nn.DataParallel(model, [0])(), expected)
        test_helper(nn.DataParallel(model, [0, 1])(), expected)
        test_helper(dp.data_parallel(model, None, [0]), expected)
        test_helper(dp.data_parallel(model, (), [0, 1]), expected)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_device_args(self):
        cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1')

        # test output_device
        l = nn.Linear(10, 5).to(cuda0, torch.float)
        i = torch.randn(20, 10, dtype=torch.float, device=cuda0, requires_grad=True)
        out = dp.data_parallel(l, i, device_ids=(0, 1), output_device=cuda0)
        self.assertEqual(out, l(i))

        # test device_ids
        l = nn.Linear(10, 5).to(cuda0, torch.float)
        i = torch.randn(20, 10, dtype=torch.float, device=cuda0, requires_grad=True)
        out = dp.data_parallel(l, i, device_ids=(cuda0, cuda1), output_device=cuda0)
        self.assertEqual(out, l(i))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_function_deletion(self):
        # this test case is originated from #16532
        def gradient_penalty(net, x):
            output = net(x)
            loss = torch.autograd.grad(
                outputs=output, inputs=x,
                grad_outputs=x.new_ones(output.size()),
                create_graph=True, retain_graph=True)[0].mean()
            return loss

        net = nn.Linear(4, 1).cuda()
        dpn = nn.DataParallel(net, [0, 1])
        x = torch.ones(2, 4, requires_grad=True).cuda()

        dpn.zero_grad()
        loss = gradient_penalty(dpn, x)
        loss.backward()
        grads = [p.grad for p in net.parameters()]
        self.assertEqual(2, len(grads))
        self.assertEqual(
            torch.tensor([[0.25, 0.25, 0.25, 0.25]], device='cuda:0'),
            grads[0])
        self.assertEqual(torch.tensor([0.0], device='cuda:0'), grads[1])

    def _test_scatter(self, tensor):
        x = tensor.detach().requires_grad_()
        result = dp.scatter(x, (0, 1))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x[:2])
        self.assertEqual(result[0].get_device(), 0)
        self.assertEqual(result[1], x[2:])
        self.assertEqual(result[1].get_device(), 1)
        grad = result[0].detach().clone().fill_(2)
        result[0].backward(grad)
        self.assertEqual(x.grad[:2], grad)
        self.assertEqual(x.grad[2:], grad.clone().zero_())
        _assertGradAndGradgradChecks(self, lambda y: dp.scatter(y, (0, 1)), (x,))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_scatter_cpu(self):
        self._test_scatter(torch.randn((4, 4)))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_scatter_gpu(self):
        self._test_scatter(torch.randn((4, 4)).cuda())

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "At least 2 CUDA GPUS needed")
    @skip_but_pass_in_sandcastle_if(NO_NCCL, "NCCL needed")
    def test_data_parallel_complex(self):
        # We expect complex parameters to be broadcast by view_as_real, e.g. move from C to R^2
        class Cplx(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cplx = torch.nn.Parameter(torch.zeros(1, 10, dtype=torch.cfloat).cuda())

            def forward(self, x):
                return x + self.cplx

        cplx = torch.nn.DataParallel(Cplx().cuda())
        input = torch.rand(1, 10, dtype=torch.cfloat).cuda()
        result = cplx(input)
        # 2 is the extra real view dimension here
        self.assertEqual(result.size(), torch.Size([1, 10, 2]))
        self.assertEqual(result, torch.view_as_real(input))

    def _test_gather(self, output_device):
        inputs = (
            torch.randn(2, 4, device='cuda:0', requires_grad=True),
            torch.randn(2, 4, device='cuda:1', requires_grad=True),
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([4, 4]))
        self.assertEqual(result[:2], inputs[0])
        self.assertEqual(result[2:], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_cuda)
        grad = torch.randn((4, 4))
        if output_device != -1:
            grad = grad.cuda(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad, grad[:2])
        self.assertEqual(inputs[1].grad, grad[2:])
        _assertGradAndGradgradChecks(self, lambda x, y: dp.gather((x, y), output_device), inputs)

        # test scalar inputs, should stack into a vector in this case
        inputs = (
            torch.randn((), device='cuda:0', requires_grad=True),
            torch.randn((), device='cuda:1', requires_grad=True),
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([2]))
        self.assertEqual(result[0], inputs[0])
        self.assertEqual(result[1], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_cuda)
        grad = torch.randn(2)
        if output_device != -1:
            grad = grad.cuda(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad, grad[0])
        self.assertEqual(inputs[1].grad, grad[1])
        _assertGradAndGradgradChecks(self, lambda x, y: dp.gather((x, y), output_device), inputs)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_cpu(self):
        self._test_gather(-1)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_gpu(self):
        self._test_gather(0)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_different_len_dicts(self):
        inputs = (
            {'a': torch.randn(1, 2, requires_grad=True, device="cuda:0")},
            {
                'b': torch.randn(1, 2, requires_grad=True, device="cuda:1"),
                'a': torch.randn(1, 2, requires_grad=True, device="cuda:1"),
            }
        )
        with self.assertRaises(ValueError):
            _ = dp.gather(inputs, target_device=0)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_replicate(self):
        module = nn.Linear(10, 5).float().cuda()
        input = torch.randn(2, 10, dtype=torch.float, device="cuda")
        expected_output = module(input)
        for devices in [(0, 1), [0, 1]]:
            replicas = dp.replicate(module, devices)
            for i, replica in enumerate(replicas):
                for p in replica.parameters():
                    self.assertEqual(p.get_device(), i)
                replica_input = input.cuda(i)
                self.assertEqual(replica(replica_input), expected_output)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_replicate_buffers(self):
        net = nn.Module()
        net.bn = nn.BatchNorm2d(10)
        net.cuda()
        for devices in [(0, 1), [0, 1]]:
            replicas = dp.replicate(net, devices)
            for i, replica in enumerate(replicas):
                self.assertEqual(replica.bn.running_mean.get_device(), i, msg='buffer on wrong device')
                self.assertEqual(replica.bn.running_var.get_device(), i, msg='buffer on wrong device')
                self.assertEqual(replica.bn.num_batches_tracked.get_device(), i, msg='buffer on wrong device')

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_zero_grad(self):
        # zero_grad should warn about using gradients inside forward

        class Net(torch.nn.Module):
            def __init__(self, testcase):
                super().__init__()
                self._testcase = testcase

            def forward(self, x):
                with self._testcase.assertWarnsRegex(
                        UserWarning,
                        r"Calling \.zero_grad\(\) from a module created with nn\.DataParallel\(\) has no effect."):
                    self.zero_grad()
                return x

        module = Net(self).cuda()
        dpm = dp.DataParallel(module)
        dpm(torch.rand(4, 3, 6, 5))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_autocast(self):
        class Model(torch.nn.Linear):
            def __init__(self):
                super().__init__(8, 8)

            @torch.cuda.amp.autocast()
            def forward(self, input):
                return super().forward(input)

        model = dp.DataParallel(Model().cuda().to(dtype=torch.float32))
        input = torch.randn((8, 8), dtype=torch.float32, device="cuda")
        self.assertTrue(model(input).dtype is torch.float16)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_save_replica_module(self):
        # DataParallel replicas can be saved (gh-37182)
        module = torch.nn.Linear(8, 8).cuda()
        dpm = torch.nn.parallel.replicate(module, devices=[0, 1], detach=False)
        data = io.BytesIO()
        torch.save(dpm, data)
        dpm = torch.nn.parallel.replicate(module, devices=[0, 1], detach=True)
        torch.save(dpm, data)

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_strided_grad_layout(self):
        class ConvNet(nn.Module):
            def __init__(self, layouts, dtype_list):
                super().__init__()
                self.dtypes = dtype_list
                self.conv0 = torch.nn.Conv2d(8, 16, (2, 2)).to(memory_format=layouts[0], dtype=dtype_list[0])
                self.conv1 = torch.nn.Conv2d(16, 32, (2, 2)).to(memory_format=layouts[1], dtype=dtype_list[1])
                self.conv2 = torch.nn.Conv2d(32, 16, (2, 2)).to(memory_format=layouts[2], dtype=dtype_list[2])
                self.conv3 = torch.nn.Conv2d(16, 8, (2, 2)).to(memory_format=layouts[3], dtype=dtype_list[3])

            def forward(self, x):
                x = x.to(self.dtypes[0])
                x = self.conv0(x).to(self.dtypes[1])
                x = self.conv1(x).to(self.dtypes[2])
                x = self.conv2(x).to(self.dtypes[3])
                x = self.conv3(x)
                return x

        layer_formats = ([torch.contiguous_format] * 4,
                         [torch.channels_last] * 2 + [torch.contiguous_format] * 2,
                         [torch.channels_last] * 4,)
        layer_dtypes = ([torch.float] * 4,
                        [torch.float] * 2 + [torch.half] * 2,
                        [torch.half] * 4,)

        ndevs = torch.cuda.device_count()
        input = torch.randn(ndevs * 8, 8, 8, 8, device="cuda:0", dtype=torch.float)
        target = torch.randn(ndevs * 8, 8, 4, 4, device="cuda:0", dtype=torch.float)
        device_ids = list(range(ndevs))

        with torch.backends.cudnn.flags(enabled=True, deterministic=True, benchmark=False):
            for formats, dtype_list in product(layer_formats, layer_dtypes):
                model_msg = "formats = {} dtypes = {}".format(formats, dtypes)
                try:
                    m = ConvNet(formats, dtype_list).cuda(device="cuda:0")
                    m_dp = dp.DataParallel(deepcopy(m), device_ids=device_ids)
                    opt = torch.optim.SGD(m.parameters(), lr=0.1)
                    opt_dp = torch.optim.SGD(m_dp.parameters(), lr=0.1)
                    has_half = any(p.dtype is torch.half for p in m.parameters())
                    tol = 1.e-3 if has_half else 1.e-5
                except BaseException:
                    # Prints case-specific debugging info to narrow down failing case.
                    print("Caught exception during model creation for " + model_msg, flush=True)
                    raise
                # 2 iters:  First iter creates grads, second iter tries zeroed grads.
                for it in range(2):
                    iter_msg = "iter = {} ".format(it) + model_msg
                    named_msg = iter_msg
                    try:
                        F.mse_loss(m(input).float(), target).backward()
                        F.mse_loss(m_dp(input).float(), target).backward()
                        for i, ((layer_name, m_child), m_dp_child) in enumerate(zip(m.named_children(),
                                                                                    m_dp.module.children())):
                            named_msg = layer_name + ".weight " + iter_msg
                            self.assertTrue(m_child.weight.grad.is_contiguous(memory_format=formats[i]), named_msg)
                            self.assertTrue(m_dp_child.weight.grad.is_contiguous(memory_format=formats[i]), named_msg)
                            for j, ((param_name, p), p_dp) in enumerate(zip(m_child.named_parameters(),
                                                                            m_dp_child.parameters())):
                                named_msg = layer_name + "." + param_name + " " + iter_msg
                                self.assertEqual(p.grad, p_dp.grad, rtol=tol, atol=tol)
                        opt.step()
                        opt_dp.step()
                        opt.zero_grad()
                        opt_dp.zero_grad()
                    except BaseException:
                        # Makes sure we still get info if an error occurred somewhere other than the asserts.
                        print("Caught exception during iterations at " + named_msg, flush=True)
                        raise

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parameter_list_dict_replica(self):
        class MyMod(torch.nn.Module):
            def __init__(self, data, check_fn):
                super().__init__()
                self.data = data
                self.check_fn = check_fn

            def forward(self, inp):
                self.check_fn(self)
                return inp

        p1 = torch.nn.Parameter(torch.rand(10))
        p2 = torch.nn.Parameter(torch.rand(10))
        key0 = 0
        key1 = 1

        def check_fn(self_):
            self.assertEqual(p1, self_.data[key0])
            self.assertEqual(p2, self_.data[key1])
            self.assertTrue(self_.data[key0].requires_grad)
            self.assertTrue(self_.data[key1].requires_grad)
            self.assertIsNotNone(self_.data[key0].grad_fn)
            self.assertIsNotNone(self_.data[key1].grad_fn)

        module = MyMod(torch.nn.ParameterList([p1, p2]), check_fn).cuda()
        model = dp.DataParallel(module)
        input = torch.randn((8, 8), device="cuda")

        # Runs the check_fn
        model(input)

        key0 = "0"
        key1 = "1"
        module = MyMod(torch.nn.ParameterDict({"0": p1, "1": p2}), check_fn).cuda()
        model = dp.DataParallel(module)
        input = torch.randn((8, 8), device="cuda")

        # Runs the check_fn
        model(input)


class TestDataParallelDeviceType(TestCase):

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module(self, device, dtype):
        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        net = nn.DataParallel(l)
        out = net(i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only(self, device, dtype):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = l

            def forward(self, input):
                return self.l(input)

        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        n = nn.DataParallel(Net())
        out = n(input=i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only_empty_list(self, device, dtype):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': []})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only_empty_dict(self, device, dtype):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': {}})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)

    @onlyCUDA
    @skipMeta
    @dtypes(torch.float, torch.double, torch.half)
    def test_data_parallel_module_kwargs_only_empty_tuple(self, device, dtype):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).to(device, dtype)
        i = torch.randn(20, 10, device=device, dtype=dtype)
        expected_out = l(i)
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': ()})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out, expected_out, atol=dtype2prec_DONTUSE[dtype], rtol=0)


instantiate_device_type_tests(TestDataParallelDeviceType, globals())

if __name__ == '__main__':
    run_tests()
