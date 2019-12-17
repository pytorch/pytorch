import contextlib
import unittest
import torch
from torch import nn
import torch.nn.parallel as dp
from torch.testing._internal.common_cuda import TEST_MULTIGPU, TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase, skipIfRocm, repeat_test_for_types, ALL_TENSORTYPES, PY3
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks
from torch.testing._internal.common_utils import dtype2prec
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict

class TestDataParallel(TestCase):

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_buffers_requiring_grad(self):
        class TestModule(nn.Module):
            def __init__(self, t):
                super(TestModule, self).__init__()
                self.register_buffer('t_rg', t)
                self.register_buffer('t_not_rg', t.clone().detach())

            def forward(self, x):
                return x * self.t_rg + self.t_not_rg

        m = TestModule(torch.randn(100, device='cuda', requires_grad=True, dtype=torch.double))
        self.assertTrue(m.t_rg.requires_grad)

        dpm = nn.DataParallel(m, [0, 1])
        inp = torch.randn(2, 100, device='cuda', dtype=torch.double)

        def fn(t):
            return dpm(inp)

        torch.autograd.gradcheck(fn, (m.t_rg,))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_rnn(self):

        class TestModule(torch.nn.Module):

            def __init__(self):
                super(TestModule, self).__init__()
                self.rnn = torch.nn.LSTM(300, 1024, 1, batch_first=True, bidirectional=True)

            def forward(self, x):
                self.rnn.flatten_parameters()
                return self.rnn(x)

        def step(model):
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
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
            p1.allclose(p2)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_parallel_apply(self):
        l1 = nn.Linear(10, 5).to("cuda:0", torch.float)
        l2 = nn.Linear(10, 5).to("cuda:1", torch.float)
        i1 = torch.randn(2, 10, device="cuda:0", dtype=torch.float)
        i2 = torch.randn(2, 10, device="cuda:1", dtype=torch.float)
        expected1 = l1(i1).data
        expected2 = l2(i2).data
        modules = (l1, l2)
        expected_outputs = (expected1, expected2)

        # each input can be either a collection of positional arguments
        #                       or an object representing the single argument
        for inputs in [((i1,), (i2,)), (i1, i2)]:
            outputs = dp.parallel_apply(modules, inputs, None)
            for out, expected in zip(outputs, expected_outputs):
                self.assertEqual(out.data, expected)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
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

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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
            var1.grad.data.fill_(0.0)
            var2.grad.data.fill_(0.0)
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

        var1.grad.data.fill_(0.0)
        var2.grad.data.fill_(0.0)
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

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_data_parallel_small_back(self):
        l = nn.Linear(10, 5).float().cuda()
        i = torch.randn(20, 10, dtype=torch.float, device="cuda")
        out = dp.data_parallel(l, i, (0, 1))
        self.assertEqual(out, l(i))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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

    @unittest.skipIf(not TEST_MULTIGPU or not PY3, "multi-GPU not supported")
    @skipIfRocm
    def test_data_parallel_model_no_refcycles(self):
        # Python 2.7 will create reference cycles with the following
        # Module on multiple GPUs, but Python 3 shouldn't unless
        # there are refcycles on the PyTorch side (or the defined module)
        import gc

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, x):
                return self.linear(x)

        gc.collect()
        model = nn.DataParallel(Model().cuda())
        data = torch.randn(1, device="cuda")
        model(data)

        refcycles = gc.collect()
        self.assertEqual(refcycles, 0)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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
                self.assertEqual(out.data, expected_out.data)
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad.data, expected.data)

        # Check for None device_ids
        l = l.cuda()
        out = dp.data_parallel(l, i)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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
                self.assertEqual(out.data, expected_out.data)
                for expected, param in zip(expected_grads, l.parameters()):
                    self.assertEqual(param.grad.data, expected.data)

        # Check for None device_ids
        l = l.cuda()
        out = dp.data_parallel(l, i)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module(self, dtype=torch.float):
        l = nn.Linear(10, 5).to("cuda", dtype)
        i = torch.randn(20, 10, device="cuda", dtype=dtype)
        expected_out = l(i).data
        net = nn.DataParallel(l)
        out = net(i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out, dtype2prec[dtype])

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module_kwargs_only(self, dtype=torch.float):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l = l

            def forward(self, input):
                return self.l(input)

        l = nn.Linear(10, 5).to("cuda", dtype)
        i = torch.randn(20, 10, device="cuda", dtype=dtype)
        expected_out = l(i).data
        n = nn.DataParallel(Net())
        out = n(input=i)
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out, dtype2prec[dtype])

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module_kwargs_only_empty_list(self, dtype=torch.float):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).to("cuda", dtype)
        i = torch.randn(20, 10, device="cuda", dtype=dtype)
        expected_out = l(i).data
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': []})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out, dtype2prec[dtype])

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module_kwargs_only_empty_dict(self, dtype=torch.float):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).to("cuda", dtype)
        i = torch.randn(20, 10, device="cuda", dtype=dtype)
        expected_out = l(i).data
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': {}})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out, dtype2prec[dtype])

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    @repeat_test_for_types(ALL_TENSORTYPES)
    def test_data_parallel_module_kwargs_only_empty_tuple(self, dtype=torch.float):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.l = l

            def forward(self, input):
                return self.l(input['data'])

        l = nn.Linear(10, 5).to("cuda", dtype)
        i = torch.randn(20, 10, device="cuda", dtype=dtype)
        expected_out = l(i).data
        n = nn.DataParallel(Net())
        out = n(input={'data': i, 'unused': ()})
        self.assertEqual(out.get_device(), 0)
        self.assertEqual(out.data, expected_out, dtype2prec[dtype])

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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
        grad = result[0].data.clone().fill_(2)
        result[0].backward(grad)
        self.assertEqual(x.grad.data[:2], grad)
        self.assertEqual(x.grad.data[2:], grad.clone().zero_())
        _assertGradAndGradgradChecks(self, lambda y: dp.scatter(y, (0, 1)), (x,))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_scatter_cpu(self):
        self._test_scatter(torch.randn((4, 4), dtype=torch.double))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_scatter_gpu(self):
        self._test_scatter(torch.randn((4, 4), dtype=torch.double).cuda())

    def _test_gather(self, output_device):
        inputs = (
            torch.randn(2, 4, device='cuda:0', requires_grad=True, dtype=torch.double),
            torch.randn(2, 4, device='cuda:1', requires_grad=True, dtype=torch.double),
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([4, 4]))
        self.assertEqual(result[:2], inputs[0])
        self.assertEqual(result[2:], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_cuda)
        grad = torch.randn((4, 4), dtype=torch.double)
        if output_device != -1:
            grad = grad.cuda(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad.data, grad[:2])
        self.assertEqual(inputs[1].grad.data, grad[2:])
        _assertGradAndGradgradChecks(self, lambda x, y: dp.gather((x, y), output_device), inputs)

        # test scalar inputs, should stack into a vector in this case
        inputs = (
            torch.randn((), device='cuda:0', requires_grad=True, dtype=torch.double),
            torch.randn((), device='cuda:1', requires_grad=True, dtype=torch.double),
        )
        result = dp.gather(inputs, output_device)
        self.assertEqual(result.size(), torch.Size([2]))
        self.assertEqual(result[0], inputs[0])
        self.assertEqual(result[1], inputs[1])
        if output_device != -1:
            self.assertEqual(result.get_device(), output_device)
        else:
            self.assertFalse(result.is_cuda)
        grad = torch.randn(2, dtype=torch.double)
        if output_device != -1:
            grad = grad.cuda(output_device)
        result.backward(grad)
        self.assertEqual(inputs[0].grad, grad[0])
        self.assertEqual(inputs[1].grad, grad[1])
        _assertGradAndGradgradChecks(self, lambda x, y: dp.gather((x, y), output_device), inputs)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_cpu(self):
        self._test_gather(-1)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_gather_gpu(self):
        self._test_gather(0)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
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

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_replicate(self):
        module = nn.Linear(10, 5).float().cuda()
        input = torch.randn(2, 10, dtype=torch.float, device="cuda")
        expected_output = module(input).data
        for devices in [(0, 1), [0, 1]]:
            replicas = dp.replicate(module, devices)
            for i, replica in enumerate(replicas):
                for p in replica.parameters():
                    self.assertEqual(p.get_device(), i)
                replica_input = input.cuda(i)
                self.assertEqual(replica(replica_input).data, expected_output)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_replicate_buffers(self):
        net = nn.Module()
        net.bn = nn.BatchNorm2d(10)
        net.cuda()
        for devices in [(0, 1), [0, 1]]:
            replicas = dp.replicate(net, devices)
            for i, replica in enumerate(replicas):
                self.assertEqual(replica.bn.running_mean.get_device(), i, 'buffer on wrong device')
                self.assertEqual(replica.bn.running_var.get_device(), i, 'buffer on wrong device')
                self.assertEqual(replica.bn.num_batches_tracked.get_device(), i, 'buffer on wrong device')



if __name__ == '__main__':
    run_tests()
