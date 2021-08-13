# Do not add this to test/run_test.py as this is run with
# test/test_autograd.py.
#
# If you add a TestCase here, import it in test/test_autograd.py.

import math
import sys
import time
from itertools import product, permutations
from functools import partial
import torch
import torch.nn

from torch._six import inf, nan
from torch.autograd.profiler import emit_nvtx
from torch.utils.checkpoint import checkpoint
from torch.testing._internal.common_utils import (TestCase, run_tests,
                                                  CudaMemoryLeakCheck,
                                                  TEST_WITH_ROCM,
                                                  gradcheck, gradgradcheck, make_tensor)
from torch.autograd import Variable, Function
from torch.testing._internal.common_methods_invocations import mask_not_all_zeros
from torch.testing._internal.common_device_type import (instantiate_device_type_tests, skipCUDAIfRocm,
                                                        onlyCPU, onlyCUDA, onlyOnCPUAndCUDA, dtypes, dtypesIfCUDA,
                                                        deviceCountAtLeast, skipCUDAIfCudnnVersionLessThan,
                                                        skipCUDAIf, skipMeta)

import test_autograd


# Generic device type autograd tests.
class TestAutogradDeviceType(TestCase):

    def test_min_max_median_backprops_to_all_values(self, device):
        for f in [torch.min, torch.max, torch.median, torch.nanmedian]:
            x1 = torch.tensor([1., 0., 1., 0., 1., 0.], device=device, requires_grad=True)
            x2 = torch.tensor([float('nan'), float('nan'), float('nan')], requires_grad=True)
            for x in [x1, x2]:
                y = f(x)
                y.backward()
                self.assertEqual(x.grad.sum(), 1.)
                self.assertEqual((x.grad == 1 / 3).sum(), 3)

    def test_cdist(self, device):
        def _test_euclidean_large_cdist(sizex, sizey=None):
            if sizey is None:
                sizey = sizex
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # to avoid extremum
            x = x - (((x - y) < eps).float() * 2 * eps)
            x.requires_grad = True
            y.requires_grad = True
            dist = torch.cdist(x, y, p=2)
            # Do a backward pass to check that it is valid for large
            # matrices
            loss = dist.sum()
            loss.backward()

        _test_euclidean_large_cdist((2000, 5))

    # Ensure that cdist backward with p<1 does not produce NaNs
    def test_cdist_grad_p_lt_1_no_nan(self, device):
        for p in [0.99, 0.7, 0.5, 0.1, 0.01]:
            x = torch.randn(1, 2, device=device)
            y = x.clone().detach() + torch.tensor([[1., 0.]], device=device)
            x.requires_grad = True
            y.requires_grad = True
            result = torch.cdist(x, y, p=p)
            result.backward(torch.ones_like(result))
            self.assertFalse(torch.isnan(x.grad).any())
            self.assertFalse(torch.isnan(y.grad).any())

    def test_cdist_same_inputs(self, device):
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            eps = 1e-6
            x.requires_grad = True
            d = torch.cdist(x, y)
            d.backward(dist_grad)
            # Check that the backward passs does not contain invalid
            # values such as nan or inf
            assert torch.isfinite(x.grad).all()

    def test_parameter_resize(self, device):
        asd = torch.nn.Parameter(torch.ones(16, dtype=torch.double, device=device))

        for i in range(2):
            with torch.no_grad():
                asd.set_(asd[1:])
                asd.grad = None

            m = torch.cat((asd, asd))
            m.sum().backward()

    @dtypes(torch.double, torch.cdouble)
    def test_sparse_ctor_getter_backward(self, device, dtype):
        # See NOTE [ Sparse: autograd and API ] on the expected behavior of this test
        def _test(size, sparse_dim, nnz, device):
            v_size = [nnz] + list(size[sparse_dim:])
            i = torch.rand(sparse_dim, nnz)
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)

            inp = torch.randn(v_size, dtype=torch.double, device=device, requires_grad=True)
            other = self.genSparseTensor(size, sparse_dim, nnz, is_uncoalesced=True, device=device,
                                         dtype=dtype)[0]

            def fn(v):
                x = torch.sparse_coo_tensor(i, v, size, dtype=dtype, device=device)
                y = (x + other).coalesce()
                yv = y.values()
                new_v = yv.tanh()
                z = torch.sparse_coo_tensor(y.indices(), new_v, y.size())
                return z.coalesce().values()

            gradcheck(fn, (inp,), check_batched_grad=False)
            # FIXME: make gradgradcheck work.
            # gradgradcheck(fn, (inp,), check_batched_grad=False)

            # assert that _values is non-differentiable
            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.detach().requires_grad_()._values().backward(torch.ones_like(other._values()))

        for empty_i, empty_v, empty_nnz in product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5
            _test(sparse_size + dense_size, len(sparse_size), nnz, device)

    @dtypes(torch.double, torch.cdouble)
    def test_sparse_backward(self, device, dtype):
        class FixedGradientFunction(Function):
            @staticmethod
            def forward(ctx, x, grad_x):
                ctx.save_for_backward(grad_x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                saved_grad_x, = ctx.saved_tensors
                return saved_grad_x, None

        size = torch.Size([6, 3, 2])
        i1 = torch.tensor([
            [0, 3, 4],
            [0, 2, 2],
        ], dtype=torch.long)
        v1 = make_tensor([3, 2], dtype=dtype, device=device)
        sparse_grad1 = torch.sparse_coo_tensor(i1, v1, size, dtype=dtype, device=device)
        i2 = torch.tensor([
            [0, 1, 3, 4],
            [0, 1, 2, 2],
        ], dtype=torch.long)
        v2 = make_tensor([4, 2], dtype=dtype, device=device)
        sparse_grad2 = torch.sparse_coo_tensor(i2, v2, size, dtype=dtype, device=device)
        dense_grad = torch.rand(size, device=device, dtype=dtype)
        fn = FixedGradientFunction

        # sparse first
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, sparse_grad1) + fn.apply(x, dense_grad) + fn.apply(x, sparse_grad2)).sum().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # dense first
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, dense_grad) + fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # sparse only
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().backward()
        self.assertEqual(x.grad, sparse_grad1 + sparse_grad2)

    # autograd tests via common_method_invocations don't allow input tensors to
    # be sparse (RuntimeError: gradcheck expects all tensor inputs are dense when
    # check_sparse_nnz is set to False.)
    def test_sparse_mask_autograd(self, device):
        tensor = torch.randn(3, requires_grad=True, device=device)
        mask = torch.ones(3, device=device)
        mask[1] = 0
        mask = mask.to_sparse()
        converted = tensor.sparse_mask(mask).to_dense()
        converted.sum().backward()
        self.assertEqual(tensor.grad, mask.to_dense())

    def test_pyscalar_conversions(self, device):
        def _test_pyscalar_conversions(t, integral_conv):
            # integral -> integral
            l = t(torch.zeros(1, 1, 1, dtype=torch.long))
            pyscalar = -12345
            l[0] = pyscalar
            self.assertEqual(integral_conv(l), pyscalar)

            # floating point -> floating point
            f = Variable(t(torch.randn(1, 1, dtype=torch.double)))
            pyscalar = -12345.1
            f[0] = pyscalar
            self.assertEqual(float(f), pyscalar)
            f[0] = nan
            self.assertTrue(math.isnan(float(f)))
            f[0] = inf
            self.assertEqual(float(f), inf)
            f[0] = -inf
            self.assertEqual(float(f), -inf)

            # integral -> floating point
            # check we can convert something that loses precision
            pyscalar = 1234567890123456789
            self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
            l[0] = pyscalar
            self.assertEqual(float(l), float(pyscalar))

            # floating point -> integral
            f[0] = nan
            self.assertRaises(ValueError, lambda: integral_conv(f[0]))
            f[0] = inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = -inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = sys.float_info.max
            self.assertEqual(integral_conv(f), sys.float_info.max)

            # bool, nonzero
            def test_nonzero(tensor, value, expected):
                tensor[0] = value
                self.assertEqual(expected, bool(tensor))
                self.assertEqual(expected, True if tensor else False)

            test_nonzero(l, 0, False)
            test_nonzero(l, -2, True)
            test_nonzero(f, 0.0, False)
            test_nonzero(f, sys.float_info.min, True)
            test_nonzero(f, nan, bool(nan))
            test_nonzero(f, inf, bool(inf))
            test_nonzero(f, -inf, bool(-inf))


        _test_pyscalar_conversions(lambda x: x.to(device), lambda x: int(x))

    @dtypesIfCUDA(torch.half, torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_set_requires_grad_only_for_floats(self, device, dtype):
        def f1():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad_()

        def f2():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad = True

        def f3():
            torch.ones(1, dtype=dtype, device=device, requires_grad=True)

        a = torch.ones(1, dtype=dtype, device=device)
        a.requires_grad = False  # should always work
        a.requires_grad_(False)

        for f in [f1, f2, f3]:
            if dtype.is_floating_point:
                f()
            else:
                with self.assertRaisesRegex(RuntimeError, 'floating point', msg="dt: {} device: {}".format(a.dtype, a.device)):
                    f()

    @onlyCUDA
    def test_advanced_indexing_backwards_large(self, device):
        # See https://github.com/pytorch/pytorch/issues/22843
        n = (1 << 16)
        x = torch.rand(n, 1, device=device, requires_grad=True)
        a = x[:, [0]]
        a.sum().backward()
        self.assertEqual(x.grad, torch.ones(n, 1, device=device))

    def test_advanced_indexing_backwards_memory_format(self, device):
        # See https://github.com/pytorch/pytorch/issues/36956
        shape = (2, 8, 1, 2)
        i = torch.randint(1, shape, device=device).contiguous(memory_format=torch.channels_last)
        x = torch.randn(shape, requires_grad=True, device=device)
        x[i].sum().backward()

    def _test_reentrant_parent_error_on_cpu(self, device):
        t1 = torch.rand([3, 3], requires_grad=True)
        t2 = torch.rand([3, 3], device=device, requires_grad=True)
        t3 = torch.rand([3, 3], device=device, requires_grad=True)

        # Parent graph cpu graph.
        t4 = t1 * t1
        t5 = test_autograd.TestAutograd.SimulateBackwardError.apply(t4)

        # Child gpu graph (much longer than parent graph).
        prev = t2 * t2
        for i in range(10):
            prev = prev * t2
        reentrant_root = prev

        class ReentrantFunc(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will take much longer.
                reentrant_root.backward()
                return grad

        # Parent gpu graph.
        t6 = ReentrantFunc.apply(t3)
        t7 = t6 * t6

        # Parent graph will error out first, while child graph will continue executing.
        with self.assertRaisesRegex(Exception, "Simulate error"):
            torch.autograd.backward([t5.sum(), t7.sum()])

        # No grads should be accumulated since child graph will stop execution
        # after parent receives error.
        self.assertIsNone(t2.grad)
        self.assertIsNone(t1.grad)
        self.assertIsNone(t3.grad)

    @onlyCUDA
    def test_reentrant_parent_error_on_cpu(self, device):
        before = CudaMemoryLeakCheck.get_cuda_memory_usage()

        # Run as separate function so that gc can clean up everything when we
        # check for memory usage.
        self._test_reentrant_parent_error_on_cpu(device)

        # Wait for autograd thread to cleanup failed tasks.
        after = CudaMemoryLeakCheck.get_cuda_memory_usage()
        start = time.time()
        while before != after and time.time() - start < 30:
            time.sleep(0.1)
            after = CudaMemoryLeakCheck.get_cuda_memory_usage()

        self.assertEqual(before, after)

    # test for backward in https://github.com/pytorch/pytorch/issues/15511
    # TODO: opinfo pdist
    def test_pdist_large(self, device):
        def func(x):
            return torch.pdist(x, p=2)

        # shape[0] should be able to be (roughly) arbitrarily large, but the kernel
        # is currently limited to smaller sizes (see issue above); this is just testing
        # a floor.
        shape = (1000, 1)
        x = torch.randn(shape, device=device).requires_grad_()
        output = torch.pdist(x, p=2)
        # just run a single backward, as gradcheck/gradgradcheck is expensive here
        output.sum().backward()

    # TODO: see if these tests can be ported to OpInfos or moved to where's test suite
    def test_where_functional(self, device):
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where(cond, x, y):
            return torch.where(cond, x, y)

        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, device=device)])

        x = torch.randn(5, 1, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, 1, dtype=torch.double, device=device, requires_grad=True)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, 5, device=device)])

    def test_where_scalar(self, device):
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        scalar = 4.
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where_scalar_first(cond, x):
            return torch.where(cond, scalar, x)

        def where_scalar_second(cond, x):
            return torch.where(cond, x, scalar)

        gradcheck(where_scalar_first, (cond, x))
        gradgradcheck(where_scalar_first, (cond, x))

        gradcheck(where_scalar_second, (cond, x))
        gradgradcheck(where_scalar_second, (cond, x))

    @skipCUDAIf(True, """Test is flaky on Linux and Windows, typical error message:
            https://github.com/pytorch/pytorch/issues/34870""")
    def test_ctc_loss(self, device):
        batch_size = 64
        num_labels = 101
        target_length = 15
        gradcheck_input_size = 10

        ZERO_NONE = 0
        ZERO_SOME = 1
        ZERO_ALL = 2

        # input_length, vary_lengths, zero_lengths
        tests = [(150, False, ZERO_NONE),
                 (150, True, ZERO_NONE),
                 (50, True, ZERO_SOME),
                 (50, True, ZERO_ALL)]

        if 'cuda' in device:
            tests += [(50, False, ZERO_NONE),
                      (50, True, ZERO_NONE),
                      (150, True, ZERO_SOME),
                      (150, True, ZERO_ALL)]

        for input_length, vary_lengths, zero_mode in tests:
            targets = torch.randint(1, num_labels, (batch_size, target_length),
                                    device=device, dtype=torch.long)
            x = torch.randn(gradcheck_input_size, dtype=torch.double, device=device, requires_grad=True)
            tile_factors = torch.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                       device=device)
            input_lengths = [(torch.randint(input_length // 2, input_length + 1, ()).item()
                              if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
            if zero_mode == ZERO_ALL:
                target_lengths = [0 for _ in range(batch_size)]
            else:
                target_lengths = [(torch.randint(target_length // 2, target_length + 1, ()).item()
                                   if vary_lengths else target_length) for _ in range(batch_size)]
                if zero_mode == ZERO_SOME:
                    idxes = torch.randint(0, batch_size, (10,))
                    for i in idxes:
                        target_lengths[i] = 0

            def ctc_after_softmax(x):
                x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                          .view(input_length, batch_size, num_labels))
                log_probs = torch.log_softmax(x_full, 2)
                return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

            gradcheck(ctc_after_softmax, [x])

    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(7600)
    def test_ctc_loss_cudnn(self, device):
        batch_size = 16
        input_length = 30
        num_labels = 101
        target_length = 15
        targets = torch.randint(1, num_labels, (batch_size * target_length,),
                                device='cuda', dtype=torch.long)
        log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='cuda', dtype=torch.float), 2)
        log_probs.requires_grad_()

        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]
        grad_out = torch.randn(batch_size, device='cuda', dtype=torch.float)
        with torch.backends.cudnn.flags(enabled=False):
            loss_native = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
            grad_native, = torch.autograd.grad(loss_native, log_probs, grad_out)
        loss_cudnn = torch.nn.functional.ctc_loss(log_probs, targets.to('cpu', torch.int32),
                                                  input_lengths, target_lengths, reduction='none')
        self.assertTrue("Cudnn" in str(loss_cudnn.grad_fn))
        grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)
        self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)

    def test_leaky_relu_inplace_with_neg_slope(self, device):
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.leaky_relu_(a.clone(), -2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.rrelu_(a.clone(), -5.0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    def test_leaky_relu_inplace_with_zero_slope(self, device):
        a = torch.tensor([-2., 0., 2.], device=device, requires_grad=True)
        b = torch.nn.functional.leaky_relu_(a.clone(), 0.0)
        b.backward(torch.ones(3, device=device))
        expected = torch.tensor([0., 0., 1.], device=device)
        self.assertEqual(a.grad, expected)

    @onlyOnCPUAndCUDA
    def test_elu_inplace_with_neg_alpha(self, device):
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.elu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.celu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    @onlyCUDA
    def test_free_unneeded_tensor(self, device):
        x = torch.randn(2, 3, 10, 10, device=device, requires_grad=True)
        m = torch.randn(1, 3, 1, 1, device=device)

        z = x.sum()
        base_mem = torch.cuda.memory_allocated()
        z = ((x + 2) * m).sum()
        end_mem = torch.cuda.memory_allocated()

        # In the end the memory usage should remain equal, because neither of
        # (x + 2) and ((x + 2) * m) should be kept alive for backward, while the
        # previous allocation of z had the same size as the current one.
        self.assertEqual(base_mem, end_mem)

    @onlyCUDA
    def test_pin_memory(self, device):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        self.assertEqual(x, x.pin_memory())
        self.assertIsNot(x, x.pin_memory())
        self.assertTrue(x.pin_memory().requires_grad)
        gradcheck(lambda x: x.pin_memory(), [x])
        gradgradcheck(lambda x: x.pin_memory(), [x])

    @skipCUDAIfRocm
    @onlyCUDA
    def test_profiler_emit_nvtx(self, device):
        # This test is not intended to ensure correctness of nvtx ranges.
        # That would require something a great deal more complex (you'd have to create a
        # profile in a subprocess, open it, and parse the sql somehow).
        # This test is merely intended to catch if emit_nvtx breaks on construction.
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with torch.cuda.profiler.profile():
            with emit_nvtx():
                a.add(1.0)

    @onlyCUDA
    def test_rnn_backward_to_input_but_not_parameters(self, device):
        # this checks whether it is possible to not require
        # weight parameters, but require inputs, see #7722
        l = torch.nn.LSTM(2, 3).to(device)
        for p in l.parameters():
            p.requires_grad = False
        s = torch.randn(1, 1, 2, requires_grad=True, device=device)
        out, _ = l(s)
        out.sum().backward()
        self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    @onlyCUDA
    def test_lstmcell_backward_only_one_output_grad(self, device):
        # checks that undefined gradients doen't hamper the backward
        # see #11872
        l = torch.nn.LSTMCell(2, 3).to(device).double()
        s = torch.randn(1, 2, device=device, dtype=torch.double, requires_grad=True)
        for i in range(2):
            out = l(s)[i]
            out.sum().backward()
            self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    def _test_rnn_mod(self, mod, inp):
        def flatten_out(mod, inp):
            out = mod(inp)
            return tuple([t if isinstance(t, torch.Tensor) else tt for t in out for tt in t])
        gradcheckfunc = partial(flatten_out, mod)
        with torch.backends.cudnn.flags(enabled=False):
            gradcheck(gradcheckfunc, inp, check_batched_grad=False)
            gradgradcheck(gradcheckfunc, inp, check_batched_grad=False)

        if inp.is_cuda and not TEST_WITH_ROCM:
            # Assert that we have good error message around unsupported CuDNN double backward
            # NB: we trigger double backward using .backward() instead of autograd.grad due to
            # https://github.com/pytorch/pytorch/issues/37874
            with torch.backends.cudnn.flags(enabled=True):
                result = gradcheckfunc(inp)
                result[0].sum().backward(create_graph=True)
                grad0 = next(mod.parameters()).grad
                with self.assertRaisesRegex(RuntimeError,
                                            "please disable the CuDNN backend temporarily"):
                    grad0.sum().backward()

                # Here we avoid the backward(create_graph=True) memory leak
                # described in https://github.com/pytorch/pytorch/issues/7343
                for param in mod.parameters():
                    param.grad = None
                inp.grad = None

    @skipMeta  # LSTM cell reuses output which was resized
    def test_LSTM_grad_and_gradgrad(self, device):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=torch.float64, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.LSTM(hsize, hsize, bias=bias).to(device).to(torch.float64)
            self._test_rnn_mod(mod, inp)

    @skipMeta  # GRU cell reuses output which was resized
    def test_GRU_grad_and_gradgrad(self, device):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=torch.float64, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.GRU(hsize, hsize, bias=bias).to(device).to(torch.float64)
            self._test_rnn_mod(mod, inp)

    def test_copysign_subgradient(self, device):
        # Input is 0.0
        x = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Input is -0.0
        x = torch.tensor([-0.0, -0.0, -0.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is 0.0
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is -0.0
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-0.0, -0.0, -0.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

    @deviceCountAtLeast(1)
    def test_grad_assignment(self, devices):
        x = torch.randn(5, 5, device=devices[0])

        # Tests that the wrong shape raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(2, 2, device=devices[0])

        # Tests that the wrong dtype raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(5, 5, dtype=torch.long, device=devices[0])

        # Tests that self-assignment raises
        with self.assertRaises(RuntimeError):
            x.grad = x

        # Tests device -> cpu grad assignment raises
        if self.device_type != 'cpu':
            with self.assertRaises(RuntimeError):
                t_cpu = torch.rand(5, 5)
                t_cpu.grad = torch.randn(5, 5, device=devices[0])

        # Tests half type on CUDA
        if self.device_type == 'cuda':
            x = x.to(dtype=torch.half, device=devices[0])
            x.grad = torch.zeros_like(x)

        # Tests cross-device assignment raises
        if len(devices) > 1:
            x = torch.randn(5, 5, device=devices[0])
            with self.assertRaises(RuntimeError):
                x.grad = torch.randn(5, 5, device=devices[1])

    @deviceCountAtLeast(1)
    @dtypes(torch.float, torch.double)
    def test_requires_grad_factory(self, devices, dtype):
        fns = [torch.ones_like, torch.randn_like]
        x = torch.randn(2, 3, dtype=dtype, device=devices[0])

        for fn in fns:
            for requires_grad in [True, False]:
                output = fn(x, dtype=dtype, device=devices[0], requires_grad=requires_grad)
                self.assertEqual(requires_grad, output.requires_grad)
                self.assertIs(dtype, output.dtype)
                self.assertEqual(devices[0], str(x.device))

    @deviceCountAtLeast(2)
    def test_unused_output_device(self, devices):
        from torch.nn.parallel._functions import Broadcast
        x = torch.randn(5, 5, dtype=torch.float, device=devices[0], requires_grad=True)
        outputs = Broadcast.apply(list(range(len(devices))), x)
        y = outputs[-1] * 2
        y.sum().backward()
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(x.grad, torch.ones(5, 5) * 2)

    @deviceCountAtLeast(2)
    def test_backward_device(self, devices):
        # check that current device matches the variable's device
        device = [None]

        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                device[0] = grad_output.device
                return grad_output.clone()

        v = torch.randn(1, device=devices[1], requires_grad=True)
        Identity.apply(v).backward()
        self.assertEqual(str(device[0]), devices[1])

    @deviceCountAtLeast(2)
    def test_inputbuffer_add_multidevice(self, devices):
        input = torch.randn(1, device=devices[0], requires_grad=True)
        output = input.to(device=devices[1]) + input.to(device=devices[1])
        output.backward()

    @onlyCPU
    def test_copy_(self, device):
        # At the time of writing this test, copy_ is not generated from native_functions.yaml
        # there was a bug that bfloat16 was not recognized as floating.
        x = torch.randn(10, device=device, requires_grad=True)
        floating_dt = [dt for dt in torch.testing.get_all_dtypes() if dt.is_floating_point]
        for dt in floating_dt:
            y = torch.empty(10, device=device, dtype=dt)
            y.copy_(x)
            self.assertTrue(y.requires_grad)
            z = x.to(torch.bfloat16)
            self.assertTrue(z.requires_grad)

    @onlyCUDA
    def test_simple_reentrant_cross_device(self, device):
        class ReentrantFunc(Function):
            _cpu_mode = True

            @staticmethod
            def forward(ctx, x):
                return x * (x + 2)

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    if ReentrantFunc._cpu_mode:
                        new_param = torch.randn(2, 2, requires_grad=True)
                        (new_param ** 2).sum().backward()
                    else:
                        new_param = torch.randn(2, 2, device=device, requires_grad=True)
                        (new_param ** 2).sum().backward()
                return grad_output

        # Reentrant starts on GPU thread, finishs on GPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on CPU thread, finishs on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        # set ReentrantFunc node to GPU to emit tasks to GPU queue
        ReentrantFunc._cpu_mode = False
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on GPU thread, finishs on CPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        # set ReentrantFunc node to CPU to emit tasks to CPU queue
        ReentrantFunc._cpu_mode = True
        out = ReentrantFunc.apply(x)
        out.sum().backward()

    @onlyCUDA
    def test_cross_device_reentrant_autograd(self, device):
        # Output on gpu so that this task will be associated with the gpu thread
        def fn_on_gpu(inp):
            # Artificially increase the priority of the next op to make sure it runs
            # as soon as we reach it before the ops of branch1.
            dummy = inp * 2 * 2 * 2 * 2
            return inp.to(device=device)

        def parent_on_cpu(inp):
            # Slow branch of ops on gpu so that the work queue for the gpu thread
            # won't empty too quickly. They also have smaller priorities than the
            # ones created by fn_on_gpu
            branch1 = inp.to(device=device)
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            # Perform checkpoint on cpu tensors. So the last op performed in the reentrant
            # autograd is an AccumulateGrad that runs on the cpu thread for the gpu thread.
            # So the cpu thread will notify the gpu thread with an empty NodeTask.
            branch2 = checkpoint(fn_on_gpu, inp)
            out = branch2 + branch1
            return out

        inp = torch.rand(2, requires_grad=True)
        out = parent_on_cpu(inp)
        # This will segfault if the empty NodeTask is not handled properly in the
        # gpu thread ReadyQueue
        out.sum().backward()

    def test_inplace_on_view_backprop_base(self, device):
        # modify view and back-prop through base
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v1.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [1, 1]])

    def test_inplace_on_view_backprop_view_of_view(self, device):
        # modify view and backprop through view-of-view
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = x.narrow(0, 0, 1)
        v1.mul_(2)
        v2.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [0, 0]])

    def test_inplace_on_view_of_view(self, device):
        # modify view-of-view and backprop through base
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1]])

    def test_inplace_on_view_then_no_grad(self, device):
        # Perform an in-place operation on a view of a non-leaf variable.
        a = torch.ones(3, 1, dtype=torch.double, device=device, requires_grad=True)
        b = a * 2
        c = b.view_as(b)
        c[0][0] = 3

        # Force a graph update with grad disabled.
        with torch.no_grad():
            c.grad_fn

        c.sum().backward()

    def test_inplace_on_view_gradcheck(self, device):
        # gradcheck modifications to views
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            x = root.clone()
            x.narrow(1, 2, 2).narrow(0, 1, 2).mul_(b)
            x.narrow(1, 0, 2).narrow(0, 1, 2).mul_(b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_multiple_outputs(self, device):
        root = torch.arange(9., dtype=torch.double).reshape(3, 3).requires_grad_()
        x = root.clone()
        v1 = x.unbind()
        with self.assertRaises(RuntimeError):
            v1[0].mul_(2)

    def test_inplace_on_view_of_multiple_output_view(self, device):
        a = torch.rand(10, dtype=torch.double, device=device, requires_grad=True).clone()
        b = a.unbind(0)
        c = b[0].view_as(b[0])
        with self.assertRaises(RuntimeError):
            c.mul_(2)

    def test_inplace_multiple_output_view_of_view(self, device):
        a = torch.rand(10, dtype=torch.double, device=device, requires_grad=True).clone()
        b = a.view_as(a)
        c = b.unbind(0)
        with self.assertRaises(RuntimeError):
            c[0].mul_(2)

    def test_inplace_on_view_makes_base_require_grad(self, device):
        # in-place modification to view makes base require grad
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=False)
        b = torch.randn(4, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            x = root.clone()
            self.assertFalse(x.requires_grad)
            x.narrow(1, 2, 2).mul_(b)
            self.assertTrue(x.requires_grad)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_backprop_view(self, device):
        # modify view and backprop through view
        a = torch.tensor([2., 5.], device=device, requires_grad=False)
        b = torch.tensor([3.], device=device, requires_grad=True)
        res = a.narrow(0, 1, 1).mul_(b)
        res.sum().backward()
        self.assertEqual(b.grad.tolist(), [5])
        self.assertIsNone(a.grad)

    def test_inplace_on_view_modify_base(self, device):
        # Test that an in-place operation on a base that forced it to require
        # grad also forces any previous views to require grad and backprop
        # correctly
        r = torch.ones(1, dtype=torch.double, device=device, requires_grad=True)

        def fn(r):
            x = torch.ones(5, dtype=torch.double, device=device)
            v = x.select(0, 1)
            self.assertFalse(v.requires_grad)
            self.assertIsNone(v.grad_fn)
            x.add_(r)  # v is now dependent on r due to the in-place op on x
            self.assertTrue(v.requires_grad)
            return v

        gradcheck(fn, [r])
        gradgradcheck(fn, [r])

    def test_inplace_on_view_python(self, device):
        # in-place modifications of Python-autograd created view
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        class PyAdd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.mark_dirty(x)
                x.add_(y)
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad, grad

        def func(root, b):
            x = root.clone()
            PyAdd.apply(x.narrow(1, 2, 2).narrow(0, 1, 2), b)
            PyAdd.apply(x.narrow(1, 0, 2).narrow(0, 1, 2), b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_non_contig(self, device):
        root = torch.ones(2, 3, 2, device=device).select(2, 1).t().requires_grad_(True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1], [1, 1]])

    def test_inplace_on_view_multi_output_unsafe(self, device):
        for f in [lambda t: t.unsafe_split(1),
                  lambda t: t.unsafe_split_with_sizes((1, 1, 1)),
                  lambda t: t.unsafe_chunk(3)]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            s1, s2, s3 = f(b)
            s1.mul_(s2)
            s1.sum().backward()

    def test_inplace_on_view_multi_output_safe(self, device):
        for f in [lambda t: t.split(1),
                  lambda t: t.split_with_sizes((1, 1, 1)),
                  lambda t: t.chunk(3)]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            s1, s2, s3 = f(b)
            error_msg = 'This view is the output of a function that returns multiple views.'
            with self.assertRaisesRegex(RuntimeError, error_msg):
                s1.mul_(s2)

    def test_mv_grad_stride_0(self, device):
        # Reference: https://github.com/pytorch/pytorch/issues/38315
        mat = torch.randn(2, 2, dtype=torch.double, device=device)
        vec = torch.randn(1, dtype=torch.double, device=device).requires_grad_(True)

        def fn(vec):
            # Expand inside the function to make sure the input to
            # gradcheck does not have overlapping memory
            vec = vec.expand(2)
            return (mat @ vec).sum()

        gradcheck(fn, (vec))
        gradgradcheck(fn, (vec))

    @onlyCUDA
    def test_gradcheck_input_output_different_device(self, device):
        x = torch.ones((1,), dtype=torch.double, device="cuda", requires_grad=True)
        gradcheck(lambda x: x.to("cpu"), (x,))

        x = torch.ones((1,), dtype=torch.double, device="cpu", requires_grad=True)
        gradcheck(lambda x: x.to("cuda"), (x,))

    # TODO: see if this can be OpInfo'd or moved to test_reductions.py
    def test_logcumsumexp_large_value(self, device):
        a = torch.rand(4, 4, 4, dtype=torch.double, requires_grad=True)
        with torch.no_grad():
            # Large Number
            a[0] = 10000

        gradcheck(lambda x: x.logcumsumexp(0), a)
        gradgradcheck(lambda x: x.logcumsumexp(0), a)

        gradcheck(lambda x: x.logcumsumexp(1), a)
        gradgradcheck(lambda x: x.logcumsumexp(1), a)

        gradcheck(lambda x: x.logcumsumexp(2), a)
        gradgradcheck(lambda x: x.logcumsumexp(2), a)

    def test_strided_leaf_grad_layout(self, device):
        # (1) If leaf is non-overlapping and dense, grad's layout should match its leaf.
        for fmt_a in (torch.contiguous_format, torch.channels_last):
            for fmt_b in (torch.contiguous_format, torch.channels_last):
                a = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_a)
                b = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_b)
                a.requires_grad_()
                b.requires_grad_()
                # checks (1) for broadcasted gradients
                a.sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                b.sum().backward()
                self.assertEqual(b.grad.stride(), b.stride())
                # checks (1) for non-broadcasted gradients
                a.grad = None
                b.grad = None
                (a * b).sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                self.assertEqual(b.grad.stride(), b.stride())

        # (2) If leaf isn't dense, checks that grads are rowmajor contiguous.
        c = torch.empty_strided((2, 2), (4, 2), device=device).copy_(torch.rand((2, 2), device=device))
        c.requires_grad_()
        d = torch.rand((2, 2), device=device)
        # checks (2) for broadcasted gradients
        c.sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))
        # checks (2) for non-broadcasted gradients
        c.grad = None
        (c * d).sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))

    # TODO: OpInfo this or move to atleast's test suite
    def _test_atleast(self, device, torch_fn):
        # 0-dim
        s = torch.tensor(0.5, dtype=torch.double, requires_grad=True)

        gradcheck(lambda x: torch_fn(x), s)
        gradgradcheck(lambda x: torch_fn(x), s)

        # 1-dim
        a = torch.rand(4, dtype=torch.double, requires_grad=True)

        gradcheck(lambda x: torch_fn(x), a)
        gradgradcheck(lambda x: torch_fn(x), a)

        # 2,3,4-dim
        b = torch.rand(4, 3, dtype=torch.double, requires_grad=True)
        c = torch.rand(4, 3, 2, dtype=torch.double, requires_grad=True)
        d = torch.rand(4, 3, 2, 1, dtype=torch.double, requires_grad=True)

        input_tuple = (s, a, b, c, d)
        gradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)
        gradgradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)

    def test_atleast(self, device):
        self._test_atleast(device, torch.atleast_1d)
        self._test_atleast(device, torch.atleast_2d)
        self._test_atleast(device, torch.atleast_3d)

    # TODO: opinfo this or move to test_binary_ufuncs.py
    def test_xlogy(self, device):

        def _tensor_tensor_helper(x, y):
            gradcheck(lambda x, y: torch.xlogy(x, y), (x, y))
            gradgradcheck(lambda x, y: torch.xlogy(x, y), (x, y))

            with torch.no_grad():
                x = x.clone()
                x[torch.rand_like(x) > 0.5] = 0

            gradcheck(lambda y: torch.xlogy(x, y), (y))
            gradgradcheck(lambda y: torch.xlogy(x, y), (y))

        shapes = ((4,), (1, 4), (1, 1, 4), (1, 1, 1, 4))

        # For broadcastible shapes and scalar.
        for x_shape, y_shape in permutations(shapes, 2):
            x = torch.rand(*x_shape, dtype=torch.double, device=device, requires_grad=True)
            y = torch.rand(*y_shape, dtype=torch.double, device=device, requires_grad=True)

            _tensor_tensor_helper(x, y)
            _tensor_tensor_helper(y, x)

            gradcheck(lambda y: torch.xlogy(0, y), (y))
            gradgradcheck(lambda y: torch.xlogy(0, y), (y))

            gradcheck(lambda y: torch.xlogy(2, y), (y))
            gradgradcheck(lambda y: torch.xlogy(2, y), (y))
            gradcheck(lambda y: torch.xlogy(y, 2), (y))
            gradgradcheck(lambda y: torch.xlogy(y, 2), (y))

        # Different shape
        x = torch.rand(2, 3, 4, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.rand(4, 5, dtype=torch.double, device=device, requires_grad=True)
        _tensor_tensor_helper(x, y)
        _tensor_tensor_helper(y, x)
        _tensor_tensor_helper(x, x)
        _tensor_tensor_helper(y, y)

        # Same shape
        x = torch.rand(4, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.rand(4, 5, dtype=torch.double, device=device, requires_grad=True)
        _tensor_tensor_helper(x, y)
        _tensor_tensor_helper(y, x)
        _tensor_tensor_helper(x, x)
        _tensor_tensor_helper(y, y)

    def test_copy_r_to_c(self, device):
        out_c = torch.empty(3, 2, dtype=torch.cdouble, device=device)
        inp_r = torch.randn(3, 2, dtype=torch.double, device=device,
                            requires_grad=True)

        def do_test():
            out_c.copy_(inp_r)
            out_c.sum().backward()
            self.assertEqual(inp_r.grad, torch.ones_like(inp_r))

        self.assertNotWarn(do_test)

    def test_non_differentiable_ops(self, device):
        # Just make sure the op doesn't raise an error
        # and resulting tensor has requires_grad=False.
        x = torch.tensor([[1, 2], [3, 4.]], requires_grad=True, device=device)
        out = torch.isin(x, torch.tensor([2, 3], device=device))
        self.assertFalse(out.requires_grad)

        x = torch.randn(3, 3, requires_grad=True)
        out = torch.signbit(x)
        self.assertFalse(out.requires_grad)


# e.g., TestAutogradDeviceTypeCPU and TestAutogradDeviceTypeCUDA
instantiate_device_type_tests(
    TestAutogradDeviceType,
    globals(),
    except_for=None
)


# Only export the generated tests, identified because they start with
# the generic instantiated above.
__all__ = [name for name in globals() if name.startswith('TestAutogradDeviceType')]


if __name__ == '__main__':
    run_tests()
