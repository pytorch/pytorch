import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import unittest

from torch.testing._internal.common_utils import suppress_warnings, num_profiled_runs, run_tests

from torch.testing._internal.te_utils import CudaCodeGenCreated, CudaCodeGenExecuted, \
    LLVMCodeGenExecuted, SimpleIREvalExecuted

from torch.testing._internal.jit_utils import JitTestCase


class BaseTestClass(JitTestCase):
    def setUp(self):
        self.old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        self.old_profiling_mode = torch._C._jit_set_profiling_mode(True)

        self.old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        self.old_gpu_fuser_state = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(True)
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)
        # TODO: CPU fuser currently is disabled when multithreading.
        self.old_fuse_parallel = torch._C._jit_texpr_parallel_cpu_enabled()
        torch._C._jit_set_texpr_parallel_cpu_enabled(True)

        self.devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']

    def tearDown(self):
        torch._C._jit_set_profiling_executor(self.old_profiling_executor)
        torch._C._jit_set_profiling_mode(self.old_profiling_mode)

        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuser_state)
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuser_state)
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)
        torch._C._jit_set_texpr_parallel_cpu_enabled(self.old_fuse_parallel)

    def assertLastGraphAllFused(self):
        self.assertAllFused(torch.jit.last_executed_optimized_graph())


def warmup_and_run_forward(f, *args):
    for _ in range(torch._C._jit_get_num_profiled_runs() + 1):
        results = f(*args)
    return results


class TestTensorExprFuser(BaseTestClass):
    def test_easy(self):
        def easy(x, y):
            aaa = torch.add(x, y)
            return aaa

        traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

        a = torch.rand(1024)
        b = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

    def test_three_arg(self):
        llvm_executed = LLVMCodeGenExecuted()
        simple_ir_eval_executed = SimpleIREvalExecuted()

        def easy(x, y, z):
            aaa = torch.add(x, y)
            bbb = torch.add(aaa, z)
            return bbb

        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )

        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = a.numpy() + b.numpy() + c.numpy()
        np.testing.assert_allclose(npr, x.numpy())
        assert (
            llvm_executed.elapsed_value() >= 1
            or simple_ir_eval_executed.elapsed_value() >= 1
        )

    def test_four_arg(self):
        def run_addcmul(x, y, z, w):
            c = torch.addcmul(torch.add(x, y), z, w)
            return c

        device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ['cpu']
        for dev in device_options:
            rand_a = torch.rand(1024, dtype=torch.float, device=dev)
            rand_b = torch.rand(1024, dtype=torch.float, device=dev)
            rand_c = torch.rand(1024, dtype=torch.float, device=dev)
            rand_d = torch.rand(1024, dtype=torch.float, device=dev)

            traced = torch.jit.trace(
                run_addcmul,
                (
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                    torch.zeros(1024, dtype=torch.float, device=dev),
                ),
            )

            x = warmup_and_run_forward(traced, rand_a, rand_b, rand_c, rand_d)
            self.assertLastGraphAllFused()
            y = run_addcmul(rand_a, rand_b, rand_c, rand_d)
            np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy(), atol=1e-6)

    def test_three_arg_cuda(self):
        if not torch.cuda.is_available():
            return
        cuda_cg_executed = CudaCodeGenExecuted()
        cuda_cg_created = CudaCodeGenCreated()

        def test(x, y, z):
            aaa = torch.add(x, y)
            bbb = torch.add(aaa, z)
            return bbb

        M = 32
        N = 32
        traced = torch.jit.trace(
            test,
            (
                torch.rand(M, N, device="cuda"),
                torch.rand(M, N, device="cuda"),
                torch.rand(M, N, device="cuda"),
            ),
        )

        a = torch.rand(M, N, device="cuda")
        b = torch.rand(M, N, device="cuda")
        c = torch.rand(M, N, device="cuda")
        x = traced(a, b, c)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
        np.testing.assert_allclose(npr, x.cpu().numpy())
        assert cuda_cg_executed.elapsed_value() >= 1
        assert cuda_cg_created.elapsed_value() >= 1

    def test_broadcast_cuda(self):
        if not torch.cuda.is_available():
            return

        def test_body(M, N, L, K):
            if not torch.cuda.is_available():
                return
            cuda_cg_executed = CudaCodeGenExecuted()
            cuda_cg_created = CudaCodeGenCreated()

            def test(x, y, z):
                v1 = torch.add(x, y)
                v2 = torch.add(v1, z)
                return v2

            a_shape = [M, N]
            b_shape = [L, M, 1]
            c_shape = [K, L, 1, 1]
            traced = torch.jit.trace(
                test,
                (
                    torch.rand(*a_shape, device="cuda"),
                    torch.rand(*b_shape, device="cuda"),
                    torch.rand(*c_shape, device="cuda"),
                ),
            )

            a = torch.rand(*a_shape, device="cuda")
            b = torch.rand(*b_shape, device="cuda")
            c = torch.rand(*c_shape, device="cuda")
            x = warmup_and_run_forward(traced, a, b, c)
            self.assertLastGraphAllFused()
            npr = a.cpu().numpy() + b.cpu().numpy() + c.cpu().numpy()
            np.testing.assert_allclose(npr, x.cpu().numpy())
            assert cuda_cg_executed.elapsed_value() >= 1
            assert cuda_cg_created.elapsed_value() >= 1

        test_configs = [[36, 17, 63, 33], [32, 32, 32, 32]]
        for test_config in test_configs:
            test_body(*test_config)

    def test_all_combos(self):
        def easy(x, y, z):
            a = torch.add(x, y)
            b = torch.add(a, z)
            c = torch.add(x, b)
            d = torch.add(c, a)
            return d

        def np_easy(x, y, z):
            a = x + y
            b = a + z
            c = x + b
            d = c + a
            return d

        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )

        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_rank_two(self):
        def easy(x, y, z):
            a = torch.add(x, y)
            b = torch.add(a, z)
            c = torch.add(x, b)
            d = torch.add(c, a)
            return d

        def np_easy(x, y, z):
            a = x + y
            b = a + z
            c = x + b
            d = c + a
            return d

        shape = 32, 32
        traced = torch.jit.trace(
            easy, (torch.rand(shape), torch.rand(shape), torch.rand(shape))
        )

        a = torch.rand(shape)
        b = torch.rand(shape)
        c = torch.rand(shape)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    def test_broadcast(self):
        def easy(x, y, z):
            a = torch.add(x, y)
            b = torch.add(a, z)
            return b

        def np_easy(x, y, z):
            a = x + y
            b = a + z
            return b

        N = 32
        traced = torch.jit.trace(easy, (torch.rand(N, N), torch.rand(N), torch.rand(N, N)))

        a = torch.rand(N, N)
        b = torch.rand(N)
        c = torch.rand(N, N)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        npr = np_easy(a.numpy(), b.numpy(), c.numpy())
        np.testing.assert_allclose(npr, x.numpy())

    @unittest.skip("temporarily disable")
    def test_broadcast_2(self):
        zero = torch.tensor([0.0], dtype=torch.float)

        def foo(x, y, z):
            aaa = torch.add(x, y)
            bbb = torch.add(zero, aaa)
            return torch.add(bbb, z)

        def foo_np(x, y, z):
            a = x + y
            b = zero.numpy() + a
            return b + z

        x = torch.rand(3, 4)
        y = torch.ones(3, 1)
        z = torch.rand(4)
        traced = torch.jit.trace(foo, (x, y, z))

        r = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()

        rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
        np.testing.assert_allclose(r, rnp)

    @unittest.skip("temporarily disable")
    def test_broadcast_big2(self):
        zero = torch.tensor([0.0], dtype=torch.float)

        def foo(x, y, z):
            aaa = torch.add(x, y)
            bbb = torch.add(zero, aaa)
            return torch.add(bbb, z)

        def foo_np(x, y, z):
            a = x + y
            b = zero.numpy() + a
            return b + z

        x = torch.rand(32, 1024)
        y = torch.ones(32, 1)
        z = torch.rand(1024)
        traced = torch.jit.trace(foo, (x, y, z))

        r = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()
        rnp = foo_np(x.numpy(), y.numpy(), z.numpy())
        np.testing.assert_allclose(r, rnp)

    def test_alpha(self):
        def alpha(x):
            aaa = torch.add(x, x, alpha=2.0)
            return aaa

        traced = torch.jit.trace(alpha, (torch.tensor([1.0])))

        a = torch.tensor([1.0])
        x = traced(a)
        np.testing.assert_allclose(a.numpy() + 2.0 * a.numpy(), x.numpy())

    @suppress_warnings
    @unittest.skip("temporarily disable")
    def test_constant(self):
        def constant(x):
            bbb = torch.tensor([1.0])
            aaa = torch.add(x, bbb)
            return aaa

        traced = torch.jit.trace(constant, (torch.tensor([1.0])))

        a = torch.tensor([1.0])
        x = warmup_and_run_forward(traced, a)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + 1.0, x.numpy())

    def test_add_sub(self):
        def easy(x, y, z):
            aaa = torch.add(x, y)
            bbb = torch.sub(aaa, z)
            return bbb

        traced = torch.jit.trace(
            easy, (torch.rand(1024), torch.rand(1024), torch.rand(1024))
        )

        a = torch.rand(1024)
        b = torch.rand(1024)
        c = torch.rand(1024)
        x = warmup_and_run_forward(traced, a, b, c)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy() - c.numpy(), x.numpy())

    def test_promotion(self):
        def easy(x, y):
            aaa = torch.add(x, y)
            return aaa

        traced = torch.jit.trace(
            easy,
            (torch.zeros(1024, dtype=torch.int32), torch.rand(1024, dtype=torch.float32)),
        )

        a = torch.zeros(1024, dtype=torch.int32)
        b = torch.rand(1024, dtype=torch.float32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(a.numpy() + b.numpy(), x.numpy())

    def test_double(self):
        TENSOR_LEN = 8

        def easy(x, y):
            aaa = torch.add(x, y)
            bbb = torch.mul(aaa, y)
            return bbb

        traced = torch.jit.trace(
            easy,
            (torch.rand(TENSOR_LEN, dtype=torch.float64), torch.full((TENSOR_LEN,), 0.5, dtype=torch.float64)),
        )

        a = torch.rand(TENSOR_LEN, dtype=torch.double)
        b = torch.full((TENSOR_LEN,), 0.5, dtype=torch.double)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_short(self):
        TENSOR_LEN = 8

        def easy(x, y):
            aaa = torch.add(x, y)
            bbb = torch.mul(aaa, y)
            return bbb

        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16),
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)),
        )

        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int16)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_char(self):
        TENSOR_LEN = 8

        def easy(x, y):
            aaa = torch.add(x, y)
            bbb = torch.mul(aaa, y)
            return bbb

        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8),
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)),
        )

        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_int64_promotion(self):
        TENSOR_LEN = 8

        def easy(x, y):
            aaa = torch.add(x, y)
            bbb = torch.mul(aaa, y)
            return bbb

        traced = torch.jit.trace(
            easy,
            (torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8),
             torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int64)),
        )

        a = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int8)
        b = torch.randint(TENSOR_LEN, (TENSOR_LEN,), dtype=torch.int64)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose((a.numpy() + b.numpy()) * b.numpy(), x.numpy())

    def test_eq(self):
        def easy(x, y):
            c = torch.eq(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        a = torch.zeros(1024, dtype=torch.int32)
        b = torch.zeros(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_ne(self):
        def easy(x, y):
            c = torch.ne(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        a = torch.zeros(1024, dtype=torch.int32)
        b = torch.ones(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_ge(self):
        def easy(x, y):
            c = torch.ge(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        aa = np.empty([1024], dtype=np.int32)
        aa.fill(5)
        a = torch.from_numpy(aa)
        b = torch.zeros(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_gt(self):
        def easy(x, y):
            c = torch.gt(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        a = torch.ones(1024, dtype=torch.int32)
        b = torch.zeros(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_le(self):
        def easy(x, y):
            c = torch.le(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.zeros(1024), torch.zeros(1024)))
        aa = np.empty([1024], dtype=np.int32)
        aa.fill(5)
        a = torch.from_numpy(aa)
        b = torch.zeros(1024, dtype=torch.int32)
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(np.zeros(1024), x.numpy())

    def test_lt(self):
        def easy(x, y):
            c = torch.lt(x, y)
            return c

        device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for dev in device_options:
            traced = torch.jit.trace(easy, (torch.zeros(1024, device=dev), torch.zeros(1024, device=dev)))
            a = torch.ones(1024, dtype=torch.int32, device=dev)
            b = torch.zeros(1024, dtype=torch.int32, device=dev)
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()
            np.testing.assert_allclose(np.zeros(1024), x.cpu().numpy())

    @suppress_warnings
    def test_min_max(self):
        def test(x, y):
            return torch.max(torch.min(x, y), torch.tensor([4.0]))

        traced = torch.jit.trace(test, (torch.zeros(1024), torch.zeros(1024)))
        a = 8.0 * torch.rand(1024)
        b = 8.0 * torch.rand(1024)
        np.testing.assert_allclose(
            warmup_and_run_forward(traced, a, b), np.maximum(np.minimum(a.numpy(), b.numpy()), [4.0])
        )
        self.assertLastGraphAllFused()

    @unittest.skip("temporarily disable")
    def test_min_max_reduction(self):
        def test(x):
            return torch.min(x) + torch.max(x)

        traced = torch.jit.trace(test, (torch.zeros(1024)))
        a = 8.0 * torch.rand(1024)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(a.numpy()) + np.amax(a.numpy()))
        self.assertLastGraphAllFused()

    @unittest.skip("temporarily disable")
    def test_min_max_reduction2(self):
        def test(x):
            return x.min() + x.max()

        traced = torch.jit.trace(test, (torch.zeros(1024)))
        a = 8.0 * torch.rand(1024)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(a.numpy()) + np.amax(a.numpy()))
        self.assertLastGraphAllFused()

    def test_min_max_reduction_dim1(self):
        def test(x):
            return torch.min(x, 1)[0] + torch.max(x, 1)[0]

        traced = torch.jit.trace(test, (torch.zeros(16, 16)))
        a = 8.0 * torch.rand(16, 16)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a), np.amin(
            a.numpy(), axis=1) + np.amax(a.numpy(), axis=1))
        self.assertLastGraphAllFused()

    @unittest.skip("temporarily disable")
    def test_min_max_reduction_dim1_2(self):
        def test(x):
            return torch.min(x, 1)

        traced = torch.jit.trace(test, (torch.zeros(16, 16)))
        a = 8.0 * torch.rand(16, 16)
        np.testing.assert_allclose(warmup_and_run_forward(traced, a)[0], np.amin(a.numpy(), axis=1))
        self.assertLastGraphAllFused()

    def test_clamp(self):
        def test(x):
            return torch.clamp(x + 3.0, 0.0, 6.0)

        device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

        for dev in device_options:
            traced = torch.jit.trace(test, (torch.zeros(1024, device=dev)))
            a = 20.0 * torch.rand(1024, device=dev) - 10.0
            an = a.cpu().numpy()
            np.testing.assert_allclose(warmup_and_run_forward(traced, a).cpu(), np.clip(an + 3.0, 0.0, 6.0))
            self.assertLastGraphAllFused()

    def test_relu(self):
        def test(x):
            return torch.clamp(F.relu(x), 0, 0.5)

        device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for dev in device_options:
            traced = torch.jit.trace(test, (torch.zeros(1024, device=dev)))
            a = 20.0 * torch.rand(1024, device=dev) - 10.0
            an = a.cpu().numpy()
            np.testing.assert_allclose(warmup_and_run_forward(traced, a).cpu(), np.clip((np.maximum(0, an)), 0, 0.5))
            self.assertLastGraphAllFused()

    def test_reps(self):
        def easy(x, y):
            c = torch.add(x, y)
            return c

        traced = torch.jit.trace(easy, (torch.rand(1024), torch.rand(1024)))

        for _ in range(32):
            a = torch.ones(1024)
            b = torch.zeros(1024)
            x = warmup_and_run_forward(traced, a, b)
            np.testing.assert_allclose(np.ones(1024), x.numpy())

    def test_add_const_rhs(self):
        def test(x):
            return x + 3.0

        traced = torch.jit.trace(test, torch.rand(4))
        x = torch.rand(4)
        y = warmup_and_run_forward(traced, x)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(x.numpy() + 3.0, y.numpy())

    def test_int_output(self):
        def test(x, y, z):
            return x * y * z

        xs = [(torch.rand(4) * 3 + 1).to(torch.int32) for i in range(3)]
        x, y, z = xs
        xn, yn, zn = [t.numpy() for t in xs]
        traced = torch.jit.trace(test, (x, y, z))
        res = warmup_and_run_forward(traced, x, y, z)
        self.assertLastGraphAllFused()
        np.testing.assert_allclose(xn * yn * zn, res.numpy())

    def test_binary_ops(self):
        def test_atan2(x, y):
            c = torch.atan2(torch.add(x, y), y)
            return c

        def test_gt(x, y):
            c = torch.gt(torch.add(x, y), y)
            return c

        def test_ge(x, y):
            c = torch.ge(torch.add(x, y), y)
            return c

        def test_lt(x, y):
            c = torch.lt(torch.add(x, y), y)
            return c

        def test_le(x, y):
            c = torch.le(torch.add(x, y), y)
            return c

        def test_lerp(x, y):
            c = torch.lerp(torch.add(x, 1), x, 2.0)
            return c

        def test_mul(x, y):
            c = torch.mul(torch.add(x, y), y)
            return c

        def test_ne(x, y):
            c = torch.ne(torch.add(x, y), y)
            return c

        def test_div(x, y):
            c = torch.div(torch.add(x, y), 2)
            return c

        def test_eq(x, y):
            c = torch.eq(torch.add(x, y), y)
            return c

        def test_fmod(x, y):
            c = torch.fmod(torch.add(x, y), 2)
            return c

        def test_sub(x, y):
            c = torch.sub(torch.add(x, y), x)
            return c

        def test_remainder(x, y):
            c = torch.remainder(torch.add(x, y), 3.0)
            return c

        def test_pow(x, y):
            c = torch.pow(torch.add(x, y), 2.0)
            return c

        def test_type_as(x, y):
            return x.type_as(torch.add(x, y))

        fns = {
            test_atan2,
            test_gt,
            test_ge,
            test_lt,
            test_le,
            test_lerp,
            test_mul,
            test_ne,
            test_div,
            test_eq,
            test_fmod,
            test_sub,
            test_remainder,
            test_pow,
            test_type_as,
        }
        device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ['cpu']
        for torch_fn in fns:
            for dev in device_options:
                rand_a = torch.rand(1024, device=dev)
                rand_b = torch.rand(1024, device=dev)
                in1 = 20 * torch.rand(1024, device=dev)
                in2 = 20 * torch.rand(1024, device=dev)
                traced = torch.jit.trace(torch_fn, (in1, in2))
                x = warmup_and_run_forward(traced, rand_a, rand_b)
                self.assertLastGraphAllFused()
                y = torch_fn(rand_a, rand_b)
                np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy(), atol=2e-3)

    def test_unary_ops(self):
        def test_cast_float(x, y):
            c = torch.ops.aten._cast_Float(torch.add(x, y))
            return c

        def test_round(x, y):
            c = torch.round(torch.add(x, y))
            return c

        def test_sin(x, y):
            c = torch.sin(torch.add(x, y))
            return c

        def test_asin(x, y):
            c = torch.asin(torch.add(x, y))
            return c

        def test_sinh(x, y):
            c = torch.sinh(torch.add(x, y))
            return c

        def test_cos(x, y):
            c = torch.cos(torch.add(x, y))
            return c

        def test_acos(x, y):
            c = torch.acos(torch.add(x, y))
            return c

        def test_cosh(x, y):
            c = torch.cosh(torch.add(x, y))
            return c

        def test_tan(x, y):
            c = torch.tan(torch.add(x, y))
            return c

        def test_atan(x, y):
            c = torch.atan(torch.add(x, y))
            return c

        def test_tanh(x, y):
            c = torch.tanh(torch.add(x, y))
            return c

        def test_sqrt(x, y):
            c = torch.sqrt(torch.add(x, y))
            return c

        def test_rsqrt(x, y):
            c = torch.rsqrt(torch.add(x, y))
            return c

        def test_floor(x, y):
            c = torch.floor(torch.add(x, y))
            return c

        def test_ceil(x, y):
            c = torch.ceil(torch.add(x, y))
            return c

        def test_trunc(x, y):
            c = torch.trunc(torch.add(x, y))
            return c

        def test_abs(x, y):
            c = torch.abs(torch.add(x, y))
            return c

        def test_log(x, y):
            c = torch.log(torch.add(x, y))
            return c

        def test_log2(x, y):
            c = torch.log2(torch.add(x, y))
            return c

        def test_log10(x, y):
            c = torch.log10(torch.add(x, y))
            return c

        def test_log1p(x, y):
            c = torch.log1p(torch.add(x, y))
            return c

        def test_rqrt(x, y):
            c = torch.rsqrt(torch.add(x, y))
            return c

        def test_erf(x, y):
            c = torch.erf(torch.add(x, y))
            return c

        def test_exp(x, y):
            c = torch.exp(torch.add(x, y))
            return c

        def test_expm1(x, y):
            c = torch.expm1(torch.add(x, y))
            return c

        def test_erfc(x, y):
            c = torch.erfc(torch.add(x, y))
            return c

        def test_frac(x, y):
            c = torch.frac(torch.add(x, y))
            return c

        def test_lgamma(x, y):
            c = torch.lgamma(torch.add(x, y))
            return c

        def test_sigmoid(x, y):
            c = torch.sigmoid(torch.add(x, y))
            return c

        def test_reciprocal(x, y):
            c = torch.reciprocal(torch.add(x, y))
            return c

        def test_neg(x, y):
            c = torch.neg(torch.add(x, y))
            return c

        def test_relu(x, y):
            c = torch.relu(torch.add(x, y))
            return c

        def test_hardtanh(x, y):
            c = F.hardtanh(torch.add(x, y), -1.0, 1.0)
            return c

        def test_threshold(x, y):
            c = F.threshold(torch.add(x, y), 0.5, 10)
            return c

        fns = {
            test_round,
            test_sin,
            test_asin,
            test_sinh,
            test_cos,
            test_acos,
            test_cosh,
            test_tan,
            test_atan,
            test_sqrt,
            test_floor,
            test_ceil,
            test_trunc,
            test_abs,
            test_log,
            test_log2,
            test_log10,
            test_log1p,
            test_rsqrt,
            test_exp,
            test_expm1,
            test_erf,
            test_erfc,
            test_frac,
            test_lgamma,
            test_reciprocal,
            test_neg,
            test_threshold,
            test_relu,
            test_tanh,
            test_hardtanh,
            test_sigmoid,
        }
        device_options = ["cpu", "cuda"] if torch.cuda.is_available() else ['cpu']

        for torch_fn in fns:
            for dev in device_options:
                # print(torch_fn, dev)
                rand_a = torch.rand(1024, device=dev)
                rand_b = torch.rand(1024, device=dev)
                ins = 20 * torch.rand(1024, device=dev)
                cc = np.empty([1024], dtype=np.float32)
                cc.fill(np.nan)
                nans = torch.from_numpy(cc).to(dev)
                traced = torch.jit.trace(torch_fn, (ins, ins))
                x = warmup_and_run_forward(traced, rand_a, rand_b)
                self.assertLastGraphAllFused()
                y = torch_fn(rand_a, rand_b)
                np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy(), atol=2e-3)
                # nans
                # TODO: reenable. Currently all of the tests fail
                # traced = torch.jit.trace(torch_fn, (ins, ins))
                # x = warmup_and_run_forward(traced, rand_a, rand_b)
                # y = torch_fn(nans, rand_b)
                # try:
                #     np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())
                #     print("Succeeded on dev=", dev, "function=", torch_fn)
                # except AssertionError:
                #     # Print extra info before exiting:
                #     print("Failed on dev=", dev, "function=", torch_fn)
                #     # np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())

    def test_rand_like(self):
        devices = ["cuda"] if torch.cuda.is_available() else []
        N = 1 << 16

        def run_rand_like(x, y):
            return torch.rand_like(torch.add(x, y))

        for device in devices:
            x = torch.rand(N, device=device)
            traced = torch.jit.trace(run_rand_like, (x, x), check_trace=False)
            x_v = warmup_and_run_forward(traced, x, x)
            self.assertLastGraphAllFused()
            x_np = x.cpu().numpy()
            x1_mean = np.mean(x_np)
            x2_mean = np.mean(x_np ** 2)
            x3_mean = np.mean(x_np ** 3)
            np.testing.assert_allclose(x1_mean, 1. / 2, rtol=2e-2)
            np.testing.assert_allclose(x2_mean, 1. / 3, rtol=2e-2)
            np.testing.assert_allclose(x3_mean, 1. / 4, rtol=2e-2)

    def test_nans(self):
        def test_max(x, y):
            return torch.max(2 * x, 2 * y)

        def test_min(x, y):
            return torch.min(2 * x, 2 * y)

        tmax = torch.jit.trace(test_max, (torch.rand(1), torch.rand(1)))
        tmin = torch.jit.trace(test_min, (torch.rand(1), torch.rand(1)))

        x = torch.tensor([np.nan])
        y = torch.tensor([1.0])

        assert np.isnan(warmup_and_run_forward(tmin, x, y).item())
        assert np.isnan(warmup_and_run_forward(tmin, y, x).item())
        self.assertLastGraphAllFused()
        assert np.isnan(warmup_and_run_forward(tmax, x, y).item())
        assert np.isnan(warmup_and_run_forward(tmax, y, x).item())
        self.assertLastGraphAllFused()

    def test_double_intrinsics(self):
        # TODO: add "cpu" device once `pow` is supported there
        devices = ["cuda"] if torch.cuda.is_available() else []

        def do_pow(x):
            return torch.pow(x, 7)

        for device in devices:
            x = torch.rand(10, dtype=torch.double, device=device)
            traced = torch.jit.trace(do_pow, (x))
            x = warmup_and_run_forward(traced, x)
            self.assertLastGraphAllFused()

    def test_remainder(self):
        def run_remainder(x, y):
            c = torch.remainder(torch.add(x, y), x)
            return c

        a = torch.rand(1024, dtype=float)
        b = torch.rand(1024, dtype=float)
        zeros = torch.zeros(1024, dtype=float)
        cc = np.array(1024, dtype=float)
        cc.fill(np.nan)
        nans = torch.from_numpy(cc)

        # random floats
        traced = torch.jit.trace(run_remainder, (torch.zeros(1024), torch.zeros(1024)))
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        y = run_remainder(a, b)
        np.testing.assert_allclose(x.numpy(), y.numpy())

        # div by 0
        traced = torch.jit.trace(run_remainder, (torch.zeros(1024), torch.zeros(1024)))
        x = warmup_and_run_forward(traced, zeros, a)
        self.assertLastGraphAllFused()
        y = run_remainder(zeros, a)
        np.testing.assert_allclose(x.numpy(), y.numpy())

        # numerators and denominatos are nan
        traced = torch.jit.trace(run_remainder, (torch.zeros(1024), torch.zeros(1024)))
        x = warmup_and_run_forward(traced, nans, a)
        self.assertLastGraphAllFused()
        y = run_remainder(nans, a)
        np.testing.assert_allclose(x.numpy(), y.numpy())

    def test_multioutput(self):
        def easy(x):
            b = x + 1
            c = b + b
            return (b, c)

        traced = torch.jit.trace(easy, (torch.zeros(1024)))

        a = torch.zeros(1024)
        b, c = warmup_and_run_forward(traced, a)
        self.assertLastGraphAllFused()
        bp = a.numpy() + 1
        cp = bp + bp
        np.testing.assert_allclose(b.numpy(), bp)
        np.testing.assert_allclose(c.numpy(), cp)

    def test_chunk(self):
        def easy(x):
            y = x + 1
            aaa, bbb = torch.chunk(y, 2)
            return aaa + bbb

        traced = torch.jit.trace(easy, (torch.zeros(1024, 1024)))

        a = torch.zeros(1024, 1024)
        x = warmup_and_run_forward(traced, a)
        self.assertLastGraphAllFused()
        npr = a.numpy()
        npr2 = npr + 1
        npr_a, npr_b = np.array_split(npr2, 2)
        np.testing.assert_allclose(npr_a + npr_b, x.numpy())

    def _test_cat(self, device):
        def foo(*args):
            args_2 = [v + i for i, v in enumerate(args)]
            v = torch.cat(args_2, dim=1)
            return v * v

        M = 16
        Ns = [128, 16, 1]
        values = [torch.zeros(M, N, device=device) for N in Ns]
        traced = torch.jit.trace(foo, values)

        x = warmup_and_run_forward(traced, *values)
        self.assertLastGraphAllFused()
        ref = foo(*values)
        np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_cpu(self):
        self._test_cat('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cat_cuda(self):
        self._test_cat('cuda')

    # This test checks that we correctly handle fusion group with just aten::cat in it.
    # Note that the test only makes sense with min_fusion_group=1, otherwise no
    # fusion groups would be formed at all.
    # TODO: Fix and re-enable the test.
    def _test_cat_only(self, device):
        def foo(*args):
            args_2 = [v + i for i, v in enumerate(args)]
            v = torch.cat(args_2, dim=1)
            return v

        M = 16
        Ns = [128, 16, 1]
        values = [torch.zeros(M, N, device=device) for N in Ns]
        traced = torch.jit.trace(foo, values)

        x = warmup_and_run_forward(traced, *values)
        self.assertLastGraphAllFused()
        ref = foo(*values)
        np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    @unittest.skip("temporarily disable")
    def test_cat_only_cpu(self):
        self._test_cat_only('cpu')

    @unittest.skip("temporarily disable")
    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cat_only_cuda(self):
        self._test_cat_only('cuda')

    def _test_cat_negative_dim(self, device):
        def foo(*args):
            v = torch.cat(args, dim=-1)
            return v * v

        M = 16
        Ns = [128, 16, 1]
        values = [torch.randn(M, N, device=device) for N in Ns]
        traced = torch.jit.trace(foo, values)

        x = warmup_and_run_forward(traced, *values)
        self.assertLastGraphAllFused()
        ref = foo(*values)
        np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_cat_negative_dim_cpu(self):
        self._test_cat_negative_dim('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cat_negative_dim_cuda(self):
        self._test_cat_negative_dim('cuda')

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cat_promote_inputs(self):
        def foo(*args):
            v = torch.cat(args, dim=1)
            return v * v

        M = 16
        Ns = [128, 16, 1]
        dtypes = [torch.half, torch.float32, torch.double]
        values = [torch.randn(M, N, device='cuda', dtype=dt) for N, dt in zip(Ns, dtypes)]
        traced = torch.jit.trace(foo, values)

        x = warmup_and_run_forward(traced, *values)
        self.assertLastGraphAllFused()
        ref = foo(*values)
        np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cat_empty_tensors(self):
        def foo(*args):
            v = torch.cat(args, dim=1)
            return v * v

        M = 16
        Ns = [128, 16, 1]
        empty = torch.tensor([], device='cuda', dtype=torch.double)
        values = [empty] + [torch.randn(M, N, device='cuda') for N in Ns]
        traced = torch.jit.trace(foo, values)

        x = warmup_and_run_forward(traced, *values)
        self.assertLastGraphAllFused()
        ref = foo(*values)
        np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

        # now test with only empty tensors
        values = [empty for i in range(3)]
        traced = torch.jit.trace(foo, values)
        x = warmup_and_run_forward(traced, *values)
        self.assertLastGraphAllFused()
        ref = foo(*values)
        np.testing.assert_allclose(ref.cpu().numpy(), x.cpu().numpy())

    def test_scalar(self):
        @torch.jit.script
        def test_float(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, a: float, b: float) -> torch.Tensor:
            return torch.add(torch.add(x, y, alpha=a), z, alpha=b)

        @torch.jit.script
        def test_int(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, a: int, b: int) -> torch.Tensor:
            return torch.add(torch.add(x, y, alpha=a), z, alpha=b)

        for test in (test_float, test_int):
            llvm = LLVMCodeGenExecuted()
            interp = SimpleIREvalExecuted()
            x, y, z = [torch.rand(4) for i in range(3)]
            a, b = 1, 2
            test(x, y, z, a, b)
            r = test(x, y, z, a, b)
            xn, yn, zn = [t.numpy() for t in (x, y, z)]
            np.testing.assert_allclose(r.numpy(), xn + yn * a + zn * b)
            # FIXME: interp.elapsed_value() also increments due to simplifier
            assert llvm.elapsed_value() == 1 or interp.elapsed_value() > 1

    def test_loop(self):
        @torch.jit.script
        def test(x: torch.Tensor, y: torch.Tensor, z: int) -> torch.Tensor:
            b = y
            for i in range(0, z):
                a = x + y
                b = b + y
            return b

        llvm = LLVMCodeGenExecuted()
        interp = SimpleIREvalExecuted()
        x, y, z = (torch.zeros(32, 32), torch.ones(32, 32), 4)
        test(x, y, z)
        r = test(x, y, z)
        assert llvm.elapsed_value == 1 or interp.elapsed_value() > 1

    @unittest.skip("no shape inference for aten::slice yet")
    def test_slice(self):
        def easy(x, y):
            a = x[0:512:2]
            b = y[0:512:2]
            return a + b

        traced = torch.jit.trace(easy, (torch.ones(1024, 1024), torch.zeros(1024, 1024)))

        llvm = LLVMCodeGenExecuted()
        interp = SimpleIREvalExecuted()

        a = torch.ones(1024, 1024)
        x = traced(a, a)
        npr = a[0:512:2]
        npr = npr + npr
        np.testing.assert_allclose(npr.numpy(), x.numpy())
        # FIXME: interp.elapsed_value() also increments due to simplifier
        assert llvm.elapsed_value() == 1 or interp.elapsed_value() > 1

    def test_unsqueeze(self, N=256):
        def easy(x, y):
            a = torch.unsqueeze(x, 0)
            b = torch.unsqueeze(y, 0)
            return a + b

        traced = torch.jit.trace(easy, (torch.ones(N, N), torch.zeros(N, N)))

        llvm = LLVMCodeGenExecuted()
        interp = SimpleIREvalExecuted()

        a = torch.rand(N, N)
        x = traced(a, a)
        npr = np.expand_dims(a, 0)
        npr = npr + npr
        np.testing.assert_allclose(npr, x.numpy())
        # FIXME: interp.elapsed_value() also increments due to simplifier
        assert llvm.elapsed_value() == 1 or interp.elapsed_value() > 1

    def _test_softmax(self, device):
        def test_softmax(x, y):
            a = F.softmax(x, dim=0, dtype=torch.float32)
            b = F.softmax(y, dim=0, dtype=torch.float32)
            c = F.softmax(x, dim=1, dtype=torch.float32)
            d = F.softmax(y, dim=1, dtype=torch.float32)
            return a + b + c + d

        def test_softmax_neg_index(x, y):
            a = F.softmax(x, dim=-2, dtype=torch.float32)
            b = F.softmax(y, dim=-2, dtype=torch.float32)
            c = F.softmax(x, dim=-1, dtype=torch.float32)
            d = F.softmax(y, dim=-1, dtype=torch.float32)
            return a + b + c + d

        def test_log_softmax(x, y):
            a = F.log_softmax(x, dim=0, dtype=torch.float32)
            b = F.log_softmax(y, dim=0, dtype=torch.float32)
            c = F.log_softmax(x, dim=1, dtype=torch.float32)
            d = F.log_softmax(y, dim=1, dtype=torch.float32)
            return a + b + c + d

        for test in (test_softmax, test_log_softmax, test_softmax_neg_index):
            old = torch._C._jit_set_texpr_reductions_enabled(True)
            traced = torch.jit.trace(test, (torch.randn(2, 3, device=device), torch.randn(2, 3, device=device)))
            inp = torch.randn(2, 3, device=device)
            res = traced(inp, inp)
            # Use eager mode as reference.
            ref = test(inp, inp)
            np.testing.assert_allclose(ref, res.cpu().numpy(), rtol=1e-06, atol=1e-06)
            torch._C._jit_set_texpr_reductions_enabled(old)

    def test_softmax_cpu(self):
        llvm = LLVMCodeGenExecuted()
        interp = SimpleIREvalExecuted()
        self._test_softmax('cpu')
        # FIXME: interp.elapsed_value() also increments due to simplifier
        assert llvm.elapsed_value() == 1 or interp.elapsed_value() > 1

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @unittest.skip("global allocs are not supported yet.")
    def test_softmax_cuda(self):
        cuda = CudaCodeGenExecuted()
        self._test_softmax('cuda')
        assert cuda.elapsed_value() == 1

    def test_half_gelu(self):
        devices = ["cuda"] if torch.cuda.is_available() else []

        @torch.jit.script
        def bias_gelu(bias, y):
            x = bias + y
            return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

        for device in devices:
            a = torch.rand(1024, dtype=torch.half, device=device)
            b = torch.rand(1024, dtype=torch.half, device=device)
            traced = torch.jit.trace(bias_gelu, (a, b))
            x = warmup_and_run_forward(traced, a, b)
            self.assertLastGraphAllFused()

    def test_exp_pow(self):
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

        @torch.jit.script
        def do_exp(x, y, z):
            return ((x * y) * 2) * torch.pow(z, 2)

        for device in devices:
            x = torch.rand(10, dtype=torch.double, device=device)
            y = torch.rand(10, dtype=torch.double, device=device)
            z = torch.rand(10, dtype=torch.double, device=device)
            traced = torch.jit.trace(do_exp, (x, y, z))
            x = warmup_and_run_forward(traced, x, y, z)
            self.assertLastGraphAllFused()

    def test_transpose(self):
        @torch.jit.script
        def test(x, y, z):
            return x.transpose(0, 1) + y + z
        llvm = LLVMCodeGenExecuted()
        interp = SimpleIREvalExecuted()
        x = torch.rand(4, 5, 2, 3)
        y = torch.rand(5, 4, 2, 3)
        z = torch.rand(5, 4, 2, 3)
        ref = test(x, y, z)
        res = test(x, y, z)
        np.testing.assert_allclose(ref.numpy(), res.numpy())
        # FIXME: interp.elapsed_value() also increments due to simplifier
        assert llvm.elapsed_value() == 1 or interp.elapsed_value() > 1

    def test_sliced_stride(self):
        @torch.jit.script
        def test(x, y, z):
            return x + y + z
        llvm = LLVMCodeGenExecuted()
        interp = SimpleIREvalExecuted()
        x = torch.rand(16, 4, 2, 3)[::2]
        y = torch.rand(8, 4, 2, 3)
        z = torch.rand(8, 4, 2, 3)
        ref = test(x, y, z)
        res = test(x, y, z)
        np.testing.assert_allclose(ref.numpy(), res.numpy())
        # FIXME: interp.elapsed_value() also increments due to simplifier
        assert llvm.elapsed_value() == 1 or interp.elapsed_value() > 1

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @unittest.skip("dynamic shapes are not quite there yet")
    def test_dynamic_shape(self):
        with num_profiled_runs(2):
            @torch.jit.script
            def test(x, y, z):
                return x * y * z
            cuda = CudaCodeGenCreated()
            x, y, z = [torch.rand(4, 8).cuda() for _ in range(3)]
            ref = test(x, y, z)
            _ = test(*[torch.rand(6, 8).cuda() for _ in range(3)])
            res = test(x, y, z)
            np.testing.assert_allclose(ref.cpu().numpy(), res.cpu().numpy())
            assert cuda.elapsed_value() == 1

            # A wild broadcast appears.
            x = torch.rand(4, 8).cuda()
            y = torch.rand(1, 8).cuda()
            z = torch.rand(4, 1).cuda()
            res = test(x, y, z)
            xn, yn, zn = [t.cpu().numpy() for t in (x, y, z)]
            np.testing.assert_allclose(res.cpu().numpy(), xn * yn * zn)
            assert cuda.elapsed_value() == 1

            # Mismatched shapes shouldn't reach codegen.
            x = torch.rand(4, 8).cuda()
            y = torch.rand(4, 8).cuda()
            z = torch.rand(5, 8).cuda()
            try:
                res = test(x, y, z)
            except RuntimeError as e:
                assert "The size of tensor a (4) must match" in e.args[0]
            assert cuda.elapsed_value() == 1

            # Changing a static dimension fails guards.
            # x, y, z = [torch.rand(4, 7).cuda() for _ in range(3)]
            # xn, yn, zn = [t.cpu().numpy() for t in (x, y, z)]
            # res = test(x, y, z)
            # print(test.graph_for(x, y, z))
            # np.testing.assert_allclose(res.cpu().numpy(), xn * yn * zn)
            # assert cuda.elapsed_value() == 1

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_guard_fails(self):
        @torch.jit.script
        def test(x, y, z):
            return x * y * z
        cuda = CudaCodeGenExecuted()
        r1 = test(*[torch.rand(4).cuda() for _ in range(3)])
        assert cuda.elapsed_value() == 0
        r2 = test(*[torch.rand(4).cuda() for _ in range(3)])
        assert cuda.elapsed_value() == 1
        r3 = test(*[torch.rand(4).cuda() for _ in range(3)])
        assert cuda.elapsed_value() == 2
        r4 = test(*[torch.rand(7).cuda() for _ in range(3)])
        assert cuda.elapsed_value() == 2

    def test_bitwise_ops(self):
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

        def run_and(x, y):
            return x & (x & y)

        def run_or(x, y):
            return x & (x | y)

        def run_xor(x, y):
            return x ^ (x ^ y)

        def run_lshift(x, y):
            return x & (x << y)

        def run_rshift(x, y):
            return x & (x >> y)

        fns = {run_and, run_or, run_xor, run_lshift, run_rshift}

        for device in devices:
            for fn in fns:
                a = torch.ones(128, dtype=torch.int32, device=device)
                b = torch.zeros(128, dtype=torch.int32, device=device)
                inp = torch.ones(128, dtype=torch.int32, device=device)
                traced = torch.jit.trace(fn, (inp, inp))
                x = warmup_and_run_forward(traced, a, b)
                self.assertLastGraphAllFused()
                y = fn(a, b)
                np.testing.assert_allclose(x.cpu().numpy(), y.cpu().numpy())

    def test_where(self):
        def run_where(x, y):
            return torch.where(torch.gt(x, y), x, y)

        a = torch.rand(1024, dtype=float)
        b = torch.rand(1024, dtype=float)
        traced = torch.jit.trace(run_where, (torch.zeros(1024), torch.zeros(1024)))
        x = warmup_and_run_forward(traced, a, b)
        self.assertLastGraphAllFused()
        y = run_where(a, b)
        np.testing.assert_allclose(x.numpy(), y.numpy())

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_multi_rand(self):
        def test(x):
            y = torch.rand_like(x)
            return (x + y) - (y - x)
        a = torch.rand(4, device="cuda")
        scripted = torch.jit.script(test)
        cx = CudaCodeGenExecuted()
        out = warmup_and_run_forward(scripted, a)
        self.assertLastGraphAllFused()
        assert torch.allclose(out, 2 * a)
        assert cx.elapsed_value() == 1

    def test_mask(self):
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

        def test(x):
            return x.unsqueeze(1) == 0

        for d in devices:
            x = torch.rand(4, device=d) > 0.5
            scripted = torch.jit.script(test)
            out = warmup_and_run_forward(scripted, x)
            self.assertLastGraphAllFused()
            assert torch.equal(out, test(x))

    def test_simple_add(self):
        val = torch._C._jit_get_te_generate_block_code()
        torch._C._jit_set_te_generate_block_code(True)
        fall_bk = torch._C._jit_texpr_fallback_allowed()
        torch._C._jit_texpr_set_fallback_allowed(True)

        def simple(a, b):
            return torch.add(a, b)

        a = torch.ones(256, 256)
        b = torch.ones(256, 256)
        traced = torch.jit.trace(simple,
                                 (torch.ones(256, 256), torch.ones(256, 256)))
        f = traced(a, b)
        f_test = np.full((256, 256), 2, dtype=float)
        np.testing.assert_allclose(f.numpy(), f_test)
        torch._C._jit_set_te_generate_block_code(val)
        torch._C._jit_texpr_set_fallback_allowed(fall_bk)

    def test_strided_output_preserved(self):
        def foo(a, b):
            return a + b - a

        # smaller, easier to debug example
        x = torch.arange(6)
        x = torch.as_strided(x, (2, 3), (1, 2))
        total = 0
        for i in range(2):
            for j in range(3):
                x[i, j] = total
                total += 1
        foo_script = torch.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)
        out_s = foo_script(x, x)
        out_eager = foo(x, x)
        self.assertEqual(out_s, out_eager)
        self.assertEqual(out_s.stride(), out_eager.stride())
        self.assertLastGraphAllFused()

        # more dims
        N, C, H, W, = 2, 3, 4, 5
        x = torch.rand(N, C, H, W).to(memory_format=torch.channels_last)
        foo_script = torch.jit.script(foo)
        foo_script(x, x)
        foo_script(x, x)
        out_s = foo_script(x, x)
        out_eager = foo(x, x)
        self.assertEqual(out_s, out_eager)
        self.assertEqual(out_s.stride(), out_eager.stride())
        self.assertLastGraphAllFused()

    def test_alias_analysis_module(self):
        class AliasModule(nn.Module):
            def __init__(self):
                super(AliasModule, self).__init__()
                torch.manual_seed(1337)
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                z = z + self.a
                self.b.add_(y)
                w = z + self.a
                z = w + x
                return z
        x = torch.randn(128, 128)

        def getModule(script):
            am = AliasModule()
            if script:
                return torch.jit.script(am)
            return am

        am = getModule(False)
        am_s = getModule(True)
        ref = am(x, x, x)
        test = am_s(x, x, x)
        torch.testing.assert_allclose(ref, test)

        # Now do the aliasing
        am.a = am.b
        ref = am(x, x, x)

        am_s.a = am_s.b
        test = am_s(x, x, x)

        torch.testing.assert_allclose(ref, test)

    def test_alias_analysis_inputs(self):
        class AliasModule(nn.Module):
            def __init__(self):
                super(AliasModule, self).__init__()
                torch.manual_seed(1337)
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                x.add_(y)
                w = z + self.a
                z = w + x
                return z

        def getModule(script):
            am = AliasModule()
            if script:
                return torch.jit.script(am)
            return am
        am = getModule(False)
        am_s = getModule(True)

        torch.manual_seed(1337)
        x = torch.randn(128, 128)
        ref = am(x, x, x)

        torch.manual_seed(1337)
        x = torch.randn(128, 128)
        test = am_s(x, x, x)

        torch.testing.assert_allclose(ref, test)

    def test_alias_analysis_input_and_module(self):
        class AliasModule(nn.Module):
            def __init__(self):
                super(AliasModule, self).__init__()
                torch.manual_seed(1337)
                self.a = torch.randn(128, 128)
                self.b = torch.randn(128, 128)
                self.c = torch.randn(128, 128)

            def forward(self, x, y, z):
                x.add_(y)
                w = z + self.b
                z = w + x
                return z

        def getModule(script):
            am = AliasModule()
            if script:
                return torch.jit.script(am)
            return am
        am = getModule(False)
        am_s = getModule(True)

        torch.manual_seed(1337)
        x = torch.randn(128, 128)
        am.b = x
        ref = am(x, x, x)

        torch.manual_seed(1337)
        x = torch.randn(128, 128)
        am_s.b = x
        test = am_s(x, x, x)

        torch.testing.assert_allclose(ref, test)

    def test_multiple_outputs(self):
        for device in self.devices:
            # A bug reported internally similar to the one reported in #48533
            def foo(a, b, c):
                t_next = c + 1
                t5 = t_next * b
                t6 = torch.unsqueeze(t_next, 1)
                t7 = a * t6
                return (t7, t5, t_next)

            a = torch.rand(20, 20, dtype=torch.float32, device=device)
            b = torch.rand(20 * 29, dtype=torch.float32, device=device).as_strided([20], [29])
            c = torch.ones(20, dtype=torch.int64, device=device)
            traced = torch.jit.trace(foo, (a, b, c))
            ref = foo(a, b, c)
            exp = traced(a, b, c)
            exp = traced(a, b, c)
            self.assertEqual(ref, exp)

if __name__ == '__main__':
    run_tests()
