# Owner(s): ["module: intel"]

import itertools
import math
import sys
import unittest

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyXPU,
    OpDTypes,
    ops,
    precisionOverride,
)
from torch.testing._internal.common_methods_invocations import ops_and_refs
from torch.testing._internal.common_utils import (
    iter_indices,
    NoTest,
    run_tests,
    suppress_warnings,
    TEST_WITH_UBSAN,
    TEST_XPU,
    TestCase,
)

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")

any_common_cpu_xpu_one = OpDTypes.any_common_cpu_cuda_one
_xpu_computation_op_list = [
    "fill",
    "zeros",
    "zeros_like",
    "clone",
    "view_as_real",
    "view_as_complex",
    "view",
    "resize_",
    "resize_as_",
    "add",
    "sub",
    "mul",
    "div",
    "abs",
]
_xpu_tensor_factory_op_list = [
    "as_strided",
    "empty",
    "empty_strided",
]
_xpu_not_test_dtype_op_list = [
    "resize_",  # Skipped by CPU
    "resize_as_",  # Skipped by CPU
    "abs",  # Not aligned dtype
]
_xpu_all_op_list = _xpu_computation_op_list + _xpu_tensor_factory_op_list
_xpu_all_ops = [op for op in ops_and_refs if op.name in _xpu_all_op_list]
_xpu_computation_ops = [
    op for op in ops_and_refs if op.name in _xpu_computation_op_list
]


class TestXpu(TestCase):
    def test_device_behavior(self):
        current_device = torch.xpu.current_device()
        torch.xpu.set_device(current_device)
        self.assertEqual(current_device, torch.xpu.current_device())

    @unittest.skipIf(not TEST_MULTIXPU, "only one GPU detected")
    def test_multi_device_behavior(self):
        current_device = torch.xpu.current_device()
        target_device = (current_device + 1) % torch.xpu.device_count()

        with torch.xpu.device(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())
        self.assertEqual(current_device, torch.xpu.current_device())

        with torch.xpu._DeviceGuard(target_device):
            self.assertEqual(target_device, torch.xpu.current_device())
        self.assertEqual(current_device, torch.xpu.current_device())

    def test_get_device_properties(self):
        current_device = torch.xpu.current_device()
        device_properties = torch.xpu.get_device_properties(current_device)
        self.assertEqual(device_properties, torch.xpu.get_device_properties(None))
        self.assertEqual(device_properties, torch.xpu.get_device_properties())

        device_name = torch.xpu.get_device_name(current_device)
        self.assertEqual(device_name, torch.xpu.get_device_name(None))
        self.assertEqual(device_name, torch.xpu.get_device_name())

        device_capability = torch.xpu.get_device_capability(current_device)
        self.assertTrue(device_capability["max_work_group_size"] > 0)
        self.assertTrue(device_capability["max_num_sub_groups"] > 0)

    def test_wrong_xpu_fork(self):
        stderr = TestCase.runWithPytorchAPIUsageStderr(
            """\
import torch
from torch.multiprocessing import Process
def run(rank):
    torch.xpu.set_device(rank)
if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # it would work fine without the line below
        torch.xpu.set_device(0)
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
"""
        )
        self.assertRegex(stderr, "Cannot re-initialize XPU in forked subprocess.")

    def test_streams(self):
        s0 = torch.xpu.Stream()
        torch.xpu.set_stream(s0)
        s1 = torch.xpu.current_stream()
        self.assertEqual(s0, s1)
        s2 = torch.xpu.Stream()
        self.assertFalse(s0 == s2)
        torch.xpu.set_stream(s2)
        with torch.xpu.stream(s0):
            self.assertEqual(s0, torch.xpu.current_stream())
        self.assertEqual(s2, torch.xpu.current_stream())

    def test_stream_priority(self):
        low, high = torch.xpu.Stream.priority_range()
        s0 = torch.xpu.Stream(device=0, priority=low)

        self.assertEqual(low, s0.priority)
        self.assertEqual(torch.device("xpu:0"), s0.device)

        s1 = torch.xpu.Stream(device=0, priority=high)

        self.assertEqual(high, s1.priority)
        self.assertEqual(torch.device("xpu:0"), s1.device)

    def test_stream_event_repr(self):
        s = torch.xpu.current_stream()
        self.assertTrue("torch.xpu.Stream" in str(s))
        e = torch.xpu.Event()
        self.assertTrue("torch.xpu.Event(uninitialized)" in str(e))
        s.record_event(e)
        self.assertTrue("torch.xpu.Event" in str(e))

    def test_events(self):
        stream = torch.xpu.current_stream()
        event = torch.xpu.Event()
        self.assertTrue(event.query())
        stream.record_event(event)
        event.synchronize()
        self.assertTrue(event.query())

    def test_generator(self):
        torch.manual_seed(2024)
        g_state0 = torch.xpu.get_rng_state()
        torch.manual_seed(1234)
        g_state1 = torch.xpu.get_rng_state()
        self.assertNotEqual(g_state0, g_state1)

        torch.xpu.manual_seed(2024)
        g_state2 = torch.xpu.get_rng_state()
        self.assertEqual(g_state0, g_state2)

        torch.xpu.set_rng_state(g_state1)
        self.assertEqual(g_state1, torch.xpu.get_rng_state())

        torch.manual_seed(1234)
        torch.xpu.set_rng_state(g_state0)
        self.assertEqual(2024, torch.xpu.initial_seed())

    @onlyXPU
    @suppress_warnings
    @ops(_xpu_computation_ops, dtypes=any_common_cpu_xpu_one)
    def test_compare_cpu(self, device, dtype, op):
        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device="cpu")
            return arg

        samples = op.reference_inputs(device, dtype)

        for sample in samples:
            cpu_sample = sample.transform(to_cpu)
            xpu_results = op(sample.input, *sample.args, **sample.kwargs)
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            xpu_results = sample.output_process_fn_grad(xpu_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

            # Lower tolerance because we are running this as a `@slowTest`
            # Don't want the periodic tests to fail frequently
            self.assertEqual(xpu_results, cpu_results, atol=1e-4, rtol=1e-4)

    @onlyXPU
    @ops(_xpu_computation_ops, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, "Test uses undefined behavior")
    def test_non_standard_bool_values(self, device, dtype, op):
        # Test boolean values other than 0x00 and 0x01 (gh-54789)
        def convert_boolean_tensors(x):
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
                return x

            # Map False -> 0 and True -> Random value in [2, 255]
            true_vals = torch.randint(
                2, 255, x.shape, dtype=torch.uint8, device=x.device
            )
            false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
            x_int = torch.where(x, true_vals, false_vals)

            ret = x_int.view(torch.bool)
            self.assertEqual(ret, x)
            return ret

        for sample in op.sample_inputs(device, dtype):
            expect = op(sample.input, *sample.args, **sample.kwargs)

            transformed = sample.transform(convert_boolean_tensors)
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)

            self.assertEqual(expect, actual)


instantiate_device_type_tests(TestXpu, globals(), only_for="xpu")


class TestBasicGEMM(TestCase):
    def _test_addmm_addmv(
        self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False, activation=None
    ):
        dtype = t.dtype
        numpy_dtype = dtype
        if dtype in {torch.bfloat16, torch.half}:
            numpy_dtype = torch.float
        if dtype.is_complex:
            alpha = 0.9 + 0.3j if alpha is None else alpha
            beta = 0.5 + 0.6j if beta is None else beta
        else:
            alpha = 1.2 if alpha is None else alpha
            beta = 0.8 if beta is None else beta
        if activation == "gelu":
            res1 = f(t, m, v, alpha=alpha, beta=beta, use_gelu=True)
        else:
            res1 = f(t, m, v, alpha=alpha, beta=beta)
        res2 = torch.full_like(res1, math.nan)
        if transpose_out:
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
        if activation == "gelu":
            f(t, m, v, alpha=alpha, beta=beta, out=res2, use_gelu=True)
        else:
            f(t, m, v, alpha=alpha, beta=beta, out=res2)
        m.to(numpy_dtype).cpu().numpy()
        v.to(numpy_dtype).cpu().numpy()
        res3 = alpha * (
            m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy()
        )
        if beta != 0:
            res3 += (beta * t).to(numpy_dtype).cpu().numpy()
        if activation == "relu":
            res3 = res3 * (res3 > 0)
        elif activation == "gelu":
            res3_t = torch.from_numpy(res3).to(dtype)
            approximate = "tanh" if t.is_cuda else "none"
            res3_t = torch.nn.functional.gelu(res3_t, approximate=approximate)
            res3 = res3_t.to(numpy_dtype).cpu().numpy()
        else:
            assert activation is None, f"unsupported activation {activation}"
        res3 = torch.from_numpy(res3).to(dtype)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

    def _test_addmm_impl(self, func, activation, device, dtype):
        M = torch.randn(10, 25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        m1 = torch.randn(10, 50, device="cpu", dtype=torch.float32).to(dtype).to(device)
        m2 = torch.randn(50, 25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        self._test_addmm_addmv(func, M, m1, m2, activation=activation)

        # vector-shaped bias and beta=1 result in epilogue fusion in CUDA
        V = torch.randn(25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        self._test_addmm_addmv(func, V, m1, m2, beta=1, activation=activation)

        # Test 0-strided
        M = (
            torch.randn(10, 1, device="cpu", dtype=torch.float32)
            .to(dtype)
            .expand(10, 25)
            .to(device)
        )
        m1 = (
            torch.randn(10, 1, device="cpu", dtype=torch.float32)
            .to(dtype)
            .expand(10, 50)
            .to(device)
        )
        m2 = torch.randn(50, 25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        self._test_addmm_addmv(func, M, m1, m2, activation=activation)

        # Test beta=0, M=nan
        M = (
            torch.full((10, 25), math.nan, device="cpu", dtype=torch.float32)
            .to(dtype)
            .to(device)
        )
        m1 = torch.randn(10, 50, device="cpu", dtype=torch.float32).to(dtype).to(device)
        m2 = torch.randn(50, 25, device="cpu", dtype=torch.float32).to(dtype).to(device)
        self._test_addmm_addmv(func, M, m1, m2, beta=0, activation=activation)

        # Test transpose
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):

            def maybe_transpose(cond, m):
                if not cond:
                    return m
                return m.t().clone(memory_format=torch.contiguous_format).t()

            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            self._test_addmm_addmv(
                func, M, m1, m2, transpose_out=t4, activation=activation
            )

            if t1:
                # use vector V instead of matrix M for epilogue fusion in CUDA (doesn't depend on t1)
                self._test_addmm_addmv(
                    func,
                    V,
                    m1,
                    m2,
                    beta=1,
                    transpose_out=t4,
                    activation=activation,
                )

    @precisionOverride(
        {
            torch.float: 1e-4,
            torch.half: 1e-1,
        }
    )
    @dtypes(torch.float32, torch.half)
    def test_addmm(self, device, dtype):
        self._test_addmm_impl(torch.addmm, None, device, dtype)

    @precisionOverride({torch.bfloat16: 1e-0, torch.half: 1e-3, torch.float: 1e-4})
    @dtypes(torch.bfloat16, torch.half, torch.float)
    def test_addmv(self, device, dtype):
        # have to use torch.randn(...).to(bfloat16) instead of
        # torch.randn(..., dtype=bfloat16). randn does not support
        # bfloat16 yet.
        # "*0.2" to reduce errors for low precision
        ts = [
            0.2 * torch.randn(50, device=device).to(dtype),
            0.2 * torch.randn(1, device=device).to(dtype).expand(50),
        ]
        vs = [
            0.2 * torch.randn(100, device=device).to(dtype),
            0.2
            * torch.ones(1, device=device)
            .to(dtype)
            .expand(100),  # to reduce errors for low precision
        ]
        ms = [
            # 0d
            0.2
            * torch.ones((), device=device)
            .to(dtype)
            .expand(50, 100),  # to reduce errors for low precision
            # 1d
            0.2 * torch.randn((1, 100), device=device).to(dtype).expand(50, 100),
            # this initialization reduces errors for low precision for broadcasted matrices
            # by making sure that intermediate and result values are exactly representable
            # in low precision type
            0.2
            * torch.randint(3, (50, 1), dtype=torch.float, device=device)
            .to(dtype)
            .expand(50, 100),
            # 2d
            0.2 * torch.randn((50, 100), device=device).to(dtype),
            0.2 * torch.randn((100, 50), device=device).to(dtype).t(),
        ]
        for m, v, t in itertools.product(ms, vs, ts):
            self._test_addmm_addmv(torch.addmv, t, m, v)
        # Test beta=0, t=nan
        t = torch.full((50,), math.nan, device=device).to(dtype)
        for m, v in itertools.product(ms, vs):
            self._test_addmm_addmv(torch.addmv, t, m, v, beta=0)

    @dtypes(torch.float32, torch.float64)
    def test_mm(self, device, dtype):
        def _test_mm(n, m, p, dtype, genf):
            # helper function
            def matrixmultiply(mat1, mat2):
                n = mat1.size(0)
                m = mat1.size(1)
                p = mat2.size(1)
                dtype_ = torch.float if dtype == torch.half else dtype
                if dtype == torch.half:
                    mat1 = mat1.float()
                    mat2 = mat2.float()
                res = torch.zeros(n, p, dtype=dtype_, device=device)
                for i, j in iter_indices(res):
                    res[i, j] = sum(mat1[i, k] * mat2[k, j] for k in range(m))
                return res.half() if dtype == torch.half else res

            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 1
            mat1 = genf(n, m)
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 2
            mat1 = genf(m, n).t()
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # test with zero stride
            mat1 = genf(n, m)
            mat2 = genf(m, 1).expand(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

    @dtypes(
        torch.half,
        torch.float32,
    )
    def test_mm(self, device, dtype):
        def _test_mm(n, m, p, dtype, genf):
            # helper function
            def matrixmultiply(mat1, mat2):
                n = mat1.size(0)
                m = mat1.size(1)
                p = mat2.size(1)
                dtype_ = torch.float if dtype == torch.half else dtype
                if dtype == torch.half:
                    mat1 = mat1.float()
                    mat2 = mat2.float()
                res = torch.zeros(n, p, dtype=dtype_, device=device)
                for i, j in iter_indices(res):
                    res[i, j] = sum(mat1[i, k] * mat2[k, j] for k in range(m))
                return res.half() if dtype == torch.half else res

            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 1
            mat1 = genf(n, m)
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 2
            mat1 = genf(m, n).t()
            mat2 = genf(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # test with zero stride
            mat1 = genf(n, m)
            mat2 = genf(m, 1).expand(m, p)
            res = torch.mm(mat1, mat2)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # contiguous case
            mat1 = genf(n, m)
            mat2 = genf(m, p)
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

            # explicitly exercise the _out variant in torch.mm().
            # non contiguous case 3
            mat1 = genf(m, n).t()
            mat2 = genf(p, m).t()
            res = genf(n, p)
            torch.mm(mat1, mat2, out=res)

            res2 = matrixmultiply(mat1, mat2)
            self.assertEqual(res, res2)

        def genf_int(x, y):
            return torch.randint(0, 100, (x, y), dtype=dtype, device=device)

        def genf_bfloat(x, y):
            return torch.randn(x, y, dtype=torch.float32, device=device).to(dtype) * 0.1

        def genf_float(x, y):
            return torch.randn(x, y, dtype=dtype, device=device)

        def genf_Half(x, y):
            return torch.randn(x, y, dtype=dtype, device=device)

        for n, m, p in [(20, 10, 15), (15, 20, 10), (25, 18, 10)]:
            if (dtype == torch.int32) or (dtype == torch.int64):
                genf = genf_int
            elif dtype == torch.bfloat16:
                genf = genf_bfloat
            elif dtype == torch.half:
                genf = genf_Half
            else:
                genf = genf_float

            _test_mm(n, m, p, dtype, genf)

    @precisionOverride({torch.half: 0.05, torch.bfloat16: 0.05})
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    def test_bmm(self, device, dtype):
        batch_sizes = [1, 10]
        M, N, O = 23, 15, 12
        numpy_dtype = dtype if dtype != torch.bfloat16 else torch.float32

        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_inputs(num_batches):
            # transposed tensors
            for perm1, perm2 in itertools.product(
                itertools.permutations((0, 1, 2)), repeat=2
            ):
                b1 = make_tensor(
                    (num_batches, M, N), dtype=dtype, device=device, low=-0.1, high=0.1
                )
                b2 = make_tensor(
                    (num_batches, N, O), dtype=dtype, device=device, low=-0.1, high=0.1
                )
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                yield b1, b2
            # broadcasting tensors
            for b1, b2, b3, b4, b5, b6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if b1 else 1, M if b2 else 1, N if b3 else 1)
                shape2 = (num_batches if b4 else 1, N if b5 else 1, O if b6 else 1)
                b1 = make_tensor(
                    shape1, dtype=dtype, device=device, low=-0.1, high=0.1
                ).expand(num_batches, M, N)
                b2 = make_tensor(
                    shape2, dtype=dtype, device=device, low=-0.1, high=0.1
                ).expand(num_batches, N, O)
                yield b1, b2
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = torch.randn(shape1, dtype=dtype, device=device)
                b2 = torch.randn(shape2, dtype=dtype, device=device)
                yield b1, b2

        for num_batches in batch_sizes:
            for (b1, b2), perm3 in itertools.product(
                generate_inputs(num_batches), itertools.permutations((0, 1, 2))
            ):
                res1 = torch.bmm(b1, b2)
                res2 = (
                    torch.full(
                        (num_batches, M, O), math.nan, dtype=dtype, device=device
                    )
                    .permute(perm3)
                    .contiguous()
                    .permute(invert_perm(perm3))
                )
                torch.bmm(b1, b2, out=res2)
                expect = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype)
                self.assertEqual(expect, res1)
                self.assertEqual(expect, res2)

                if self.device_type == "cuda":
                    # check that mixed arguments are rejected
                    self.assertRaises(RuntimeError, lambda: torch.bmm(b1, b2.cpu()))
                    self.assertRaises(RuntimeError, lambda: torch.bmm(b1.cpu(), b2))
                    self.assertRaises(
                        RuntimeError, lambda: torch.bmm(b1, b2, out=res2.cpu())
                    )

    def _test_addbmm_baddbmm(self, func, b1, b2, ref, out_tensor):
        getattr(out_tensor, func + "_")(b1, b2)
        self.assertEqual(out_tensor, ref)
        res3 = out_tensor.clone()

        with self.assertWarnsOnceRegex(
            UserWarning, f"This overload of {func}_ is deprecated"
        ):
            getattr(out_tensor, func + "_")(1, b1, b2)
        self.assertEqual(out_tensor, ref * 2),
        getattr(res3, func + "_")(b1, b2, beta=1)
        self.assertEqual(out_tensor, res3)

        with self.assertWarnsOnceRegex(
            UserWarning, f"This overload of {func}_ is deprecated"
        ):
            getattr(out_tensor, func + "_")(1.0, 0.5, b1, b2)
        self.assertEqual(out_tensor, ref * 2.5)
        getattr(res3, func + "_")(b1, b2, beta=1.0, alpha=0.5)
        self.assertEqual(out_tensor, res3)

        with self.assertWarnsOnceRegex(
            UserWarning, f"This overload of {func} is deprecated"
        ):
            self.assertEqual(out_tensor, getattr(torch, func)(1, out_tensor, 0, b1, b2))

        res4 = getattr(torch, func)(out_tensor, b1, b2, beta=1, alpha=0.5)
        self.assertEqual(res4, ref * 3),

        nan = torch.full_like(out_tensor, math.nan)
        res5 = getattr(torch, func)(nan, b1, b2, beta=0, alpha=1)
        self.assertEqual(res5, ref)

        if b1.is_complex():
            res6 = getattr(torch, func)(out_tensor, b1, b2, beta=0.1j, alpha=0.5j)
            self.assertEqual(res6, out_tensor * 0.1j + 0.5j * ref)
        else:
            res6 = getattr(torch, func)(out_tensor, b1, b2, beta=0.1, alpha=0.5)
            self.assertEqual(res6, out_tensor * 0.1 + 0.5 * ref)

        res7 = torch.full_like(out_tensor, math.nan)
        getattr(torch, func)(nan, b1, b2, beta=0, out=res7)
        self.assertEqual(res7, ref)

    @precisionOverride({torch.half: 0.05, torch.bfloat16: 0.05})
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    def test_addbmm(self, device, dtype):
        num_batches = 2
        M, N, O = 16, 17, 18

        is_supported = True

        if not is_supported:
            b1 = make_tensor(
                (num_batches, M, N), dtype=dtype, device=device, low=-1, high=1
            )
            b2 = make_tensor(
                (num_batches, N, O), dtype=dtype, device=device, low=-1, high=1
            )
            t = make_tensor((M, O), dtype=dtype, device=device, low=-1, high=1)
            self.assertRaisesRegex(
                RuntimeError,
                "type|Type|not implemented|CUBLAS_STATUS_NOT_SUPPORTED",
                lambda: torch.addbmm(t, b1, b2),
            )
            return

        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_tensor():
            numpy_dtype = dtype if dtype != torch.bfloat16 else torch.float32
            # transposed tensors
            for perm1, perm2 in itertools.product(
                itertools.permutations((0, 1, 2)), repeat=2
            ):
                for perm3 in itertools.permutations((0, 1)):
                    b1 = (
                        make_tensor(
                            (num_batches, M, N),
                            dtype=dtype,
                            device=device,
                            low=-1,
                            high=1,
                        )
                        * 0.1
                    )
                    b2 = (
                        make_tensor(
                            (num_batches, N, O),
                            dtype=dtype,
                            device=device,
                            low=-1,
                            high=1,
                        )
                        * 0.1
                    )
                    b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                    b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                    ref = (
                        torch.from_numpy(
                            b1.to(numpy_dtype).cpu().numpy()
                            @ b2.to(numpy_dtype).cpu().numpy()
                        )
                        .to(device=device, dtype=dtype)
                        .sum(0)
                    )
                    out_tensor = (
                        torch.zeros_like(ref).permute(perm3).contiguous().permute(perm3)
                    )
                    yield b1, b2, ref, out_tensor
            # broadcasting tensors
            for s1, s2, s3, s4, s5, s6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if s1 else 1, M if s2 else 1, N if s3 else 1)
                shape2 = (num_batches if s4 else 1, N if s5 else 1, O if s6 else 1)
                b1 = (
                    make_tensor(
                        shape1, dtype=dtype, device=device, low=-1, high=1
                    ).expand(num_batches, M, N)
                    * 0.1
                )
                b2 = (
                    make_tensor(
                        shape2, dtype=dtype, device=device, low=-1, high=1
                    ).expand(num_batches, N, O)
                    * 0.1
                )
                ref = (
                    torch.from_numpy(
                        b1.to(numpy_dtype).cpu().numpy()
                        @ b2.to(numpy_dtype).cpu().numpy()
                    )
                    .to(device=device, dtype=dtype)
                    .sum(0)
                )
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = (
                    make_tensor(shape1, dtype=dtype, device=device, low=-1, high=1)
                    * 0.1
                )
                b2 = (
                    make_tensor(shape2, dtype=dtype, device=device, low=-1, high=1)
                    * 0.1
                )
                ref = (
                    torch.from_numpy(
                        b1.to(numpy_dtype).cpu().numpy()
                        @ b2.to(numpy_dtype).cpu().numpy()
                    )
                    .to(device=device, dtype=dtype)
                    .sum(0)
                )
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor

        for b1, b2, ref, out_tensor in generate_tensor():
            self._test_addbmm_baddbmm("addbmm", b1, b2, ref, out_tensor)

    @precisionOverride({torch.half: 0.1, torch.bfloat16: 0.5})
    @dtypes(torch.float32, torch.bfloat16, torch.half)
    def test_baddbmm(self, device, dtype):
        num_batches = 10
        M, N, O = 12, 8, 50

        def invert_perm(p):
            d = {x: i for i, x in enumerate(p)}
            return (d[0], d[1], d[2])

        def generate_tensor():
            numpy_dtype = (
                dtype if dtype not in [torch.bfloat16, torch.half] else torch.float32
            )
            # transposed tensors
            for perm1, perm2, perm3 in itertools.product(
                itertools.permutations((0, 1, 2)), repeat=3
            ):
                b1 = make_tensor(
                    (num_batches, M, N), dtype=dtype, device=device, low=-1, high=1
                )
                b2 = make_tensor(
                    (num_batches, N, O), dtype=dtype, device=device, low=-1, high=1
                )
                b1 = b1.permute(perm1).contiguous().permute(invert_perm(perm1))
                b2 = b2.permute(perm2).contiguous().permute(invert_perm(perm2))
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype)
                out_tensor = torch.zeros_like(ref)
                out_tensor = (
                    out_tensor.permute(perm3).contiguous().permute(invert_perm(perm3))
                )
                yield b1, b2, ref, out_tensor
            # broadcasting tensors
            for s1, s2, s3, s4, s5, s6 in itertools.product((True, False), repeat=6):
                shape1 = (num_batches if s1 else 1, M if s2 else 1, N if s3 else 1)
                shape2 = (num_batches if s4 else 1, N if s5 else 1, O if s6 else 1)
                b1 = make_tensor(
                    shape1, dtype=dtype, device=device, low=-1, high=1
                ).expand(num_batches, M, N)
                b2 = make_tensor(
                    shape2, dtype=dtype, device=device, low=-1, high=1
                ).expand(num_batches, N, O)
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor
            # zero-sized tensors
            for z1, z2, z3, z4 in itertools.product((True, False), repeat=4):
                shape1 = (num_batches if z1 else 0, M if z2 else 0, N if z3 else 0)
                shape2 = (num_batches if z1 else 0, N if z3 else 0, O if z4 else 0)
                b1 = make_tensor(shape1, dtype=dtype, device=device, low=-2, high=2)
                b2 = make_tensor(shape2, dtype=dtype, device=device, low=-2, high=2)
                ref = torch.from_numpy(
                    b1.to(numpy_dtype).cpu().numpy() @ b2.to(numpy_dtype).cpu().numpy()
                ).to(device=device, dtype=dtype)
                out_tensor = torch.zeros_like(ref)
                yield b1, b2, ref, out_tensor

        for b1, b2, ref, out_tensor in generate_tensor():
            self._test_addbmm_baddbmm("baddbmm", b1, b2, ref, out_tensor)

    def test_tensordot(self, device):
        a = torch.arange(60.0, device=device).reshape(3, 4, 5)
        b = torch.arange(24.0, device=device).reshape(4, 3, 2)
        c = torch.tensordot(a, b, dims=([1, 0], [0, 1])).cpu()
        cn = torch.from_numpy(
            np.tensordot(a.cpu().numpy(), b.cpu().numpy(), axes=([1, 0], [0, 1]))
        )
        self.assertEqual(c, cn)

        cout = torch.zeros((5, 2), device=device)
        torch.tensordot(a, b, dims=([1, 0], [0, 1]), out=cout).cpu()
        self.assertEqual(c, cout)

        a = torch.randn(2, 3, 4, 5, device=device)
        b = torch.randn(4, 5, 6, 7, device=device)
        c = torch.tensordot(a, b, dims=2).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(), axes=2))

        with self.assertRaisesRegex(RuntimeError, "expects dims >= 0"):
            torch.tensordot(a, b, dims=-1)

        self.assertEqual(c, cn)
        c = torch.tensordot(a, b).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy()))
        self.assertEqual(c, cn)

        a = torch.tensordot(torch.tensor(0.0), torch.tensor(0.0), 0)
        an = torch.from_numpy(
            np.tensordot(
                np.zeros((), dtype=np.float32), np.zeros((), dtype=np.float32), 0
            )
        )
        self.assertEqual(a, an)


instantiate_device_type_tests(TestBasicGEMM, globals(), only_for="xpu")

if __name__ == "__main__":
    run_tests()
