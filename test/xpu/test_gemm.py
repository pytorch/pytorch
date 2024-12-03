# Owner(s): ["module: intel"]

import itertools
import math
import random
from functools import partial
from itertools import product

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    precisionOverride,
)
from torch.testing._internal.common_utils import iter_indices, run_tests, TestCase


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

    @precisionOverride({torch.float: 1e-4, torch.double: 1e-6})
    @dtypes(torch.float32, torch.half, torch.double)
    def test_addmm(self, device, dtype):
        self._test_addmm_impl(torch.addmm, None, device, dtype)

    @precisionOverride({torch.bfloat16: 1e-0, torch.half: 1e-3, torch.float: 1e-4})
    @dtypes(torch.bfloat16, torch.half, torch.float, torch.double)
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

    @dtypes(
        torch.half,
        torch.float32,
        torch.float64,
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
    @dtypes(torch.float32, torch.bfloat16, torch.half, torch.float64)
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
    @dtypes(torch.float64, torch.float32, torch.bfloat16, torch.half)
    def test_addbmm(self, device, dtype):
        num_batches = 2
        M, N, O = 16, 17, 18

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

    @precisionOverride({torch.half: 0.1, torch.bfloat16: 0.5, torch.float64: 1e-6})
    @dtypes(torch.float64, torch.float32, torch.bfloat16, torch.half)
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

    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 1e-4})
    def test_1_sized_with_0_strided(self, device, dtype):
        a = make_tensor((8, 1, 64), dtype=dtype, device=device)
        a_strided = torch.as_strided(a, size=[8, 1, 64], stride=[64, 0, 1])
        b = make_tensor((8, 64, 512), dtype=dtype, device=device)
        b_strided = torch.as_strided(b, size=[8, 64, 512], stride=[64, 1, 512])
        res = torch.bmm(a_strided, b_strided)
        expect = torch.from_numpy(a_strided.cpu().numpy() @ b_strided.cpu().numpy()).to(
            device=device, dtype=dtype
        )
        self.assertEqual(expect, res)

    def _select_broadcastable_dims(self, dims_full=None):
        # select full dimensionality
        if dims_full is None:
            dims_full = []
            ndims = random.randint(1, 4)
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # select actual dimensions for ops:
        # larger: full ndims, individual sizes may be reduced
        # smaller: possibly reduced ndims, sizes may be reduced
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # no reduced singleton dimension
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # larger may have reduced singleton dimension
                ds = dims_full[i]
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # smaller may have reduced singleton dimension
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        return (dims_small, dims_large, dims_full)

    def test_broadcast_fused_matmul(self, device):
        fns = ["baddbmm", "addbmm", "addmm", "addmv", "addr"]

        for fn in fns:
            batch_dim = random.randint(1, 8)
            n_dim = random.randint(1, 8)
            m_dim = random.randint(1, 8)
            p_dim = random.randint(1, 8)

            def dims_full_for_fn():
                if fn == "baddbmm":
                    return (
                        [batch_dim, n_dim, p_dim],
                        [batch_dim, n_dim, m_dim],
                        [batch_dim, m_dim, p_dim],
                    )
                elif fn == "addbmm":
                    return (
                        [n_dim, p_dim],
                        [batch_dim, n_dim, m_dim],
                        [batch_dim, m_dim, p_dim],
                    )
                elif fn == "addmm":
                    return ([n_dim, p_dim], [n_dim, m_dim], [m_dim, p_dim])
                elif fn == "addmv":
                    return ([n_dim], [n_dim, m_dim], [m_dim])
                elif fn == "addr":
                    return ([n_dim, m_dim], [n_dim], [m_dim])
                else:
                    raise AssertionError("unknown function")

            (t0_dims_full, t1_dims, t2_dims) = dims_full_for_fn()
            (t0_dims_small, _, _) = self._select_broadcastable_dims(t0_dims_full)

            t0_small = torch.randn(*t0_dims_small, device=device).float()
            t1 = torch.randn(*t1_dims, device=device).float()
            t2 = torch.randn(*t2_dims, device=device).float()

            t0_full = t0_small.expand(*t0_dims_full).to(device)

            fntorch = getattr(torch, fn)
            r0 = fntorch(t0_small, t1, t2)
            r1 = fntorch(t0_full, t1, t2)
            self.assertEqual(r0, r1)

    @dtypes(torch.float32, torch.float64)
    def test_strided_mm_bmm(self, device, dtype):
        # Tests strided view case with stride smaller than corresponding dimension size
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype, device=device)
        new_shape = [2, 2, 2]
        new_stride = [3, 1, 1]
        sx = torch.as_strided(x, size=new_shape, stride=new_stride)

        torch_fn = lambda x: torch.bmm(x, x)  # noqa: E731
        np_fn = lambda x: np.matmul(x, x)  # noqa: E731
        self.compare_with_numpy(torch_fn, np_fn, sx)

        torch_fn = lambda x: torch.mm(x, x)  # noqa: E731
        self.compare_with_numpy(torch_fn, np_fn, sx[0])

    def test_mm_empty_inputs_mixed_dtype_errors(self, device):
        a = torch.randint(0, 10, [1, 10], dtype=torch.int16, device=device)
        b = torch.randn(10, 20, dtype=torch.float32, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "expected .* and .* to have the same dtype, but got:"
        ):
            torch.mm(a, b)

    def test_matmul_45724(self, device):
        # https://github.com/pytorch/pytorch/issues/45724
        a = torch.rand(65537, 22, 64, device=device, dtype=torch.half)
        b = torch.rand(65537, 64, 22, device=device, dtype=torch.half)
        c = torch.full((65537, 22, 22), math.nan, dtype=torch.half, device=device)
        cpu_result = torch.matmul(a.cpu().float(), b.cpu().float()).half()
        torch.matmul(a, b, out=c)
        self.assertEqual(c, cpu_result)

    @dtypes(
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    )
    def test_baddbmm_input_dtypes_compatibility(self, device, dtype):
        batch1 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
        batch2 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
        input_tensor = torch.rand((1, 2, 2), device=device).to(dtype)
        if dtype != torch.float32:
            with self.assertRaisesRegex(RuntimeError, "Input dtypes must be the same"):
                y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0)
        else:
            out = torch.randn((1, 2, 2), dtype=dtype, device=device).fill_(torch.nan)
            y_ref = torch.bmm(batch1, batch2)
            y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0, out=out)
            self.assertEqual(out, y_ref)

    @dtypes(torch.float)
    def test_baddbmm_nan_input_with_zero_beta(self, device, dtype):
        for shape in [[3, 2, 2], [2, 20, 20]]:
            mat1, mat2 = (
                torch.randn(shape, dtype=dtype, device=device) for _ in range(2)
            )
            inputs = [
                torch.randn(shape, dtype=dtype, device=device),
                torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan),
            ]
            outs = [
                None,
                torch.randn(shape, dtype=dtype, device=device),
                torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan),
            ]
            options = itertools.product(inputs, outs)
            for input, out in options:
                y_ref = torch.bmm(mat1, mat2)
                y = torch.baddbmm(input, mat1, mat2, beta=0.0, out=out)
                self.assertEqual(y_ref, y)

    @precisionOverride({torch.double: 1e-6})
    @dtypes(torch.float, torch.double)
    def test_addmm_sizes(self, device, dtype):
        for m in [0, 1, 25]:
            for n in [0, 1, 10]:
                for k in [0, 1, 8]:
                    M = torch.randn(n, m, device=device).to(dtype)
                    m1 = torch.randn(n, k, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    self._test_addmm_addmv(torch.addmm, M, m1, m2)

                    m1 = torch.randn(n, k + 1, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    self.assertRaisesRegex(
                        RuntimeError,
                        f"{n}x{k + 1}.*{k}x{m}",
                        lambda: torch.addmm(M, m1, m2),
                    )
                    self.assertRaisesRegex(
                        RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.mm(m1, m2)
                    )

    @precisionOverride(
        {
            torch.double: 1e-6,
            torch.float: 1e-4,
            torch.bfloat16: 5e-2,
            torch.half: 5e-2,
            torch.cfloat: 1e-4,
            torch.cdouble: 1e-8,
        }
    )
    @dtypes(torch.double, torch.float32, torch.bfloat16, torch.half)
    def test_addmm_gelu(self, device, dtype):
        self._test_addmm_impl(torch._addmm_activation, "gelu", device, dtype)

    @precisionOverride(
        {
            torch.double: 1e-6,
            torch.float: 1e-4,
            torch.bfloat16: 5e-2,
            torch.half: 5e-2,
            torch.cfloat: 1e-4,
            torch.cdouble: 1e-8,
        }
    )
    @dtypes(torch.double, torch.float32, torch.bfloat16, torch.half)
    def test_addmm_relu(self, device, dtype):
        self._test_addmm_impl(torch._addmm_activation, "relu", device, dtype)

    @dtypes(torch.float, torch.bfloat16, torch.half, torch.double)
    def test_addmv_rowmajor_colmajor_incx_incy_lda(self, device, dtype):
        # tests (o, s)*(s).  o is output size, s is summed size.
        o = 5
        s = 3
        a_data = torch.arange(1, o * s + 1, device=device, dtype=dtype).view(o, s)
        x_data = torch.arange(1, s + 1, 1, device=device, dtype=dtype)
        y_data = torch.ones(o, device=device, dtype=dtype)
        control = torch.tensor(
            [15.0, 33.0, 51.0, 69.0, 87.0], device=device, dtype=dtype
        )

        def _test(row_major, incx, incy, lda_tail):
            if row_major:
                a_storage = torch.full(
                    (o, s + lda_tail), float("nan"), device=device, dtype=dtype
                )
            else:
                a_storage = torch.full(
                    (s, o + lda_tail), float("nan"), device=device, dtype=dtype
                ).permute(1, 0)
            a = a_storage[:o, :s].copy_(a_data)

            x_storage = torch.full((s, incx), float("nan"), device=device, dtype=dtype)
            x = x_storage[:, 0].copy_(x_data)

            y_storage = torch.full((o, incy), float("nan"), device=device, dtype=dtype)
            y = y_storage[:, 0].copy_(y_data)

            self._test_addmm_addmv(torch.addmv, y, a, x)

        for row_major, incx, incy, lda_tail in itertools.product(
            (False, True), (1, 2), (1, 2), (0, 1)
        ):
            _test(row_major, incx, incy, lda_tail)

    @precisionOverride(
        {
            torch.double: 1e-8,
            torch.float: 1e-4,
            torch.bfloat16: 0.6,
            torch.half: 1e-1,
            torch.cfloat: 1e-4,
            torch.cdouble: 1e-8,
        }
    )
    @dtypes(torch.double, torch.bfloat16, torch.half, torch.float32)
    def test_corner_cases_of_cublasltmatmul(self, device, dtype):
        # common case
        M = torch.randn(128, device=device).to(dtype)
        m1 = torch.randn(2048, 2400, device=device).to(dtype)
        m2 = torch.randn(128, 2400, device=device).to(dtype)
        torch.nn.functional.linear(m1, m2, M)
        # Ntrans_B has ld >> rows
        m1 = torch.rand([128, 2400]).to(dtype).to(device).t()
        m2 = torch.rand([2048, 25272]).to(dtype).to(device).t()[21940:24340]
        M = torch.rand([128]).to(dtype).to(device)
        torch.addmm(M, m2.t(), m1)
        # trans_A has ld >> rows
        m1 = torch.rand([128, 25272]).to(dtype).to(device)[:, 21940:24340].t()
        m2 = torch.randn(2048, 2400, device=device).to(dtype)
        M = torch.rand([128]).to(dtype).to(device)
        torch.addmm(M, m2, m1)
        # large tensor dim > 65535
        M = torch.randn(16, device=device).to(dtype)
        m1 = torch.randn(32, 131071, device=device).to(dtype)
        m2 = torch.randn(16, 131071, device=device).to(dtype)
        torch.nn.functional.linear(m1, m2, M)

    def test_blas_empty(self, device):
        def fn(torchfn, *args, test_out=False, **kwargs):
            def call_torch_fn(*args, **kwargs):
                return torchfn(
                    *tuple(
                        torch.randn(shape, device=device)
                        if isinstance(shape, tuple)
                        else shape
                        for shape in args
                    ),
                    **kwargs,
                )

            result = call_torch_fn(*args, **kwargs)
            if not test_out:
                return result
            else:
                out = torch.full_like(result, math.nan)
                out1 = call_torch_fn(*args, **kwargs, out=out)
                return out

        # mm, addmm
        self.assertEqual((0, 0), fn(torch.mm, (0, 0), (0, 0)).shape)
        self.assertEqual((0, 5), fn(torch.mm, (0, 0), (0, 5)).shape)
        self.assertEqual((5, 0), fn(torch.mm, (5, 0), (0, 0)).shape)
        self.assertEqual((3, 0), fn(torch.mm, (3, 2), (2, 0)).shape)
        self.assertEqual(
            torch.zeros((5, 6), device=device), fn(torch.mm, (5, 0), (0, 6))
        )
        self.assertEqual(
            torch.zeros((5, 6), device=device),
            fn(torch.mm, (5, 0), (0, 6), test_out=True),
        )

        self.assertEqual((0, 0), fn(torch.addmm, (0, 0), (0, 0), (0, 0)).shape)
        self.assertEqual((0, 1), fn(torch.addmm, (1,), (0, 17), (17, 1)).shape)
        t = torch.randn((5, 6), device=device)
        self.assertEqual(t, fn(torch.addmm, t, (5, 0), (0, 6)))
        self.assertEqual(t, fn(torch.addmm, t, (5, 0), (0, 6), test_out=True))

        # mv, addmv
        self.assertEqual((0,), fn(torch.mv, (0, 0), (0,)).shape)
        self.assertEqual((0,), fn(torch.mv, (0, 2), (2,)).shape)
        self.assertEqual(torch.zeros((3,), device=device), fn(torch.mv, (3, 0), (0,)))
        self.assertEqual(
            torch.zeros((3,), device=device), fn(torch.mv, (3, 0), (0,), test_out=True)
        )

        self.assertEqual((0,), fn(torch.addmv, (0,), (0, 0), (0,)).shape)
        t = torch.randn((3,), device=device)
        self.assertEqual(t, fn(torch.addmv, t, (3, 0), (0,)))
        self.assertEqual(t, fn(torch.addmv, t, (3, 0), (0,), test_out=True))

        # bmm, baddbmm
        self.assertEqual((0, 0, 0), fn(torch.bmm, (0, 0, 0), (0, 0, 0)).shape)
        self.assertEqual((3, 0, 5), fn(torch.bmm, (3, 0, 0), (3, 0, 5)).shape)
        self.assertEqual((0, 5, 6), fn(torch.bmm, (0, 5, 0), (0, 0, 6)).shape)
        self.assertEqual(
            torch.zeros((3, 5, 6), device=device), fn(torch.bmm, (3, 5, 0), (3, 0, 6))
        )
        self.assertEqual(
            torch.zeros((3, 5, 6), device=device),
            fn(torch.bmm, (3, 5, 0), (3, 0, 6), test_out=True),
        )

        self.assertEqual(
            (0, 0, 0), fn(torch.baddbmm, (0, 0, 0), (0, 0, 0), (0, 0, 0)).shape
        )
        self.assertEqual(
            (3, 0, 5), fn(torch.baddbmm, (3, 0, 5), (3, 0, 0), (3, 0, 5)).shape
        )
        self.assertEqual(
            (0, 5, 6), fn(torch.baddbmm, (0, 5, 6), (0, 5, 0), (0, 0, 6)).shape
        )
        self.assertEqual(
            (3, 5, 6), fn(torch.baddbmm, (3, 5, 6), (3, 5, 0), (3, 0, 6)).shape
        )
        c = torch.arange(30, dtype=torch.float32, device=device).reshape(3, 2, 5)
        self.assertEqual(
            -2 * c, fn(torch.baddbmm, c, (3, 2, 0), (3, 0, 5), beta=-2)
        )  # Issue #33467
        self.assertEqual(
            -2 * c, fn(torch.baddbmm, c, (3, 2, 0), (3, 0, 5), beta=-2, test_out=True)
        )  # Issue #33467

        # addbmm
        self.assertEqual((0, 0), fn(torch.addbmm, (0, 0), (0, 0, 0), (0, 0, 0)).shape)
        self.assertEqual((0, 5), fn(torch.addbmm, (0, 5), (3, 0, 0), (3, 0, 5)).shape)
        t = torch.randn((5, 6), device=device)
        self.assertEqual(t, fn(torch.addbmm, t, (0, 5, 0), (0, 0, 6)))
        self.assertEqual(t, fn(torch.addbmm, t, (0, 5, 0), (0, 0, 6), test_out=True))

        # matmul
        self.assertEqual(torch.tensor(0.0, device=device), fn(torch.matmul, (0,), (0,)))
        self.assertEqual(
            torch.tensor(0.0, device=device),
            fn(torch.matmul, (0,), (0,), test_out=True),
        )
        self.assertEqual((0, 0), fn(torch.matmul, (0, 0), (0, 0)).shape)
        self.assertEqual((0, 0, 0), fn(torch.matmul, (0, 0, 0), (0, 0, 0)).shape)
        self.assertEqual((5, 0, 0), fn(torch.matmul, (5, 0, 0), (5, 0, 0)).shape)
        self.assertEqual(
            torch.zeros((5, 3, 4), device=device),
            fn(torch.matmul, (5, 3, 0), (5, 0, 4)),
        )
        self.assertEqual(
            torch.zeros((5, 3, 4), device=device),
            fn(torch.matmul, (5, 3, 0), (5, 0, 4), test_out=True),
        )

        # dot
        self.assertEqual(torch.tensor(0.0, device=device), fn(torch.dot, (0,), (0,)))
        self.assertEqual(
            torch.tensor(0.0, device=device), fn(torch.dot, (0,), (0,), test_out=True)
        )

    def test_large_bmm_backward(self, device):
        A = torch.randn([1024, 2, 1024], device=device).mT.contiguous().mT
        B = torch.randn([1, 1024, 65536], device=device, requires_grad=True)
        G = torch.randn([1024, 2, 65536], device=device)

        # Should not create an intermediary tensor of size [1024, 1024, 65536] (256GB of memory) and OOM
        (A @ B).backward(G)

    def test_large_bmm_mm_backward(self, device):
        A = torch.randn([1024, 2, 1024], device=device).mT.contiguous().mT
        B = torch.randn([1024, 65536], device=device, requires_grad=True)
        G = torch.randn([1024, 2, 65536], device=device)

        # Should not create an intermediary tensor of size [1024, 1024, 65536] (256GB of memory) and OOM
        (A @ B).backward(G)

    def check_single_matmul(self, x, y):
        def assertEqual(answer, expected):
            if x.dtype.is_floating_point or x.dtype.is_complex:
                k = max(x.shape[-1], 1)  # Scale the atol with the size of the matrix
                self.assertEqual(
                    answer,
                    expected,
                    msg=f"{x.shape} x {y.shape} = {answer.shape}",
                    atol=k * 5e-5,
                    rtol=1e-4,
                )
            else:
                self.assertEqual(
                    answer, expected, msg=f"{x.shape} x {y.shape} = {answer.shape}"
                )

        # test x @ y
        expected = np.matmul(x.cpu(), y.cpu())
        ans = torch.matmul(x, y)
        self.assertTrue(ans.is_contiguous())
        assertEqual(ans, expected)

        # test out
        out = torch.empty_like(ans)
        ans = torch.matmul(x, y, out=out)
        self.assertIs(ans, out)
        self.assertTrue(ans.is_contiguous())
        assertEqual(ans, expected)

    def gen_sizes_matmul(self, x_dim, y_dim=4, matrix_size=4, batch_size=3):
        """
        Generates sequences of tuples (x, y) of with size(x) = x_dim and
        size(y) <= y_dim that are compatible wrt. matmul
        """
        assert x_dim >= 1
        assert y_dim >= 2
        x = x_dim
        for y in range(1, y_dim + 1):
            for batch, mn in product(
                product(range(batch_size), repeat=max(x - 2, y - 2, 0)),
                product(range(matrix_size), repeat=min(y, 2)),
            ):
                if x == 1:
                    size_x = mn[:1]
                    size_y = batch + mn
                    yield size_x, size_y
                else:
                    for k in range(matrix_size):
                        size_x = (k,) + mn[:1]
                        if x > 2:
                            size_x = batch[-(x - 2) :] + size_x
                        size_y = mn
                        if y > 2:
                            size_y = batch[-(y - 2) :] + size_y
                        yield size_x, size_y

    @dtypes(torch.float)
    def test_matmul_small_brute_force_1d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(
            self.gen_sizes_matmul(1), (True, False), (True, False)
        ):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

    @dtypes(torch.float)
    def test_matmul_small_brute_force_2d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(
            self.gen_sizes_matmul(2), (True, False), (True, False)
        ):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

    @dtypes(torch.float)
    def test_matmul_small_brute_force_3d_Nd(self, device, dtype):
        make_arg = partial(make_tensor, device=device, dtype=dtype)

        for (size_x, size_y), nctg_x, nctg_y in product(
            self.gen_sizes_matmul(3), (True, False), (True, False)
        ):
            x = make_arg(size_x, noncontiguous=nctg_x)
            y = make_arg(size_y, noncontiguous=nctg_y)
            self.check_single_matmul(x, y)

    @dtypes(torch.float)
    def test_matmul_out_kernel_errors_with_autograd(self, device, dtype):
        a = torch.empty(
            (256, 512), device=device, dtype=dtype, requires_grad=True
        ).unsqueeze(0)
        b = torch.empty(
            (4, 128, 512), device=device, dtype=dtype, requires_grad=True
        ).transpose(-1, -2)
        c = torch.empty((256, 4, 128), device=device, dtype=dtype).movedim(1, 0)

        torch.matmul(a.detach(), b.detach(), out=c)

        with self.assertRaisesRegex(
            RuntimeError,
            "functions with out=... arguments don't support automatic differentiation",
        ):
            torch.matmul(a, b, out=c)

        with torch.no_grad():
            torch.matmul(a, b, out=c)


instantiate_device_type_tests(TestBasicGEMM, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
