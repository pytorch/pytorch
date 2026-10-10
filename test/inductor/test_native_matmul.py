# Owner(s): ["module: inductor"]


from collections.abc import Callable

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification
from torch.utils._triton import has_triton


aten = torch.ops.aten


class TestTritonDotReduction(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def _check_equal(
        self, f: Callable, example_inputs: tuple[torch.Tensor], tol: float = 1e-4
    ):
        compiled = torch.compile(f)
        actual = compiled(*example_inputs)
        expect = f(*example_inputs)
        self.assertTrue(same(expect, actual, tol=tol))

    def _check_code(
        self,
        f: Callable,
        example_inputs: tuple[torch.Tensor],
        kernel_count: int,
        dot_count: int,
    ):
        f = torch.compile(f)
        code = run_and_get_triton_code(f, *example_inputs)
        FileCheck().check_regex(r"triton.*mm.*\.run\(").run(code)

        FileCheck().check_count("@triton.jit", kernel_count, exactly=True).check_count(
            "tl.dot", dot_count, exactly=True
        ).run(code)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_matmul(self, device):
        def f(x, y):
            z = x @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=device)
        y = rand_strided((K, N), (N, 1), device=device)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_mm_1d_expand(self, device):
        def f(x, y, M, K):
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M,), (1,), device=device)
        y = rand_strided((K, N), (N, 1), device=device)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_mm_2_expand(self, device):
        def f(x, y, M, K):
            z = x[:, None].expand(M, K) @ y
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((1,), (0,), device=device)
        y = rand_strided((K, N), (N, 1), device=device)

        self._check_equal(f, (x, y, M, K))
        self._check_code(f, (x, y, M, K), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_matmul_fp16(self, device):
        def f(x, y):
            z = x @ y.to(x.dtype)
            return z

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), dtype=torch.float16, device=device)
        y = rand_strided((K, N), (N, 1), dtype=torch.float32, device=device)

        # _check_equal calls torch._dynamo.utils.same with kwarg tol=1e-4.
        # For fp16 dtype, torch.allclose() defaults to atol=1e-3 rtol=1e-5,
        # but same() uses the single value to assign both, resulting in
        # Accuracy failed: allclose not within tol=0.0001.
        self._check_equal(f, (x, y), tol=1e-3)
        self._check_code(f, (x, y), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_reduction_mask_zeroout(self, device):
        def f(x, y):
            return (x + 1) @ (y - 2)

        M, K, N = 62, 62, 62
        x = rand_strided((M, K), (K, 1), device=device)
        y = rand_strided((K, N), (N, 1), device=device)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_3mm_add(self, device):
        def f(x, y, z, w, r, t):
            return x @ y + z @ w + r @ t

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=device)
        y = rand_strided((K, N), (N, 1), device=device)
        w = rand_strided((M, K), (K, 1), device=device)
        z = rand_strided((K, N), (N, 1), device=device)
        r = rand_strided((M, K), (K, 1), device=device)
        t = rand_strided((K, N), (N, 1), device=device)

        self._check_equal(f, (x, y, z, w, r, t))
        self._check_code(f, (x, y, z, w, r, t), 1, 3)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_mm_complex(self, device):
        def f(x, y, z, w):
            return x[z] @ y + w + 3

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=device)
        y = rand_strided((K, N), (N, 1), device=device)

        z = torch.randint(M, (M, K), dtype=torch.long, device=device)
        w = rand_strided((M, N), (N, 1), device=device)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_batchmatmul(self, device):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 256, 128, 128, 128
        x = rand_strided((B, M, K), (M * K, K, 1), device=device)
        y = rand_strided((B, K, N), (K * N, N, 1), device=device)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_bmm_vertical_fusion(self, device):
        def f(x, y):
            z = torch.bmm(x, y)
            w = torch.nn.functional.relu(z)
            v = w + z * z
            return v

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=device)
        y = rand_strided((B, K, N), (K * N, N, 1), device=device)

        self._check_equal(f, (x, y))
        self._check_code(f, (x, y), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_bmm_horizontal_fusion(self, device):
        def f(x, y, z, w):
            bmm1 = torch.bmm(x, y)
            bmm2 = torch.bmm(z, w)
            return bmm1 - bmm2 + bmm1 * bmm2

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=device)
        y = rand_strided((B, K, N), (K * N, N, 1), device=device)
        z = rand_strided((B, M, K), (M * K, K, 1), device=device)
        w = rand_strided((B, K, N), (K * N, N, 1), device=device)

        self._check_equal(f, (x, y, z, w))
        self._check_code(f, (x, y, z, w), 1, 2)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_bmm_fusion_complex1(self, device):
        # Out[O[b,i],j] += Input[I[b,i],r] * Weight[W[b],r,j] * Val[b,i]
        def f(inp, weight, val, I_idx, W_idx, O_idx, out):
            x = inp[I_idx]  # [B,I,R]
            y = weight[W_idx]  # [B,R,J]
            t = torch.einsum("bir,brj,bi->bij", x, y, val)
            out.index_add_(0, O_idx.reshape(-1), t.reshape(-1, J))
            return out

        B, I, R, J = 32, 16, 128, 16
        N_in = 128
        N_w = 64
        N_out = 128

        inp = rand_strided((N_in, R), (R, 1), device=device)
        weight = rand_strided((N_w, R, J), (R * J, J, 1), device=device)
        val = rand_strided((B, I), (I, 1), device=device)

        I_idx = torch.randint(0, N_in, (B, I), device=device, dtype=torch.int64)
        W_idx = torch.randint(0, N_w, (B,), device=device, dtype=torch.int64)
        O_idx = torch.randint(0, N_out, (B, I), device=device, dtype=torch.int64)

        out = torch.zeros((N_out, J), device=device)

        self._check_equal(f, (inp, weight, val, I_idx, W_idx, O_idx, out))
        self._check_code(f, (inp, weight, val, I_idx, W_idx, O_idx, out), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_bmm_fusion_complex2(self, device):
        # out[Ai[g],m,n] += Av[g,m,k,p] * B[Ak[g,p],k,n]
        def f(Av, B, Ai, Ak, out):
            Bg = B[Ak]  # [G,P,K,N]
            Cg = torch.einsum("gmkp,gpkn->gmn", Av, Bg)  # [G,M,N]
            out.index_add_(0, Ai, Cg)
            return out

        G, M, K, P, N = 128, 16, 64, 8, 16
        NB = 128
        NA = 128

        Av = rand_strided((G, M, K, P), (M * K * P, K * P, P, 1), device=device)
        B = rand_strided((NB, K, N), (K * N, N, 1), device=device)

        Ai = torch.randint(0, NA, (G,), device=device, dtype=torch.int64)
        Ak = torch.randint(0, NB, (G, P), device=device, dtype=torch.int64)

        out = torch.zeros((NA, M, N), device=device)

        self._check_equal(f, (Av, B, Ai, Ak, out))
        self._check_code(f, (Av, B, Ai, Ak, out), 1, 1)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_bmm_large_batch_reversed_pid(self, device):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 65537, 16, 16, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=device)
        y = rand_strided((B, K, N), (K * N, N, 1), device=device)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check("zoffset = tl.program_id(0)").check(
            "yoffset = tl.program_id(1)"
        ).check("xoffset = tl.program_id(2)").run(code)

    @inductor_config.patch({"triton.native_matmul": True})
    def test_bmm_no_z_broadcast(self, device):
        def f(x, y):
            z = torch.bmm(x, y)
            return z

        B, M, K, N = 128, 16, 128, 16
        x = rand_strided((B, M, K), (M * K, K, 1), device=device)
        y = rand_strided((B, K, N), (K * N, N, 1), device=device)

        f = torch.compile(f)
        code = run_and_get_triton_code(f, x, y)

        FileCheck().check_not("tl.arange(0, ZBLOCK)[:, None, None, None]").check(
            "tl.arange(0, ZBLOCK)"
        ).check("tl.arange(0, YBLOCK)[None, :, None, None]").check(
            "tl.arange(0, XBLOCK)[None, None, :, None]"
        ).check("tl.arange(0, R0_BLOCK)[None, None, None, :]").run(code)


instantiate_device_type_tests(
    TestTritonDotReduction, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    if has_triton():
        run_tests()
