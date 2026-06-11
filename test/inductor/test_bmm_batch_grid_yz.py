# Owner(s): ["module: inductor"]
# PR #178617 — BMM batch dimension exceeds CUDA gridDim.y (~65535): launch uses (grid_y, grid_z)
# and Triton `triton_bmm` reconstructs the batch index from tl.program_id(1,2).
r"""
Regression contract (three layers):

1. **Grid math** — ``bmm_grid`` invariants (``grid_y`` cap, ``grid_z`` split, PID linearization).
2. **Codegen** — emitted Triton for ``aten.bmm`` includes the ``idx_q`` reconstruction using
   ``tl.program_id(2)`` and ``tl.num_programs(1)`` (with ``triton.native_matmul`` disabled so
   we exercise ``templates/triton_bmm.py.jinja``, not the native-matmul PID layout).
3. **Semantics** — K=1, values in ``{0, 1}`` only: each output entry is one exact multiply;
   ``torch.equal`` vs a hand-built expected tensor catches wrong batch slices (wrong ``idx_q``).
"""

from __future__ import annotations

import unittest

import torch
from torch._inductor import config as inductor_config
from torch._inductor.kernel.bmm import bmm_grid
from torch._inductor.runtime.runtime_utils import get_max_y_grid
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


def sparse_probe_bmm_tensors(
    *,
    device: str,
    dtype: torch.dtype,
    B: int,
    M: int = 64,
    N: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K=1 BMM: zeros except unit spikes at probe batch rows (see PR #178617).

    For ``B > 65535``, probes include ``65534, 65535, 65536`` (around the CUDA grid.y limit).
    ``expected`` is the exact dense result (no ``torch.bmm``), so ``torch.equal(out, expected)``
    is a tolerance-free oracle for correct batch addressing.
    """
    row_idxs = {0, B - 1}
    if B > 65535:
        row_idxs |= {65534, 65535, 65536}
    row_idxs = {i for i in row_idxs if i < B}
    a = torch.zeros(B, M, 1, device=device, dtype=dtype)
    b = torch.zeros(B, 1, N, device=device, dtype=dtype)
    expected = torch.zeros(B, M, N, device=device, dtype=dtype)
    for i in row_idxs:
        a[i, 0, 0] = 1
        b[i, 0, 0] = 1
        expected[i, 0, 0] = 1
    return a, b, expected


class TestBmmBatchGridYz(TestCase):
    def test_bmm_grid_yz_invariants_and_pid_linearization(self):
        """Algebraic contract for PR #178617 grid + the idx_q layout ``q = pid_y + pid_z * grid_y``.

        ``grid_y <= max_y`` follows from ``grid_z >= b / max_y`` so ``b/grid_z <= max_y``.
        ``gy * gz >= b`` ensures every batch row index appears in the Y/Z program grid.
        Triton uses ``num_programs(1) == grid_y`` at launch, so the kernel's ``idx_q`` matches
        ``q % gy`` + ``(q // gy) * gy`` == ``q`` for all ``q < b``.
        """
        max_y = get_max_y_grid()
        meta = {"BLOCK_M": 64, "BLOCK_N": 64}
        for b in (
            1,
            max_y - 1,
            max_y,
            max_y + 1,
            max_y + 2,
            70_000,
            max_y * 2,
        ):
            tiles, gy, gz = bmm_grid(b, 64, 64, meta)
            self.assertEqual(tiles, 1)  # m=n=64 with BLOCK_M=BLOCK_N=64
            self.assertLessEqual(
                gy, max_y, msg=f"b={b}: grid_y must not exceed CUDA gridDim.y cap"
            )
            self.assertGreaterEqual(
                gy * gz, b, msg=f"b={b}: Y/Z grid must cover all batch rows"
            )
            if b <= max_y:
                self.assertEqual(gz, 1)
            else:
                self.assertGreaterEqual(gz, 2)
            for q in range(b):
                pid_y = q % gy
                pid_z = q // gy
                self.assertLess(
                    pid_z,
                    gz,
                    msg=f"b={b} q={q} gy={gy} gz={gz}: batch row must map into Z launch bound",
                )
                self.assertEqual(
                    pid_y + pid_z * gy,
                    q,
                    msg="Kernel idx_q uses pid_y + pid_z * num_programs(1) with num_programs(1)==grid_y",
                )

    @unittest.skipIf(not HAS_GPU, "requires CUDA/XPU Triton")
    @inductor_config.patch(
        {
            "triton.native_matmul": False,
            "max_autotune": False,
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_triton_bmm_kernel_reconstructs_batch_from_yz_program_ids(self):
        """Emitted Triton must contain the idx_q formula from triton_bmm.py.jinja."""
        if GPU_TYPE == "cpu":
            raise unittest.SkipTest("Triton BMM codegen test is GPU-only")

        max_y = get_max_y_grid()
        B = max_y + 1
        M = K = N = 16
        dtype = torch.float16

        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.bmm(a, b)

        x = torch.randn(B, M, K, device=GPU_TYPE, dtype=dtype)
        y = torch.randn(B, K, N, device=GPU_TYPE, dtype=dtype)
        opt = torch.compile(fn)
        code = run_and_get_triton_code(opt, x, y)
        # Two identical idx_q assignments in triton_bmm (load path + store path).
        needle = "tl.program_id(2) * tl.num_programs(1)"
        self.assertGreaterEqual(
            code.count(needle),
            2,
            msg="Expected triton_bmm idx_q reconstruction (program_id(2) * num_programs(1))",
        )
        FileCheck().check("idx_q").check("tl.program_id(1)").check(needle).run(code)

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @inductor_config.patch(
        {
            "triton.native_matmul": False,
            "max_autotune": False,
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_bmm_sparse_probe_exact_equal_inductor(self):
        """End-to-end: torch.compile Inductor path matches hand-built expected (no allclose)."""
        if GPU_TYPE == "cpu":
            raise unittest.SkipTest("GPU-only")

        dtype = torch.float16
        a0, b0, e0 = sparse_probe_bmm_tensors(device=GPU_TYPE, dtype=dtype, B=100)
        a1, b1, e1 = sparse_probe_bmm_tensors(
            device=GPU_TYPE, dtype=dtype, B=max(get_max_y_grid() + 1, 70_000)
        )

        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.bmm(a, b)

        opt = torch.compile(fn, backend="inductor", dynamic=False)
        # Large batch first so the first compile sees grid_z > 1.
        self.assertTrue(torch.equal(opt(a1, b1), e1))
        self.assertTrue(torch.equal(opt(a0, b0), e0))


if __name__ == "__main__":
    if HAS_GPU:
        torch.set_default_device(GPU_TYPE)
    run_tests()
