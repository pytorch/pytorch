# Owner(s): ["module: inductor"]


import unittest

import torch
from torch import Tensor
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_env import is_datacenter_blackwell_arch
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import ensure_cute_available
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@unittest.skipIf(
    not (ensure_cute_available() and is_datacenter_blackwell_arch()),
    "CuTeDSL library or Blackwell device not available",
)
@instantiate_parametrized_tests
class TestCuTeDSLGroupedGemm(InductorTestCase):
    def _get_inputs(
        self,
        group_size: int,
        M_hint: int,
        K: int,
        N: int,
        device: str,
        dtype: torch.dtype,
        alignment: int = 16,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # --- Random, tile-aligned M sizes ---
        M_sizes = (
            torch.randint(1, (M_hint // alignment) + 1, (group_size,), dtype=torch.int)
            * alignment
        )

        M_total = torch.sum(M_sizes).item()

        # --- Construct input tensors ---
        A = torch.randn(int(M_total), K, dtype=dtype, device=device) * 0.1
        B = torch.randn((group_size, K, N), dtype=dtype, device=device) * 0.01

        # --- Build offsets (no leading zero, strictly increasing) ---
        offsets = torch.cumsum(M_sizes, dim=0).to(dtype=torch.int32, device=device)

        return (A, B, offsets)

    @parametrize("group_size", (2, 8))
    @parametrize("M_hint", (256, 1024))
    @parametrize("K", (64, 128))
    @parametrize("N", (128, 256))
    def test_grouped_gemm_basic(self, group_size: int, M_hint: int, K: int, N: int):
        device = "cuda"
        dtype = torch.bfloat16

        A, B, offsets = self._get_inputs(group_size, M_hint, K, N, device, dtype)

        def grouped_gemm_fn(A_packed, B_batched, offs):
            return F.grouped_mm(A_packed, B_batched, offs=offs)

        # Eager execution
        c_eager = grouped_gemm_fn(A, B, offsets)

        # Test with Cute backend
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "test_configs.autotune_choice_name_regex": "cutedsl",
                "autotune_fallback_to_aten": False,
            }
        ):
            grouped_gemm_compiled = torch.compile(
                grouped_gemm_fn, backend="inductor", dynamic=False
            )
            c_compiled = grouped_gemm_compiled(A, B, offsets)

        self.assertEqual(c_eager.dtype, dtype)
        self.assertEqual(c_compiled.dtype, dtype)
        torch.testing.assert_close(c_eager, c_compiled)

    def _tensor_before_guard_page(self, rows, cols, dtype):
        # The granule after the tensor stays reserved but unmapped, so any read past its end faults.
        from torch.cuda._utils import (
            _check_cuda_bindings as check,
            _cuda_bindings_driver as cu,
        )

        nbytes = rows * cols * dtype.itemsize
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = torch.cuda.current_device()
        granularity = check(
            cu.cuMemGetAllocationGranularity(
                prop,
                cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        self.assertEqual(nbytes % granularity, 0)
        va = check(cu.cuMemAddressReserve(nbytes + granularity, 0, 0, 0))
        self.addCleanup(lambda: check(cu.cuMemAddressFree(va, nbytes + granularity)))
        handle = check(cu.cuMemCreate(nbytes, prop, 0))
        self.addCleanup(lambda: check(cu.cuMemRelease(handle)))
        check(cu.cuMemMap(va, nbytes, 0, handle, 0))
        self.addCleanup(lambda: check(cu.cuMemUnmap(va, nbytes)))
        self.addCleanup(torch.cuda.synchronize)
        access = cu.CUmemAccessDesc()
        access.location = prop.location
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        check(cu.cuMemSetAccess(va, nbytes, [access], 1))

        class Mapped:
            __cuda_array_interface__ = {
                "shape": (rows, cols),
                "typestr": "<i2",
                "data": (int(va), False),
                "version": 3,
            }

        return torch.as_tensor(Mapped(), device="cuda").view(dtype)

    @parametrize("b_transposed", (False, True))
    @parametrize("group_sizes", ((1279, 1), (1272, 8, 0)))
    def test_grouped_gemm_small_last_group_before_guard_page(
        self, group_sizes, b_transposed
    ):
        # TMA descriptors seeded from packed A made a small last group read past A (#176178).
        K, N, G = 4096, 1024, len(group_sizes)
        A = self._tensor_before_guard_page(sum(group_sizes), K, torch.bfloat16)
        A.copy_(torch.randn_like(A) * 0.1)
        if b_transposed:
            B = (torch.randn(G, N, K, dtype=torch.bfloat16, device="cuda") * 0.01).mT
        else:
            B = torch.randn(G, K, N, dtype=torch.bfloat16, device="cuda") * 0.01
        offsets = torch.tensor(group_sizes, device="cuda").cumsum(0, dtype=torch.int32)

        expected = F.grouped_mm(A, B, offs=offsets)
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "test_configs.autotune_choice_name_regex": "cutedsl",
                "autotune_fallback_to_aten": False,
            }
        ):
            actual = torch.compile(F.grouped_mm, dynamic=False)(A, B, offs=offsets)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @parametrize("layout_A", ("contiguous", "offset", "padded", "view"))
    @parametrize("layout_B", ("contiguous", "broadcasted"))
    def test_grouped_gemm_assorted_layouts(
        self,
        layout_A: str,
        layout_B: str,
    ):
        device = "cuda"
        dtype = torch.bfloat16

        G, K, N = 8, 64, 128
        M_sizes = [128] * G
        sum_M = sum(M_sizes)
        offsets = torch.tensor(
            [sum(M_sizes[: i + 1]) for i in range(G)], dtype=torch.int32, device=device
        )

        A_base = torch.randn(sum_M, K, device=device, dtype=dtype)
        A = A_base

        if layout_A == "offset":
            # allocate bigger buffer than needed, use nonzero storage offset
            storage = torch.randn(sum_M * K + 512, device=device, dtype=dtype)
            offset = 128  # skip first 128 elements
            A = torch.as_strided(storage[offset:], (sum_M, K), (K, 1))
        elif layout_A == "padded":
            # simulate row pitch > K (row_stride = K + pad)
            row_pitch = K + 8
            storage = torch.randn(sum_M * row_pitch, device=device, dtype=dtype)
            A = torch.as_strided(storage, (sum_M, K), (row_pitch, 1))
        elif layout_A == "view":
            A_storage = torch.randn(sum_M * K, device=device, dtype=dtype)
            A = A_storage.view(sum_M, K)
            if A._base is None:
                raise AssertionError
            if A.shape != (sum_M, K):
                raise AssertionError

        B = torch.randn((G, K, N), dtype=dtype, device=device) * 0.01

        if layout_B == "broadcasted":
            # Broadcast B across groups (zero stride along G)
            B = B[0].expand(G, K, N)
            if B.stride(0) != 0:
                raise AssertionError

        def grouped_gemm_fn(A_packed, B_batched, offs):
            return F.grouped_mm(A_packed, B_batched, offs=offs)

        # --- eager ---
        c_eager = grouped_gemm_fn(A, B, offsets)

        # --- compiled (CUTE backend) ---
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTEDSL",
                "test_configs.autotune_choice_name_regex": "cutedsl",
                "autotune_fallback_to_aten": False,
            }
        ):
            grouped_gemm_compiled = torch.compile(
                grouped_gemm_fn, backend="inductor", dynamic=False
            )
            c_compiled = grouped_gemm_compiled(A, B, offsets)

        self.assertEqual(c_eager.dtype, dtype)
        self.assertEqual(c_compiled.dtype, dtype)
        torch.testing.assert_close(c_eager, c_compiled)


if __name__ == "__main__":
    run_tests()
