import os
import random
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any

from benchmark_runner import BenchmarkRunner  # type: ignore[import-not-found]
from benchmark_utils import (  # type: ignore[import-not-found]
    fits_in_memory,
    get_mm_tensors,
    set_precision,
    transpose_tensors,
)

import torch
from torch._inductor.fx_passes.pad_mm import (  # type: ignore[import-not-found]
    get_alignment_size_dtype,
)
from torch._inductor.utils import fresh_inductor_cache


class BenchmarkRunnerPadMM(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    """
    BenchmarkRunner for pad_mm. Used to generate collect training data with AutoHeuristic to learn a heuristic.
    """

    def __init__(self) -> None:
        super().__init__("pad_mm")

    def create_input(self) -> tuple[Any, ...]:
        dtype = self.get_dtype()
        set_precision(dtype)
        m, k, n = self.get_m_k_n(dtype)

        (transpose_left, transpose_right) = transpose_tensors()
        prepadded_left = self.prepadded()
        prepadded_right = self.prepadded()
        return (
            m,
            k,
            n,
            transpose_left,
            transpose_right,
            dtype,
            prepadded_left,
            prepadded_right,
        )

    def run_benchmark(
        self,
        m: int,
        k: int,
        n: int,
        transpose_left: bool,
        transpose_right: bool,
        dtype: Any,
        prepadded_left: bool,
        prepadded_right: bool,
    ) -> None:
        a, b = get_mm_tensors(
            m,
            k,
            n,
            transpose_left,
            transpose_right,
            dtype_left=dtype,
            dtype_right=dtype,
        )

        print("Benchmarking the following input:")
        print(f"m={m} k={k} n={n} dtype={dtype}")
        print(f"transpose_left={transpose_left} transpose_right={transpose_right}")
        print(f"prepadded_left={prepadded_left} prepadded_right={prepadded_right}")

        with fresh_inductor_cache():

            def mm(a: Any, b: Any) -> Any:
                return torch.mm(a, b)

            def mm_mat1_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a + 1, b)

            def mm_mat2_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a, b + 1)

            def mm_mat1_mat2_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a + 1, b + 1)

            if prepadded_left and prepadded_right:
                cf = torch.compile(mm_mat1_mat2_prepadded)
            elif prepadded_left:
                cf = torch.compile(mm_mat1_prepadded)
            elif prepadded_right:
                cf = torch.compile(mm_mat2_prepadded)
            else:
                cf = torch.compile(mm)
            cf(a, b)
            torch.compiler.reset()

    def get_random_dim(
        self, min_power2: int = 1, max_power2: int = 16, p_unaligned: float = 0.25
    ) -> int:
        aligned = random.choices([True, False], [1 - p_unaligned, p_unaligned])[0]
        if aligned:
            return 2 ** random.randint(min_power2, max_power2)  # type: ignore[no-any-return]
        else:
            # choose a random number between 2^i and 2^(i+1)
            return self.get_random_between_pow2(min_power2, max_power2)  # type: ignore[no-any-return]

    def is_aligned(self, dim: int, align_size: int) -> bool:
        return dim % align_size == 0

    def get_m_k_n(self, dtype: Any) -> tuple[int, int, int]:
        uniform = random.choices([True, False])[0]
        align_size = get_alignment_size_dtype(dtype)

        # repeat until tensors fit in memory
        while True:
            if uniform:
                m = random.randint(1, 65536)
                k = random.randint(1, 65536)
                n = random.randint(1, 65536)
            else:
                m = self.get_random_dim()
                k = self.get_random_dim()
                n = self.get_random_dim()

            if all(self.is_aligned(dim, align_size) for dim in [m, k, n]):
                # skip if already aligned
                continue

            if fits_in_memory(dtype, m, k, n):
                return (m, k, n)

    def prepadded(self, p_prepadded: float = 0.2) -> bool:
        # p_prepadded: probability that a tensor is "prepadded", i.e. pad_mm excludes time it takes to pad from benchmarking
        return random.choices([True, False], [p_prepadded, 1 - p_prepadded])[0]

    def get_dtype(self) -> Any:
        dtype_choices = [torch.float16, torch.bfloat16, torch.float32]
        return random.choices(dtype_choices)[0]


if __name__ == "__main__":
    runner = BenchmarkRunnerPadMM()
    runner.run()
