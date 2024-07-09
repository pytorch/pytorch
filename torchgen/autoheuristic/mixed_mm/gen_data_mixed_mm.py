# mypy: ignore-errors
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Tuple

from benchmark_runner import BenchmarkRunner  # type: ignore[import-not-found]
from benchmark_utils import (  # type: ignore[import-not-found]
    fits_in_memory,
    get_mm_tensors,
    transpose_tensors,
)

import torch

from torch._inductor.utils import fresh_inductor_cache


class BenchmarkRunnerMixedMM(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    def __init__(self) -> None:
        super().__init__("mixed_mm")

    def create_input(self) -> Tuple[Any, ...]:
        dtype1, dtype2 = self.get_dtypes()
        m, k, n = self.get_m_k_n(dtype1)
        transpose_left, transpose_right = transpose_tensors()
        return (m, k, n, transpose_left, transpose_right, dtype1, dtype2)

    def run_benchmark(
        self,
        m: int,
        k: int,
        n: int,
        transpose_left: bool,
        transpose_right: bool,
        dtype_left: Any,
        dtype_right: Any,
    ) -> Any:
        a, b = get_mm_tensors(
            m,
            k,
            n,
            transpose_left,
            transpose_right,
            dtype_left=dtype_left,
            dtype_right=torch.float32,
        )
        b = b.to(dtype=dtype_right)

        with fresh_inductor_cache():

            def mixed_mm(A, B):
                return torch.mm(A, B.to(A.dtype))

            cf = torch.compile(mixed_mm, mode="max-autotune-no-cudagraphs")
            cf(a, b)
            torch.compiler.reset()

    def random_multiple_of_128(self, min_num=7, max_num=17):
        ran_pow2 = random.randint(min_num, max_num - 1)
        start = (2**ran_pow2) // 128
        end = (2 ** (ran_pow2 + 1)) // 128
        random_multiple = random.randint(start, end)
        return random_multiple * 128

    def get_random_pow2_weighted(self, min_power2: int = 1, max_power2: int = 17):
        choices = []
        choices.append(range(1, 6))
        choices.append(range(6, 10))
        choices.append(range(10, max_power2 + 1))
        weights = [0.05, 0.1, 0.85]
        group = random.choices(choices, weights)[0]
        return 2 ** random.choice(group)

    def get_distr_type(self) -> str:
        # 85%: choose a random multiple of 128 between 2^7 and 2^17
        # 10%: choose a random power of 2 between 2^1 and 2^17 favoring larger values
        #  4%: choose a random number between 1 and 131072
        #  1%: choose a random number between 2^i and 2^(i+1) with i in [1, 16]
        return random.choices(
            ["mult_128", "pow2", "uniform", "uniform-between-pow2"],
            [0.85, 0.1, 0.04, 0.01],
        )[0]

    def get_random_dim(self):
        distr_type = self.get_distr_type()
        if distr_type == "mult_128":
            return self.random_multiple_of_128()
        if distr_type == "pow2":
            return self.get_random_pow2_weighted()
        elif distr_type == "uniform-between-pow2":
            min_power2 = 1
            max_power2 = 17
            i = random.randint(min_power2, max_power2 - 1)
            lower = 2**i + 1
            upper = 2 ** (i + 1) - 1
            return random.randint(lower, upper)
        elif distr_type == "uniform":
            return random.randint(1, 131072)
        print(f"random_type {distr_type} not supported")
        sys.exit(1)

    def get_m_k_n(self, dtype: Any) -> Tuple[int, int, int]:
        numel_max = 2**31

        # repeat until tensors fit in memory
        while True:
            m = self.get_random_dim()
            k = self.get_random_dim()
            n = self.get_random_dim()

            if m * k >= numel_max or m * n >= numel_max or k * n >= numel_max:
                # autotuning will not happen for tensors that are this large
                continue

            if fits_in_memory(dtype, m, k, n):
                return (m, k, n)

    def get_dtypes(self) -> Any:
        while True:
            dtype_floats = [torch.float16, torch.bfloat16]
            dtype_ints = [torch.int8, torch.uint8]
            mat1_dtype = random.choices(dtype_floats)[0]
            mat2_dtype = random.choices(dtype_ints)[0]
            if mat1_dtype == torch.bfloat16 and mat2_dtype == torch.uint8:
                # this combination seems to cause issues with mixed_mm
                continue
            return (mat1_dtype, mat2_dtype)


if __name__ == "__main__":
    runner = BenchmarkRunnerMixedMM()
    runner.run()
