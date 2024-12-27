import itertools
import os
import random
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any

from benchmark_runner import BenchmarkRunner  # type: ignore[import-not-found]
from benchmark_utils import (  # type: ignore[import-not-found]
    fits_in_memory,
    get_mm_tensors,
    get_random_between_pow2,
    set_precision,
)

import torch
from torch._inductor.utils import fresh_inductor_cache


class BenchmarkRunnerMM(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    """
    BenchmarkRunner for mm.
    """

    def __init__(self) -> None:
        super().__init__("mm")

    def create_input(self) -> tuple[Any, ...]:
        dtype = random.choices([torch.float32, torch.float16, torch.bfloat16])[0]
        set_precision(dtype)
        m, k, n = self.get_m_k_n(dtype)
        return (m, k, n, dtype)

    def run_benchmark(
        self,
        m: int,
        k: int,
        n: int,
        dtype: Any,
    ) -> Any:
        # for a given shape, test all possible combinations of transpose_left and transpose_right
        for transpose_left, transpose_right in itertools.product(
            [False, True], repeat=2
        ):
            print(
                f"m: {m}, k: {k}, n: {n}, transpose_left: {transpose_left}, transpose_right: {transpose_right}, dtype: {dtype}"
            )
            a, b = get_mm_tensors(
                m,
                k,
                n,
                transpose_left,
                transpose_right,
                dtype_left=dtype,
                dtype_right=dtype,
            )

            with fresh_inductor_cache():

                def mixed_mm(A: Any, B: Any) -> Any:
                    return torch.mm(A, B)

                cf = torch.compile(mixed_mm, mode="max-autotune-no-cudagraphs")
                cf(a, b)
                torch.compiler.reset()

    def random_multiple_of_128(self, min_num: int = 7, max_num: int = 17) -> int:
        # generates a random number ran_pow2 between min_num and max_num -1
        # and returns a random multiple of 128 between 2^ran_pow2 and 2^(ran_pow2+1)
        ran_pow2 = random.randint(min_num, max_num - 1)
        start = (2**ran_pow2) // 128
        end = (2 ** (ran_pow2 + 1)) // 128
        random_multiple = random.randint(start, end)
        return random_multiple * 128

    def get_distr_type(self) -> str:
        # 85%: choose a random multiple of 128 between 2^10 and 2^17
        # 10%: choose a random power of 2 between 2^0 and 2^17
        #  4%: choose a random number between 1 and 131072
        #  1%: choose a random number between 2^i and 2^(i+1) with i in [1, 16]
        return random.choices(
            ["mult_128", "pow2", "uniform", "uniform-between-pow2"],
            [0.85, 0.1, 0.04, 0.01],
        )[0]

    def get_random_dim(self) -> int:
        distr_type = self.get_distr_type()
        if distr_type == "mult_128":
            return self.random_multiple_of_128(min_num=10, max_num=17)
        if distr_type == "pow2":
            return int(2 ** random.randint(0, 17))
        elif distr_type == "uniform-between-pow2":
            # TODO(AlnisM): make mypy work for torchgen/_autoheuristic/
            return int(get_random_between_pow2(min_power2=1, max_power2=17))
        elif distr_type == "uniform":
            return random.randint(1, 131072)
        print(f"random_type {distr_type} not supported")
        sys.exit(1)

    def get_m_k_n(self, dtype: Any) -> tuple[int, int, int]:
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


if __name__ == "__main__":
    runner = BenchmarkRunnerMM()
    runner.run()
