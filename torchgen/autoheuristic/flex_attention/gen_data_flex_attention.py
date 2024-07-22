# mypy: ignore-errors
import os
import random
import sys
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial
from typing import Any, Callable, List, Tuple

from benchmark_runner import BenchmarkRunner  # type: ignore[import-not-found]

import torch
from torch._inductor.utils import fresh_inductor_cache
from torch.nn.attention.flex_attention import flex_attention


class BenchmarkRunnerFlexAttention(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    def __init__(self) -> None:
        super().__init__("flex_attention")

    def generate_score_mods(self, score_mods: List[str]) -> List[Callable]:
        def noop(score, b, h, m, n):
            return score

        def causal_mask(score, b, h, token_q, token_kv):
            return torch.where(token_q >= token_kv, score, float("-inf"))

        def relative_bias(score, b, h, m, n):
            return score + (m - n)

        def head_bias(score, b, h, m, n):
            return score + 2 * h

        function_dict = {
            "noop": noop,
            "causal": causal_mask,
            "rel": relative_bias,
            "head_bias": head_bias,
        }
        return [function_dict[name] for name in score_mods]

    def random_multiple(self, multiple, min_num=7, max_num=17):
        ran_pow2 = random.randint(min_num, max_num - 1)
        start = (2**ran_pow2) // multiple
        end = (2 ** (ran_pow2 + 1)) // multiple
        random_multiple = random.randint(start, end)
        return random_multiple * multiple

    def get_random_dim(self, min, max):
        pow2 = random.choices([True, False], [0.9, 0.1])[0]
        if pow2:
            return 2 ** random.randint(min, max)
        else:
            i = random.randint(min, max - 1)
            lower = 2**i + 1
            upper = 2 ** (i + 1) - 1
            if lower > upper:
                return lower
            return random.randint(lower, upper)

    def get_random_sequence_length(self, min, max):
        pow2 = random.choices([True, False], [0.9, 0.1])[0]
        if pow2:
            return 2 ** random.randint(min, max)
        else:
            return self.random_multiple(128, min, max)

    def create_input(self) -> Tuple[Any, ...]:
        numel_max = 2**31
        while True:
            batch_size = self.get_random_dim(0, 8)
            q_heads = self.get_random_dim(0, 6)
            kv_heads = q_heads

            # slen must be multiple of block size
            q_seq_len = self.get_random_sequence_length(7, 13)
            kv_seq_len = q_seq_len

            # head dims (must be >= 16)
            d = 2 ** random.randint(4, 8)
            dtype = random.choices([torch.bfloat16, torch.float16, torch.float32])[0]
            device = torch.device("cuda")
            requires_grad = False
            if batch_size * q_heads * q_seq_len * d >= numel_max:
                continue
            print(
                batch_size,
                q_heads,
                kv_heads,
                q_seq_len,
                kv_seq_len,
                d,
                dtype,
                device,
                requires_grad,
            )
            return (
                batch_size,
                q_heads,
                q_seq_len,
                kv_heads,
                kv_seq_len,
                d,
                dtype,
                device,
                requires_grad,
            )

    def generate_inputs(
        self,
        batch_size: int,
        q_heads: int,
        q_sequence_length: int,
        kv_heads: int,
        kv_sequence_length: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool,
    ):
        q_shape = (batch_size, q_sequence_length, q_heads * head_dim)
        kv_shape = (batch_size, kv_sequence_length, kv_heads * head_dim)

        assert q_heads % kv_heads == 0

        num_h_groups = q_heads // kv_heads

        make_q = partial(
            torch.rand, q_shape, device=device, dtype=dtype, requires_grad=requires_grad
        )
        make_kv = partial(
            torch.rand,
            kv_shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        query = (
            make_q()
            .view(batch_size, num_h_groups * q_sequence_length, kv_heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            make_kv()
            .view(batch_size, kv_sequence_length, kv_heads, head_dim)
            .transpose(1, 2)
        )
        value = (
            make_kv()
            .view(batch_size, kv_sequence_length, kv_heads, head_dim)
            .transpose(1, 2)
        )
        return query, key, value

    def run_benchmark(
        self,
        batch_size: int,
        q_heads: int,
        q_sequence_length: int,
        kv_heads: int,
        kv_sequence_length: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool,
    ) -> Any:
        query, key, value = self.generate_inputs(
            batch_size,
            q_heads,
            q_sequence_length,
            kv_heads,
            kv_sequence_length,
            head_dim,
            dtype,
            device,
            requires_grad,
        )

        for score_mod in self.generate_score_mods(
            ["noop", "causal", "rel", "head_bias"]
        ):
            start_time = time.time()
            with fresh_inductor_cache():
                compiled_sdpa = torch.compile(
                    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
                )
                compiled_sdpa(query, key, value, score_mod)
                torch.compiler.reset()
            end_time = time.time()
            print(f"{end_time - start_time} seconds")


if __name__ == "__main__":
    runner = BenchmarkRunnerFlexAttention()
    runner.run()
