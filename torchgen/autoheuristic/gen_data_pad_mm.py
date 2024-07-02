import argparse
import random
import time

from typing import Any, Tuple

from tqdm import tqdm  # type: ignore[import-untyped]

import torch

torch.set_default_device("cuda")

from torch._inductor.fx_passes.pad_mm import get_alignment_size_dtype
from torch._inductor.utils import fresh_inductor_cache

# A100: 81920MiB
# without a threshold we sometimes run out of memory
threshold_memory = 85899345920 / 4

# probability that a dimension is unaligned
p_unaligned = 0.25

# probability that a tensor is "prepadded", i.e. pad_mm excludes time it takes to pad from benchmarking
p_prepadded = 0.2

# probability that we pick from uniform distribution
p_uniform = 0.5

# probability that a tensor is transposed
p_transposed = 0.1

p_float32_prec_highest = 0.8


def benchmark(
    m: int,
    k: int,
    n: int,
    tranpose_left: bool,
    tranpose_right: bool,
    dtype: Any,
    prepadded_left: bool,
    prepadded_right: bool,
) -> None:
    if tranpose_left:
        a = torch.randn(k, m, dtype=dtype).t()
    else:
        a = torch.randn(m, k, dtype=dtype)
    if tranpose_right:
        b = torch.randn(n, k, dtype=dtype).t()
    else:
        b = torch.randn(k, n, dtype=dtype)

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


def fits_in_memory(dtype: Any, m: int, k: int, n: int) -> Any:
    return dtype.itemsize * (m * k + k * n + m * n) < threshold_memory


def get_random_dim(min_power2: int = 1, max_power2: int = 16) -> int:
    aligned = random.choices([True, False], [1 - p_unaligned, p_unaligned])[0]
    if aligned:
        return 2 ** random.randint(min_power2, max_power2)  # type: ignore[no-any-return]
    else:
        # choose a random number between 2^i and 2^(i+1)
        i = random.randint(min_power2, max_power2 - 1)
        lower = 2**i + 1
        upper = 2 ** (i + 1) - 1
        return random.randint(lower, upper)


def get_m_k_n(dtype: Any) -> Tuple[int, int, int]:
    uniform = random.choices([True, False], [0.5, 0.5])[0]

    # repeat until tensors fit in memory
    while True:
        if uniform:
            m = random.randint(1, 65536)
            k = random.randint(1, 65536)
            n = random.randint(1, 65536)
        else:
            m = get_random_dim()
            k = get_random_dim()
            n = get_random_dim()

        if fits_in_memory(dtype, m, k, n):
            return (m, k, n)


def transposed() -> bool:
    return random.choices([True, False], [p_transposed, 1 - p_transposed])[0]


def prepadded() -> bool:
    return random.choices([True, False], [p_prepadded, 1 - p_prepadded])[0]


def get_dtype() -> Any:
    dtype_choices = [torch.float16, torch.bfloat16, torch.float32]
    return random.choices(dtype_choices)[0]


def set_precision(dtype: Any) -> None:
    if dtype == torch.float32:
        precisions = ["high", "highest"]
        weights = [1 - p_float32_prec_highest, p_float32_prec_highest]
        precision = random.choices(precisions, weights)[0]
    else:
        precision = "high"
    torch.set_float32_matmul_precision(precision)


def main(num_samples: int) -> None:
    for i in tqdm(range(num_samples)):
        dtype = get_dtype()
        set_precision(dtype)
        m, k, n = get_m_k_n(dtype)

        align_size = get_alignment_size_dtype(dtype)
        if m % align_size == 0 and k % align_size == 0 and n % align_size == 0:
            # skip if already aligned
            continue

        tranpose_left = transposed()
        tranpose_right = transposed()
        prepadded_left = prepadded()
        prepadded_right = prepadded()

        print(
            f"{m} {k} {n} mat1.t()={tranpose_left} mat2.t()={tranpose_right} dtype={dtype}"
        )
        print("prepadded_left:", prepadded_left, "prepadded_right:", prepadded_right)
        for i in range(3):
            benchmark(
                m,
                k,
                n,
                tranpose_left,
                tranpose_right,
                dtype,
                prepadded_left,
                prepadded_right,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="torch.cuda.set_device(device) will be used",
    )
    parser.add_argument(
        "--autoheuristic-mode",
        type=str,
        default="COLLECT_DATA",
        help="COLLECT_DATA to collect Data. USE_HEURISTIC to test heuristic.",
    )
    parser.add_argument(
        "-o",
        type=str,
        default="a100_data.txt",
        help="Path to file where AutoHeuristic will log results.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to collect.",
    )

    args = parser.parse_args()
    torch._inductor.config.autoheuristic_mode = args.autoheuristic_mode
    torch._inductor.config.autoheuristic_log_path = args.o
    if args.device is not None:
        torch.cuda.set_device(args.device)
    random.seed(time.time())
    main(args.num_samples)
