from itertools import product, combinations
import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from tqdm import tqdm
import pandas as pd
import argparse
import random
from torch.ao.pruning import SemiStructuredSparseTensor


torch.set_printoptions(
    precision=2,
    threshold=None,
    edgeitems=16,
    linewidth=480,
    profile=None,
    sci_mode=False,
)


# helper model definition for pruner
class Model(nn.Module):
    def __init__(self, m, k, dtype=None):
        super().__init__()
        # transposed so reversed
        self.linear = nn.Linear(k, m)

    def forward(self, x):
        return self.linear(x)


def gen_two_four_sparse_mask(r, c, dtype=torch.float16, device="cuda"):
    def random_mask_choice(i=None):
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
        return choices[random.randint(0, len(choices) - 1) if i is None else i]

    mask_entries = [random_mask_choice() for i in range(r * c // 4)]
    return (
        torch.tensor(mask_entries, dtype=dtype, device=device).view(r, c).contiguous()
    )


def test_linear(m, k, n, dtype):
    sparse_weight = torch.rand(m, k).to(dtype).cuda() * gen_two_four_sparse_mask(m, k, dtype=dtype)
    input_tensor = torch.zeros(n, k).to(dtype).cuda()
    model = Model(m, k).to(dtype).cuda().eval()
    model.weight = sparse_weight

    dense_measurement = benchmark.Timer(
        stmt="model(input_tensor)",
        globals=locals(),
    ).blocked_autorange()

    dense_output = model(B)

    # sparsify weights
    model.linear.weight = nn.Parameter(SemiStructuredSparseTensor(model.linear.weight))

    sparse_output = model(B)

    sparse_measurement = benchmark.Timer(
        stmt="model(input_tensor)",
        globals=locals(),
    ).blocked_autorange()

    correct = torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3)

    return {
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype),
        "sparse_latency (ms)": sparse_measurement.median * 1000,
        "dense_latency (ms)": dense_measurement.median * 1000,
        "speedup (d/s)": dense_measurement.median / sparse_measurement.median,
        "correct": correct,
    }


def test_tensor(m, k, n, dtype):
    A = gen_two_four_sparse_mask(m, k, dtype=dtype)
    B = torch.zeros(k, n).to(dtype).cuda()
    bias = torch.rand(n).to(dtype).cuda()

    sA = SemiStructuredSparseTensor(A)

    # torch.mm calculation
    # sparse_output_addmm = torch.addmm(bias, sA, B)
    # dense_output_addmm = torch.addmm(bias, A, B)
    # correct_addmm = torch.allclose(
        # sparse_output_addmm, dense_output_addmm, rtol=1e-3, atol=1e-3
    # )

    # int8 cuda mm is now using 
    if dtype is torch.int8:
        dense_output = torch._int_mm(A, B)

        dense_measurement = benchmark.Timer(
            stmt="torch._int_mm(A, B)",
            globals=locals(),
        ).blocked_autorange()

    else:
        dense_output = torch.mm(A, B)

        dense_measurement = benchmark.Timer(
            stmt="torch.mm(A, B)",
            globals=locals(),
        ).blocked_autorange()

    sparse_output = torch.mm(sA, B)
    sparse_measurement = benchmark.Timer(
        stmt="torch.mm(sA, B)",
        globals=locals(),
    ).blocked_autorange()

    return {
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype),
        "sparse_latency (ms)": sparse_measurement.median * 1001,
        "dense_latency (ms)": dense_measurement.median * 1000,
        "speedup (d/s)": dense_measurement.median / sparse_measurement.median,
        "correct": torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3),
    }


if __name__ == "__main__":
    dtype_lookup = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16, 
        "fp32": torch.float32,
    }

    parser = argparse.ArgumentParser(description="cuSPARSELt Benchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "nvidia-bert",
            "nvidia-fixed-k",
            "nvidia-fixed-mn",
            "llama-shapes",
            "int8",
            "test",
        ],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=dtype_lookup.keys(),
        default="fp16",
    )
    parser.add_argument(
        "-v",
        action="store_true"
    )
    args = parser.parse_args()

    print(f"Started benchmark: {args.mode} | dtype: {args.dtype}")
    dtype = dtype_lookup[args.dtype]

    if args.mode == "nvidia-bert":
        bert_shapes = [
            (3072, 1024, 16384),
            (4096, 1024, 16384),
            (1024, 1024, 16384),
            (1024, 4096, 16384),
        ]
        results = (
            test_tensor(m, k, n, dtype) for (m, k, n) in tqdm(bert_shapes)
        )

    elif args.mode == "nvidia-fixed-k":
        mn_vals = [
            3072,
            4096,
            5120,
            6144,
            7168,
            8192,
            9216,
            10240,
            11264,
            12288,
            13312,
            14336,
            15360,
            16384,
            17408,
            18432,
            19456,
            20480,
        ]
        results = (test_tensor(mn, 10240, mn, dtype) for mn in tqdm(mn_vals))

    elif args.mode == "nvidia-fixed-mn":
        k_vals = [
            2560,
            3840,
            5120,
            6400,
            7680,
            8960,
            10240,
            11520,
            12800,
            14080,
            15360,
            16640,
            17920,
            19200,
            20480,
        ]
        results = (test_tensor(10240, k, 10240, dtype) for k in tqdm(k_vals))

    df = pd.DataFrame.from_records(results)
    if args.v:
        save_file = f"{args.mode}.csv"
        df.to_csv(save_file)
        print(f"Finished benchmark: {args.mode} saved results to {save_file}")
    print(df)
