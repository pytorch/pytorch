import torch
import torch.utils.benchmark as benchmark
from torch import nn
from tqdm import tqdm
import pandas as pd
import argparse
from torch.sparse import to_semi_structured_sparse
from torch.sparse.semi_structured.utils import gen_two_four_sparse_mask


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



def test_linear(m, k, n, dtype, contiguous, backend):
    mask = gen_two_four_sparse_mask(
        m, k, dtype=dtype
    )
    sparse_weight = torch.rand(m, k).to(dtype).cuda() * mask
    input_tensor = torch.zeros(n, k).to(dtype).cuda()
    model = Model(m, k).to(dtype).cuda().eval()

    dense_measurement = benchmark.Timer(
        stmt="model(input_tensor)",
        globals=locals(),
    ).blocked_autorange()

    dense_output = model(input_tensor)

    # sparsify weights
    model.linear.weight = nn.Parameter(to_semi_structured_sparse(sparse_weight))

    sparse_output = model(input_tensor)

    sparse_measurement = benchmark.Timer(
        stmt="model(input_tensor)",
        globals=locals(),
    ).blocked_autorange()

    correct = torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3)

    return {
        "test_function": "linear",
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype),
        "backend": backend,
        "sparse_latency (ms)": sparse_measurement.median * 1000,
        "dense_latency (ms)": dense_measurement.median * 1000,
        "speedup (d/s)": dense_measurement.median / sparse_measurement.median,
        "correct": correct,
        "contiguous": sparse_output.is_contiguous(),
    }


def test_tensor(m, k, n, dtype, contiguous, backend):
    A = gen_two_four_sparse_mask(m, k, dtype=dtype)
    B = torch.zeros(k, n).to(dtype).cuda()
    bias = torch.rand(n).to(dtype).cuda()

    sA = to_semi_structured_sparse(A)

    # torch.mm calculation
    if dtype is not torch.int8:
        dense_output = torch.mm(A, B)

        dense_measurement = benchmark.Timer(
            stmt="torch.mm(A, B)",
            globals=locals(),
        ).blocked_autorange()

    else:
        print("int8 baseline not supported")
        dense_output = torch.mm(sA, B)

        dense_measurement = benchmark.Timer(
            stmt="torch.mm(sA, B)",
            globals=locals(),
        ).blocked_autorange()

    sparse_output = torch.mm(sA, B)
    sparse_measurement = benchmark.Timer(
        stmt="torch.mm(sA, B)",
        globals=locals(),
    ).blocked_autorange()

    correct = torch.allclose(dense_output, sparse_output, rtol=1e-3, atol=1e-3)

    return {
        "test_function": "tensor",
        "m": m,
        "k": k,
        "n": n,
        "dtype": str(dtype),
        "backend": backend,
        "sparse_latency (ms)": sparse_measurement.median * 1000,
        "dense_latency (ms)": dense_measurement.median * 1000,
        "speedup (d/s)": dense_measurement.median / sparse_measurement.median,
        "correct": correct,
        "contiguous": sparse_output.is_contiguous(),
    }


if __name__ == "__main__":
    dtype_lookup = {
        "int8": torch.int8,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    parser = argparse.ArgumentParser(description="Semi-Structured Sparsity Benchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "nvidia-bert",
            "nvidia-fixed-k",
            "nvidia-fixed-mn",
        ],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=dtype_lookup.keys(),
        default="fp16",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cutlass", "cusparselt"],
        default="cusparselt")
    parser.add_argument("-contiguous", action="store_true")
    parser.add_argument("-e2e", action="store_true")
    parser.add_argument("-save", action="store_true")
    args = parser.parse_args()

    if args.e2e:
        eval_fn = test_linear
    else:
        eval_fn = test_tensor

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
            eval_fn(m, k, n, dtype, args.contiguous, args.backend)
            for (m, k, n) in tqdm(bert_shapes)
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
        results = (
            eval_fn(mn, 10240, mn, dtype, args.contiguous, args.backend) for mn in tqdm(mn_vals)
        )

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
        results = (
            eval_fn(10240, k, 10240, dtype, args.contiguous, args.backend) for k in tqdm(k_vals)
        )

    df = pd.DataFrame.from_records(results)
    if args.save:
        save_file = f"{args.mode}_{args.dtype}_{args.backend}.csv"
        df.to_csv(save_file)
        print(f"Finished benchmark: {args.mode} saved results to {save_file}")
    print(df)
