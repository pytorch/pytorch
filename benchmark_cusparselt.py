import os
import sys
from itertools import product, combinations, combinations_with_replacement, permutations
import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormPruner
from torch.ao.nn.sparse.cusparselt_linear import cuSPARSELtLinear, cuSPARSELtLinearInt8
from torch.profiler import profile, record_function, ProfilerActivity
from pprint import pprint
from time import time
from tqdm import tqdm
import pandas as pd
import argparse
import gc

DEVICE = "cuda"
torch.set_printoptions(
    precision=3,
    threshold=None,
    edgeitems=32,
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


# function to compare dense vs cusparselt linear for given m, k, n, batch_size
def compare_linear(m, k, n, batch_size, init_batch_size, dtype, assert_correct=False):

    temp = cuSPARSELtLinear if dtype is torch.float16 else cuSPARSELtLinearInt8

    # print(m, k, n, batch_size, init_batch_size, dtype, temp)
    # create dense fp16 model
    model = Model(m, k).half().cuda().eval()
    # create input tensor
    input_tensor = torch.randint(
        5, 
        (init_batch_size, n, k),
        device=DEVICE,
        dtype=dtype,
    )

    # get sparse model
    pruner = WeightNormPruner(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )

    pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
    pruner.step()
    sparse_model = pruner.convert(model, mapping={nn.Linear: temp})
    pruner.squash_mask()

    # suppress stdout
    devnull = open("/dev/null", "w")
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)

    correct = torch.allclose(
        model(input_tensor.half()).to(dtype), sparse_model(input_tensor), rtol=1e-3, atol=1e-3
    )

    input_tensor = torch.randint(
        5, 
        (batch_size, n, k),
        device=DEVICE, 
        dtype=dtype,
    )
    # get latency
    sparse_measurement = benchmark.Timer(
        stmt="sparse_model(input_tensor)",
        globals={"input_tensor": input_tensor, "sparse_model": sparse_model},
    ).blocked_autorange()
    dense_measurement = benchmark.Timer(
        stmt="model(input_tensor)",
        globals={"input_tensor": input_tensor.half(), "model": model},
    ).blocked_autorange()

    os.dup2(oldstdout_fno, 1)

    return {
        "m": m,
        "k": k,
        "n": n,
        "eval_batch_size": batch_size,
        "init_batch_size": init_batch_size,
        "dtype": str(dtype),
        "sparse_latency (ms)": sparse_measurement.median * 1000,
        "dense_latency (ms)": dense_measurement.median * 1000,
        "speedup (d/s)": dense_measurement.median / sparse_measurement.median,
        # "correct": correct,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cuSPARSELt Benchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "nvidia-bert",
            "nvidia-fixed-k",
            "nvidia-fixed-mn",
            "distilbert-shapes",
            "alg-id-sweep",
            "int8-fp16-linear",
        ],
    )
    args = parser.parse_args()

    print(f"Started benchmark: {args.mode}")

    if args.mode == "nvidia-bert":
        bert_shapes = [
            (3072, 1024, 16384),
            (4096, 1024, 16384),
            (1024, 1024, 16384),
            (1024, 4096, 16384),
        ]
        results = (compare_linear(m, k, n, 1, 1, torch.float16) for (m, k, n) in tqdm(bert_shapes))

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
        results = (compare_linear(mn, 10240, mn, 1, 1, torch.float16) for mn in tqdm(mn_vals))

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
        results = (compare_linear(10240, k, 10240, 1, 1, torch.float16) for k in tqdm(k_vals))

    elif args.mode == "distilbert-shapes":
        shapes = [
            # distilbert shapes
            (768, 3072, 768),
            (3072, 768, 3072),
            # jiecao shapes
            # (1024, 1536, 2048),
            # (1024, 9408, 2048),
            # (1024, 3200, 2048),
            # (1024, 256, 9472),
            # (1024, 10240, 256),
            # (1024, 256, 12608),
            # (1024, 2560, 1024),
            # (1024, 512, 10240),
            # (1024, 10240, 512),
            # (1024, 2048, 1024),
            # (1024, 512, 512),
            # (1024, 1024, 1024),
            # (1024, 2048, 2048),
            # (2048, 1536, 2048),
            # (2048, 9408, 2048),
            # (2048, 3200, 2048),
            # (2048, 256, 9472),
            # (2048, 10240, 256),
            # (2048, 256, 12608),
            # (2048, 2560, 1024),
            # (2048, 512, 10240),
            # (2048, 10240, 512),
            # (2048, 2048, 1024),
            # (2048, 512, 512),
            # (2048, 1024, 1024),
            # (2048, 2048, 2048),
        ]
        batch_sizes = [4, 16, 64, 256]
        results = (
            compare_linear(m, k, n, batch_size, batch_size, torch.float16)
            for (m, k, n), batch_size in tqdm(
                product(shapes, batch_sizes), total=len(shapes) * len(batch_sizes)
            )
        )


    elif args.mode == "int8-fp16-linear":
        MP = 2
        BS = 512
        print(f"Working on MP: {MP}, BS: {BS}")
        shapes = [
            (8192 // MP, 8192, BS),
            (8192, 8192 // MP, BS),
            (22016 // MP, 8192, BS),
            (8192, 22016 // MP, BS),
        ]
        dtypes = [torch.int8, torch.float16]
        batch_sizes = [1, 16]
        results = (
            compare_linear(m, k, n, batch_size, batch_size, dtype)
            for dtype, batch_size, (m, k, n) in tqdm(
                product(dtypes, batch_sizes, shapes), total=len(dtypes) * len(batch_sizes) * len(shapes)
            )
        )

    save_file = f"{args.mode}.csv"
    df = pd.DataFrame.from_records(results)
    df.to_csv(save_file)
    print(f"Finished benchmark: {args.mode} saved results to {save_file}")
    print(df)
