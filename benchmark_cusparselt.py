import os
import sys
import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from itertools import product
from torch.profiler import profile, record_function, ProfilerActivity
from torch.ao.nn.sparse.cusparselt_linear import cuSPARSELtLinear
from pprint import pprint
from tqdm import tqdm
import pandas as pd
import argparse

device = "cuda"
dtype = torch.float16
torch.set_printoptions(
    precision=3,
    threshold=None,
    edgeitems=4,
    linewidth=460,
    profile=None,
    sci_mode=False,
)


# helper model definition for pruner
class Model(nn.Module):
    def __init__(self, m, k):
        super().__init__()
        # transposed so reversed
        self.linear = nn.Linear(k, m)

    def forward(self, x):
        return self.linear(x)


# function to compare dense vs cusparselt linear for given m, k, n, batch_size
def compare_linear(m, k, n, batch_size, alg_id=None, suppress_stdout=True):
    # hack to get around extra printouts b/c engineering build
    if suppress_stdout:
        devnull = open("/dev/null", "w")
        oldstdout_fno = os.dup(sys.stdout.fileno())
        os.dup2(devnull.fileno(), 1)

    # create input tensor
    input_tensor = torch.randn(batch_size, n, k, device=device, dtype=dtype)
    # create dense model
    model = Model(m, k).half().cuda().eval()

    # get sparse model
    pruner = WeightNormSparsifier(
        sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2
    )
    pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
    pruner.step()
    sparse_model = pruner.convert(model, mapping={nn.Linear: cuSPARSELtLinear})

    # zero out dense tensor weights for correctness check
    pruner.squash_mask()
    assert torch.allclose(
        model(input_tensor), sparse_model(input_tensor), rtol=1e-3, atol=1e-3
    )

    # get alg_id
    alg_id = sparse_model.linear.cslt.get_alg_id()

    # get latency
    sparse_measurement = benchmark.Timer(
        stmt="sparse_model(input_tensor)",
        globals={"input_tensor": input_tensor, "sparse_model": sparse_model},
    ).blocked_autorange()

    dense_measurement = benchmark.Timer(
        stmt="model(input_tensor)",
        globals={"input_tensor": input_tensor, "model": model},
    ).blocked_autorange()

    # end hack
    if suppress_stdout:
        os.dup2(oldstdout_fno, 1)

    return {
        "m": m,
        "k": k,
        "n": n,
        "batch_size": batch_size,
        "alg_id": alg_id,
        "sparse_latency (ms)": sparse_measurement.median * 1000,
        "dense_latency (ms)": dense_measurement.median * 1000,
        "speedup (d/s)": dense_measurement.median / sparse_measurement.median,
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
            "compare-alg-id",
        ],
        metavar="",
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
        results = (compare_linear(m, k, n, 1) for (m, k, n) in tqdm(bert_shapes))

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
        results = (compare_linear(mn, 10240, mn, 1) for mn in tqdm(mn_vals))

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
        results = (compare_linear(10240, k, 10240, 1) for k in tqdm(k_vals))

    elif args.mode == "distilbert-shapes":
        shapes = [
            # distilbert shapes
            (768, 3072, 768),
            # (3072, 768, 3072),
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
        batch_sizes = list(range(16, 128 + 1, 16))
        results = (
            compare_linear(m, k, n, batch_size)
            for (m, k, n), batch_size in tqdm(
                product(shapes, batch_sizes), total=len(shapes) * len(batch_sizes)
            )
        )

    elif args.mode == "compare-alg-id":
        dim_range = list(range(128, 3072 + 1, 128))
        batch_sizes = list(range(16, 128 + 1, 16))
        results = (
            compare_linear(768, 3072, n, batch_size)
            for n, batch_size in tqdm(
                product(dim_range, batch_sizes), total=len(dim_range) * len(batch_sizes)
            )

            )
    else:
        raise ValueError(f"--mode set to unrecognized value {args.mode}")

    print(f"Finished benchmark: {args.mode} ")
    df = pd.DataFrame.from_records(results)
    df.to_csv(f"{args.mode}.csv")
    print(df)
