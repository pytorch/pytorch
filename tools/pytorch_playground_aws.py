# %%
import csv
import json


# scuba columns:
# input_shapes,Hits,Samples,CPU time Us (sum)
# Load the scuba dataset


def load_shapes(filename):
    with open(filename, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        scuba_data = list(reader)

    return [json.loads(x["input_shapes"]) for x in scuba_data]


def matmul_shapes():
    matmul_shapes = load_shapes("scuba-data-2021-09-09T22_15_45.513Z.csv")

    # Filter out all matmuls with special outputs
    # Filter out all shapes with dim < 2
    matmul_shapes = [
        x for x in matmul_shapes if len(x) == 2 and all(len(item) >= 2 for item in x)
    ]
    return matmul_shapes


def linear_shapes():
    raw_shapes = load_shapes("scuba-data-2021-09-10_linear.csv")
    linear_shapes = []
    for shape in raw_shapes:
        if len(shape) != 3:
            continue
        if len(shape[0]) < 2 or len(shape[1]) < 2:
            continue
        if len(shape[2]) != 1:
            continue
        linear_shapes.append(shape)
    return linear_shapes


# %%

# Now do the benchmark
import torch
import time
import pprint


def matmul_setup(device, sizes):
    inp = torch.rand(sizes[0]).to(device)
    weights = torch.rand(sizes[1]).to(device)
    args = (inp, weights)
    return args


def matmul(*args):
    inp, weights = args
    return inp @ weights


def matmul_transposed(*args):
    inp, weights = args
    return (weights.transpose(-1, -2) @ inp.transpose(-1, -2)).transpose(-1, -2)


# Linear Nodes
def linear_setup(device, sizes):
    inp = torch.rand(sizes[0]).to(device)
    weights = torch.rand(sizes[1]).to(device)
    biases = torch.rand(sizes[2]).to(device)
    return inp, weights, biases


def linear(inp, weights, biases):
    return torch._C._nn.linear(inp, weights, biases)


def linear_matmul(inp, weights, biases):
    return inp @ weights.transpose(-1, -2) + biases


def linear_transposed(inp, weights, biases):
    return (weights @ inp.transpose(-1, -2)).transpose(-1, -2) + biases


device = "cpu"
# device = "cuda"


def benchmark(func1, func2, setup, inp_sizes):
    results = []
    for sizes in inp_sizes:
        args = setup(device, sizes)

        NITER = 40
        NWARMUP = 10

        # First test compatible
        # torch.testing.assert_allclose(func1(*args), func2(*args))

        def benchmark(func):
            if device == "cuda":
                torch.cuda.synchronize()
            for _ in range(NWARMUP):
                func(*args)
                if device == "cuda":
                    torch.cuda.synchronize()
            s = time.time()
            for _ in range(NITER):
                func(*args)
                if device == "cuda":
                    torch.cuda.synchronize()
            e = time.time()
            return (e - s) / NITER

        res1_list, res2_list = [], []
        for _ in range(5):
            res1_list.append(benchmark(func1))
            res2_list.append(benchmark(func2))

        # Based on the advice from the link, I am using the min
        # https://docs.python.org/3/library/timeit.html#timeit.Timer.timeit
        res1 = min(res1_list)
        res2 = min(res2_list)

        results.append(
            {
                "device": device,
                "input_size": sizes,
                "orig": res1,
                "new": res2,
                "speedup": res1 / res2,
            }
        )
    return results


# res = benchmark(matmul, matmul_transposed, matmul_setup, matmul_shapes)
linear_res = benchmark(linear, linear_transposed, linear_setup, linear_shapes())
ordered_res = sorted(linear_res, key=lambda x: x["speedup"], reverse=True)

with open(f"linear_benchmark_{device}.json", "w") as f:
    f.write(pprint.pformat(ordered_res))

# Save the benchmark results


# Test the exact examples given in the concat examples
# Note that dimms are x3 to account for the concatting of 3 tensors.

github_issue_sizes = [
    [[14, 768], [768 * 3, 768], [768 * 3]],
    [[64, 1024], [4096 * 3, 1024], [4096 * 3]],
]


github_res = benchmark(linear, linear_transposed, linear_setup, github_issue_sizes)
pprint.pprint(github_res)
