# %%
import csv
import json


# scuba columns:
# input_shapes,Hits,Samples,CPU time Us (sum)
# Load the scuba dataset


# TODO: check that the improvements still show up in Eval mode. -- Module Eval mode is almost always slower than the C binding
# TODO: Distribution of speedup both weighted and not weighted on frequency.
# TODO: Also test FP16 for GPU
# TODO Less important: test single threaded CPU


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
print(*torch.__config__.show().split("\n"), sep="\n")


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


class MatmulLinear(torch.nn.Module):
    def __init__(self, weight, bias):
        super(MatmulLinear, self).__init__()
        self.weight_T = torch.nn.Parameter(weight.transpose(-1, -2).contiguous())
        self.bias = torch.nn.Parameter(bias)

    def forward(self, inp):
        return inp @ self.weight_T + self.bias


# Linear Nodes
def linear_setup(device, sizes):
    inp = torch.rand(sizes[0]).to(device)
    weights = torch.rand(sizes[1]).to(device)
    weights_T = weights.T.contiguous()
    biases = torch.rand(sizes[2]).to(device)
    linear_layer = torch.nn.Linear(sizes[1][0], sizes[1][1], bias=True)
    linear_layer.weight = torch.nn.parameter.Parameter(weights, False)
    linear_layer.bias = torch.nn.parameter.Parameter(biases, False)
    if device == "cpu":
        # Also prepare nnc tests
        linear_script = torch.jit.script(linear_layer)
        linear_frozen = torch.jit.optimize_for_inference(linear_script)
        
        # Converting to MKLDNN is costly - To 
        dense_conv = linear_frozen.graph.findNode("aten::to_dense")
        dense_conv.output().replaceAllUsesWith(list(dense_conv.inputs())[0])
        torch._C._jit_pass_dce(linear_frozen.graph)

        matmul_linear = MatmulLinear(weights, biases)
        linear_m_script = torch.jit.script(matmul_linear)
        linear_m_frozen = torch.jit.optimize_for_inference(linear_m_script)
        # dense_conv = linear_m_frozen.graph.findNode("aten::to_dense")
        # dense_conv.output().replaceAllUsesWith(list(dense_conv.inputs())[0])
        torch._C._jit_pass_dce(linear_m_frozen.graph)

    return locals()


def linear(*, inp, weights, biases, **kwargs):
    return torch._C._nn.linear(inp, weights, biases)


# def linear_layer(*, inp, linear_layer, **kwargs):
#     return linear_layer.forward(inp)


def linear_matmul(*, inp, biases, weights_T, **kwargs):
    # TODO: We want to make the weight contiguous at setup time
    return inp @ weights_T + biases


def linear_transposed(*, inp, weights, biases, **kwargs):
    return (weights @ inp.transpose(-1, -2)).transpose(-1, -2) + biases


def linear_module(*, inp, linear_frozen, **kwargs):
    return linear_frozen.forward(inp)


def linear_matmul_module(*, inp, linear_m_frozen, **kwargs):
    return linear_m_frozen(inp)


def benchmark(func1, func2, setup, inp_sizes, device):
    results = []
    for sizes in inp_sizes:
        # print(f"Benchmarking {sizes} on {device}")
        kwargs = setup(device, sizes)

        NITER = 40
        NWARMUP = 10

        # First test compatible
        # torch.testing.assert_allclose(func1(*args), func2(*args))

        def benchmark(func):
            if device == "cuda":
                torch.cuda.synchronize()
            for _ in range(NWARMUP):
                func(**kwargs)
                if device == "cuda":
                    torch.cuda.synchronize()
            s = time.time()
            for _ in range(NITER):
                func(**kwargs)
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


tests = [
    [linear, linear_module, "mkldnn_1conv_vs_orig"],
    # [linear, linear_matmul, "orig_mm_vs_orig"],
    # [linear, linear_matmul_module, "mkldnn_mm_1conv_vs_orig"],
    # [linear_module, linear_matmul_module, "mkldnn_mm_vs_mlkdnn"],
    # [linear_matmul, linear_matmul_module, "mkldnn_mm_vs_orig_mm"],
]
set_1cpu = False

for ref_func, test_func, test_name in tests:
    # test_name = "mkldnn_2"
    # torch.set_default_dtype(torch.float16)
    # ref_func = linear_module
    # test_func = linear_t_module
    
    print(f"starting tests for {test_name}")
    github_issue_sizes = [
        [[14, 768], [768 * 3, 768], [768 * 3]],
        [[64, 1024], [4096 * 3, 1024], [4096 * 3]],
    ]

    # devices = ["cpu", "cuda"]
    devices = ["cpu"]

    for raw_device in devices:
        if set_1cpu and raw_device != "1cpu":
            raise Exception("Can't run benchmark after threads have already been set to 1")
        
        if raw_device == "1cpu":
            device = "cpu"

            if not set_1cpu:
                set_1cpu = True
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
        else:
            device = raw_device

        # res = benchmark(matmul, matmul_transposed, matmul_setup, matmul_shapes)
        linear_res = benchmark(ref_func, test_func, linear_setup, linear_shapes(), device)
        ordered_res = sorted(linear_res, key=lambda x: x["speedup"], reverse=True)

        with open(f"linear_{test_name}_{raw_device}.json", "w") as f:
            f.write(pprint.pformat(ordered_res).replace("'", '"'))

        # Test the exact examples given in the concat examples
        # Note that dimms are x3 to account for the concatting of 3 tensors.
        github_res = benchmark(
            ref_func, test_func, linear_setup, github_issue_sizes, device
        )
        pprint.pprint(github_res)


# %%
# %%
