import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from itertools import product

class Model(nn.Module):

    def __init__(self, m, k):
        super().__init__()
        # transposed so reversed
        self.linear = nn.Linear(k, m)

    def forward(self, x):
        return self.linear(x)

class cuSPARSELtLinear(nn.Linear):

    def forward(self, x):
        self.cslt.masked_mm(x)
        return self.res

    @classmethod
    def from_dense(cls, mod, sample_input):
        """
        convert from nn.Linear
        """
        weight_tensor = mod.weight.data
        bias_tensor = mod.bias.data

        cusparselt = cls(mod.in_features,
                         mod.out_features)

        res = torch.zeros(mod.out_features, sample_input.size(1), dtype=dtype, device=device)
        # need to be separate otherwise numeric issues
        offset = torch.zeros(mod.out_features, sample_input.size(1), dtype=dtype, device=device)

        # set up cusparselt
        cslt = torch.classes.cusparselt.CusparseLtLinear(weight_tensor)
        cslt.init(res, sample_input, bias_tensor, offset)
        cslt.prune()
        cslt.compress()
        cslt.search_matmul_algo()

        cusparselt.cslt = cslt
        cusparselt.res = res
        return cusparselt


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f},
    )
    return t0.blocked_autorange()


def get_linear(m, k, n):
    model = Model(m, k)
    model.linear.bias.data.zero_()
    model.half()
    model.cuda()

    pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(4, 1), zeros_per_block=2)
    pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
    pruner.step()
    pruner.squash_mask()

    input_tensor = torch.randn(k, n, device=device, dtype=dtype)
    sparse_linear = cuSPARSELtLinear.from_dense(model.linear, input_tensor)
    dense_linear = model.linear.weight.data.matmul

    for i in range(5):
        input_tensor = torch.randn(k, n, device=device, dtype=dtype)
        sparse_output = sparse_linear(input_tensor)
        dense_output = model.linear.weight.data.matmul(input_tensor)
        torch.testing.assert_close(sparse_output, dense_output, rtol=1e-2, atol=1e-3)

    return input_tensor, sparse_linear, dense_linear


results = []
device = "cuda"
dtype = torch.float16
torch.set_printoptions(precision=3, threshold=None, edgeitems=4, linewidth=460, profile=None, sci_mode=False)

shapes = [
    # distilbert shapes
    (768, 3072, 768),
    (3072, 768, 3072),
    # jiecao shapes
    (1024, 1536, 2048),
    (1024, 9408, 2048),
    (1024, 3200, 2048),
    (1024, 256, 9472),
    (1024, 10240, 256),
    (1024, 256, 12608),
    (1024, 2560, 1024),
    (1024, 512, 10240),
    (1024, 10240, 512),
    (1024, 2048, 1024),
    (1024, 512, 512),
    (1024, 1024, 1024),
    (1024, 2048, 2048),
    (2048, 1536, 2048),
    (2048, 9408, 2048),
    (2048, 3200, 2048),
    (2048, 256, 9472),
    (2048, 10240, 256),
    (2048, 256, 12608),
    (2048, 2560, 1024),
    (2048, 512, 10240),
    (2048, 10240, 512),
    (2048, 2048, 1024),
    (2048, 512, 512),
    (2048, 1024, 1024),
    (2048, 2048, 2048),
]
batch_sizes = [1] # [1, 4, 16, 64, 256]

for batch_size, (m, k, n) in product(batch_sizes, shapes):
    # label and sub_label are the rows
    # description is the column
    label = 'cuSPARSELt Sparse MM vs. torch.matmul'
    sub_label = f'batch_size: {batch_size:2d} | m:{m:6d} | k:{k:6d} | n:{n:6d}'

    try:
        input_tensor, sparse_linear, dense_linear = get_linear(m, k, n * batch_size)

        result = benchmark.Timer(
            stmt='sparse_linear(input_tensor)',
            globals={'input_tensor': input_tensor, 'sparse_linear': sparse_linear},
            label=label,
            sub_label=sub_label,
            description='sparse latency',
        )
        results.append(result.blocked_autorange(min_run_time=1))

        result = benchmark.Timer(
            stmt='dense_linear(input_tensor)',
            globals={'input_tensor': input_tensor, 'dense_linear': dense_linear},
            label=label,
            sub_label=sub_label,
            description='dense latency',
        )
        results.append(result.blocked_autorange(min_run_time=1))
    except Exception:
        print("FAILED: ", sub_label)

compare = benchmark.Compare(results)
compare.colorize(rowwise=True)
compare.print()

# sparse_t = benchmark_torch_function_in_microseconds(sparse_linear, input_tensor)
# dense_t = benchmark_torch_function_in_microseconds(model.linear.weight.data.matmul, input_tensor)

# print(f"Benchmark | (m, n, k): ({m} {n} {k}) sparse_t: {sparse_t:.0f}us dense_t: {dense_t:.0f}us speedup: {dense_t/sparse_t:.2f}x")
