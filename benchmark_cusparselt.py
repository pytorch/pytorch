import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from itertools import product
from torch.profiler import profile, record_function, ProfilerActivity


device = "cuda"
dtype = torch.float16
torch.set_printoptions(precision=3, threshold=None, edgeitems=4, linewidth=460, profile=None, sci_mode=False)

class Model(nn.Module):

    def __init__(self, m, k):
        super().__init__()
        # transposed so reversed
        self.linear = nn.Linear(k, m)

    def forward(self, x):
        return self.linear(x)

class cuSPARSELtLinear(nn.Linear):

    def forward(self, x):
        return self.cslt.masked_mm(x.T).T

    @classmethod
    def from_dense(cls, mod):
        """
        convert from nn.Linear
        """
        weight_tensor = mod.weight.data
        bias_tensor = mod.bias.data

        cusparselt = cls(mod.in_features,
                         mod.out_features)

        # set up cusparselt
        cslt = torch.classes.cusparselt.CusparseLtLinear()
        cslt.init(weight_tensor, bias_tensor)
        cslt.prune()
        cslt.compress()

        cusparselt.cslt = cslt
        return cusparselt

def get_linear(m, k, n):
    model = Model(m, k)
    model.half()
    model.cuda()

    pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(4, 1), zeros_per_block=2)
    pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
    pruner.step()
    pruner.squash_mask()

    # converted_model = pruner.convert(model, mapping={nn.Linear: cuSPARSELtLinear})
    # sparse_linear = converted_model
    # dense_linear = model
    sparse_linear = cuSPARSELtLinear.from_dense(model.linear)
    dense_linear = model.linear

    for i in range(5):
        input_tensor = torch.randn(n, k, device=device, dtype=dtype)
        dense_output = dense_linear(input_tensor)
        sparse_output = sparse_linear(input_tensor)
        assert torch.allclose(sparse_output, dense_output, rtol=1e-3, atol=1e-3)

    return input_tensor, pruner, sparse_linear, dense_linear

def get_matmul(m, k, n):
    model = Model(m, k)
    model.linear.bias.data.zero_()
    model.half()
    model.cuda()

    pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(4, 1), zeros_per_block=2)
    pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
    pruner.step()
    pruner.squash_mask()

    sparse_matmul = cuSPARSELtLinear.from_dense(model.linear)

    def dense_matmul(x):
        return torch.matmul(x, model.linear.weight.data.T)

    for i in range(5):
        input_tensor = torch.randn(n, k, device=device, dtype=dtype)
        sparse_output = sparse_matmul(input_tensor)
        dense_output = dense_matmul(input_tensor)
        assert torch.allclose(sparse_output, dense_output, rtol=1e-2, atol=1e-3)

    return input_tensor, pruner, sparse_matmul, dense_matmul

if __name__ == "__main__":
    results = []

    shapes = [
        # distilbert shapes
        (768, 3072, 768),
        # (3072, 768, 3072),
        # # jiecao shapes
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
    batch_sizes = [1] # [1, 4, 16, 64, 256]


    for batch_size, (m, k, n) in product(batch_sizes, shapes):
        # label and sub_label are the rows
        # description is the column
        label = 'cuSPARSELt Linear vs. nn.Linear'
        sub_label = f'batch_size: {batch_size:2d} | m:{m:6d} | k:{k:6d} | n:{n:6d}'

        input_tensor, pruner, sparse_linear, dense_linear = get_linear(m, k, n)

        measurement = benchmark.Timer(
            stmt='sparse_linear(input_tensor)',
            globals={'input_tensor': input_tensor, 'sparse_linear': sparse_linear},
            label=label,
            sub_label=sub_label,
            description='sparse latency',
        ).blocked_autorange(min_run_time=1)
        results.append(measurement)
        measurement.metadata = {
            'device': device,
        }
        results.append(measurement)

        measurement = benchmark.Timer(
            stmt='dense_linear(input_tensor)',
            globals={'input_tensor': input_tensor, 'dense_linear': dense_linear},
            label=label,
            sub_label=sub_label,
            description='dense latency',
        ).blocked_autorange(min_run_time=1)
        results.append(measurement)

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
            with record_function("cusparselt"):
                sparse_linear(input_tensor)

        prof.export_stacks(f"{m}_{k}_{n}_cusparselt_profiler_stacks.txt", "self_cuda_time_total")

    compare = benchmark.Compare(results)
    compare.colorize(rowwise=True)
    compare.print()
