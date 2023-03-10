import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from itertools import product
from torch.profiler import profile, record_function, ProfilerActivity
from torch.ao.nn.sparse.cusparselt_linear import cuSPARSELtLinear


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

if __name__ == "__main__":
    results = []

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
    batch_sizes = [1, 4, 16, 64, 256]


    for (m, k, n), batch_size in product(shapes, batch_sizes):
        print(m, k, n, batch_size)
        # label and sub_label are the rows
        # description is the column
        # label = "cusparselt vs linear"
        label = f'm, k, n: {m:5d}, {k:5d}, {n:5d} ]   [ cuSPARSELtLinear vs nn.Linear'
        sub_label = f'batch_size: {batch_size:4d}'

        model = Model(m, k)
        model.half()
        model.cuda()
        model.eval()

        pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2)
        pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
        pruner.step()
        sparse_linear = pruner.convert(model, mapping={nn.Linear: cuSPARSELtLinear})

        pruner.squash_mask()
        dense_linear = model

        for i in range(5):
            input_tensor = torch.randn(batch_size, n, k, device=device, dtype=dtype)
            dense_output = dense_linear(input_tensor)
            sparse_output = sparse_linear(input_tensor)
            assert torch.allclose(sparse_output, dense_output, rtol=1e-3, atol=1e-3)


        measurement = benchmark.Timer(
            stmt='sparse_linear(input_tensor)',
            globals={'input_tensor': input_tensor, 'sparse_linear': sparse_linear},
            label=label,
            sub_label=sub_label,
            description='sparse latency',
        ).blocked_autorange()
        results.append(measurement)

        measurement = benchmark.Timer(
            stmt='dense_linear(input_tensor)',
            globals={'input_tensor': input_tensor, 'dense_linear': dense_linear},
            label=label,
            sub_label=sub_label,
            description='dense latency',
        ).blocked_autorange()
        results.append(measurement)

        # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
            # with record_function("cusparselt"):
                # sparse_linear(input_tensor)

        # prof.export_stacks(f"{m}_{k}_{n}_cusparselt_profiler_stacks.txt", "self_cuda_time_total")

    compare = benchmark.Compare(results)
    compare.colorize(rowwise=True)
    compare.print()
