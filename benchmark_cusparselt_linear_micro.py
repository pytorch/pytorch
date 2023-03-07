import torch
import torch.utils.benchmark as benchmark
from torch.ao.pruning import WeightNormSparsifier
from itertools import product

if __name__ == "__main__":
    results = []
    device = "cuda"
    dtype = torch.float16
    torch.set_printoptions(precision=3, threshold=None, edgeitems=4, linewidth=460, profile=None, sci_mode=False)

    from benchmark_cusparselt_linear_e2e import Model, cuSPARSELtLinear
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

    def get_linear(m, k, n):
        model = Model(m, k)
        model.linear.bias.data.zero_()
        model.half()
        model.cuda()

        pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(4, 1), zeros_per_block=2)
        pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
        pruner.step()
        pruner.squash_mask()

        input_tensor = torch.randn(n, k, device=device, dtype=dtype)
        model.linear.sample_input = input_tensor.T
        sparse_linear = cuSPARSELtLinear.from_dense(model.linear)
        
        def dense_linear(x):
            return torch.matmul(x, model.linear.weight.data.T)

        for i in range(5):
            input_tensor = torch.randn(n, k, device=device, dtype=dtype)
            sparse_output = sparse_linear(input_tensor)
            dense_output = dense_linear(input_tensor)
            assert torch.allclose(sparse_output, dense_output, rtol=1e-2, atol=1e-3)
            print("ok")

        return input_tensor, sparse_linear, dense_linear

    for batch_size, (m, k, n) in product(batch_sizes, shapes):
        # label and sub_label are the rows
        # description is the column
        label = 'cuSPARSELt Sparse MM vs. torch.matmul'
        sub_label = f'batch_size: {batch_size:2d} | m:{m:6d} | k:{k:6d} | n:{n:6d}'

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

    compare = benchmark.Compare(results)
    compare.colorize(rowwise=True)
    compare.print()
