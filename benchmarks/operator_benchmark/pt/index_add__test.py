import numpy

import operator_benchmark as op_bench

import torch


"""Microbenchmarks for index_add_ operator."""


configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "dim"],
    attrs=[[8, 32, 1, 0], [256, 512, 1, 1], [512, 512, 1, 2]],
    cross_product_configs={"device": ["cpu"], "dtype": [torch.float]},
    tags=["short"],
)


configs_long = op_bench.cross_product_configs(
    M=[1, 128, 1024],
    N=[2, 256, 512],
    K=[1, 2, 8],
    dim=[0, 1, 2],
    device=["cpu", "cuda"],
    dtype=[torch.float],
    tags=["long"],
)


class IndexAddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, dim, dtype, device):
        # creating the original tensor
        tensor = torch.rand(M, N, K, dtype=dtype, device=device)

        # creating index
        index_max_len = tensor.shape[dim]
        index_len = numpy.random.randint(1, index_max_len + 1)
        index = torch.tensor(
            numpy.random.choice(index_max_len, index_len, replace=False), device=device
        )

        src_dims = [M, N, K]
        src_dims[dim] = index_len
        source = torch.rand(*src_dims, dtype=dtype, device=device)

        self.inputs = {
            "tensor": tensor,
            "dim": dim,
            "index": index,
            "source": source,
        }
        self.set_module_name("index_add_")

    def forward(self, tensor, dim, index, source):
        return tensor.index_add_(dim, index, source)


op_bench.generate_pt_test(configs_short + configs_long, IndexAddBenchmark)


configs_contiguous = op_bench.cross_product_configs(
    M=[4096],
    N=[8192],
    D=[32, 128, 130, 1024, 4096],
    alpha=[1, -0.5],
    pattern=["uniform", "hot"],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16],
    index_dtype=[torch.int32, torch.int64],
    tags=["long"],
)


class IndexAddContiguousBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, D, alpha, pattern, device, dtype, index_dtype):
        generator = torch.Generator(device=device).manual_seed(142)
        tensor = torch.zeros(M, D, dtype=dtype, device=device)
        source = torch.randint(-4, 5, (N, D), device=device, generator=generator)
        source = source.to(dtype) / 128
        index = torch.randint(
            0,
            M if pattern == "uniform" else 8,
            (N,),
            dtype=index_dtype,
            device=device,
            generator=generator,
        )
        self.inputs = {
            "tensor": tensor,
            "index": index,
            "source": source,
            "alpha": alpha,
        }
        self.set_module_name("index_add_contiguous_")

    def forward(self, tensor, index, source, alpha):
        return tensor.index_add_(0, index, source, alpha=alpha)


op_bench.generate_pt_test(configs_contiguous, IndexAddContiguousBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
