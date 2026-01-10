import operator_benchmark as op_bench
import torch


"""Microbenchmarks for add_(matmul) operator. Supports both Caffe2/PyTorch."""

# Configs for PT add operator
addmm_long_configs = op_bench.cross_product_configs(
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    device=["cuda"],
    tags=["long"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
)


addmm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
        "dtype": [torch.float],
    },
    tags=["short"],
)


"""Mircobenchmark for addmm operator."""


class AddmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                M, K, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "mat1": torch.rand(
                M, N, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "mat2": torch.rand(
                N, K, device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.set_module_name("addmm")

    def forward(self, input_one, mat1, mat2):
        return torch.addmm(input_one, mat1, mat2)

    def get_memory_traffic_bytes(self):
        """Override for addmm: input + (mat1 @ mat2) -> (M, K)
        addmm computes: input_one (M, K) + mat1 (M, N) @ mat2 (N, K)
        Memory traffic: read(M*K + M*N + N*K) + write(M*K)
        """
        input_one = self.inputs["input_one"]
        mat1 = self.inputs["mat1"]
        mat2 = self.inputs["mat2"]

        M, K = input_one.shape
        M_check, N = mat1.shape
        N_check, K_check = mat2.shape
        assert M == M_check and K == K_check and N == N_check, (
            "Matrix dimensions must match"
        )

        bytes_per_element = input_one.element_size()
        total_elements = M * K + M * N + N * K + M * K
        return total_elements * bytes_per_element


op_bench.generate_pt_test(addmm_short_configs + addmm_long_configs, AddmmBenchmark)
op_bench.generate_pt_gradient_test(addmm_long_configs, AddmmBenchmark)

"""Mircobenchmark for addbmm operator."""


class AddbmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                (M, N), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "batch1": torch.rand(
                (B, M, K), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "batch2": torch.rand(
                (
                    B,
                    K,
                    N,
                ),
                device=device,
                requires_grad=self.auto_set(),
                dtype=dtype,
            ),
        }
        self.set_module_name("addbmm")

    def forward(self, input_one, batch1, batch2):
        return torch.addbmm(input_one, batch1, batch2)

    def get_memory_traffic_bytes(self):
        """Override for addbmm: input + sum(batch1[i] @ batch2[i]) -> (M, N)
        addbmm computes: input_one (M, N) + sum over batch of batch1 (B, M, K) @ batch2 (B, K, N)
        Memory traffic: read(M*N + B*M*K + B*K*N) + write(M*N)
        """
        input_one = self.inputs["input_one"]
        batch1 = self.inputs["batch1"]
        batch2 = self.inputs["batch2"]

        M, N = input_one.shape
        B, M_check, K = batch1.shape
        B_check, K_check, N_check = batch2.shape
        assert M == M_check and N == N_check and B == B_check and K == K_check, (
            "Dimensions must match"
        )

        bytes_per_element = input_one.element_size()
        total_elements = M * N + B * M * K + B * K * N + M * N
        return total_elements * bytes_per_element


addbmm_long_configs = op_bench.cross_product_configs(
    B=[8, 32],
    M=[256, 1024],
    N=[256, 1024],
    K=[64, 128],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["long"],
)
addbmm_short_configs = op_bench.cross_product_configs(
    B=[1, 8],
    M=[8, 128],
    N=[32, 64],
    K=[256, 512],
    device=["cpu", "cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["short"],
)

op_bench.generate_pt_test(addbmm_long_configs + addbmm_short_configs, AddbmmBenchmark)
op_bench.generate_pt_gradient_test(addbmm_long_configs, AddbmmBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
