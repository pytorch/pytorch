import operator_benchmark as op_bench
import torch


"""Microbenchmarks for MatMul operator"""

# Configs for PT Matmul operator
mm_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K", "trans_a", "trans_b"],
    attrs=[
        [1, 1, 1, True, False],
        [128, 128, 128, True, False],
        [256, 256, 256, False, True],
    ],
    cross_product_configs={"device": ["cpu", "cuda"]},
    tags=["short"],
)


mm_long_configs = op_bench.cross_product_configs(
    M=[256, 1024, 3000],
    N=[512, 4096],
    K=[512, 4096],
    trans_a=[False, True],
    trans_b=[True, False],
    device=["cuda"],
    dtype=[torch.float16, torch.bfloat16, torch.float32],
    tags=["long"],
)


class MatMulBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, trans_a, trans_b, device, dtype=torch.float):
        # Create tensors without requires_grad first, then set it separately
        # This avoids creating graph leaves that cannot be deep copied
        if trans_a:
            input_one = torch.rand(M, N, device=device, dtype=dtype)
        else:
            input_one = torch.rand(N, M, device=device, dtype=dtype).t()

        if trans_b:
            input_two = torch.rand(N, K, device=device, dtype=dtype)
        else:
            input_two = torch.rand(K, N, device=device, dtype=dtype).t()

        # Set requires_grad after tensor creation to avoid graph leaf issues
        if self.auto_set():
            input_one.requires_grad_(True)
        if self.auto_set():
            input_two.requires_grad_(True)

        self.inputs = {
            "input_one": input_one,
            "input_two": input_two,
        }
        self.set_module_name("matmul")

    def forward(self, input_one, input_two):
        return torch.matmul(input_one, input_two)

    def get_memory_traffic_bytes(self):
        """Override for matmul: (M, N) @ (N, K) -> (M, K)
        Memory traffic: read(M*N + N*K) + write(M*K)
        """
        input_one = self.inputs["input_one"]
        input_two = self.inputs["input_two"]

        # input_one and input_two are properly shaped for matmul regardless of transpose
        M, N = input_one.shape
        N_check, K = input_two.shape
        assert N == N_check, "Matrix dimensions must match for matmul"

        bytes_per_element = input_one.element_size()
        total_elements = M * N + N * K + M * K
        return total_elements * bytes_per_element


op_bench.generate_pt_test(mm_long_configs + mm_short_configs, MatMulBenchmark)
op_bench.generate_pt_gradient_test(mm_long_configs, MatMulBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
