import operator_benchmark as op_bench
import torch
import torch.optim as optim


"""Microbenchmarks for optimizer operators."""


optimizer_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["adamw", optim.AdamW],
        ["adam", optim.Adam],
        ["sgd", optim.SGD],
        ["rmsprop", optim.RMSprop],
        ["adagrad", optim.Adagrad],
    ],
)

optimizer_configs_long = op_bench.cross_product_configs(
    shape=[(100000,), (1000000,), (10000000,)],
    device=["cuda"],
    tags=["long"],
)


class OptimizerBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, op_func, device, shape):
        self.op_func = op_func
        self.param = torch.randn(
            shape, device=device, requires_grad=True, dtype=torch.float32
        )
        self.param.grad = torch.randn(shape, device=device)

        kwargs = {"momentum": 0.9} if op_func == optim.SGD else {}
        self.optimizer = op_func([self.param], lr=0.001, **kwargs)

        self.inputs = {"dummy": self.param}

    def forward(self, dummy):
        self.optimizer.step()
        return self.param

    def get_memory_traffic_bytes(self):
        # Memory traffic calculation for bandwidth
        total_elements = self.param.numel()
        bytes_per_element = self.param.element_size()
        # SGD w/ momentum: read(param, grad, momentum) + write(param, momentum) = 5x
        # Adam/AdamW: read(param, grad, exp_avg, exp_avg_sq) + write(param, exp_avg, exp_avg_sq) = 7x
        # Adagrad/RMSprop: read(param, grad, state) + write(param, state) = 5x
        if self.op_func in (optim.Adam, optim.AdamW):
            memory_multiplier = 7
        else:
            memory_multiplier = 5
        return total_elements * bytes_per_element * memory_multiplier


op_bench.generate_pt_tests_from_op_list(
    optimizer_list, optimizer_configs_long, OptimizerBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
