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
    num_params=[1, 10, 100],
    param_size=[100000, 1000000, 10000000],
    device=["cuda"],
    tags=["long"],
)


class OptimizerBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, op_func, device, shape=None, num_params=None, param_size=None):
        if shape is not None:
            num_params = num_params if num_params is not None else 1
            self.params = [
                torch.randn(shape, device=device, requires_grad=True)
                for _ in range(num_params)
            ]
            for param in self.params:
                param.grad = torch.randn(shape, device=device)
        else:
            self.params = [
                torch.randn(param_size, device=device, requires_grad=True)
                for _ in range(num_params)
            ]
            for param in self.params:
                param.grad = torch.randn_like(param)

        kwargs = {"momentum": 0.9} if op_func == optim.SGD else {}
        self.optimizer = op_func(self.params, lr=0.001, **kwargs)

        # Memory traffic calculation for bandwidth
        self.total_elements = sum(p.numel() for p in self.params)
        self.bytes_per_element = self.params[0].element_size()
        # SGD w/ momentum: read(param, grad, momentum) + write(param, momentum) = 5x
        # Adam/AdamW: read(param, grad, exp_avg, exp_avg_sq) + write(param, exp_avg, exp_avg_sq) = 7x
        # Adagrad/RMSprop: read(param, grad, state) + write(param, state) = 5x
        if op_func in (optim.Adam, optim.AdamW):
            self.memory_multiplier = 7
        else:
            self.memory_multiplier = 5

        self.inputs = {"dummy": self.params[0]}

    def forward(self, dummy):
        self.optimizer.step()
        for param in self.params:
            param.grad = torch.randn_like(param)
        return self.params[0]

    def get_memory_traffic_bytes(self):
        return self.total_elements * self.bytes_per_element * self.memory_multiplier


op_bench.generate_pt_tests_from_op_list(
    optimizer_list, optimizer_configs_long, OptimizerBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
