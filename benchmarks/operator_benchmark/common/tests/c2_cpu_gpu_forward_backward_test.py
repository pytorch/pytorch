from __future__ import absolute_import, division, print_function, unicode_literals
import operator_benchmark as op_bench
from caffe2.python import core 


add_configs = op_bench.cross_product_configs(
    M=[8],
    N=[8],
    K=[8],
    tags=["short"],
    device=["cuda", "cpu"]
)

class AddBenchmark(op_bench.Caffe2BenchmarkBase):
    def init(self, M, N, K, device): 
        self.set_module_name("add")
        self.input_one = self.tensor([M, N, K], device=device) 
        self.input_two = self.tensor([M, N, K], device=device) 
        self.input_one_grad = self.tensor([M, N, K], device=device) 
        self.input_two_grad = self.tensor([M, N, K], device=device) 
        self.output = self.tensor([M, N, K], device=device)

    def forward(self):
        op = core.CreateOperator(
            "Add", [self.input_one, self.input_two], self.output, **self.args 
        )
        return op

    def backward(self):
        grad_op = core.CreateOperator(
            "AddGradient", [self.output, self.input_one, self.input_two], 
            [self.input_one_grad, self.input_two_grad], **self.args 
        )
        return grad_op


op_bench.generate_c2_test(add_configs, AddBenchmark)
op_bench.generate_c2_gradient_test(add_configs, AddBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
