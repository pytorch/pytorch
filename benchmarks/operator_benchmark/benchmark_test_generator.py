from benchmark_core import _register_test
from benchmark_pytorch import create_pytorch_op_test_case


def generate_pt_test(configs, pt_bench_op):
    """ This function creates PyTorch op test based on the given operator
    """
    _register_test(configs, pt_bench_op, create_pytorch_op_test_case, False)


def generate_pt_gradient_test(configs, pt_bench_op):
    """ This function creates PyTorch op test based on the given operator
    """
    _register_test(configs, pt_bench_op, create_pytorch_op_test_case, True)


def generate_pt_tests_from_op_list(ops_list, configs, pt_bench_op):
    """ This function creates pt op tests one by one from a list of dictionaries.
        ops_list is a list of dictionary. Each dictionary includes
        the name of the operator and the math operation. Here is an example of using this API:
        unary_ops_configs = op_bench.config_list(
            attrs=[...],
            attr_names=["M", "N"],
        )
        unary_ops_list = op_bench.op_list(
            attr_names=["op_name", "op_func"],
            attrs=[
                ["abs", torch.abs],
            ],
        )
        class UnaryOpBenchmark(op_bench.TorchBenchmarkBase):
            def init(self, M, N, op_name, op_func):
                ...
            def forward(self):
                ...
        op_bench.generate_pt_tests_from_op_list(unary_ops_list, unary_ops_configs, UnaryOpBenchmark)
    """
    for op in ops_list:
        _register_test(configs, pt_bench_op, create_pytorch_op_test_case, False, op)

def generate_pt_gradient_tests_from_op_list(ops_list, configs, pt_bench_op):
    for op in ops_list:
        _register_test(configs, pt_bench_op, create_pytorch_op_test_case, True, op)
