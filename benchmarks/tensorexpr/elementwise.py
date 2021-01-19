from . import benchmark
import itertools
import numpy as np
import torch
import scipy.special

# A template class for elementwise operations.
# A derived class will override the class instance to customize its behavior.
class ElementBench(benchmark.Benchmark):
    # List of customization class variables.
    op_str = None
    binary_op_pt_func = None
    binary_op_np_func = None
    unary_op_pt_func = None
    unary_op_np_func = None
    split_input = True

    def __init__(self, mode, device, dtype, N):
        super().__init__(mode, device, dtype)
        self.N = N
        self.d1 = self.rand([N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d2 = self.rand([N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d3 = self.rand([N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d4 = self.rand([N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.inputs = [self.d1, self.d2, self.d3, self.d4]
        self.deterministic = "rand" not in self.op_str

    def _eval(self, d1, d2, d3, d4, binary_op, unary_op):
        if not binary_op:
            def binary_op(x, y):
                return x + y
        if not unary_op:
            def unary_op(x):
                return x

        if self.split_input:
            d1 = unary_op(d1)
            d2 = unary_op(d2)
            d3 = unary_op(d3)
            d4 = unary_op(d4)
        else:
            d2 = unary_op(d1 + 0.001)
            d3 = unary_op(d1 + 0.002)
            d4 = unary_op(d1 + 0.003)
            d1 = unary_op(d1)
        a = binary_op(d1, d2)
        b = binary_op(d3, d4)
        c = a + b
        return c

    def forward(self, d1, d2, d3, d4):
        binary_op = self.__class__.binary_op_pt_func
        unary_op = self.__class__.unary_op_pt_func
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def reference(self):
        binary_op = self.__class__.binary_op_np_func
        unary_op = self.__class__.unary_op_np_func
        [d1, d2, d3, d4] = [self.numpy(d) for d in [self.d1, self.d2, self.d3, self.d4]]
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def config(self):
        return [self.N]

    @classmethod
    def module(cls):
        return "element_" + cls.op_str

    def memory_workload(self):
        input_count = len(self.inputs)
        if self.mode == "fwd":
            if self.split_input:
                sol_count = input_count + 1
                algorithmic_count = input_count + 1
            else:
                sol_count = 1 + 1
                algorithmic_count = 1 + 1
            if "rand" in self.op_str:
                sol_count = 1
                algorithmic_count = 1
        else:
            if self.split_input:
                sol_count = (input_count + 1) + (1 + input_count)
                algorithmic_count = (input_count + 1) + ((2 + 1) * input_count)
            else:
                sol_count = 1 + 1
                algorithmic_count = 1 + 1
            if "rand" in self.op_str:
                sol_count = 1
                algorithmic_count = 1

        buffer_size = self.N
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [[1 << 25]]


def register_element_ops():
    binary_op_list = [
        ["mul", lambda a, b: a * b],
        ["add", lambda a, b: a + b],
        ["sub", lambda a, b: a - b],
        ["div", lambda a, b: a / (b + 1e-4)],
        [
            "pow",
            lambda a, b: torch.pow(a, b),
            lambda a, b: np.power(a, b),
        ],  # no fuson triggered
        ["max", lambda a, b: torch.max(a, b), lambda a, b: np.maximum(a, b)],
        ["min", lambda a, b: torch.min(a, b), lambda a, b: np.minimum(a, b)],
    ]

    unary_op_list = [
        ["erf", lambda x: torch.erf(x), lambda x: scipy.special.erf(x)],
        ["exp", lambda x: torch.exp(x), lambda x: np.exp(x)],
        ["sin", lambda x: torch.sin(x), lambda x: np.sin(x)],
        ["cos", lambda x: torch.cos(x), lambda x: np.cos(x)],
        ["rand_like", lambda x: torch.rand_like(x), lambda x: np.random.rand(*x.shape)],
    ]

    for split_input, binary_op in itertools.product([True, False], binary_op_list):
        # Make a copy of ElementBench
        if len(binary_op) == 2:
            [op_str, op_pt_func] = binary_op
            op_np_func = op_pt_func
        elif len(binary_op) == 3:
            [op_str, op_pt_func, op_np_func] = binary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("ElementBench_" + op_str, (ElementBench,), {})
        bm_cls.op_str = op_str
        bm_cls.binary_op_pt_func = op_pt_func
        bm_cls.binary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)

    for split_input, unary_op in itertools.product([True, False], unary_op_list):
        # Make a copy of ElementBench
        if len(unary_op) == 2:
            [op_str, op_pt_func] = unary_op
            op_np_func = op_pt_func
        elif len(unary_op) == 3:
            [op_str, op_pt_func, op_np_func] = unary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("ElementBench_" + op_str, (ElementBench,), {})
        bm_cls.op_str = op_str
        bm_cls.unary_op_pt_func = op_pt_func
        bm_cls.unary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)


# benchmark.register_benchmark_class(ElementMulBench)
register_element_ops()


class SimpleElementBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, N):
        super().__init__(mode, device, dtype)
        self.N = N
        self.data = self.rand([N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.inputs = [self.data]

    def forward(self, data):
        a = data + 0.001
        b = a + 0.002
        return b

    def reference(self):
        binary_op = self.__class__.binary_op_np_func
        unary_op = self.__class__.unary_op_np_func
        [d1, d2, d3, d4] = [self.numpy(d) for d in [self.d1, self.d2, self.d3, self.d4]]
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def config(self):
        return [self.N]

    @staticmethod
    def input_iterable():
        return True

    @classmethod
    def module(cls):
        return "simple_element"

    def memory_workload(self):
        input_count = len(self.inputs)
        if self.mode == "fwd":
            sol_count = 2
            algorithmic_count = 2
        else:
            sol_count = 2
            algorithmic_count = 2

        buffer_size = self.N
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [[1 << 25]]


benchmark.register_benchmark_class(SimpleElementBench)


class DynamicSimpleElementBench(benchmark.DynamicShape, SimpleElementBench):
    def __init__(self, mode, device, dtype, N):
        benchmark.DynamicShape.__init__(self)
        SimpleElementBench.__init__(self, mode, device, dtype, N)

    @classmethod
    def module(cls):
        return "simple_dynamic_element"

    def instantiate_input(self):
        N, = self.rand_shape([self.N])
        data = self.rand([N], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        self.inputs = [data]


benchmark.register_benchmark_class(DynamicSimpleElementBench)
