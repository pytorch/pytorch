from . import benchmark
import itertools
import numpy as np
import torch


class BroadcastMulBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, case, M, N, K):
        super().__init__(mode, device, dtype)
        self.case = case
        self.M = M
        self.N = N
        self.K = K

        if case == "row":
            self.d1 = self.rand(
                [M, N, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(
                [M, 1, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        elif case == "mid":
            self.d1 = self.rand(
                [M, N, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(
                [1, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        elif case == "col":
            self.d1 = self.rand(
                [M, 1, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(
                [1, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        else:
            raise ValueError("invalid case: %s" % (case))

        self.inputs = [self.d1, self.d2]

    def forward(self, d1, d2):
        y = d1 + d2
        return y

    def reference(self):
        return self.numpy(self.d1) + self.numpy(self.d2)

    def config(self):
        return [self.M, self.N, self.K]

    @staticmethod
    def default_configs():
        return [[128, 256, 128]]

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + (1 + 1)

        buffer_size = self.M * self.N * self.K
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }


class BroadcastRowBench(BroadcastMulBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super().__init__(mode, device, dtype, "row", M, N, K)

    @staticmethod
    def module():
        return "broadcast_row"


class BroadcastMidBench(BroadcastMulBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super().__init__(mode, device, dtype, "mid", M, N, K)

    @staticmethod
    def module():
        return "broadcast_mid"


class BroadcastColBench(BroadcastMulBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super().__init__(mode, device, dtype, "col", M, N, K)

    @staticmethod
    def module():
        return "broadcast_col"


class BroadcastThreeArgs(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, M, N, K, L):
        super().__init__(mode, device, dtype)
        self.M = M
        self.N = N
        self.K = K
        self.L = L

        self.d1 = self.rand([M, N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d2 = self.rand([K, M, 1], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d3 = self.rand(
            [L, K, 1, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
        )

        self.inputs = [self.d1, self.d2, self.d3]

    def forward(self, d1, d2, d3):
        y = d1 + d2 + d3
        return y

    def reference(self):
        return self.numpy(self.d1) + self.numpy(self.d2) + self.numpy(self.d3)

    def config(self):
        return [self.M, self.N, self.K, self.L]

    @staticmethod
    def default_configs():
        return [[32, 16, 64, 128]]

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + (1 + 1 + 1)

        buffer_size = self.M * self.N * self.K * self.L * 4
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def module():
        return "broadcast_3args"


# benchmark.register_benchmark_class(BroadcastRowBench)
# benchmark.register_benchmark_class(BroadcastMidBench)
# benchmark.register_benchmark_class(BroadcastColBench)
# benchmark.register_benchmark_class(BroadcastThreeArgs)

# TODO: merge this with elementwise bench
# A template class for elementwise operations.
# A derived class will override the class instance to customize its behavior.
class BroadcastBench(benchmark.Benchmark):
    # List of customization class variables.
    op_str = None
    binary_op_pt_func = None
    binary_op_np_func = None
    unary_op_pt_func = None
    unary_op_np_func = None
    split_input = True

    def __init__(self, mode, device, dtype, M, N, K):
        super().__init__(mode, device, dtype)
        self.M = M
        self.N = N
        self.K = K
        self.d1 = self.rand([M, N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d2 = self.rand([K, 1, N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d3 = self.rand([M, N], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.d4 = self.rand([K, M, 1], device=device, dtype=dtype, requires_grad=self.requires_grad)
        self.inputs = [self.d1, self.d2, self.d3, self.d4]

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
            d1, d2, d3, d4 = (
                unary_op(d1),
                unary_op(d2),
                unary_op(d1 + 0.001),
                unary_op(d4),
            )
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
        return [self.M, self.N, self.K]

    @classmethod
    def module(cls):
        return "broadcast_" + cls.op_str

    def memory_workload(self):
        input_count = len(self.inputs)
        if self.mode == "fwd":
            if self.split_input:
                sol_count = 1
                algorithmic_count = 1
            else:
                sol_count = 1
                algorithmic_count = 1
        else:
            if self.split_input:
                sol_count = 1
                algorithmic_count = input_count
            else:
                sol_count = 1
                algorithmic_count = input_count

        buffer_size = self.M * self.N * self.K * 4
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    @staticmethod
    def default_configs():
        return [[1 << 8, 1 << 7, 1 << 9]]


def register_broadcast_ops():
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
        ["erf", lambda x: torch.erf(x), lambda x: np.erf(x)],
        ["exp", lambda x: torch.exp(x), lambda x: np.exp(x)],
        ["sin", lambda x: torch.sin(x), lambda x: np.sin(x)],
        ["cos", lambda x: torch.cos(x), lambda x: np.cos(x)],
    ]

    for split_input, binary_op in itertools.product([True, False], binary_op_list):
        # Make a copy of BroadcastBench
        if len(binary_op) == 2:
            [op_str, op_pt_func] = binary_op
            op_np_func = op_pt_func
        elif len(binary_op) == 3:
            [op_str, op_pt_func, op_np_func] = binary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("BroadcastBench_" + op_str, (BroadcastBench,), {})
        bm_cls.op_str = op_str
        bm_cls.binary_op_pt_func = op_pt_func
        bm_cls.binary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)

    for split_input, unary_op in itertools.product([True, False], unary_op_list):
        # Make a copy of BroadcastBench
        if len(unary_op) == 2:
            [op_str, op_pt_func] = unary_op
            op_np_func = op_pt_func
        elif len(unary_op) == 3:
            [op_str, op_pt_func, op_np_func] = unary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("BroadcastBench_" + op_str, (BroadcastBench,), {})
        bm_cls.op_str = op_str
        bm_cls.unary_op_pt_func = op_pt_func
        bm_cls.unary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)


register_broadcast_ops()
