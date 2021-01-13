from . import benchmark


class ReduceBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, case, M, N, K):
        super().__init__(mode, device, dtype)
        self.case = case
        self.M = M
        self.N = N
        self.K = K

        self.inputs = [self.randn(
            [M, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
        )]
        if case == "row":
            self.dims = [1, 2]
        elif case == "mid":
            self.dims = [0, 2]
        elif case == "col":
            self.dims = [0, 1]
        else:
            raise ValueError("invalid case: %s" % case)

    def forward(self, inputs):
        x = self.add(inputs, 0.001)
        y = self.sum(x, self.dims)
        return y

    def config(self):
        return [self.M, self.N, self.K]

    @staticmethod
    def default_configs():
        return [
            # [512, 512, 512],
            [512, 64, 512],
        ]

    @staticmethod
    def module():
        return "reduce"

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + 1

        buffer_size = self.M * self.N * self.K
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }


class ReduceRowBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super(ReduceRowBench, self).__init__(mode, device, dtype, "row", M, N, K)

    @staticmethod
    def module():
        return "reduce_row"


class ReduceMidBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super(ReduceMidBench, self).__init__(mode, device, dtype, "mid", M, N, K)

    @staticmethod
    def module():
        return "reduce_mid"


class ReduceColBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super(ReduceColBench, self).__init__(mode, device, dtype, "col", M, N, K)

    @staticmethod
    def module():
        return "reduce_col"

class Reduce2DBench(benchmark.Benchmark):
    '''
    A benchmark class to validate 2 dimensional reduction performance.
    Only a simple add is fused to induce the fuser and isolate reduction perf.
    '''
    def __init__(self, mode, device, dtype, red_dim, dim0, dim1):
        super().__init__(mode, device, dtype)
        self.red_dim = red_dim
        self.dim0 = dim0
        self.dim1 = dim1

        self.inputs = [self.randn(
            [dim0, dim1], device=device, dtype=dtype, requires_grad=self.requires_grad
        )]

        if red_dim != 0 and red_dim != 1 :
            raise ValueError("invalid reduction dimension: {}".format(red_dim))

    def forward(self, inputs):
        x = self.add(inputs, 0.001)
        y = self.sum(x, [self.red_dim])
        return y

    def config(self):
        return [self.red_dim, self.dim0, self.dim1]

    @staticmethod
    def default_configs():
        return [
            [1, 640, 524288],
        ]

    @staticmethod
    def module():
        return "reduce2d"

    @staticmethod
    def input_iterable() :
        return True

    def memory_workload(self):
        assert self.mode == "fwd", "Only the forward operation is modeled!"

        buffer_size = self.dim0 * self.dim1
        if self.red_dim == 0 :
            buffer_size += self.dim1
        else :
            buffer_size += self.dim0
        return {
            "sol": buffer_size,
            "algorithmic": buffer_size,
        }

class Reduce2DInnerBench(Reduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        super(Reduce2DInnerBench, self).__init__(mode, device, dtype, 1, dim0, dim1)

    @staticmethod
    def module():
        return "reduce2d_inner"

class Reduce2DOuterBench(Reduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        super(Reduce2DOuterBench, self).__init__(mode, device, dtype, 0, dim0, dim1)

    @staticmethod
    def module():
        return "reduce2d_outer"
benchmark.register_benchmark_class(ReduceRowBench)
benchmark.register_benchmark_class(ReduceMidBench)
benchmark.register_benchmark_class(ReduceColBench)
benchmark.register_benchmark_class(Reduce2DInnerBench)
benchmark.register_benchmark_class(Reduce2DOuterBench)


class DynamicReduce2DBench(benchmark.DynamicShape, Reduce2DBench):
    '''
    A benchmark class to validate 2 dimensional reduction performance.
    Only a simple add is fused to induce the fuser and isolate reduction perf.
    '''

    def __init__(self, mode, device, dtype, red_dim, dim0, dim1):
        benchmark.DynamicShape.__init__(self)
        Reduce2DBench.__init__(self, mode, device, dtype, red_dim, dim0, dim1)

    def instantiate_input(self):
        dim0, dim1 = self.rand_shape([self.dim0, self.dim1])

        self.inputs = [self.randn(
            [dim0, dim1], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad
        )]

    @staticmethod
    def module():
        return "dynamicreduce2d"


class DynamicReduce2DInnerBench(DynamicReduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        super().__init__(mode, device, dtype, 1, dim0, dim1)

    @staticmethod
    def module():
        return "reduce2d_dynamic_inner"


class DynamicReduce2DOuterBench(DynamicReduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        super().__init__(mode, device, dtype, 0, dim0, dim1)

    @staticmethod
    def module():
        return "reduce2d_dynamic_outer"

benchmark.register_benchmark_class(DynamicReduce2DInnerBench)
benchmark.register_benchmark_class(DynamicReduce2DOuterBench)
