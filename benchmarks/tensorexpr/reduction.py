from . import benchmark


class ReduceBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, case, M, N, K, skip_input_transform):
        super().__init__(mode, device, dtype)
        self.case = case
        self.M = M
        self.N = N
        self.K = K
        self._set_skip_input_transform(skip_input_transform)

        self.inputs = [self.randn(
            [M, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
        )]
        if case == "row":
            self.dims = [1, 2]
        elif case == "mid":
            self.dims = [0, 2]
        elif case == "col":
            self.dims = [0, 1]
        elif case == "full":
            self.dims = [0, 1, 2]
        else:
            raise ValueError("invalid case: %s" % case)

    def forward(self, inputs):
        if self.skip_input_transform:
            x = inputs
        else:
            x = self.add(inputs, 0.001)
        y = self.sum(x, self.dims)
        return y

    def config(self):
        if self.case == "full":
            return [self.M * self.N * self.K, self._skip_input_transform_str()]
        return [self.M, self.N, self.K, self._skip_input_transform_str()]

    @staticmethod
    def default_configs():
        return [
            # [512, 512, 512],
            [512, 64, 512, "s0"],
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

    def _set_skip_input_transform(self, input_str):
        # In the test setting, s1 will skip the input transformation, and s0 will not.
        if input_str == "s0":
            self.skip_input_transform = False
        elif input_str == "s1":
            self.skip_input_transform = True
        else:
            raise ValueError('invalid skip_input_transform: %s' % (input_str))

    def _skip_input_transform_str(self):
        if self.skip_input_transform:
            return "s1"
        else:
            return "s0"


class ReduceRowBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        super(ReduceRowBench, self).__init__(mode, device, dtype, "row", M, N, K, skip_input_transform)

    @staticmethod
    def module():
        return "reduce_row"


class ReduceMidBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        super(ReduceMidBench, self).__init__(mode, device, dtype, "mid", M, N, K, skip_input_transform)

    @staticmethod
    def module():
        return "reduce_mid"


class ReduceColBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, N, K, skip_input_transform):
        super(ReduceColBench, self).__init__(mode, device, dtype, "col", M, N, K, skip_input_transform)

    @staticmethod
    def module():
        return "reduce_col"


class ReduceFullBench(ReduceBench):
    def __init__(self, mode, device, dtype, M, skip_input_transform):
        super(ReduceFullBench, self).__init__(mode, device, dtype, "full", M, 1, 1, skip_input_transform)

    def config(self):
        return [self.M * self.N * self.K, self._skip_input_transform_str()]

    @staticmethod
    def default_configs():
        return [
            [1 << 24, "s1"],
        ]

    @staticmethod
    def module():
        return "reduce_full"


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
    def default_configs():
        parent_config = Reduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        parent_config = super(Reduce2DInnerBench, self).config()
        return parent_config[1:]

    @staticmethod
    def module():
        return "reduce2d_inner"

class Reduce2DOuterBench(Reduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        super(Reduce2DOuterBench, self).__init__(mode, device, dtype, 0, dim0, dim1)

    @staticmethod
    def default_configs():
        parent_config = Reduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        parent_config = super(Reduce2DOuterBench, self).config()
        return parent_config[1:]

    @staticmethod
    def module():
        return "reduce2d_outer"

benchmark.register_benchmark_class(ReduceRowBench)
benchmark.register_benchmark_class(ReduceMidBench)
benchmark.register_benchmark_class(ReduceColBench)
benchmark.register_benchmark_class(Reduce2DInnerBench)
benchmark.register_benchmark_class(Reduce2DOuterBench)
benchmark.register_benchmark_class(ReduceFullBench)

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
    def default_configs():
        parent_config = DynamicReduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        parent_config = super(DynamicReduce2DInnerBench, self).config()
        return parent_config[1:]

    @staticmethod
    def module():
        return "reduce2d_dynamic_inner"


class DynamicReduce2DOuterBench(DynamicReduce2DBench):
    def __init__(self, mode, device, dtype, dim0, dim1):
        super().__init__(mode, device, dtype, 0, dim0, dim1)

    @staticmethod
    def default_configs():
        parent_config = DynamicReduce2DBench.default_configs()[0]
        return [parent_config[1:]]

    def config(self):
        parent_config = super(DynamicReduce2DInnerBench, self).config()
        return parent_config[1:]

    @staticmethod
    def module():
        return "reduce2d_dynamic_outer"

benchmark.register_benchmark_class(DynamicReduce2DInnerBench)
benchmark.register_benchmark_class(DynamicReduce2DOuterBench)
