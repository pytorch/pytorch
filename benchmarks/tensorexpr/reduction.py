import framework


class ReduceBench(framework.Benchmark):
    def __init__(self, mode, device, case, M, N, K):
        super().__init__(mode, device)
        self.case = case
        self.M = M
        self.N = N
        self.K = K

        self.data = self.rand([M, N, K], device=device, requires_grad=self.requires_grad)
        if case == 'row':
            self.dims = [1, 2]
        elif case == 'mid':
            self.dims = [0, 2]
        elif case == 'col':
            self.dims = [0, 1]
        else:
            raise ValueError('invalid case: %s' % case)
        
    def forward(self):
        y = self.sum(self.data, self.dims)
        return y

    def config(self):
        return [self.M, self.N, self.K]

    @staticmethod
    def default_configs():
        return [
            #[512, 512, 512],
            [512, 64, 512],
        ]

    @staticmethod
    def module():
        return 'reduce'

    def memory_workload(self):
        if self.mode == 'fwd':
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + 1

        buffer_size = self.M * self.N * self.K * 4
        return {'sol': buffer_size * sol_count, 'algorithmic': buffer_size * algorithmic_count}


class ReduceRowBench(ReduceBench):
    def __init__(self, mode, device, M, N, K):
        super(ReduceRowBench, self).__init__(mode, device, 'row', M, N, K)

    @staticmethod
    def module():
        return 'reduce_row'


class ReduceMidBench(ReduceBench):
    def __init__(self, mode, device, M, N, K):
        super(ReduceMidBench, self).__init__(mode, device, 'mid', M, N, K)

    @staticmethod
    def module():
        return 'reduce_mid'


class ReduceColBench(ReduceBench):
    def __init__(self, mode, device, M, N, K):
        super(ReduceColBench, self).__init__(mode, device, 'col', M, N, K)

    @staticmethod
    def module():
        return 'reduce_col'


framework.register_benchmark_class(ReduceRowBench)
framework.register_benchmark_class(ReduceMidBench)
framework.register_benchmark_class(ReduceColBench)
