import framework


class ConvImplBench(framework.Benchmark):
    def __init__(self, case, mode, device, kernel_size, N, iC, H, W, oC):
        super().__init__(mode, device)
        self.case = case
        self.kernel_size = kernel_size
        self.N = N
        self.iC = iC
        self.H = H
        self.W = W
        self.oC = oC
        self.data = self.rand([N, iC, H, W], device=device, requires_grad=self.requires_grad)
        if case == 'conv':
            self.groups = 1
        elif case == 'depthwise_conv':
            self.groups = iC
        else:
            raise ValueError('invalid case: %s' % (case))

        self.conv = self.conv2d_layer(iC, oC, kernel_size, groups=self.groups)
        if device != 'cpu':
            self.to_device(self.conv, device)
 
    def forward(self):
        y = self.conv(self.data)
        return y

    def config(self):
        return [self.kernel_size, self.N, self.iC, self.H, self.W, self.oC]

    def memory_workload(self):
        if self.mode == 'fwd':
            sol_count = {'i': 1, 'o': 1, 'k': 1}
            algorithmic_count = {'i': 1, 'o': 1, 'k': 1}
        else:
            sol_count = {
                'i': 1 + 1,
                'o': 1 + 1,
                'k': 1 + 1
            }
            algorithmic_count = {
                'i': 1 + (1 + 1),
                'o': 1 + (1 + 1),
                'k': 1 + (1 + 1)
            }

        buffer_size = {
            'i': self.N * self.iC * self.H * self.W * 4,
            'o': self.N * self.oC * self.H * self.W * 4,
            'k': self.oC * (self.iC / self.groups) * self.kernel_size * self.kernel_size * 4,
        }
        sol_size = 0
        algorithmic_size = 0
        for key in sol_count:
            sol_size += buffer_size[key] * sol_count[key]
            algorithmic_size += buffer_size[key] * algorithmic_count[key]
        return {
            'sol': sol_size,
            'algorithmic': algorithmic_size
        }

    def compute_workload(self):
        if self.mode == 'fwd':
            count = 1
        elif self.mode == 'both':
            count = 1 + (1 + 1)
        else:
            raise ValueError('invalid mode: %s' % (self.mode))

        op_count = self.N * self.iC / self.groups * self.oC * self.kernel_size * self.kernel_size * self.H * self.W
        op_count *= 2

        return op_count * count

    @staticmethod
    def default_configs():
        return [
            [3, 64, 32, 128, 128, 64],
        ]


class ConvBench(ConvImplBench):
    def __init__(self, *args):
        super().__init__('conv', *args) 

    @staticmethod
    def module():
        return 'conv'

    
class DepthwiseConvBench(ConvImplBench):
    def __init__(self, *args):
        super().__init__('depthwise_conv', *args) 

    @staticmethod
    def module():
        return 'depthwise_conv'

    
framework.register_benchmark_class(ConvBench)
framework.register_benchmark_class(DepthwiseConvBench)
