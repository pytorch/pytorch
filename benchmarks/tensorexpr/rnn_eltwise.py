from . import benchmark
import torch

class RNNEltwise(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, b, hs):
        super().__init__(mode, device, dtype)
        self.b = b
        self.hs = hs
        self.input = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.hx = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.cx = self.rand(
            [b, hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.b_ih = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.b_hh = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [
            self.input,
            self.hx,
            self.cx,
            self.b_ih,
            self.b_hh,
        ]

    def forward(self, input, hx, cx, b_ih, b_hh):
        gates = input + hx + b_ih + b_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

    def config(self):
        return [self.b, self.hs]

    @staticmethod
    def module():
        return "rnn_eltwise"

    def memory_workload(self):
        def memsize(t):
            return t.numel() * t.element_size()

        input_size = sum([memsize(t) for t in self.inputs])
        output_size = 2 * memsize(self.cx)
        io_size = input_size + output_size
        return {"sol": io_size, "algorithmic": io_size}

    @staticmethod
    def default_configs():
        return [[64, 512]]

benchmark.register_benchmark_class(RNNEltwise)


class DynamicLSTM(benchmark.DynamicShape, RNNEltwise):
    def __init__(self, mode, device, dtype, b, hs):
        benchmark.DynamicShape.__init__(self)
        RNNEltwise.__init__(self, mode, device, dtype, b, hs)

    def instantiate_input(self):
        b, hs = self.rand_shape([self.b, self.hs])

        self.input = self.rand(
            [b, 4 * hs], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad
        )
        self.hx = self.rand(
            [b, 4 * hs], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad
        )
        self.cx = self.rand(
            [b, hs], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad
        )
        self.b_ih = self.rand(
            [b, 4 * hs], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad
        )
        self.b_hh = self.rand(
            [b, 4 * hs], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad
        )
        self.inputs = [
            self.input,
            self.hx,
            self.cx,
            self.b_ih,
            self.b_hh,
        ]

    @staticmethod
    def module():
        return "dynamic_lstm"

benchmark.register_benchmark_class(DynamicLSTM)
