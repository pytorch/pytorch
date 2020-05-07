# This is a copy of rnn_attention from MLPerf, with some common sizes hardcoded
# for benchmarking and some control flow stripped out.
# https://github.com/mlperf/training/blob/master/rnn_translator/pytorch/seq2seq/models/attention.py

from . import benchmark
import torch


class BahdanauAttention(benchmark.Benchmark):
    def __init__(self, mode, device, b, t_q, t_k, n):
        super().__init__(mode, device)
        self.b = b
        self.t_q = t_q
        self.t_k = t_k
        self.n = n
        self.att_query = self.rand(
            [b, t_q, n], device=device, requires_grad=self.requires_grad
        )
        self.att_keys = self.rand(
            [b, t_k, n], device=device, requires_grad=self.requires_grad
        )
        self.normalize_bias = self.rand(
            [n], device=device, requires_grad=self.requires_grad
        )
        self.linear_att = self.rand(
            [n], device=device, requires_grad=self.requires_grad
        )
        self.inputs = [
            self.att_query,
            self.att_keys,
            self.normalize_bias,
            self.linear_att,
        ]

    def forward(self, att_query, att_keys, normalize_bias, linear_att):
        """
        Calculate Bahdanau score

        :param att_query: b x t_q x n
        :param att_keys: b x t_k x n

        return b x t_q x t_k scores
        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys + normalize_bias
        out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    def reference(self):
        return self.numpy(self.forward(*self.inputs))

    def config(self):
        return [self.b, self.t_q, self.t_k, self.n]

    @staticmethod
    def module():
        return "attention"

    def memory_workload(self):
        def memsize(t):
            return t.numel() * t.element_size()

        input_size = (
            memsize(self.att_query)
            + memsize(self.att_keys)
            + memsize(self.normalize_bias)
            + memsize(self.linear_att)
        )
        output_size = 4 * torch.Size([self.b, self.t_q, self.t_k]).numel()
        io_size = input_size + output_size

        # If matmul is not fused, must write and then read `sum_qk`.
        intermediate_size = (
            2 * 4 * torch.Size([self.b, self.t_q, self.t_k, self.n]).numel()
        )
        return {"sol": io_size, "algorithmic": io_size + intermediate_size}

    @staticmethod
    def default_configs():
        mlperf_inference = [1280, 1, 66, 1024]
        nvidia = [128, 10, 128, 1024]
        return [mlperf_inference, nvidia]


benchmark.register_benchmark_class(BahdanauAttention)
