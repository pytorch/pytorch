#! /usr/bin/env python

import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from itertools import product

device = 'cuda'
dtype = torch.float16
torch.set_printoptions(precision=3, threshold=None, edgeitems=4, linewidth=460, profile=None, sci_mode=False)

class SwiGLU(nn.Module):

    def __init__(self, m, k, dtype):
        super().__init__()

        # to properly initialize weights and biases
        linear1 = nn.Linear(k, m, dtype=dtype)
        linear2 = nn.Linear(k, m, dtype=dtype)
        linear3 = nn.Linear(m, m, dtype=dtype)

        self.m = m
        self.k = k
        self.dtype = dtype
        self.w1 = linear1.weight
        self.b1 = linear1.bias
        self.w2 = linear2.weight
        self.b2 = linear2.bias
        self.w3 = linear3.weight
        self.b3 = linear3.bias

    def forward(self, x):
        x1 = x @ self.w1.transpose(-2, -1) + self.b1
        x2 = x @ self.w2.transpose(-2, -1) + self.b2
        x3 = torch.nn.functional.silu(x1)
        x4 = x3 * x2
        x5 = x4 @ self.w3.transpose(-2, -1) + self.b3
        return x5

class CUTLASSSwiGLU1(SwiGLU):

    def forward(self, x):
        b = x.view(-1, x.shape[-1]).T

        #x1 = x @ self.w1.transpose(-2, -1) + self.b1
        c1 = self.b1.view(torch.numel(self.b1), 1)
        if self.mask1 is not None:
            x3, self.meta1 = torch._cutlass_linear(self.w1, b, c1, self.mask1, activation='silu')
            self.mask = None
        else:
            x3, _ = torch._cutlass_linear(self.w1, b, c1, self.meta1, activation='silu')
        x3 = x3.T.view(*x.shape[:-1], -1)

        #x2 = x @ self.w2.transpose(-2, -1) + self.b2
        c2 = self.b2.view(torch.numel(self.b2), 1)
        if self.mask2 is not None:
            x2, self.meta2 = torch._cutlass_linear(self.w2, b, c2, self.mask2)
            self.mask = None
        else:
            x2, _ = torch._cutlass_linear(self.w2, b, c2, self.meta2)
        x2 = x2.T.view(*x.shape[:-1], -1)

        x4 = x3 * x2
        x5 = x4 @ self.w3.transpose(-2, -1) + self.b3
        return x5

    @classmethod
    def from_dense(cls, mod):
        cutlass_swiglu = cls(mod.m, mod.k, mod.dtype)

        m, k = mod.m, mod.k

        mask1 = mod.w1.data != 0
        mask2 = mod.w2.data != 0

        cutlass_swiglu.w1 = torch.nn.parameter.Parameter(mod.w1.data.masked_select(mask1).view(m, k // 2))
        cutlass_swiglu.b1 = torch.nn.parameter.Parameter(mod.b1)
        cutlass_swiglu.w2 = torch.nn.parameter.Parameter(mod.w2.data.masked_select(mask2).view(m, k // 2))
        cutlass_swiglu.b2 = torch.nn.parameter.Parameter(mod.b2)
        cutlass_swiglu.w3 = torch.nn.parameter.Parameter(mod.w3)
        cutlass_swiglu.b3 = torch.nn.parameter.Parameter(mod.b3)

        cutlass_swiglu.mask1 = mask1
        cutlass_swiglu.meta1 = None
        cutlass_swiglu.mask2 = mask2
        cutlass_swiglu.meta2 = None

        return cutlass_swiglu

class CUTLASSSwiGLU2(SwiGLU):

    def forward(self, x):
        #x12 = x @ self.w12.transpose(-2, -1) + self.b12
        b = x.view(-1, x.shape[-1]).T
        c = self.b12.view(torch.numel(self.b12), 1)
        if self.mask12 is not None:
            x12, self.meta12 = torch._cutlass_linear(self.w12, b, c, self.mask12)
            self.mask12 = None
        else:
            x12, _ = torch._cutlass_linear(self.w12, b, c, self.meta12)

        x1, x2 = torch.split(x12, m)
        x1 = x1.T.view(*x.shape[:-1], -1)
        x2 = x2.T.view(*x.shape[:-1], -1)
        x3 = torch.nn.functional.silu(x1)
        x4 = x3 * x2
        x5 = x4 @ self.w3.transpose(-2, -1) + self.b3
        return x5

    @classmethod
    def from_dense(cls, mod):
        cutlass_swiglu = cls(mod.m, mod.k, mod.dtype)

        m, k = mod.m, mod.k

        w12 = torch.cat((mod.w1, mod.w2))
        b12 = torch.cat((mod.b1, mod.b2))
        mask12 = w12.data != 0

        cutlass_swiglu.w12 = torch.nn.parameter.Parameter(w12.data.masked_select(mask12).view(2 * m, k // 2))
        cutlass_swiglu.b12 = torch.nn.parameter.Parameter(b12)
        cutlass_swiglu.w3 = torch.nn.parameter.Parameter(mod.w3)
        cutlass_swiglu.b3 = torch.nn.parameter.Parameter(mod.b3)

        cutlass_swiglu.mask12 = mask12
        cutlass_swiglu.meta12 = None

        return cutlass_swiglu

class Model(nn.Module):

    def __init__(self, m, k, dtype):
        super().__init__()
        self.swiglu = SwiGLU(m, k, dtype)

    def forward(self, x):
        return self.swiglu(x)

if __name__ == '__main__':
    results = []

    shapes = [
        ###(32, 64, 8),
        # distilbert shapes
        (768, 3072, 768),
        (3072, 768, 3072),
        # jiecao shapes
        (1024, 1536, 2048),
        (1024, 9408, 2048),
        (1024, 3200, 2048),
        (1024, 256, 9472),
        (1024, 10240, 256),
        (1024, 256, 12608),
        (1024, 2560, 1024),
        (1024, 512, 10240),
        (1024, 10240, 512),
        (1024, 2048, 1024),
        (1024, 512, 512),
        (1024, 1024, 1024),
        (1024, 2048, 2048),
        (2048, 1536, 2048),
        (2048, 9408, 2048),
        (2048, 3200, 2048),
        (2048, 256, 9472),
        (2048, 10240, 256),
        (2048, 256, 12608),
        (2048, 2560, 1024),
        (2048, 512, 10240),
        (2048, 10240, 512),
        (2048, 2048, 1024),
        (2048, 512, 512),
        (2048, 1024, 1024),
        (2048, 2048, 2048),
    ]
    batch_sizes = [4, 16, 64, 256]


    for (m, k, n), batch_size in product(shapes, batch_sizes):
        print(m, k, n, batch_size)
        # label and sub_label are the rows
        # description is the column
        try:
            label = 'CUTLASSSwiGLU vs SwiGLU'
            sub_label = f'm:{m:5d} | k:{k:5d} | n:{n:5d} | batch_size: {batch_size:4d}'

            model = Model(m, k, dtype)
            model.half()
            model.cuda()
            model.eval()

            pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2)
            pruner.prepare(model, [{'tensor_fqn': 'swiglu.w1'}, {'tensor_fqn': 'swiglu.w2'}])
            pruner.step()

            sparse_linear1 = pruner.convert(model, mapping={SwiGLU: CUTLASSSwiGLU1})
            print(sparse_linear1)

            sparse_linear2 = pruner.convert(model, mapping={SwiGLU: CUTLASSSwiGLU2})
            print(sparse_linear2)

            pruner.squash_mask()
            dense_linear = model
            print(dense_linear)

            for i in range(2):
                input_tensor = torch.randn(batch_size, n, k, device=device, dtype=dtype)
                dense_output = dense_linear(input_tensor)
                sparse_output1 = sparse_linear1(input_tensor)
                sparse_output2 = sparse_linear2(input_tensor)

                assert torch.allclose(sparse_output1, dense_output, rtol=1e-3, atol=1e-3)
                assert torch.allclose(sparse_output2, dense_output, rtol=1e-3, atol=1e-3)


            measurement = benchmark.Timer(
                stmt='sparse_linear(input_tensor)',
                globals={'input_tensor': input_tensor, 'sparse_linear': sparse_linear1},
                label=label,
                sub_label=sub_label,
                description='sparse latency 1',
            ).blocked_autorange()
            results.append(measurement)

            measurement = benchmark.Timer(
                stmt='sparse_linear(input_tensor)',
                globals={'input_tensor': input_tensor, 'sparse_linear': sparse_linear2},
                label=label,
                sub_label=sub_label,
                description='sparse latency 2',
            ).blocked_autorange()
            results.append(measurement)

            measurement = benchmark.Timer(
                stmt='dense_linear(input_tensor)',
                globals={'input_tensor': input_tensor, 'dense_linear': dense_linear},
                label=label,
                sub_label=sub_label,
                description='dense latency',
            ).blocked_autorange()
            results.append(measurement)

        except Exception:
            continue

        # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
        #     with record_function('CUTLASS'):
        #         sparse_linear(input_tensor)

        # prof.export_stacks(f'{m}_{k}_{n}_cutlass_profiler_stacks.txt', 'self_cuda_time_total')

    compare = benchmark.Compare(results)
    compare.colorize(rowwise=True)
    compare.print()
