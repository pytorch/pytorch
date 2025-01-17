import time

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._inductor.test_case import run_tests, TestCase

from padded_tensor import *
from utils import *


class TransformerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Define model parameters
        self.n_local_heads = 8
        self.head_dim = 128
        self.dim = 4096
        self.n_head = 32
        self.vocab_size = 128256

        self.dtype = torch.bfloat16

        # Initialize token embeddings and layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim).to(
            device="cuda", dtype=self.dtype
        )
        total_head_dim = (self.n_head + 2 * self.n_local_heads) * self.head_dim
        self.wqkv = nn.Linear(self.dim, total_head_dim, bias=False).to(
            device="cuda", dtype=self.dtype
        )
        self.wo = nn.Linear(self.dim, self.dim, bias=False).to(
            device="cuda", dtype=self.dtype
        )

    def create_kv_cache(self):
        max_batch_size = 15
        max_seq_length = 1024
        cache_shape = (
            max_batch_size,
            self.n_local_heads,
            max_seq_length,
            self.head_dim,
        )
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=self.dtype, device="cuda")
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=self.dtype, device="cuda")
        )

    def reset_kv_cache(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def f_1(self, idx):
        x = self.tok_embeddings(idx)

        kv_size = self.n_local_heads * self.head_dim
        yy = self.wqkv(x)
        q, k, v = yy.split([self.dim, kv_size, kv_size], dim=-1)

        return q, k, v

    def f_2(self, q, k, v, bsz, seqlen):
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        return q, k, v

    def apply_rotary_emb(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * freqs_cis[..., 0]
                - xshaped[..., 1] * freqs_cis[..., 1],
                xshaped[..., 1] * freqs_cis[..., 0]
                + xshaped[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        )

        x_out2 = x_out2.flatten(3)
        return x_out2.type_as(x)

    def f_3(self, q, k, v, freqs_cis):
        q = self.apply_rotary_emb(q, freqs_cis)
        k = self.apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        return q, k, v

    def kv_update(self, input_pos, k_val, v_val, k_cache, v_cache):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

    def f_4(self, k, v, input_pos, k_cache, v_cache):
        k, v = self.kv_update(input_pos, k, v, k_cache, v_cache)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        return k, v

    def f_5(self, q, k, v, mask):
        outs = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0
        )
        y = outs[0]
        return (y,)

    def f_6(self, y, bsz, seqlen):
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return (y,)

    def f_7(self, y):
        y = self.wo(y)
        return (y,)

    def f_attention(self, x, freqs_cis, mask, input_pos, k_cache, v_cache):
        bsz, seqlen = x.shape

        q, k, v = self.f_1(x)
        q, k, v = self.f_2(q, k, v, bsz, seqlen)
        q, k, v = self.f_3(q, k, v, freqs_cis)
        k, v = self.f_4(k, v, input_pos, k_cache, v_cache)
        (y,) = self.f_5(q, k, v, mask)
        (y,) = self.f_6(y, bsz, seqlen)
        (y,) = self.f_7(y)

        return (y,)


class TestAttention(TestCase):
    def setUp(self):
        super().setUp()

    def are_equal(self, outputs, outputs_p):
        for x, x_p in zip(outputs, outputs_p):
            x_p = x_p.tensor if isinstance(x_p, PaddedTensor) else x_p

            slice_idx = [slice(0, s) for s in x.shape]
            self.assertEqual(x, x_p[tuple(slice_idx)])

    def run_unpadded_padded(self, fn, inputs, inputs_p):
        # Run the function on unpadded and padded inputs
        inputs = pytree.tree_map(lambda x: x.clone(), inputs)
        outputs = fn(*inputs)

        inputs_p = pytree.tree_map(lambda x: x.clone(), inputs_p)
        outputs_p = fn(*inputs_p)

        # Check the outputs are equal
        self.are_equal(outputs, outputs_p)

    def run_unpadded_padded_bench(self, model, fn, inputs, inputs_p, outputs_p_shapes):
        # Precondition: Are inputs and inputs_p equal?
        self.are_equal(inputs, inputs_p)

        def median(x):
            return sorted(x)[len(x) // 2]

        def bench(fn, inputs, n_iter=10):
            times = []
            outputs = None

            for _ in range(n_iter):
                model.reset_kv_cache()
                inps = pytree.tree_map(lambda x: x.clone(), inputs)

                start = time.time()
                outputs = fn(*inps)
                times.append(time.time() - start)

            outputs = pytree.tree_map(lambda x: x.clone(), outputs)
            return median(times), outputs

        # Run eager
        time_eager, outputs_eager = bench(fn, inputs)
        time_eager_padded, outputs_eager_padded = bench(fn, inputs_p)

        self.are_equal(outputs_eager, outputs_eager_padded)

        # Run compiled
        fn_compiled = torch.compile(fn, mode="reduce-overhead")

        time_compiled, outputs_compiled = bench(fn_compiled, inputs)
        time_compiled_padded, outputs_compiled_padded = bench(fn_compiled, inputs_p)

        self.are_equal(outputs_eager, outputs_compiled)
        self.are_equal(outputs_eager, outputs_compiled_padded)

        # Report results
        print(f"Unpadded time (eager): {time_eager}")
        print(f"Padded time (eager): {time_eager_padded}")
        print(f"Unpadded time (compiled): {time_compiled}")
        print(f"Padded time (compiled): {time_compiled_padded}")

    def test_attention_1(self):
        model = TransformerModel()

        inputs = [torch.ones(4, 2, dtype=torch.int32)]
        MULTIPLIERS = {0: 128, 1: 128, 2: 1}
        inputs_p = pytree.tree_map(
            lambda x: PaddedTensor(x, {0: 128, 1: 128, 2: 1}, None), inputs
        )

        self.run_unpadded_padded(model.f_1, inputs, inputs_p)

    def test_attention_2(self):
        model = TransformerModel()

        inputs = [torch.randn(4, 2, 10), torch.randn(4, 2, 10), torch.randn(4, 2, 10)]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)

        self.run_unpadded_padded(model.f_2, inputs + [4, 2], inputs_p + [8, 8])

    def test_attention_3(self):
        model = TransformerModel()

        inputs = [
            torch.randn(4, 2, 5, 2),
            torch.randn(4, 2, 5, 2),
            torch.randn(4, 2, 5, 2),
            torch.randn(2, 1, 2),
        ]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = [PaddedTensor(x, MULTIPLIERS, None) for x in inputs[:-1]] + [
            PaddedTensor(inputs[-1], {0: 8, 1: 1, 2: 1}, None)
        ]

        self.run_unpadded_padded(model.f_3, inputs, inputs_p)

    def test_attention_4(self):
        model = TransformerModel()

        k_cache, v_cache = self.create_kv_cache()

        inputs = [
            torch.randn(4, 5, 2, 2),
            torch.randn(4, 5, 2, 2),
            torch.ones(2, dtype=torch.int32),
            k_cache,
            v_cache,
        ]
        MULTIPLIERS = {0: 1, 1: 1, 2: 1, 3: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)

        self.run_unpadded_padded(model.f_4, inputs, inputs_p)

    def test_attention_5(self):
        model = TransformerModel()

        inputs = [
            torch.randn([4, 5, 2, 2]),
            torch.randn([4, 5, 16, 2]),
            torch.randn([4, 5, 16, 2]),
            torch.ones([1, 1, 2, 16]),
        ]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)

        self.run_unpadded_padded(model.f_5, inputs, inputs_p)

    def test_attention_6(self):
        model = TransformerModel()

        inputs = [torch.randn([4, 5, 2, 2])]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)

        self.run_unpadded_padded(model.f_6, inputs + [4, 2], inputs_p + [4, 2])

    def test_attention_7(self):
        model = TransformerModel()

        inputs = [torch.randn([4, 2, 10])]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)

        self.run_unpadded_padded(model.f_7, inputs, inputs_p)

    def gen_rand_inputs(self, model, batchsize, seqlen):
        x = torch.randint(0, 128256, (batchsize, seqlen)).to(device="cuda")
        freqs_cis = torch.randn(seqlen, 64, 2).to(device="cuda", dtype=model.dtype)
        mask = torch.ones([batchsize, 1, seqlen, 15], device="cuda")
        input_pos = torch.arange(0, seqlen, dtype=torch.int32, device="cuda")

        model.create_kv_cache()

        inputs = [x, freqs_cis, mask, input_pos, model.k_cache, model.v_cache]

        return inputs

    def pad_inputs(self, inputs, N):
        x, freqs_cis, mask, input_pos, k_cache, v_cache = inputs

        inputs_p = [
            PaddedTensor(x, {0: N, 1: N}, None),
            PaddedTensor(freqs_cis, {0: 1, 1: N, 2: 1}, None),
            mask,
            PaddedTensor(input_pos, {0: 1, 1: 1, 2: N}, None),
            PaddedTensor(k_cache, {0: 1, 1: 1, 2: N}, None),
            PaddedTensor(v_cache, {0: 1, 1: 1, 2: N}, None),
        ]

        return inputs_p

    def test_attention_all(self):
        model = TransformerModel()

        inputs = self.gen_rand_inputs(model, 15, 991)
        inputs_p = self.pad_inputs(inputs, 16)

        self.run_unpadded_padded(model.f_attention, inputs, inputs_p)

    def test_attention_bench(self):
        model = TransformerModel()

        with torch.no_grad():
            inputs = self.gen_rand_inputs(model, 15, 991)
            inputs_p = self.pad_inputs(inputs, 16)

            self.run_unpadded_padded_bench(
                model, model.f_attention, inputs, inputs_p, None
            )


if __name__ == "__main__":
    run_tests()
