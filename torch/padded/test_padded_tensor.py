import sys
import time

import torch
import torch.utils._pytree as pytree
from numpy import dtype, outer
from torch._inductor.test_case import run_tests, TestCase

from padded_tensor import *
from transformer_model import *
from helpers import *


class TestAttention(TestCase):
    def setUp(self):
        super().setUp()

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
        max_batch_size = 16
        max_seq_length = 1024
        cache_shape = (
            max_batch_size,
            self.n_local_heads,
            max_seq_length,
            self.head_dim,
        )
        k_cache, v_cache = torch.zeros(cache_shape), torch.zeros(cache_shape)

        return k_cache, v_cache

    def run_unpadded_padded(self, fn, inputs, inputs_p, outputs_p_shapes):
        # Run the function on unpadded and padded inputs
        outputs = fn(*inputs)
        outputs_p = fn(*inputs_p)

        ## Check the shapes of the padded outputs
        # for x_p, shape in zip(outputs_p, outputs_p_shapes):
        #    self.assertEqual(x_p.shape, shape)

        # Check the non-padded values are the same
        for x, x_p in zip(outputs, outputs_p):
            slice_idx = [slice(0, s) for s in x.shape]
            self.assertEqual(x, x_p.tensor[tuple(slice_idx)])

    def run_unpadded_padded_bench(self, fn, inputs, inputs_p, outputs_p_shapes):
        # Run the function on unpadded and padded inputs
        times_unpadded = []
        for _ in range(10):
            start = time.time()
            outputs = fn(*inputs)
            times_unpadded.append(time.time() - start)

        times_padded = []
        for _ in range(10):
            start = time.time()
            outputs_p = fn(*inputs_p)
            times_padded.append(time.time() - start)

        fn_compiled = torch.compile(fn, mode="reduce-overhead")

        times_compiled_unpadded = []
        for _ in range(10):
            start = time.time()
            outputs = fn_compiled(*inputs)
            times_compiled_unpadded.append(time.time() - start)

        times_compiled_padded = []
        for _ in range(10):
            start = time.time()
            outputs_p = fn_compiled(*inputs_p)
            times_compiled_padded.append(time.time() - start)

        def median(x):
            return sorted(x)[len(x) // 2]

        print(f"Unpadded time (eager): {median(times_unpadded)}")
        print(f"Padded time (eager): {median(times_padded)}")
        print(f"Unpadded time (compiled): {median(times_compiled_unpadded)}")
        print(f"Padded time (compiled): {median(times_compiled_padded)}")

        ## Check the shapes of the padded outputs
        # for x_p, shape in zip(outputs_p, outputs_p_shapes):
        #    self.assertEqual(x_p.shape, shape)

        # Check the non-padded values are the same
        for x, x_p in zip(outputs, outputs_p):
            slice_idx = [slice(0, s) for s in x.shape]
            self.assertEqual(x, x_p.tensor[tuple(slice_idx)])

    def f_1(self, idx):
        x = self.tok_embeddings(idx)

        kv_size = self.n_local_heads * self.head_dim
        yy = self.wqkv(x)
        q, k, v = yy.split([self.dim, kv_size, kv_size], dim=-1)

        return q, k, v

    def test_attention_1(self):
        inputs = [torch.ones(4, 2, dtype=torch.int32)]
        MULTIPLIERS = {0: 128, 1: 128, 2: 1}
        inputs_p = pytree.tree_map(
            lambda x: PaddedTensor(x, {0: 128, 1: 128, 2: 1}, None), inputs
        )
        output_shapes_p = [torch.Size([128, 128, 10]) for _ in range(3)]

        self.run_unpadded_padded(self.f_1, inputs, inputs_p, output_shapes_p)

    def f_2(self, q, k, v, bsz, seqlen):
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        return q, k, v

    def test_attention_2(self):
        inputs = [torch.randn(4, 2, 10), torch.randn(4, 2, 10), torch.randn(4, 2, 10)]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 8, 5, 2]) for _ in range(3)]

        self.run_unpadded_padded(
            self.f_2, inputs + [4, 2], inputs_p + [8, 8], output_shapes_p
        )

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

    def test_attention_3(self):
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
        output_shapes_p = [torch.Size([8, 5, 8, 2]) for _ in range(3)]

        self.run_unpadded_padded(self.f_3, inputs, inputs_p, output_shapes_p)

    def kv_update(self, input_pos, k_val, v_val, k_cache, v_cache):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = k_cache
        v_out = v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

    def f_4(self, k, v, input_pos, k_cache, v_cache):
        k, v = self.kv_update(input_pos, k, v, k_cache, v_cache)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        return k, v

    def test_attention_4(self):
        k_cache, v_cache = self.create_kv_cache()

        inputs = [
            torch.randn(4, 5, 2, 2),
            torch.randn(4, 5, 2, 2),
            torch.ones(2, dtype=torch.int32),
            k_cache,
            v_cache,
        ]
        # MULTIPLIERS = {0: 8, 1: 1, 2: 8, 3: 1}
        MULTIPLIERS = {0: 1, 1: 1, 2: 1, 3: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 40, 8, 2]) for _ in range(2)]

        self.run_unpadded_padded(self.f_4, inputs, inputs_p, output_shapes_p)

    def f_5(self, q, k, v, mask):
        # y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        outs = torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0
        )
        y = outs[0]
        return (y,)

    def test_attention_5(self):
        inputs = [
            torch.randn([4, 5, 2, 2]),
            torch.randn([4, 5, 16, 2]),
            torch.randn([4, 5, 16, 2]),
            torch.ones([1, 1, 2, 16]),
        ]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 8, 2, 2])]

        self.run_unpadded_padded(self.f_5, inputs, inputs_p, output_shapes_p)

    def f_6(self, y, bsz, seqlen):
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return (y,)

    def test_attention_6(self):
        inputs = [torch.randn([4, 5, 2, 2])]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 8, 10])]
        self.run_unpadded_padded(
            self.f_6, inputs + [4, 2], inputs_p + [4, 2], output_shapes_p
        )

    def f_7(self, y):
        y = self.wo(y)
        return (y,)

    def test_attention_7(self):
        inputs = [torch.randn([4, 2, 10])]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([4, 2, 10])]
        self.run_unpadded_padded(self.f_7, inputs, inputs_p, output_shapes_p)

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

    def create_inputs(self, SEQLEN):
        x = torch.ones(16, SEQLEN, dtype=torch.int32).to(device="cuda")
        freqs_cis = torch.randn(SEQLEN, 64, 2).to(device="cuda", dtype=self.dtype)
        mask = torch.ones([16, 1, SEQLEN, 16], device="cuda")
        input_pos = torch.ones([SEQLEN], dtype=torch.int32, device="cuda")

        k_cache, v_cache = self.create_kv_cache()
        k_cache = k_cache.to(device="cuda", dtype=self.dtype)
        v_cache = v_cache.to(device="cuda", dtype=self.dtype)

        inputs = [x, freqs_cis, mask, input_pos, k_cache, v_cache]

        return inputs

    def test_attention_all(self):
        inputs = self.create_inputs(992)
        x, freqs_cis, mask, input_pos, k_cache, v_cache = inputs

        N = 16

        inputs_p = [
            PaddedTensor(x, {0: N, 1: N}, None),
            PaddedTensor(freqs_cis, {0: N, 1: N, 2: 1}, None),
            mask,
            PaddedTensor(input_pos, {0: 1, 1: 1, 2: N}, None),
            PaddedTensor(k_cache, {0: 1, 1: 1, 2: N}, None),
            PaddedTensor(v_cache, {0: 1, 1: 1, 2: N}, None),
        ]

        self.run_unpadded_padded(self.f_attention, inputs, inputs_p, None)

        dot_aten_graph(
            self.f_attention, inputs, "/tmp/alex/graph.pdf", max_nodes=sys.maxsize
        )

    def test_attention_bench(self):
        inputs = self.create_inputs(992)
        x, freqs_cis, mask, input_pos, k_cache, v_cache = inputs

        N = 16

        inputs_p = [
            PaddedTensor(x, {0: N, 1: N}, None),
            PaddedTensor(freqs_cis, {0: N, 1: N, 2: 1}, None),
            mask,
            PaddedTensor(input_pos, {0: 1, 1: 1, 2: N}, None),
            PaddedTensor(k_cache, {0: 1, 1: 1, 2: N}, None),
            PaddedTensor(v_cache, {0: 1, 1: 1, 2: N}, None),
        ]

        self.run_unpadded_padded_bench(self.f_attention, inputs, inputs_p, None)


if __name__ == "__main__":
    run_tests()

# cls = TestAttention()
# cls.setUp()
# cls.test_attention_all()
