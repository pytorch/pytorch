import torch
import torch.utils._pytree as pytree
from numpy import outer

from padded_tensor import *
from transformer_model import *
from helpers import *

from torch._inductor.test_case import run_tests, TestCase


class TestAttention(TestCase):
    def setUp(self):
        super().setUp()

        # Define model parameters
        self.n_local_heads = 5
        self.head_dim = 2
        self.dim = 10
        self.n_head = 5
        self.vocab_size = 32000

        # Initialize token embeddings and layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim)
        total_head_dim = (self.n_head + 2 * self.n_local_heads) * self.head_dim
        self.wqkv = nn.Linear(self.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

    def run_unpadded_padded(self, fn, inputs, inputs_p, outputs_p_shapes):
        # Run the function on unpadded and padded inputs
        outputs = fn(*inputs)
        outputs_p = fn(*inputs_p)

        # Check the shapes of the padded outputs
        for x_p, shape in zip(outputs_p, outputs_p_shapes):
            self.assertEqual(x_p.shape, shape)

        # Check the non-padded values are the same
        for x, x_p in zip(outputs, outputs_p):
            slice_idx = [slice(0, s) for s in x.shape]
            self.assertEqual(x, x_p.tensor[tuple(slice_idx)])

    def test_attention_1(self):
        def f(idx):
            x = self.tok_embeddings(idx)

            kv_size = self.n_local_heads * self.head_dim
            yy = self.wqkv(x)
            q, k, v = yy.split([self.dim, kv_size, kv_size], dim=-1)

            return q, k, v

        inputs = [torch.ones(4, 2, dtype=torch.int32)]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 8, 10]) for _ in range(3)]

        self.run_unpadded_padded(f, inputs, inputs_p, output_shapes_p)

    def test_attention_2(self):
        def f(q, k, v, bsz, seqlen):
            q = q.view(bsz, seqlen, self.n_head, self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

            return q, k, v

        inputs = [torch.randn(4, 2, 10), torch.randn(4, 2, 10), torch.randn(4, 2, 10)]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 8, 5, 2]) for _ in range(3)]

        self.run_unpadded_padded(f, inputs + [4, 2], inputs_p + [8, 8], output_shapes_p)

    def test_attention_3(self):
        def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
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

        def f(q, k, v, freqs_cis):
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

            q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

            return q, k, v

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

        self.run_unpadded_padded(f, inputs, inputs_p, output_shapes_p)

    def test_attention_4(self):
        def f(k, v):
            k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

            return k, v

        inputs = [torch.randn(4, 5, 2, 2), torch.randn(4, 5, 2, 2)]
        MULTIPLIERS = {0: 8, 1: 1, 2: 8, 3: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 40, 8, 2]) for _ in range(2)]

        self.run_unpadded_padded(f, inputs, inputs_p, output_shapes_p)

    def test_attention_5(self):
        def f(q, k, v, mask):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
            return (y,)

        inputs = [
            torch.randn([4, 5, 2, 2]),
            torch.randn([4, 5, 16, 2]),
            torch.randn([4, 5, 16, 2]),
            torch.ones([1, 1, 2, 16]),
        ]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 8, 2, 2])]

        self.run_unpadded_padded(f, inputs, inputs_p, output_shapes_p)

    def test_attention_6(self):
        def f(y, bsz, seqlen):
            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
            return (y,)

        inputs = [torch.randn([4, 5, 2, 2])]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([8, 8, 10])]
        self.run_unpadded_padded(f, inputs + [4, 2], inputs_p + [4, 2], output_shapes_p)

    def test_attention_7(self):
        def f(y):
            y = self.wo(y)
            return (y,)

        inputs = [torch.randn([4, 2, 10])]
        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
        output_shapes_p = [torch.Size([4, 2, 10])]
        self.run_unpadded_padded(f, inputs, inputs_p, output_shapes_p)


if __name__ == "__main__":
    run_tests()
