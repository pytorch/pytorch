import math
import time
from dataclasses import dataclass
from typing import Optional

import torch

import torch.nn as nn
from torch import Tensor
from torch._inductor.experimental.padded_tensor import PaddedTensor
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch.nn import functional as F
from torch.utils import _pytree as pytree


class PaddedTensorFunctionalTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_dynamic_inner(self):
        """
        Test that we can use padded dimension in inner dimension.
        This results in a non-symbolic dimension in the output shape.
        """

        def f(a, b, c):
            return a @ b + c

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16, 1: 16}

        for i in range(3, 9):
            a = torch.randn([3, i])
            b = torch.randn([i, 7])
            c = torch.randn([3, 7])

            a_p = PaddedTensor.from_tensor(a, multipliers)
            b_p = PaddedTensor.from_tensor(b, multipliers)
            c_p = PaddedTensor.from_tensor(c, multipliers)

            y = f(a, b, c)
            y_p = f(a_p, b_p, c_p)

            self.assertEqual(y_p.shape, (16, 16))
            self.assertEqual(y_p.original_tensor.shape, (3, 7))
            self.assertEqual(y, y_p.unpad())

    def test_dynamic_outer(self):
        """
        Test that we can use padded dimension in outer dimension.
        This results in a symbolic dimension in the output shape.
        """

        def f(a, b, c):
            return a @ b + c

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16, 1: 16}

        for i in range(3, 9):
            a = torch.randn([3, i])
            b = torch.randn([i, 7])
            c = torch.randn([3, 7])

            a_p = PaddedTensor.from_tensor(a, multipliers)
            b_p = PaddedTensor.from_tensor(b, multipliers)
            c_p = PaddedTensor.from_tensor(c, multipliers)

            y = f(a, b, c)
            y_p = f(a_p, b_p, c_p)

            self.assertEqual(y_p.shape, (16, 16))
            self.assertEqual(y_p.original_tensor.shape, (3, 7))
            self.assertEqual(y, y_p.unpad())

    def test_bucketing(self):
        """
        Test that we 1. compile a new graph on a shape that is larger than the original bucket.
        2. don't compile on a shape that is smaller than the original bucket.
        """

        def f(a, b, c):
            return a @ b + c

        f = torch.compile(f, fullgraph=True)
        multipliers = {0: 16, 1: 16}

        for i in range(3, 22):
            # Every multiple of 16, we allow recompilation.
            if i % 16 == 0:
                torch._dynamo.config.error_on_recompile = False
            # It takes 2 iterations to trace graph with symbolic shapes. So after 2 iterations
            # after multiples of 16, we disallow recompilation.
            if i % 16 == 2:
                torch._dynamo.config.error_on_recompile = True

            a = PaddedTensor.from_tensor(torch.randn([3, 5]), multipliers)
            b = PaddedTensor.from_tensor(torch.randn([5, i]), multipliers)
            c = PaddedTensor.from_tensor(torch.randn([3, i]), multipliers)

            y = f(a, b, c)

            self.assertEqual(y.shape, (16, 16 if i <= 16 else 32))
            self.assertEqual(y.original_tensor.shape, (3, i))

        torch._dynamo.config.error_on_recompile = False


class AtenOpTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_sum(self):
        def f(a):
            return a.sum()

        f = torch.compile(f, fullgraph=True)
        multipliers = {1: 16}

        a = torch.randn([2, 121])
        a_p = PaddedTensor.from_tensor(a, multipliers)

        y = f(a_p)

        self.assertEqual(y.shape, ())
        self.assertEqual(y.original_tensor.shape, ())


class NNOpTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_linear_on_3d(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(7, 9)

            def forward(self, x):
                return self.linear(x)

        mod = TestModule()
        mod = torch.compile(mod, fullgraph=True)

        multipliers = {0: 16, 1: 16}

        for i in range(3, 9):
            x = torch.randn((i, i, 7))
            x_p = PaddedTensor.from_tensor(x, multipliers)

            y = mod(x)
            y_p = mod(x_p)

            self.assertEqual(y_p.shape, (16, 16, 9))
            self.assertEqual(y_p.original_tensor.shape, (i, i, 9))
            self.assertEqual(y, y_p.unpad())


class ModelTests(TestCase):
    def setUp(self):
        super().setUp()

    def test_transformer_equiv(self):
        """
        Test that transformer model produces equivalent results on regular and padded tensors.
        """

        with torch.device("cuda"), torch.no_grad():
            bsz, seqlen_max = 2, 32
            seqlen_multiple = 16

            # Set up transformer
            args = ModelArgs.from_name("mini")
            transformer = Transformer(args)
            transformer.setup_caches(bsz, seqlen_max)

            transformer = torch.compile(
                transformer, fullgraph=True, mode="reduce-overhead"
            )

            for seqlen in range(3, 15):
                print("seqlen =", seqlen)
                # Set error_on_recompile to True after 3rd iteration.
                if seqlen == 5:
                    torch._dynamo.config.error_on_recompile = True

                # Run unpadded
                inputs = (
                    torch.randint(0, 3, (bsz, seqlen)),
                    torch.arange(0, seqlen, dtype=torch.int32),
                )

                torch.compiler.cudagraph_mark_step_begin()

                out = transformer(*inputs)
                out = out.clone()

                # Run padded
                inputs_p = [
                    PaddedTensor.from_tensor(
                        inputs[0], multipliers={0: 1, 1: seqlen_multiple}
                    ),
                    PaddedTensor.from_tensor(
                        inputs[1], multipliers={0: seqlen_multiple}, neutral_element=-1
                    ),
                ]

                torch.compiler.cudagraph_mark_step_begin()

                out_p = transformer(*inputs_p)
                out_p = out_p.clone()

                # Check
                self.assertEqual(out, out_p.unpad())

        torch._dynamo.config.error_on_recompile = False

    def test_transformer_bucketing(self):
        """
        Test that we 1. compile a new graph on a shape that is larger than the original bucket.
        2. don't compile on a shape that is smaller than the original bucket.
        """

        with torch.device("cuda"), torch.no_grad():
            bsz, seqlen_max = 2, 64
            seqlen_multiple = 16

            # Set up transformer
            args = ModelArgs.from_name("mini")
            transformer = Transformer(args)
            transformer.setup_caches(bsz, seqlen_max)

            transformer = torch.compile(
                transformer, fullgraph=True, mode="reduce-overhead"
            )

            for seqlen in range(3, 30):
                print("seqlen =", seqlen)

                # Set error_on_recompile to True after 3rd or 5th iteration for each bucket.
                if seqlen % 16 > 5:
                    torch._dynamo.config.error_on_recompile = True
                else:
                    torch._dynamo.config.error_on_recompile = False

                # Run unpadded
                inputs = (
                    torch.randint(0, 3, (bsz, seqlen)),
                    torch.arange(0, seqlen, dtype=torch.int32),
                )

                # Run padded
                inputs_p = [
                    PaddedTensor.from_tensor(
                        inputs[0], multipliers={0: 1, 1: seqlen_multiple}
                    ),
                    PaddedTensor.from_tensor(
                        inputs[1], multipliers={0: seqlen_multiple}, neutral_element=-1
                    ),
                ]

                torch.compiler.cudagraph_mark_step_begin()

                out_p = transformer(*inputs_p)
                out_p = out_p.clone()

        torch._dynamo.config.error_on_recompile = False


# Transformer Model Implementation
def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config.lower() in str(name).lower()
        ]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(
                config[1]
            ), name  # make sure only one 'best' match

        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
    "Mistral-7B": dict(
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
    "mini": dict(n_layer=2, n_head=2, dim=288),
    "mini_primes": dict(n_layer=1, n_head=5, dim=10),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "llama-3-8b": dict(
        block_size=8192,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3-70b": dict(
        block_size=8192,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3.1-8b": dict(
        block_size=131072,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "llama-3.1-70b": dict(
        block_size=131072,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "llama-3.1-405b": dict(
        block_size=131072,
        n_layer=126,
        n_head=128,
        n_local_heads=8,
        dim=16384,
        intermediate_size=53248,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
}


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                head_dim,
                dtype,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            dtype,
            self.config.rope_scaling,
        )
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        yy = self.wqkv(x)
        q, k, v = yy.split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, -1, self.n_head, self.head_dim)
        k = k.view(bsz, -1, self.n_local_heads, self.head_dim)
        v = v.view(bsz, -1, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, -1, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    xshaped = x.float().reshape(x.shape[0], -1, x.shape[2], x.shape[3] // 2, 2)
    freqs_cis = freqs_cis.view(1, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


if __name__ == "__main__":
    run_tests()
