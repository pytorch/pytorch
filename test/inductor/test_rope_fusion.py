# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config as config
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA_AND_TRITON


class TestRoPEFusionTemplate(TestCase):
    device = None

    @config.patch(expand_dimension_for_pointwise_nodes=True)
    def test_rope_F_api_fusion(self):
        batch_size, seq_length = 4, 128
        num_heads, head_dim = 8, 64

        q = torch.randn(
            (batch_size, seq_length, num_heads, head_dim), device=self.device
        )
        k = torch.randn(
            (batch_size, seq_length, num_heads, head_dim), device=self.device
        )
        cos, sin = F.rotary_embedding_frequencies(
            head_dim, seq_length, device=self.device
        )

        def apply_rope(q, k, cos, sin):
            return F.apply_rotary_emb(q, k, cos, sin, seq_dim=1)

        compiled_fn = torch.compile(apply_rope)
        compiled_out = compiled_fn(q, k, cos, sin)
        eager_out = apply_rope(q, k, cos, sin)
        self.assertEqual(compiled_out, eager_out)

        code = run_and_get_triton_code(compiled_fn, q, k, cos, sin)
        # All RoPE ops (mul, neg, cat, add, chunk, unsqueeze) fuse into 1 kernel
        self.assertEqual(code.count(".run("), 1)

    @config.patch(expand_dimension_for_pointwise_nodes=True)
    def test_rope_linear_epilogue_fusion(self):
        batch_size, seq_length = 4, 128
        num_heads, head_dim = 8, 64
        embed_dim = num_heads * head_dim

        linear_q = nn.Linear(embed_dim, embed_dim, device=self.device)
        linear_k = nn.Linear(embed_dim, embed_dim, device=self.device)
        cos, sin = F.rotary_embedding_frequencies(
            head_dim, seq_length, device=self.device
        )

        def proj_and_rope(x, cos, sin):
            q = linear_q(x).view(batch_size, seq_length, num_heads, head_dim)
            k = linear_k(x).view(batch_size, seq_length, num_heads, head_dim)
            return F.apply_rotary_emb(q, k, cos, sin, seq_dim=1)

        x = torch.randn(batch_size, seq_length, embed_dim, device=self.device)

        compiled_fn = torch.compile(proj_and_rope)
        compiled_out = compiled_fn(x, cos, sin)
        eager_out = proj_and_rope(x, cos, sin)
        self.assertEqual(compiled_out, eager_out)

        # 2 matmuls + RoPE epilogue fused: expect ≤3 kernels
        code = run_and_get_triton_code(compiled_fn, x, cos, sin)
        self.assertLessEqual(code.count(".run("), 3)


if HAS_CUDA_AND_TRITON:

    class RoPEFusionGpuTests(TestRoPEFusionTemplate):
        device = GPU_TYPE


if IS_LINUX and HAS_CUDA_AND_TRITON:
    pass


if __name__ == "__main__":
    run_tests()
