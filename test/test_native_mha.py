# Owner(s): ["module: nn"]
import math
import copy

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

class TestMHADeviceType(TestCase):
    @torch.no_grad()
    def _test_transform_bias_rescale_qkv_impl(
        self, device, dtype, use_nt, use_padding=False
    ):
        tests = [
            (64, 4, 16, 8),
            # dim_per_head = 12 does not divide evenly by CPU vectorization length of 8
            (24, 2, 4, 2),
            # Make sure CUDA can handle small input sizes
            (2, 2, 2, 2),
            # dim_per_head = 6 does not divide evenly by CUDA vectorization length of 4,
            # causes alignment issues
            (24, 4, 4, 2),
            (48, 4, 16, 8),
        ]
        for (embed_dim, num_heads, bs, sl) in tests:
            with self.subTest(embed_dim=embed_dim, num_heads=num_heads, bs=bs, sl=sl):
                torch.manual_seed(9343)
                dense_x = x = (
                    torch.randn(bs, sl, 3 * embed_dim, device=device, dtype=dtype) * 10
                )
                if use_padding:
                    x[0][-1] = torch.full(x[0][-1].shape, float("-Inf"))
                if use_nt:
                    xs = list(torch.unbind(x))
                    if use_padding:
                        xs[0] = xs[0][:-1]
                    x = torch.nested.nested_tensor(xs, device=device, dtype=dtype)
                qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)

                # We have to use inference_mode here because q/k/v are
                # all views of the same Tensor, which autograd doesn't
                # like. This is fine because this function is only
                # exposed to Python for purposes of writing this test.
                with torch.inference_mode():
                    (q, k, v) = torch._transform_bias_rescale_qkv(
                        x, qkv.bias, num_heads=num_heads
                    )

                    def simple_transform_bias_rescale_qkv(qkv, bias):
                        (q, k, v) = torch.split(qkv, embed_dim, dim=-1)
                        (q_bias, k_bias, v_bias) = torch.split(bias, embed_dim, dim=-1)

                        def embiggen(x):
                            if not use_nt:
                                return x
                            b, t, d = x.size()
                            t = t + (8 - t % 8) % 8
                            newsize = (b, t, d)
                            new_x = torch.zeros(newsize, device=device, dtype=dtype)
                            new_x[:x.size()[0], :x.size()[1], :x.size()[2]] = x
                            return new_x
                        return tuple(
                            embiggen(x).reshape(
                                (bs, -1, num_heads, embed_dim // num_heads)
                            ).transpose(2, 1)
                            for x in (
                                (q + q_bias) / math.sqrt(embed_dim // num_heads),
                                (k + k_bias),
                                (v + v_bias),
                            )
                        )

                    correct_q, correct_k, correct_v = simple_transform_bias_rescale_qkv(
                        dense_x, qkv.bias
                    )
                    if use_nt and use_padding:
                        for t in (correct_q, correct_k, correct_v):
                            t[t == float("-Inf")] = 0

                self.assertEqual(q.size(), correct_q.size())
                torch.testing.assert_close(q, correct_q)
                torch.testing.assert_close(k, correct_k)
                torch.testing.assert_close(v, correct_v)

    @dtypesIfCUDA(torch.float)
    @dtypes(torch.float)
    @skipMeta
    def test_transform_bias_rescale_qkv(self, device, dtype):
        for use_padding in (False, True):
            with self.subTest(use_padding=use_padding):
                self._test_transform_bias_rescale_qkv_impl(
                    device, dtype, use_nt=False, use_padding=use_padding
                )

    @dtypesIfCUDA(torch.float)
    @dtypes(torch.float)
    @skipMeta
    @onlyCUDA
    def test_transform_bias_rescale_qkv_nested(self, device, dtype):
        for use_padding in (False, True):
            with self.subTest(use_padding=use_padding):
                self._test_transform_bias_rescale_qkv_impl(
                    device, dtype, use_nt=True, use_padding=use_padding
                )

    def _test_multihead_attention_impl(
        self, device, dtype, mode, use_nt, need_weights, average_attn_weights, use_padding=False, pad_all=False
    ):
        embed_dim = 64
        num_heads = 4
        bs = 16
        sl = 8

        q = 6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32) - 3
        if use_padding:
            if pad_all:
                for q_i in q:
                    q_i[-1] = torch.zeros_like(q[0][-1], device=device, dtype=torch.float32)
                mask = torch.zeros(q.shape[:-1], device=device, dtype=torch.bool)
                for mask_i in mask:
                    mask_i[-1] = True
            else:
                q[0][-1] = torch.zeros_like(q[0][-1], device=device, dtype=torch.float32)
                mask = torch.zeros(q.shape[:-1], device=device, dtype=torch.bool)
                mask[0][-1] = True
        if mode == "self":
            k = q
            v = q
        elif mode == "encdec":
            k = 6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32) - 3
            v = k
        elif mode == "generic":
            k = 6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32) - 3
            v = 6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32) - 3
        else:
            self.fail(f"invalid mode `{mode}`!")

        qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=torch.float32)
        native_qkv = copy.deepcopy(qkv).to(dtype=dtype)

        proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=torch.float32)
        native_proj = copy.deepcopy(proj).to(dtype=dtype)

        pt = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, device=device, dtype=torch.float32
        )

        pt.in_proj_weight = qkv.weight
        pt.in_proj_bias = qkv.bias
        pt.out_proj.weight = proj.weight
        pt.out_proj.bias = proj.bias

        class NativeMHA(torch.nn.Module):
            def __init__(self, embed_dim, num_heads, qkv, proj):
                super().__init__()
                self.qkv = qkv
                self.proj = proj
                self.embed_dim = embed_dim
                self.num_heads = num_heads

            def forward(self, q, k, v, key_padding_mask):
                return torch._native_multi_head_attention(
                    q,
                    k,
                    v,
                    self.embed_dim,
                    self.num_heads,
                    self.qkv.weight,
                    self.qkv.bias,
                    self.proj.weight,
                    self.proj.bias,
                    key_padding_mask,
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                    mask_type=1,   # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
                )

        npt = NativeMHA(
            embed_dim=embed_dim, num_heads=num_heads, qkv=native_qkv, proj=native_proj
        ).to(dtype)

        if device == "cuda":
            pt = pt.cuda()
            npt = npt.cuda()

        ypt, weight_pt = pt(
            q,
            k,
            v,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            key_padding_mask=mask if use_padding else None,
        )
        if use_nt:
            qs = list(torch.unbind(q))
            if use_padding:
                if pad_all:
                    qs = [x[:-1] for x in qs]
                else:
                    qs[0] = qs[0][:-1]
            q = torch.nested.nested_tensor(qs, device=device, dtype=dtype)
            if mode == "self":
                k = v = q
            elif mode == "encdec":
                k = torch.nested.nested_tensor(torch.unbind(k), device=device, dtype=dtype)
                v = k
            else:
                k = torch.nested.nested_tensor(torch.unbind(k), device=device, dtype=dtype)
                v = torch.nested.nested_tensor(torch.unbind(v), device=device, dtype=dtype)

        native_q = q.to(dtype=dtype)
        native_k = k.to(dtype=dtype)
        native_v = v.to(dtype=dtype)

        ynpt, weight_npt = npt(
            native_q, native_k, native_v, key_padding_mask=mask if use_padding and not use_nt else None
        )
        if use_nt:
            ynpt = ynpt.to_padded_tensor(0)
            if pad_all:
                ynpt_final = torch.zeros_like(ypt)
                ynpt_final[:, :ynpt.shape[1], :] = ynpt
                ynpt = ynpt_final

        def do_pad_all(tensors):
            for t in tensors:
                for t_i in t:
                    t_i[-1] = torch.zeros_like(t_i[-1], device=device, dtype=dtype)

        # PyTorch implementation returns non-zero junk in the padding
        # locations; overwrite it so that the comparison works out.
        if use_padding:
            ypt[0][-1] = torch.zeros_like(ypt[0][-1], device=device, dtype=dtype)
            ynpt[0][-1] = torch.zeros_like(ynpt[0][-1], device=device, dtype=dtype)
            if pad_all:
                do_pad_all((ypt, ynpt))
            # Zero the last row of each TxT weight matrix
            if need_weights:
                if average_attn_weights:
                    weight_pt[0][-1] = torch.zeros_like(weight_pt[0][-1], device=device, dtype=dtype)
                    weight_npt[0][-1] = torch.zeros_like(weight_npt[0][-1], device=device, dtype=dtype)
                    if pad_all:
                        do_pad_all((weight_pt, weight_npt))
                else:
                    for nh in range(num_heads):
                        weight_pt[0][nh][-1] = torch.zeros_like(weight_pt[0][nh][-1], device=device, dtype=dtype)
                        weight_npt[0][nh][-1] = torch.zeros_like(weight_npt[0][nh][-1], device=device, dtype=dtype)

        if dtype == torch.half:
            torch.testing.assert_close(ypt, ynpt.to(torch.float32), atol=1e-3, rtol=1e-3)
        else:
            # High rtol seems necessary for
            # test_native_multihead_attention_cpu_float32 on Windows,
            # otherwise 2e-4 would likely be fine.
            torch.testing.assert_close(ypt, ynpt, atol=2e-5, rtol=2e-3)

        if need_weights:
            torch.testing.assert_close(weight_pt, weight_npt.to(torch.float32), atol=5e-4, rtol=5e-4)
        else:
            self.assertEqual(weight_pt, weight_npt)

    @dtypesIfCUDA(torch.float, torch.half)
    @dtypes(torch.float)
    @skipMeta
    @parametrize("use_nt", [False, True])
    @parametrize("use_padding, pad_all", [(False, False), (True, False), (True, True)])
    @parametrize("need_weights", [False])
    @parametrize("average_attn_weights", [False, True])
    @parametrize("fused", [False, True])
    @torch.no_grad()
    def test_native_multihead_self_attention(self, device, dtype, use_nt,
                                             need_weights, average_attn_weights, use_padding, pad_all, fused):
        for need_weights in (False, not pad_all):
            with self.subTest(use_padding=use_padding, pad_all=pad_all,
                              use_nt=use_nt, need_weights=need_weights,
                              average_attn_weights=average_attn_weights):
                with torch.backends.cuda.sdp_kernel(
                        enable_flash=False, enable_mem_efficient=False
                ) if not fused else torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_mem_efficient=True
                ):
                    self._test_multihead_attention_impl(
                        device,
                        dtype,
                        "self",
                        use_nt=use_nt,
                        use_padding=use_padding,
                        pad_all=pad_all,
                        need_weights=need_weights,
                        average_attn_weights=average_attn_weights,
                    )

    @dtypesIfCUDA(torch.float, torch.half)
    @dtypes(torch.float)
    @skipMeta
    @torch.no_grad()
    def test_native_multihead_encoder_decoder_attention(self, device, dtype):
        self._test_multihead_attention_impl(
            device,
            dtype,
            "encdec",
            use_nt=False,
            need_weights=False,
            average_attn_weights=False,
        )

    @dtypesIfCUDA(torch.float, torch.half)
    @dtypes(torch.float)
    @skipMeta
    @torch.no_grad()
    def test_native_multihead_attention(self, device, dtype):
        self._test_multihead_attention_impl(
            device,
            dtype,
            "generic",
            use_nt=False,
            need_weights=False,
            average_attn_weights=False,
        )


instantiate_device_type_tests(TestMHADeviceType, globals())

if __name__ == "__main__":
    run_tests()
