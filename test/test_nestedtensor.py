# Owner(s): ["module: nestedtensor"]

import torch
import torch.nn
import unittest
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_utils import TestCase, IS_FBCODE, parametrize
from torch import nested_tensor

# Tests are ported from pytorch/nestedtensor.
# This makes porting as_nested_tensor easier in the future.
def _iter_constructors():
    # yield as_nested_tensor
    yield nested_tensor


class TestNestedTensor(TestCase):
    @torch.inference_mode()
    def _test_unbind_case(self, a, b):
        nt = nested_tensor([a, b])
        a1, b1 = nt.unbind()
        self.assertTrue(a is not a1)
        self.assertTrue(b is not b1)

        nt = nested_tensor([a, b], dtype=a.dtype)
        a1, b1 = nt.unbind(0)
        self.assertEqual(a, a1)
        self.assertEqual(b, b1)

        a = torch.randn((2, 3)).add_(1)
        nt = nested_tensor([a])
        self.assertEqual(a, nt.unbind(0)[0])

    @torch.inference_mode()
    def test_unbind_0(self):
        self._test_unbind_case(
            torch.tensor([1, 2]), torch.tensor([7, 8]),
        )

    @torch.inference_mode()
    def test_unbind_1(self):
        self._test_unbind_case(
            torch.tensor([1]), torch.tensor([7]),
        )

    # @torch.inference_mode()
    # def test_unbind_2(self):
    #     self._test_unbind_case(
    #         torch.tensor(1), torch.tensor(7),
    #     )

    @torch.inference_mode()
    def test_unbind_3(self):
        self._test_unbind_case(
            torch.tensor([1.0]), torch.tensor([]),
        )

    @torch.inference_mode()
    def test_unbind_4(self):
        self._test_unbind_case(
            torch.tensor([]), torch.tensor([]),
        )

    @torch.inference_mode()
    def test_unbind_dim(self):
        def _test_fn(unbind_fn):
            a = torch.rand(3, 2)
            b = torch.rand(2, 3)
            nt = nested_tensor([a, b])
            self.assertRaises(RuntimeError, lambda: unbind_fn(nt, 1))

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        # TODO: Re-enable this once using torch_dispatch
        # _test_fn(lambda x, dim: torch.unbind(x, dim))

    @torch.inference_mode()
    def test_nested_tensor(self):
        self.assertRaises(TypeError, lambda: nested_tensor([3.0]))
        self.assertRaises(TypeError, lambda: nested_tensor(torch.tensor([3.0])))
        self.assertRaises(TypeError, lambda: nested_tensor(4.0))

    @torch.inference_mode()
    def test_nested_tensor_matching_dim(self):
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 1 and dimension 0 for Tensor at index 0.",
            lambda: nested_tensor([torch.tensor(1.0), torch.tensor([])]),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 2 and dimension 0 for Tensor at index 1.",
            lambda: nested_tensor(
                [torch.tensor(1.0), torch.tensor(2.0), torch.tensor([])]
            ),
        )

    @torch.inference_mode()
    def test_default_nested_tensor(self):
        self.assertRaises(TypeError, lambda: nested_tensor())
        default_nested_tensor = nested_tensor([])
        default_tensor = torch.tensor([])
        # self.assertEqual(default_nested_tensor.nested_dim(), 1)
        # self.assertEqual(default_nested_tensor.nested_size(), ())
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(
            default_nested_tensor.requires_grad, default_tensor.requires_grad
        )
        self.assertIsNone(default_tensor.grad)
        # TODO: Re-enable once we have a performance driven
        # use case and implementation.
        # self.assertEqual(default_nested_tensor.is_pinned(),
        #                  default_tensor.is_pinned())

    @torch.inference_mode()
    def test_dim(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor(3.0)])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor([1, 2, 3, 4])])
            self.assertEqual(a1.dim(), 2)

    @unittest.skipIf(IS_FBCODE, "numel is not virtual in fbcode.")
    @torch.inference_mode()
    def test_numel(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError, "numel is disabled", lambda: a1.numel(),
            )

    @torch.inference_mode()
    def test_size(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "Tensors of type NestedTensorImpl do not have sizes"
                if IS_FBCODE
                else "NestedTensorImpl doesn't support sizes",
                lambda: a1.size(),
            )

    @unittest.skipIf(IS_FBCODE, "stride is not virtual in fbcode.")
    @torch.inference_mode()
    def test_stride(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support strides",
                lambda: a1.stride(),
            )

    @unittest.skipIf(IS_FBCODE, "is_contiguous is not virtual in fbcode.")
    @torch.inference_mode()
    def test_is_contiguous(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError, "is_contiguous is disabled", lambda: a1.is_contiguous()
            )

    @torch.inference_mode()
    def test_repr_string(self):
        a = nested_tensor([])
        expected = "nested_tensor([" "\n\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = nested_tensor([torch.tensor(1.0)])
        expected = "nested_tensor([" "\n  tensor(1.)" "\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = nested_tensor([torch.tensor([[1, 2]]), torch.tensor([[4, 5]])])
        expected = (
            "nested_tensor([" "\n  tensor([[1, 2]])" "," "\n  tensor([[4, 5]])" "\n])"
        )
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

    @torch.inference_mode()
    def test_activations(self):
        for func in (torch.nn.functional.relu, torch.nn.functional.relu_, torch.nn.functional.gelu, torch._C._nn.gelu_):
            t = torch.tensor([-1, 0, 1], dtype=torch.float)
            nt = nested_tensor([t])
            nested_result = func(nt)
            self.assertTrue(nested_result.is_nested)
            self.assertEqual(func(t), nested_result.unbind()[0])

    def test_to_padded_tensor_on_empty_tensor(self):
        nt = torch.nested_tensor([])
        empty = nt.to_padded_tensor(4)
        self.assertEqual(empty, torch.tensor([]))

class TestNestedTensorDeviceType(TestCase):
    @dtypes(torch.float)
    @skipMeta
    def test_to_then_from_padded_tensor_no_transform0213(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested_tensor(ts, device=device, dtype=dtype)
        padded = nt.to_padded_tensor(0)

        nt_to = torch._nested_from_padded_and_nested_example(padded, nt)

        for (t1, t2) in zip(nt.unbind(), nt_to.unbind()):
            self.assertEqual(t1, t2)
        self.assertEqual(nt.device, nt_to.device)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    @skipMeta
    @torch.inference_mode()
    def test_layer_norm(self, device, dtype):
        def _test(size):
            t0 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            t1 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            ts = [t0, t1, t0, t1]
            nt = torch.nested_tensor(ts, device=device, dtype=dtype)
            layer_norm = torch.nn.LayerNorm(size, device=device, dtype=dtype)
            nt_result = nt._nested_tensor_layer_norm(
                layer_norm.weight, layer_norm.bias, 1e-5
            )
            for (nt_subresult, t) in zip(nt_result.unbind(), ts):
                t_result = layer_norm(t.reshape(1, -1, size).squeeze(0))
                self.assertEqual(nt_subresult, t_result)

        for size in (1024, 1023, 513, 512, 256, 128, 2, 4, 32):
            _test(size)

    @skipMeta
    @torch.inference_mode()
    def test_embedding(self, device):
        inputs = [
            torch.randint(100, (L,), device=device, dtype=torch.int64)
            for L in torch.randint(5, 50, (8,))
        ]
        x = torch.nested_tensor(inputs, device=device, dtype=torch.int64)
        emb = torch.nn.Embedding(100, 8, device=device)
        y = emb(x)
        ys = y.unbind()
        for i, inp in enumerate(inputs):
            self.assertEqual(emb(inp), ys[i])

    def test_to_padded_tensor_simple(self, device):
        t = torch.randn(4, 4, 4, device=device)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested_tensor(ts, device=device)
        for padding_value in (0, 1):
            padded = nt.to_padded_tensor(padding_value)

            correct_output = t.clone()
            if padding_value == 0:
                correct_output[0][-1] = torch.zeros_like(correct_output[0][-1])
            else:
                correct_output[0][-1] = torch.ones_like(correct_output[0][-1])

            self.assertEqual(padded, correct_output)
            self.assertEqual(padded.device, torch.device(device))

    def test_to_padded_tensor_unrelated_shapes(self, device):
        ts = [
            torch.randn(1, 2, 3, device=device),
            torch.randn(2, 3, 4, device=device),
            torch.randn(4, 5, 6, device=device),
        ]
        nt = torch.nested_tensor(ts, device=device)
        pad = 42
        correct_output = torch.cat(
            [torch.nn.ConstantPad3d((0, 6 - x.shape[2], 0, 5 - x.shape[1], 0, 4 - x.shape[0]), pad)(x.unsqueeze(0)) for x in ts])
        padded = nt.to_padded_tensor(pad)
        self.assertEqual(padded, correct_output)


class TestMHADeviceType(TestCase):
    def _test_multihead_attention_impl(
        self, device, dtype, mode, use_nt, need_weights, average_attn_weights, use_padding=False, pad_all=False
    ):
        embed_dim = 64
        num_heads = 4
        bs = 16
        sl = 8

        q = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype) * 10
        if use_padding:
            if pad_all:
                for q_i in q:
                    q_i[-1] = torch.zeros_like(q[0][-1], device=device, dtype=dtype)
                mask = torch.zeros(q.shape[:-1], device=device, dtype=torch.bool)
                for mask_i in mask:
                    mask_i[-1] = True
            else:
                q[0][-1] = torch.zeros_like(q[0][-1], device=device, dtype=dtype)
                mask = torch.zeros(q.shape[:-1], device=device, dtype=torch.bool)
                mask[0][-1] = True
        if mode == "self":
            k = q
            v = q
        elif mode == "encdec":
            k = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype) * 10
            v = k
        elif mode == "generic":
            k = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype) * 10
            v = torch.randn(bs, sl, embed_dim, device=device, dtype=dtype) * 10
        else:
            self.fail(f"invalid mode `{mode}`!")

        qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
        proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

        pt = torch.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, device=device, dtype=dtype
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

            def forward(self, query, key, value, key_padding_mask):
                # 0.
                # return torch._native_multi_head_attention(

                # 1.
                # r, w = torch.nn.functional.multi_head_attention_forward(
                #     q.transpose(1, 0),
                #     k.transpose(1, 0),
                #     v.transpose(1, 0),
                #     self.embed_dim,
                #     self.num_heads,
                #     self.qkv.weight,
                #     self.qkv.bias,
                #     None,
                #     None,
                #     False,
                #     0,
                #     self.proj.weight,
                #     self.proj.bias,
                #     training=False,
                #     key_padding_mask=key_padding_mask,
                #     need_weights=need_weights,
                #     average_attn_weights=average_attn_weights,
                # )
                # return r.transpose(0, 1), w

                in_proj_weight = self.qkv.weight
                in_proj_bias = self.qkv.bias

                # set up shape vars
                bsz, tgt_len, embed_dim = query.shape
                _, src_len, _ = key.shape
                # assert embed_dim == embed_dim_to_check, \
                #     f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
                head_dim = embed_dim // num_heads
                assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
                assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

                q, k, v = torch.nn.functional._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)


                attn_mask = None
                if key_padding_mask is not None:
                    attn_mask = key_padding_mask.to(torch.bool)
                    # attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

                #
                # reshape q, k, v for multihead attention and make em batch first
                #
                k_shape_1 = k.shape[1]
                v_shape_1 = v.shape[1]
                q = q.transpose(1, 0).reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
                k = k.transpose(1, 0).reshape(k_shape_1, bsz * num_heads, head_dim).transpose(0, 1)
                v = v.transpose(1, 0).reshape(v_shape_1, bsz * num_heads, head_dim).transpose(0, 1)

                # update source sequence length after adjustments
                src_len = k.size(1)

                # merge key padding and attention masks
                if key_padding_mask is not None:
                    assert key_padding_mask.shape == (bsz, src_len), \
                        f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
                    key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                        expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
                    attn_mask = key_padding_mask

                # convert mask to float
                if attn_mask is not None and attn_mask.dtype == torch.bool:
                    new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                    new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                    attn_mask = new_attn_mask

                #
                # (deep breath) calculate attention and out projection
                #
                dropout_p = 0.0
                attn_output, attn_output_weights = torch.nn.functional._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
                attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)

                out_proj_weight = self.proj.weight
                out_proj_bias = self.proj.bias
                attn_output = torch.nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
                attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

                # optionally average attention weights over heads
                attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
                if average_attn_weights:
                    attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

                if not need_weights:
                    attn_output_weights = None

                return attn_output.transpose(0, 1), attn_output_weights

        npt = NativeMHA(
            embed_dim=embed_dim, num_heads=num_heads, qkv=qkv, proj=proj
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
            q = torch.nested_tensor(qs, device=device, dtype=dtype)
            if mode == "self":
                k = v = q
            elif mode == "encdec":
                k = torch.nested_tensor(torch.unbind(k), device=device, dtype=dtype)
                v = k
            else:
                k = torch.nested_tensor(torch.unbind(k), device=device, dtype=dtype)
                v = torch.nested_tensor(torch.unbind(v), device=device, dtype=dtype)

        ynpt, weight_npt = npt(
            q, k, v, key_padding_mask=mask if use_padding and not use_nt else None
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
            torch.testing.assert_close(ypt, ynpt, atol=1e-3, rtol=1e-3)
        else:
            torch.testing.assert_close(ypt, ynpt)

        if need_weights:
            torch.testing.assert_close(weight_pt, weight_npt)
        else:
            self.assertEqual(weight_pt, weight_npt)

    @parametrize("use_nt", [True, False])
    @parametrize("pad_all", [True, False])
    @parametrize("use_padding", [True, False])
    @dtypesIfCUDA(torch.float, torch.half)
    @dtypes(torch.float)
    @skipMeta
    @torch.inference_mode()
    def test_native_multihead_self_attention(self, device, dtype, use_padding, pad_all, use_nt):
        # Figuring out exactly which elements of the weights are garbage in this
        # case eludes me, and it's not particularly enlightening to test anyway
        # because padding doesn't especially affect the intermediate weights.
        for need_weights in (False, not pad_all):
            for average_attn_weights in (False, True):
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
    @torch.inference_mode()
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
    @torch.inference_mode()
    def test_native_multihead_attention(self, device, dtype):
        self._test_multihead_attention_impl(
            device,
            dtype,
            "generic",
            use_nt=False,
            need_weights=False,
            average_attn_weights=False,
        )


instantiate_device_type_tests(TestNestedTensorDeviceType, globals())
instantiate_device_type_tests(TestMHADeviceType, globals())
