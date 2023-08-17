# Owner(s): ["module: nn"]

import contextlib
from functools import partial
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import unittest
from unittest.mock import patch, MagicMock, ANY
import math
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.optim as optim
from torch.testing._internal.common_dtype import floating_types_and_half
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA, onlyCPU
from typing import List, Tuple, Union, Optional
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    TEST_FAIRSEQ,
    run_tests,
    parametrize,
    freeze_rng_state,
    TEST_WITH_CROSSREF,
    slowTest,
    set_default_dtype,
    gradcheck
)


from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from torch.testing._internal.common_cuda import SM75OrLater, SM80OrLater, PLATFORM_SUPPORTS_FUSED_SDPA

if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer


@contextlib.contextmanager
def use_deterministic_algorithims(mode: bool, warn_only: bool):
    r"""
    This context manager can be used to temporarily enable or disable deterministic algorithms.
    Upon exiting the context manager, the previous state of the flag will be restored.
    """
    previous_mode: bool = torch.are_deterministic_algorithms_enabled()
    previous_warn_only: bool = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(mode, warn_only=warn_only)
        yield {}
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

isSM86or89Device = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(8, 6), (8, 9)]
isSM90Device = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
isSM5xDevice = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 5

def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    deviation = torch.abs(deviation / true_value)
    # Fill in the nans with the default rtol
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()


def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    atol = torch.abs(deviation).max().item()
    return atol


def get_tolerances(
    true_value: torch.Tensor,
    computed_value: torch.Tensor,
    fudge_factor: Optional[float] = None,
) -> Tuple[float, float]:
    """Returns the absolute and relative tolerances for comparing two tensors."""
    fudge_factor = fudge_factor if fudge_factor is not None else 1.0
    atol = get_atol(true_value, computed_value)
    rtol = get_rtol(true_value, computed_value)

    atol = fudge_factor * max(atol, default_atol[computed_value.dtype])
    rtol = fudge_factor * max(rtol, default_rtol[computed_value.dtype])
    # torch.isclose() has weird behavior around see:
    # https://github.com/pytorch/pytorch/issues/102400
    if rtol > 1e30:
        rtol = default_rtol[computed_value.dtype]
    return atol, rtol

backend_map = {
    SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
    SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
}


def rand_sdpa_tensor(shape: Tuple[Union[int, List[int]]], device: str, dtype: torch.dtype, type: str,
                     requires_grad: bool = False, packed: bool = False) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        type (str): Nested or Dense
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    batch, seq_len, num_heads, head_dim = shape
    if type == "nested":
        if isinstance(seq_len, list):
            def _size(i):
                return (seq_len[i], num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)

            return torch.nested.nested_tensor([
                torch.randn(_size(i), device=device, dtype=dtype, requires_grad=requires_grad)
                for i in range(batch)])
        else:
            size = (seq_len, num_heads, head_dim) if not packed else (seq_len, 3 * num_heads * head_dim)
            return torch.nested.nested_tensor([
                torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
                for _ in range(batch)])
    else:
        assert (isinstance(seq_len, int))
        size = (batch, seq_len, num_heads, head_dim) if not packed else (batch, seq_len, 3 * num_heads * head_dim)
        return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)


class TestTransformers(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @onlyCUDA
    @unittest.skip("4D mask not supported yet - activate when 4D mask supported")
    def test_self_attn_TxT_attn_mask(self, device):
        embed_dim = 16
        num_heads = 4
        batch_size = 10
        tgt_len = 16

        query = torch.rand(batch_size, tgt_len, embed_dim, device=device)  # [N, T, D]
        attn_mask = torch.randint(0, 2, (tgt_len, tgt_len)).cuda().float()  # [T, T]
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        attn_mask_4d = attn_mask.expand(batch_size, num_heads, tgt_len, tgt_len)

        mta_model = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).cuda()
        mta_model.eval()

        # Generate 3D results
        with torch.inference_mode():
            output_mask_4d = mta_model(query, query, query, attn_mask=attn_mask_4d)[0]
            output_mask_4d = output_mask_4d.transpose(0, 1)  # [N, T, D]

            output_mask_TxT = mta_model(query, query, query, attn_mask=attn_mask)[0]
            output_mask_TxT = output_mask_TxT.transpose(0, 1)  # [N, T, D]

            self.assertEqual(output_mask_4d, output_mask_TxT)

    @slowTest
    def test_train_with_pad_and_catch_error(self, device):
        iters = 100
        pad_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool).to(device)
        layer = nn.TransformerEncoderLayer(
            d_model=2,
            dim_feedforward=4,
            nhead=2,
            batch_first=True,
            activation="gelu",
            dropout=0,
        )
        criterion = nn.MSELoss()
        encoder = nn.TransformerEncoder(layer, 2).to(device)
        optimizer = optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
        encoder.train()
        for i in range(iters):
            encoder.train()
            optimizer.zero_grad()
            inputs = torch.cat([torch.randn(1, 2, 2), torch.zeros(1, 2, 2)], dim=1).to(device)

            outputs = encoder(inputs, src_key_padding_mask=pad_mask)

            loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                test = torch.cat([torch.randn(1, 2, 2), torch.zeros(1, 2, 2)], dim=1).to(device)

                # Expect uint8 type not supported
                ex = None
                try:
                    test_train_uint8 = encoder(test, src_key_padding_mask=pad_mask.to(torch.uint8))
                except AssertionError as e:
                    continue
                self.assertFalse(e, "Failed to catch unsupported uint8 type exception")

                test_train_bool = encoder(test, src_key_padding_mask=pad_mask)
                encoder.eval()

                # Expect long type not supported
                ex = None
                try:
                    test_eval_uint8 = encoder(test, src_key_padding_mask=pad_mask.to(torch.int64))
                except AssertionError as e:
                    continue
                self.assertFalse(e, "Failed to catch unsupported Long type exception")

                test_eval_bool = encoder(test, src_key_padding_mask=pad_mask)
                l1_bool = nn.L1Loss()(test_train_bool[:, 0:2, :], test_eval_bool[:, 0:2, :]).item()
                self.assertTrue(l1_bool < 1e-4, "Eval/Train difference in pad_mask BOOL")

    @parametrize("attn_mask_dim", [2, 3, None])
    @parametrize("key_padding_mask_dim", [2, None])
    def test_multiheadattention_fastpath_attn_mask(self, device, attn_mask_dim, key_padding_mask_dim):
        with torch.no_grad():
            B = 2
            L = 4
            D = 8
            H = 4

            if attn_mask_dim == 2:
                attn_mask = torch.randn(L, L, device=device) > 0
            elif attn_mask_dim == 3:
                attn_mask = torch.randn(B * H, L, L, device=device) > 0
            elif attn_mask_dim is None:
                attn_mask = None

            if key_padding_mask_dim == 2:
                key_padding_mask = torch.randn(B, L, device=device) > 0
            elif key_padding_mask_dim is None:
                key_padding_mask = None

            mha = nn.MultiheadAttention(D, H, batch_first=True, device=device)
            X = torch.randn(B, L, D, device=device)

            mha.train()  # disable fast path
            out, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            mha.eval()  # enable fast path
            out, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)

    @parametrize("nhead", [1, 4, 8])
    def test_transformerencoderlayer_src_mask(self, device, nhead):
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        model(src, src_mask=src_mask)
        model.eval()
        with torch.no_grad():
            model(src, src_mask=src_mask)

    @parametrize("use_torchscript", [False])
    @parametrize("enable_nested_tensor", [True, False])
    @parametrize("use_autocast", [True, False])
    @parametrize("d_model", [12, 256])
    def test_transformerencoder_fastpath(self, device, use_torchscript, enable_nested_tensor, use_autocast, d_model):
        """
        Test TransformerEncoder fastpath output matches slowpath output
        """
        torch.manual_seed(1234)
        nhead = 4
        dim_feedforward = d_model
        batch_first = True

        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=batch_first),
            num_layers=2,
            enable_nested_tensor=enable_nested_tensor
        ).to(device).eval()

        if use_torchscript:
            model = torch.jit.script(model)

        # each input is (input, mask)
        input_mask_pairs = [
            (
                torch.rand(3, 2, d_model),
                [
                    [0, 1],
                    [0, 1],
                    [1, 1]
                ]
            ),
            (
                torch.rand(2, 100, d_model),
                [
                    [0] * 98 + [1] * 2,
                    [0] * 90 + [1] * 10
                ]
            ),
            # softmax.cu switches from fast->slowpath at masked seqlen 1024. test 1024.
            (
                torch.rand(2, 1024, d_model),
                [
                    [0] * 1020 + [1] * 4,
                    [0] * 1024,
                ]
            ),
            (
                torch.rand(1, 1026, d_model),
                [[0] * 1024 + [1] * 2]
            ),
            # softmax.cu switches from fast->slowpath at masked seqlen 1024. test range of masks above 1024.
            (
                torch.rand(4, 1040, d_model),
                [
                    [0] * 1024 + [1] * 16,
                    [0] * 1025 + [1] * 15,
                    [0] * 1031 + [1] * 9,
                    [0] * 1040,
                ]
            )
        ]
        input_mask_pairs = [
            (
                torch.tensor(pair[0], device=device, dtype=torch.get_default_dtype()),  # float input
                torch.tensor(pair[1], device=device, dtype=torch.bool)  # bool mask
            ) for pair in input_mask_pairs
        ]

        maybe_autocast = torch.autocast("cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()
        with maybe_autocast:
            for input, src_key_padding_mask in input_mask_pairs:
                with torch.no_grad():
                    fastpath_output = model(input, src_key_padding_mask=src_key_padding_mask)
                slowpath_output = model(input, src_key_padding_mask=src_key_padding_mask)  # reference
                # Make sure fastpath_output is same shape as slowpath_output and mask.
                # When enable_nested_tensor=true, fastpath_output may be smaller than input tensor.
                # Eg if input bs=1, seqlen=6, and we mask out 2 tokens, fastpath_output will have bs=1, seqlen=4.
                # Expand back to old size to match.
                bs, true_seqlen, embed_dim = fastpath_output.shape
                expanded_seqlen = src_key_padding_mask.shape[1]
                fastpath_output_expanded = torch.zeros(bs, expanded_seqlen, embed_dim, device=device)
                fastpath_output_expanded[:, :true_seqlen, :] = fastpath_output
                # no garauntees on output corresponding to masked tokens, so they may vary between slow/fast path. set all to 0.
                fastpath_output_expanded = fastpath_output_expanded.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
                slowpath_output = slowpath_output.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
                torch.testing.assert_close(fastpath_output_expanded, slowpath_output, rtol=1e-7, atol=1e-5)

    @parametrize("with_no_grad", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [False])
    def test_transformerencoder_square_input(self, with_no_grad, training, enable_nested_tensor, device):
        """
        Test for edge cases when input of shape (batch size, sequence length, embedding dimension) has
        batch size == sequence length
        """
        model = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True),
            num_layers=2,
            enable_nested_tensor=enable_nested_tensor
        ).to(device)

        with torch.no_grad():
            # set constant weights of the model
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

        if training:
            model = model.train()
        else:
            model = model.eval()
        x = torch.arange(0, 16).reshape(2, 2, 4).to(torch.get_default_dtype()).to(device)
        src_mask = torch.Tensor([[0, 1], [0, 0]]).to(torch.bool).to(device)

        if with_no_grad:
            cm = torch.no_grad()
        else:
            cm = contextlib.nullcontext()
        with cm:
            result = model(x, mask=src_mask)

        ref_output = torch.Tensor([[[2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351],
                                    [2.420306205749512, 0.017629241570830, -0.607857942581177, -0.085519507527351]],
                                   [[2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689],
                                    [2.419836044311523, 0.017548924311996, -0.608187675476074, -0.085347734391689]]]
                                  ).to(device)
        self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
        torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

    @parametrize("batch_first", [True, False])
    @parametrize("training", [True, False])
    @parametrize("enable_nested_tensor", [True, False])
    def test_transformerencoder(self, batch_first, training, enable_nested_tensor, device):
        def get_a_test_layer(activation, batch_first=False):
            d_model = 4
            nhead = 2
            dim_feedforward = 16
            dropout = 0.0

            layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
            ).to(device)

            with torch.no_grad():
                # set constant weights of the model
                for idx, p in enumerate(layer.parameters()):
                    x = p.data
                    sz = x.view(-1).size(0)
                    shape = x.shape
                    x = torch.cos(torch.arange(0, sz).float().view(shape))
                    p.data.copy_(x)

            return layer

        # this is a deterministic test for TransformerEncoder
        activation = F.relu

        def _test(batch_first, training, enable_nested_tensor):
            def perm_fn(x):
                return x.transpose(1, 0) if batch_first else x

            encoder_layer = get_a_test_layer(activation=activation,
                                             batch_first=batch_first)

            model = nn.TransformerEncoder(
                encoder_layer, 1, enable_nested_tensor=enable_nested_tensor
            ).to(device)

            if not training:
                model = model.eval()

            # deterministic input
            encoder_input = perm_fn(torch.tensor([[[0.7462, 0.6653, 0.5679, 0.4891],
                                                   [0.5387, 0.1655, 0.3565, 0.0471]],
                                                  [[0.8335, 0.2799, 0.5031, 0.2947],
                                                   [0.1402, 0.0318, 0.7636, 0.1346]],
                                                  [[0.6333, 0.9344, 0.1376, 0.9938],
                                                   [0.8924, 0.2872, 0.6692, 0.2944]],
                                                  [[0.9897, 0.6915, 0.3154, 0.1733],
                                                   [0.8645, 0.3513, 0.3064, 0.0767]],
                                                  [[0.8117, 0.2366, 0.4838, 0.7881],
                                                   [0.3718, 0.4945, 0.9511, 0.0864]]]
                                                 )).to(device)
            result = model(encoder_input)
            ref_output = perm_fn(torch.tensor([[[2.428589, 0.020835, -0.602055, -0.085249],
                                                [2.427987, 0.021213, -0.602496, -0.084103]],
                                               [[2.424689, 0.019155, -0.604793, -0.085672],
                                                [2.413863, 0.022211, -0.612486, -0.072490]],
                                               [[2.433774, 0.021598, -0.598343, -0.087548],
                                                [2.425104, 0.019748, -0.604515, -0.084839]],
                                               [[2.436185, 0.022682, -0.596625, -0.087261],
                                                [2.433556, 0.021891, -0.598509, -0.086832]],
                                               [[2.416246, 0.017512, -0.610712, -0.082961],
                                                [2.422901, 0.024187, -0.606178, -0.074929]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # all 0 src_mask
            src_mask = torch.zeros([5, 5]).to(device) == 1
            result = model(encoder_input, mask=src_mask)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # all 0
            mask = torch.zeros([2, 5]).to(device) == 1
            result = model(encoder_input, src_key_padding_mask=mask)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            mask[0, 1] = 1
            mask[1, 3] = 1
            mask[1, 4] = 1
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.429026, 0.020793, -0.601741, -0.085642],
                                                [2.428811, 0.021445, -0.601912, -0.084252]],
                                               [[2.425009, 0.019155, -0.604566, -0.085899],
                                                [2.415408, 0.02249, -0.611415, -0.073]],
                                               [[2.434199, 0.021682, -0.598039, -0.087699],
                                                [2.42598, 0.019941, -0.603896, -0.085091]],
                                               [[2.436457, 0.022736, -0.59643, -0.08736],
                                                [2.434021, 0.022093, -0.598179, -0.08679]],
                                               [[2.416531, 0.017498, -0.610513, -0.083181],
                                                [2.4242, 0.024653, -0.605266, -0.074959]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # test case 2, multiple layers no norm
            model = nn.TransformerEncoder(encoder_layer, 2, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.419051, 0.017446, -0.608738, -0.085003],
                                                [2.419102, 0.017452, -0.608703, -0.085026]],
                                               [[2.419043, 0.017445, -0.608744, -0.084999],
                                                [2.419052, 0.017446, -0.608738, -0.085004]],
                                               [[2.419067, 0.017448, -0.608727, -0.085010],
                                                [2.419098, 0.017452, -0.608706, -0.085024]],
                                               [[2.419072, 0.017449, -0.608724, -0.085012],
                                                [2.419119, 0.017455, -0.608691, -0.085034]],
                                               [[2.419019, 0.017442, -0.608761, -0.084989],
                                                [2.419075, 0.017449, -0.608722, -0.085014]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            model = nn.TransformerEncoder(encoder_layer, 6, enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]],
                                               [[2.419101, 0.017453, -0.608703, -0.085025],
                                                [2.419101, 0.017453, -0.608704, -0.085025]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            # test case 3, multiple layers with norm
            # d_model = 4
            norm = nn.LayerNorm(4)
            model = nn.TransformerEncoder(encoder_layer, 2, norm=norm,
                                          enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[1.695949, -0.357635, -0.893077, -0.445238],
                                                [1.695955, -0.357639, -0.893050, -0.445266]],
                                               [[1.695948, -0.357634, -0.893082, -0.445233],
                                                [1.695950, -0.357635, -0.893077, -0.445238]],
                                               [[1.695951, -0.357636, -0.893069, -0.445246],
                                                [1.695955, -0.357639, -0.893052, -0.445264]],
                                               [[1.695952, -0.357636, -0.893066, -0.445249],
                                                [1.695957, -0.357641, -0.893041, -0.445276]],
                                               [[1.695946, -0.357632, -0.893095, -0.445220],
                                                [1.695952, -0.357637, -0.893065, -0.445251]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

            model = nn.TransformerEncoder(encoder_layer, 6, norm=norm,
                                          enable_nested_tensor=enable_nested_tensor).to(device)
            if not training:
                model = model.eval()
            result = model(encoder_input, src_key_padding_mask=mask)
            ref_output = perm_fn(torch.tensor([[[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]],
                                               [[1.695955, -0.357639, -0.893051, -0.445265],
                                                [1.695955, -0.357639, -0.893051, -0.445265]]]
                                              )).to(device)
            self.assertEqual(tuple(result.shape), tuple(ref_output.shape))
            torch.testing.assert_close(result, ref_output, rtol=1e-7, atol=1e-5)

        # TODO: remove set default dtype to double by making ref_output more precise.
        # Added because this test was copied from test_nn.py, which has default
        # dtype double. If default dtype is float, tests will say tensors not close because
        # ref output precision too low
        with set_default_dtype(torch.double):
            if training:
                cm = contextlib.nullcontext()
            else:
                cm = torch.no_grad()  # transformer fast path requires no grad
            with cm:
                _test(batch_first, training, enable_nested_tensor)

    @unittest.skipIf(sys.version_info < (3, 11), "not supported on pre-3.11 Python")
    def test_encoder_padding_and_src_mask_bool(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(16)
        encoder = nn.TransformerEncoder(
            encoder_layer, 2, encoder_norm
        )

        inputs = torch.randn(2, 3, 16)

        src_mask = torch.ones(3, 3, dtype=torch.bool).triu_(diagonal=1)
        input_seq_len = torch.tensor([3, 2])
        padding_mask = (
            torch.arange(3)[None, :].cpu() >= input_seq_len[:, None]
        )

        with self.assertNoLogs(None):
            encoder(
                inputs,
                mask=src_mask,
                src_key_padding_mask=padding_mask,
            )

    @unittest.skipIf(sys.version_info < (3, 11), "not supported on pre-3.11 Python")
    def test_decoder_padding_and_src_mask_bool(self):

        def transformer_decoder(inputs, input_seq_len, memory):
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=16,
                nhead=2,
                dim_feedforward=32,
                dropout=0.1,
                activation='relu',
                batch_first=True,
            )
            decoder_norm = nn.LayerNorm(16)
            decoder = nn.TransformerDecoder(
                decoder_layer, 2, decoder_norm
            )

            src_mask = torch.ones(
                inputs.shape[1], inputs.shape[1], dtype=torch.bool
            ).triu_(diagonal=1)
            padding_mask = (
                torch.arange(inputs.shape[1])[None, :].cpu()
                >= input_seq_len[:, None]
            )

            return decoder(
                inputs,
                memory,
                tgt_mask=src_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )

        inputs = torch.randn(2, 3, 16)
        memory = torch.randn(2, 3, 16)
        input_seq_len = torch.tensor([3, 2])

        with self.assertNoLogs(None):
            transformer_decoder(inputs, input_seq_len, memory)

    def test_encoder_is_causal(self):

        d_model = 3
        layer = torch.nn.TransformerEncoderLayer(d_model, 1, 6, batch_first=True)
        layer.eval()
        x = torch.randn(1, 5, d_model)
        unmasked_output = layer(x)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        is_causal_output = layer(x, src_mask=mask, is_causal=True)
        masked_output = layer(x, src_mask=mask)

        self.assertEqual(masked_output, is_causal_output)

    @onlyCUDA
    @parametrize("nb_heads", [1, 8])
    @parametrize("bias", [True, False])
    def test_mha_native_args(self, nb_heads, bias):

        B, L, F = 8, 100, 128
        batch_first = True
        fast_path = True
        use_pad_mask = (bias % 2) == 1

        mha = nn.MultiheadAttention(
            embed_dim=F,
            num_heads=nb_heads,
            batch_first=batch_first,
            bias=bias
        ).cuda()
        mha.eval()

        ctx = torch.no_grad if fast_path else contextlib.nullcontext
        with ctx():
            x = torch.randn(B, L, F).cuda()
            if not batch_first:
                x = x.transpose(0, 1)

            pad_mask = None
            if use_pad_mask:
                pad_mask = torch.zeros((B, L), dtype=torch.bool).cuda()

            mha(query=x, key=x, value=x, key_padding_mask=pad_mask)

    def test_kpm_mask_trailing_column_with_nested_tensor(self, device):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation='gelu',
            norm_first=False,
            batch_first=False,
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=True).to(device)

        x = torch.randn(10, 6, 256).to(device)
        mask = torch.ones(6, 10)
        mask[0, :] = 0  # here I masked 5 columns instead of just one
        mask = mask.bool().to(device)
        out = transformer_encoder(src=x, src_key_padding_mask=mask)
        self.assertEqual(out.shape[1], 6)

    # CPU unit test has_torch_functions in test environment,
    #   preventing successful completion
    @onlyCUDA
    def test_with_nested_tensor_input(self, device):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            activation='gelu',
            norm_first=False,
            batch_first=True,
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=True).to(device)

        transformer_encoder.eval()
        with torch.no_grad():
            x = torch.randn(6, 10, 256).to(device)
            mask = torch.ones(6, 10)
            mask[0, 0:] = 0  # here I masked 5 columns instead of just one
            mask[2, 2:] = 0  # here I masked 5 columns instead of just one
            mask[4, 4:] = 0  # here I masked 5 columns instead of just one
            mask[5, 8:] = 0  # here I masked 5 columns instead of just one
            mask = mask.bool().to(device)
            x = torch._nested_tensor_from_mask(x, mask.logical_not(), mask_check=False)
            out = transformer_encoder(src=x, src_key_padding_mask=None)

        self.assertEqual(out.is_nested, True)



    def test_script_encoder_subclass(self, device):
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        encoder = nn.TransformerEncoder(
            MyCustomLayer(d_model=256, nhead=8), num_layers=6
        ).to(device=device)
        torch.jit.script(encoder)

    # brazenly adapted from test_transformerencoderlayer_src_mask to test execution of
    # torchscripted transformerencoderlayer subclass
    def test_transformerencoderlayer_subclass(self, device):
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        nhead = 4
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        model = MyCustomLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True).to(device)
        script_model = torch.jit.script(model)

        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        torch.manual_seed(42)
        result = model(src, src_mask=src_mask)
        torch.manual_seed(42)
        scripted_result = script_model(src, src_mask=src_mask)
        self.assertEqual(result, scripted_result)

        model.eval()
        script_model = torch.jit.script(model)

        with torch.no_grad():
            result = model(src, src_mask=src_mask)
            scripted_result = script_model(src, src_mask=src_mask)
            self.assertEqual(result, scripted_result)


    def test_transformerencoderlayer_subclass_model(self, device):
        class MyCustomLayer(nn.TransformerEncoderLayer):
            pass

        nhead = 4
        batch_size = 2
        seqlen = 4
        d_model = 8
        dim_feedforward = 32

        layer = MyCustomLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True)
        model = nn.TransformerEncoder(
            layer, num_layers=6
        ).to(device=device)
        script_model = torch.jit.script(model)

        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model
        src_mask = torch.zeros(seqlen, seqlen).to(torch.bool).to(device)

        torch.manual_seed(42)
        result = model(src, mask=src_mask)
        torch.manual_seed(42)
        scripted_result = script_model(src, mask=src_mask)
        self.assertEqual(result, scripted_result)

        model.eval()
        script_model = torch.jit.script(model)

        with torch.no_grad():
            result = model(src, mask=src_mask)
            scripted_result = script_model(src, mask=src_mask)
            self.assertEqual(result, scripted_result)


    @onlyCUDA
    @unittest.skipIf(not TEST_FAIRSEQ, "Fairseq not found")
    def test_decoder_only_layer(self):
        DEFAULT_PADDING_IDX = 0

        class FairseqDecoder(torch.nn.Module):
            def __init__(
                self,
                embed_dim,
                attention_heads,
                ffn_embed_dim,
                num_layers,
                embedding_layer,  # torch.nn.Embedding. Must have a padding_idx field
                dropout=0,
                normalize_before=False,
                torch_encoder=None,  # torch encoder that you can map weights from
                activation="relu",
            ):
                super().__init__()

                cfg = fairseq_transformer.TransformerConfig()
                cfg.decoder.embed_dim = embed_dim
                cfg.decoder.output_dim = embed_dim
                cfg.decoder.attention_heads = attention_heads
                cfg.decoder.ffn_embed_dim = ffn_embed_dim
                cfg.dropout = dropout
                cfg.decoder.normalize_before = normalize_before
                cfg.decoder.layers = num_layers
                # make embedding behavior same as other encoders
                cfg.no_token_positional_embeddings = True
                cfg.no_scale_embedding = True
                cfg.activation_fn = activation

                dictionary = {}  # TODO: verify what this is

                self.decoder = fairseq_transformer.TransformerDecoder(
                    cfg,
                    dictionary,
                    embedding_layer,
                    no_encoder_attn=True,
                    output_projection=None,
                )

                if torch_encoder is not None:
                    self.decoder = torch_to_fairseq(torch_encoder, self.decoder)
                self.decoder = self.decoder.eval().cuda().half()

            def forward(
                self,
                tokens,
                src_lengths=None,
                with_triangle_mask=False,
                incremental_state=None,
            ):
                return self.decoder(
                    prev_output_tokens=tokens,
                    encoder_out=None,
                    incremental_state=incremental_state,
                    features_only=True,
                    full_context_alignment=not with_triangle_mask,
                    alignment_layer=None,
                    alignment_heads=None,
                    src_lengths=src_lengths,
                    return_all_hiddens=False,
                )[0]

    @parametrize("input_dim,attn_mask_dim,is_causal",
                 [(3, None, False), (3, 2, False), (3, 2, True), (3, 3, False), (3, 3, True),
                  (4, None, False), (4, 2, False), (4, 2, True), (4, 4, False), (4, 4, True)],
                 name_fn=lambda input_dim, attn_dim, is_causal: (
                     f"{input_dim}D_input_dim_" + (
                         f"{attn_dim}D_{'causal_' if is_causal else ''}attn_mask"
                         if attn_dim is not None else "no_attn_mask")))
    @parametrize("dropout_p", [0.0, 0.2, 0.5])
    @sdp_kernel(enable_flash=False, enable_mem_efficient=False)
    def test_scaled_dot_product_attention(self, device, input_dim, attn_mask_dim, is_causal, dropout_p):
        def sdp_ref(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0):
            E = q.size(-1)
            q = q / math.sqrt(E)
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            if attn_mask is not None:
                attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
            else:
                attn = torch.bmm(q, k.transpose(-2, -1))

            attn = torch.nn.functional.softmax(attn, dim=-1)
            if dropout_p > 0.0:
                attn = torch.nn.functional.dropout(attn, p=dropout_p)
            # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
            output = torch.bmm(attn, v)
            return output
        # TODO: Support cross-device / dtype testing properly when instantiate_device_type_tests() is used.
        dtypes = [torch.double, torch.float]
        for dtype in dtypes:

            def rand_tensor(*shape):
                return torch.randn(shape, device=device, dtype=dtype)

            # This test compares python and C++ implementations of SDP.
            N, N_prime, L, S, E = 5, 2, 4, 3, 6
            if input_dim == 3:
                query = rand_tensor(N, L, E)
                key = rand_tensor(N, S, E)
                value = rand_tensor(N, S, E)
            elif input_dim == 4:
                query = rand_tensor(N, N_prime, L, E)
                key = rand_tensor(N, N_prime, S, E)
                value = rand_tensor(N, N_prime, S, E)
            else:
                self.fail(f'Invalid input_dim {input_dim} encountered in SDP test')

            attn_mask = None
            if attn_mask_dim is not None:
                assert attn_mask_dim in [2, input_dim]
                mask_size = (L, S) if attn_mask_dim == 2 else ((N, L, S) if input_dim == 3 else (N, N_prime, L, S))
                attn_mask = (torch.ones(mask_size, device=device, dtype=torch.bool).tril() if is_causal
                             else torch.randint(0, 2, size=mask_size, device=device, dtype=torch.bool))

            with freeze_rng_state():
                # Python impl only supports float mask and 3D inputs.
                attn_mask_float = attn_mask
                if attn_mask_float is not None:
                    attn_mask_float = torch.zeros_like(attn_mask, dtype=query.dtype)
                    attn_mask_float.masked_fill_(attn_mask.logical_not(), float("-inf"))
                q, k, v = query.view(-1, L, E), key.view(-1, S, E), value.view(-1, S, E)
                a = attn_mask_float
                if a is not None and attn_mask_dim > 3:
                    a = a.view(-1, L, S)
                expected = sdp_ref(q, k, v, attn_mask=a, dropout_p=dropout_p)
                if input_dim > 3:
                    expected = expected.view(-1, N_prime, L, E)

            with freeze_rng_state():
                if is_causal:
                    # NB: Don't pass attn_mask here
                    actual = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, None, dropout_p, is_causal)

                    # Error case: both explicit attn_mask and is_causal are set
                    with self.assertRaisesRegex(RuntimeError,
                                                "Explicit attn_mask should not be set when is_causal=True"):
                        torch.nn.functional.scaled_dot_product_attention(
                            query, key, value, attn_mask, dropout_p, is_causal)
                else:
                    actual = torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask, dropout_p, is_causal)

                self.assertEqual(actual, expected)

        if attn_mask_dim is None:
            q = q.double().clone()
            k = k.double().clone()
            v = v.double().clone()
            q.requires_grad_()
            k.requires_grad_()
            v.requires_grad_()

            assert gradcheck(lambda *args, **kwargs: wrapper_set_seed(sdp_ref, *args, **kwargs),
                             (q, k, v, attn_mask, dropout_p))
            assert gradcheck(lambda *args, **kwargs:
                             wrapper_set_seed(torch.nn.functional.scaled_dot_product_attention, *args, **kwargs),
                             (q, k, v, attn_mask, dropout_p))

        def test_incompatible_mask(self, device):
            def ones_tensor(*shape):
                return torch.ones(shape, dtype=torch.float32)
            S, L, E, H = 1, 2, 4, 1
            qkv = ones_tensor(S, L, E)

            mha = nn.MultiheadAttention(E, H)
            mha.in_proj_weight = Parameter(torch.ones((E * 3, E)))
            mha.out_proj.weight = Parameter(torch.ones((E, E)))
            qkv = qkv.to(float)
            kpm = ones_tensor(S, L) * float("-inf")
            am = ones_tensor(L, L).to(bool)

            def func():
                return mha(qkv, qkv, qkv, need_weights=False, key_padding_mask=kpm, attn_mask=am)

            self.assertRaises(RuntimeError, func)

    @unittest.skipIf(TEST_WITH_CROSSREF, 'Fastpath not available with crossref')
    @torch.no_grad()
    def test_mask_check_fastpath(self):
        """
        Test that fastpath is executed independently of the masks that are passed.
        If the passed key padding mask is left aligned or mask_check=False, test that nested tensors are used
        (sparsity fastpath), otherwise use fastpath with traditional tensors.
        Also test that fast path is executed with both key padding mask and attention mask passed at the same time.
        """

        x = torch.Tensor([[[1, 2], [3, 4], [5, 6]]]).to(torch.float)

        def _test_fastpath(model, key_padding_mask, mock_return_value, attn_mask=None, nested_tensors=True):
            with patch('torch._transformer_encoder_layer_fwd') as fastpath_mock:
                fastpath_mock.return_value = mock_return_value
                model(x, src_key_padding_mask=key_padding_mask, mask=attn_mask)

                # If mock was called, fastpath was taken
                self.assertTrue(fastpath_mock.called)

                # If mock was called with nested tensors, sparsity fastpath was taken
                for call_args, _ in fastpath_mock.call_args_list:
                    self.assertEqual(call_args[0].is_nested, nested_tensors)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=2, nhead=2, dim_feedforward=8, batch_first=True)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=True)
        model.eval()

        aligned_key_padding_mask = torch.Tensor([[0, 0, 1]]).to(torch.bool)
        not_aligned_key_padding_mask = torch.Tensor([[1, 0, 1]]).to(torch.bool)
        attn_mask = torch.Tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).to(torch.bool)
        nested_tensor_return_value = torch.nested.nested_tensor([torch.ones((2, 2), dtype=torch.float)])
        tensor_return_value = torch.ones((1, 3, 2), dtype=torch.float)

        # Left aligned mask results in sparsity fastpath
        _test_fastpath(model, aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)

        # Not aligned mask results in fastpath
        _test_fastpath(model, not_aligned_key_padding_mask, tensor_return_value, nested_tensors=False)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False, mask_check=True)
        model.eval()

        # If nested tensor disabled, fastpath is always taken
        _test_fastpath(model, aligned_key_padding_mask, tensor_return_value, nested_tensors=False)
        _test_fastpath(model, not_aligned_key_padding_mask, tensor_return_value, nested_tensors=False)
        # Fast path is taken if both attention mask and key padding mask are present
        _test_fastpath(model, aligned_key_padding_mask, tensor_return_value, attn_mask=attn_mask, nested_tensors=False)

        model = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=False)
        model.eval()

        # Mask check disabled results in sparisty fastpath, independently of the mask
        _test_fastpath(model, aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)
        _test_fastpath(model, not_aligned_key_padding_mask, nested_tensor_return_value, nested_tensors=True)

    # Test failing MHA when bias was NoneType
    def test_bias_is_none(self):
        x = torch.rand((1, 5, 10))
        model = torch.nn.modules.activation.MultiheadAttention(10, 1, bias=False, batch_first=True)
        model.eval()
        model(x, x, x)
        # completes without error

    def test_train_with_is_causal(self, device):
        # training with is_causal
        S, L, E, H = 1, 2, 2, 1
        layer = nn.TransformerEncoderLayer(
            d_model=2,
            dim_feedforward=4,
            nhead=H,
            batch_first=True,
            activation="gelu",
            dropout=0,
        )
        criterion = nn.MSELoss()
        encoder = nn.TransformerEncoder(layer, 2).to(device)
        optimizer = optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9)
        encoder.train()

        encoder.train()
        optimizer.zero_grad()
        inputs = torch.randn(S, L, E).to(device)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            inputs.size(1), device=device
        )

        outputs = encoder(inputs, mask=mask, is_causal=True)

        loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
        loss.backward()
        optimizer.step()

        # inference with is_causal
        t_qvk = torch.randn((S, L, E), device=device, dtype=torch.float32)
        mha = nn.MultiheadAttention(E, H).to(device)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            S, device=device
        )

        attn_out, _ = mha(t_qvk, t_qvk, t_qvk, attn_mask=mask, is_causal=True)

        # Can't give only is_causal
        attn_mask = torch.randint(0, 2, size=(L, L), device=device, dtype=torch.bool)
        with self.assertRaises(RuntimeError):
            _ = mha(t_qvk, t_qvk, t_qvk, is_causal=True)

        # # Passing a causal mask sets is_causal to 1
        causal_mask = torch.triu(
            torch.ones(L, L, device=inputs.device) * float('-inf'), diagonal=1
        ).to(torch.bool)

        mock_layer = MagicMock(torch.nn.MultiheadAttention(E, H), return_value=inputs)
        encoder.layers[1] = mock_layer
        outputs = encoder(inputs, mask=causal_mask)
        mock_layer.assert_called_with(ANY, src_mask=ANY, is_causal=True, src_key_padding_mask=ANY)

        # check expected numerical values with all kernels
        self.is_causal_kernels(["math"], device)

    def is_causal_kernels(self, kernels, device):
        def ones_tensor(*shape):
            return torch.ones(shape, device=device, dtype=torch.float32).to(device)
        S, L, E, H = 1, 2, 4, 1
        qkv = ones_tensor(S, L, E)

        mha = nn.MultiheadAttention(E, H).to(device)
        mha.in_proj_weight = Parameter(torch.ones((E * 3, E), device=device))
        mha.out_proj.weight = Parameter(torch.ones((E, E), device=device))
        expected = torch.ones(size=(S, L, E)).to(device) * 16
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            qkv.size(1), device=device
        )

        for kernel in kernels:
            with torch.backends.cuda.sdp_kernel(
                enable_math=(kernel == 'math'),
                enable_flash=(kernel == 'flash'),
                enable_mem_efficient=(kernel == 'meff')
            ):
                actual, _ = mha(qkv, qkv, qkv, attn_mask=mask, need_weights=False, is_causal=True)
                self.assertTrue(torch.equal(actual, expected))

                if kernel != 'math':
                    # fails with embedding size not multiple of 4
                    with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                        qkv_f, mha_f = ones_tensor(S, L, 2), nn.MultiheadAttention(2, H).to(device)
                        mask = torch.nn.Transformer.generate_square_subsequent_mask(
                            qkv_f.size(1), device=device
                        )
                        _ = mha_f(qkv_f, qkv_f, qkv_f, attn_mask=mask, need_weights=False, is_causal=True)
                        torch.cuda.synchronize()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Platform does not supposrt fused SDPA or pre-SM80 hardware"
    )
    def test_is_causal_gpu(self):
        device = 'cuda'
        self.is_causal_kernels(["math", "meff"], device)

    def test_script_mha_in_proj_weight_none(self):
        mha = torch.nn.MultiheadAttention(
            embed_dim=128, num_heads=8, kdim=256, vdim=256
        ).eval()

        torch.jit.script(mha)


class TestSDPAFailureModes(NNTestCase):
    """ Used to test the failure modes of scaled_dot_product_attention
    """
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not isSM86or89Device,
                     "Does not support fused SDPA or not SM86+ hardware")
    @parametrize("head_dim", [72, 96, 128])
    def test_flash_backward_failure_sm86plus(self, device, head_dim: int):
        dtype = torch.float16
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype)
        # See check_requires_grad_and_head_dim_gt64_and_sm_ge86 in pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.h
        size = (2, 2, 4, head_dim)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)

        with sdp_kernel(enable_mem_efficient=False, enable_flash=False, enable_math=True):
            math_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

        with sdp_kernel(enable_mem_efficient=False, enable_flash=True, enable_math=False):
            # Should not fail because inputs don't require grad
            flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

            self.assertEqual(math_ref, flash_ref, atol=1e-3, rtol=1e-3)

            # Should fail because inputs require grad
            q = make_tensor(size, requires_grad=True)
            k = make_tensor(size, requires_grad=True)
            v = make_tensor(size, requires_grad=True)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    def test_dispatch_fails_no_backend(self, device):
        dtype = torch.float16
        with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=False):
            size = (2, 3, 4)
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch._fused_sdp_choice(q, k, v))
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if SM80OrLater
        else [SDPBackend.EFFICIENT_ATTENTION],
    )
    def test_invalid_fused_inputs_dim_3(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            # Dim is not 4
            size = (2, 3, 8)
            dtype = torch.float16
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            with self.assertWarnsRegex(UserWarning, "Both fused kernels requires query, key and value to be 4 dimensional"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if SM80OrLater
        else [SDPBackend.EFFICIENT_ATTENTION],
    )
    def test_invalid_fused_inputs_broadcast(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            #  Fused Kernels don't support broadcasting for dense inputs
            dtype = torch.float16
            size = (2, 4, 3, 8)
            size_broadcast = (1, 4, 3, 8)
            q = torch.randn(size_broadcast, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 SM80OrLater else [SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_sequence_lengths(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            # Passing in a q,k,v with 0 length sequences will error
            dtype = torch.float16
            make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype)
            size = (2, 2, 0, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)

            with self.assertWarnsRegex(UserWarning, "Both fused kernels do not support zero seq_len_q or seq_len_kv."):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 SM80OrLater else [SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_last_dim_stride(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            # Passing in a q,k,v with 0 length sequences will error
            dtype = torch.float16
            make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype)
            size = (2, 2, 8, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            q.as_strided_(size, [2, 2, 2, 2])
            with self.assertWarnsRegex(UserWarning, "Both fused kernels require the last dimension of the input to have stride 1."):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support fused scaled dot product attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_fused_inputs_head_dim(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            # The embed dim per head is not divisible by 8 for flash attention
            dtype = torch.float16
            make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype)
            size = (2, 2, 3, 9)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if SM80OrLater
        else [SDPBackend.EFFICIENT_ATTENTION],
    )
    def test_invalid_fused_inputs_invalid_dtype(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            # Invalid dtype for both Flash Attention and Mem Efficient Attention
            size = (2, 2, 3, 16)
            make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float64)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support fused scaled dot product attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION])
    def test_invalid_fused_inputs_attn_mask_present(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            # Failures for unsupported SDP args
            size = (2, 2, 3, 16)
            make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float16)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            # Non-None attention mask
            mask = torch.ones((2, 2, 3, 3), device=device, dtype=q.dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, mask, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support fused SDPA or pre-SM80 hardware")
    def test_unaligned_tensors(self, device):
        # The alignment is depdent on arch so we specifiy SM80OrLater
        dtype = torch.float16
        shape = (2, 2, 8, 5)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False):
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support fused SDPA or pre-SM80 hardware")
    def test_flash_fail_fp32(self, device):
        dtype = torch.float
        shape = (16, 16, 32, 32)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, BFloat16}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_float16(self, device):
        dtype = torch.float
        shape = (16, 16, 32, 32)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_bfloat16(self, device):
        dtype = torch.float
        shape = (16, 16, 32, 32)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_datatypes(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            # Different datatypes
            shape = (1, 4, 8, 16)
            query = torch.randn(shape, dtype=torch.float32, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_devices(self, device, kernel: SDPBackend):
        # Different devices
        shape = (1, 4, 8, 16)
        query = torch.randn(shape, dtype=torch.float32, device=device)
        key = torch.randn(shape, dtype=torch.float16, device='cpu')
        value = torch.randn(shape, dtype=torch.float16, device='cpu')
        self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_1_dimensional_inputs(self, device, kernel: SDPBackend):
        with sdp_kernel(**backend_map[kernel]):
            # 1 dimensional input
            shape = (1, 4)
            query = torch.randn(4, dtype=torch.float16, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_error_cases(self, device):
        # one of k,v needs to be broadcasted and other has non consistent seq_len dim
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        batch, num_heads, head_dim = 32, 8, 64
        seq_lens_q = torch.randint(low=1, high=32, size=(batch,)).tolist()
        seq_lens_v = torch.randint(low=1, high=32, size=(batch,)).tolist()

        q_shape = (batch, seq_lens_q, num_heads, head_dim)
        k_shape = (1, 1, num_heads, head_dim)
        v_shape = (batch, seq_lens_v, num_heads, head_dim)

        query = rand_nested_tensor(q_shape).transpose(1, 2)
        key = rand_nested_tensor(k_shape).transpose(1, 2)
        value = rand_nested_tensor(v_shape).transpose(1, 2)

        with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not isSM5xDevice, "Does not support fused SDPA or not SM50 hardware")
    def test_mem_efficient_fail_bfloat16_sm50(self, device):
        dtype = torch.bfloat16
        shape = (16, 16, 32, 32)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, Float}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

class TestSDPA(NNTestCase):
    """ Used to test generic functionality of scaled_dot_product_attention
    Summary:
        If you are adding a new test to this class, make sure that it runs
        for both cpu and cuda. If you're test is only applicable to cuda,
        add it to TestSDPACudaOnly.
    """
    @parametrize("contiguous_inputs", [True, False])
    def test_sdp_math_gradcheck(self, device, contiguous_inputs: bool):

        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor((batch_size, seq_len, num_heads, head_dim))
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False):
            assert gradcheck(lambda *args, **kwargs:
                             wrapper_set_seed(torch.nn.functional.scaled_dot_product_attention, *args, **kwargs),
                             (query, key, value, None, 0.0, False)
                             )

    @onlyCPU
    @parametrize("type", ["dense", "nested"])
    @parametrize("dropout", [0.0, 0.7])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.half])
    def test_fused_sdp_choice_cpu(self, device, type: str, dropout: float, dtype: torch.dtype):
        # Test that cpu and nestedtensor cpu return MATH backend
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype)
        size = (2, 128, 8, 64)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        if type == "nested" \
                or dropout > 0.0 \
                or dtype not in [torch.float32, torch.float64, torch.bfloat16]:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.MATH
        else:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.FLASH_ATTENTION

    @onlyCPU
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
    @parametrize("batch_size", [2, 12])
    @parametrize("seq_len", [267, 1030])
    @parametrize("n_head", [1, 3])
    @parametrize("head_dim", [8, 16])
    @parametrize("causal", [True, False])
    @parametrize("train", [True, False])
    def test_scaled_dot_product_fused_attention_vs_math_cpu(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        seq_len,
        n_head,
        head_dim,
        causal,
        train,
    ):
        atol = 1e-5
        rtol = 5e-6
        if dtype is torch.bfloat16:
            atol = 1e-2
            rtol = 1e-2
        if dtype is torch.bfloat16 and causal and train:
            atol = 2e-2
            rtol = 2e-2

        n_embd = n_head * head_dim
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, packed=True, requires_grad=False)
        shape = (batch_size, seq_len, n_head, head_dim)
        x = make_tensor(shape)
        x2 = x.clone()

        if train:
            x.requires_grad_(True)
            x2.requires_grad_(True)

        q, k, v = x.split(n_embd, dim=2)
        q2, k2, v2 = x2.split(n_embd, dim=2)

        if dtype is torch.bfloat16:
            q2 = q2.float()
            k2 = k2.float()
            v2 = v2.float()

        # (B, nh, T, hs)
        k = k.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        q2 = q2.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)

        with sdp_kernel(**backend_map[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
        with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                q2, k2, v2, attn_mask=None, dropout_p=0.0, is_causal=causal)

        if dtype is torch.bfloat16:
            math_ref = math_ref.bfloat16()

        self.assertEqual(actual, math_ref, atol=atol, rtol=rtol)

        if train:
            actual.sum().backward()
            math_ref.sum().backward()

            grad_x, grad_x2 = x.grad, x2.grad
            grad_q_actual, grad_k_actual, grad_v_actual = grad_x.split(n_embd, dim=2)
            grad_q_ref, grad_k_ref, grad_v_ref = grad_x2.split(n_embd, dim=2)

            self.assertEqual(grad_q_actual, grad_q_ref, atol=atol, rtol=rtol)
            self.assertEqual(grad_k_actual, grad_k_ref, atol=atol, rtol=rtol)
            self.assertEqual(grad_v_actual, grad_v_ref, atol=atol, rtol=rtol)

    @parametrize("kernel", [SDPBackend.MATH])
    def test_scaled_dot_product_attention_math_with_negative_scale(self, device, kernel: SDPBackend):
        # https://github.com/pytorch/pytorch/issues/105190.
        def ref(x):
            v1 = torch.matmul(x, x.transpose(-1, -2))
            v2 = v1 / -0.0001
            v3 = v2.softmax(dim=-1)
            v4 = torch.matmul(v3, x)
            return v4

        x = torch.randn(1, 3, 64, 64, device=device)
        ref_result = ref(x)
        with sdp_kernel(**backend_map[kernel]):
            sdp_math = torch.nn.functional.scaled_dot_product_attention(x, x, x, scale=-1.0 / 0.0001)
        self.assertEqual(ref_result, sdp_math)

class TestSDPACudaOnly(NNTestCase):
    """ Used to test CUDA only functionality of scaled_dot_product_attention
    Quarks:
        There is some trickiness with this function. It's runtime behavior
        is dependent on the CUDA architecture you are testing it on. See
        `PLATFORM_SUPPORTS_FUSED_SDPA` at the top of the file.
        Summary:
            Math: always supported
            FlashAttention: Supported on sm80 or newer hardware
            MemEfficientAttention: Supported on sm50 or newer hardware
    """
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def convert_flash_attn_S_to_softmax(self, S, query_padding_mask, key_padding_mask, head_dim, causal=False):
        """FlashAttention stores the S matrix in a different way.
        Arguments:
            S: (batch_size, nheads, seqlen_q, seqlen_k)
            query_padding_mask: (batch_size, seqlen_q)
            key_padding_mask: (batch_size, seqlen_k)
        """
        def _get_block_size(head_dim):
            assert head_dim % 8 == 0 and head_dim <= 128
            return 256 if head_dim <= 64 else 128
        S_flat = S.view(S.shape[0], S.shape[1], S.shape[2] * S.shape[3])
        seqlen_q, seqlen_k = S.shape[-2:]
        block_size = _get_block_size(head_dim)
        loop_steps = math.ceil(seqlen_k / block_size)
        warps_n = 4
        mmas_n = (seqlen_k // warps_n //
                  16) if seqlen_k <= block_size else (block_size // warps_n // 16)

        S_converted = S_flat.view(S_flat.shape[0], S_flat.shape[1], loop_steps,
                                  seqlen_q // 16, mmas_n, warps_n, 8, 4, 2, 2, 2)
        S_converted = S_converted.permute(0, 1, 3, 8, 6, 2, 4, 5, 9, 7, 10)
        S_converted = S_converted.reshape(S_flat.shape[0],
                                          S_flat.shape[1], (seqlen_q // 16 * 2 * 8), (loop_steps * mmas_n * warps_n * 2 * 4 * 2))
        # Need to zero out things not in attention_mask in case S was initialized with random values
        # and some of those values aren't overwritten.
        seqlen_q_og = query_padding_mask.shape[-1]
        if seqlen_q_og < seqlen_q:
            query_padding_mask = F.pad(
                query_padding_mask, (0, seqlen_q - seqlen_q_og))
        else:
            query_padding_mask = query_padding_mask[:, :seqlen_q]
        q_mask_fill = ~query_padding_mask.view(query_padding_mask.shape[0], 1, query_padding_mask.shape[1], 1)
        S_converted = S_converted.masked_fill(q_mask_fill, 0.0)
        seqlen_k_og = key_padding_mask.shape[-1]
        if seqlen_k_og < seqlen_k:
            key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k - seqlen_k_og))
        else:
            key_padding_mask = key_padding_mask[:, :seqlen_k]

        k_mask_fill = ~key_padding_mask.view(key_padding_mask.shape[0], 1, 1, key_padding_mask.shape[1])
        S_converted = S_converted.masked_fill(k_mask_fill, 0.0)

        if causal:
            causal_mask = torch.triu(torch.ones(
                seqlen_q, seqlen_k, dtype=torch.bool, device=S.device), 1)
            S_converted.masked_fill_(causal_mask, 0.0)
        if seqlen_q_og < seqlen_q:
            S_converted = S_converted[:, :, :seqlen_q_og, :]
        else:
            S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q))
        if seqlen_k_og < seqlen_k:
            S_converted = S_converted[:, :, :, :seqlen_k_og]
        else:
            S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k))
        return S_converted

    def query_key_value_clones(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype):
        """ Clones the query, key, and value tensors and moves them to the specified dtype. """
        query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
        key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
        value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
        return query_ref, key_ref, value_ref

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("mask_dim", [1, 2, 3, 4])
    def test_mem_efficient_attetntion_mask_variants(self, device, mask_dim: List[int]):
        dtype = torch.float16
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 8, 8, 64
        seq_len_q, seq_len_kv = 64, 32
        query = make_tensor((batch, num_heads, seq_len_q, head_dim))
        kv_shape = (batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)

        if mask_dim == 1:
            mask = torch.randn((seq_len_kv,), device=device, dtype=dtype)
        elif mask_dim == 2:
            mask = torch.randn((seq_len_q, seq_len_kv), device=device, dtype=dtype)
        elif mask_dim == 3:
            mask = torch.randn((num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        elif mask_dim == 4:
            mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("dtype", [torch.float, torch.float16])
    def test_mem_eff_attention_pad_mask(self, device, dtype):
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 8, 8, 64
        seq_len_q, seq_len_kv = 64, 15
        query = make_tensor((batch, num_heads, seq_len_q, head_dim))
        kv_shape = (batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("dtype", [torch.float, torch.float16])
    def test_mem_eff_attention_non_contiguous_mask(self, device, dtype):
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 8, 8, 64
        seq_len_q, seq_len_kv = 64, 16
        query = make_tensor((batch, num_heads, seq_len_q, head_dim))
        kv_shape = (batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        mask = torch.as_strided(mask, (batch, num_heads, seq_len_q, seq_len_kv), (0, 0, 0, 1))
        with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("is_contiguous", [True, False])
    @parametrize("head_dims_match", [True, False])
    def test_scaled_dot_product_attention_fused_kernels(self, device, type: str, is_contiguous: bool, head_dims_match: bool):
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16)

        batch, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = (batch, seq_len, num_heads, head_dim)
        if head_dims_match:
            shape_v = shape
        else:
            head_dim_v = 96
            shape_v = (batch, seq_len, num_heads, head_dim_v)

        query = make_tensor(shape)
        key = make_tensor(shape)
        value = make_tensor(shape_v)

        # Lets switch seq_len and num_heads
        # B x S X H X D -> B x H x S x D
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if is_contiguous:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual[0].contiguous(), math_ref[0].contiguous(), atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("is_contiguous", [True, False])
    def test_scaled_dot_product_attention_fused_kernels_packed(self, device, type: str, is_contiguous: bool):
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = (batch_size, seq_len, num_heads, head_dim)

        # Test Packed
        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if is_contiguous:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_scaled_dot_product_attention_fused_kernels_packed_accuracy(self, device, type: str, fused_kernel: str):
        if (not SM80OrLater) and fused_kernel == SDPBackend.FLASH_ATTENTION:
            return

        def rand_nt(shape):
            batch, seq_len, num_heads, head_dim = shape
            tensors = [6 * torch.rand((seq_len, 3 * num_heads * head_dim), device=device, dtype=torch.float32) - 3
                       for _ in range(batch)]
            return (torch.nested.nested_tensor(tensors, device=device, dtype=torch.float32),
                    torch.nested.nested_tensor(tensors, device=device, dtype=torch.float16))

        def rand_tensor(shape):
            batch, seq_len, num_heads, head_dim = shape
            tensor = 6 * torch.rand((batch, seq_len, 3 * num_heads * head_dim), device=device, dtype=torch.float32) - 3
            return tensor, tensor.to(dtype=torch.float16)

        batch_size, seq_len, num_heads, head_dim = 16, 8, 4, 64
        shape = (batch_size, seq_len, num_heads, head_dim)

        # Test Packed
        qkv, qkv_low_precision = rand_tensor(shape) if type == "dense" else rand_nt(shape)
        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_low_precision.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        with sdp_kernel(**backend_map[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, attn_mask=None, dropout_p=0.0, is_causal=False)

        with sdp_kernel(**backend_map[SDPBackend.MATH]):
            math_ref_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp.contiguous(), key_lp.contiguous(), value_lp.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

            math_query = query.contiguous()
            math_key = key.contiguous()
            math_value = value.contiguous()

            math_ref = torch.nn.functional.scaled_dot_product_attention(
                math_query, math_key, math_value, attn_mask=None, dropout_p=0.0, is_causal=False)

        actual_test = actual
        math_ref_test = math_ref
        math_ref_lp_test = math_ref_lp

        if actual_test.is_nested:
            actual_test = torch.nested.to_padded_tensor(actual_test.contiguous(), padding=0.0)
            math_ref_test = torch.nested.to_padded_tensor(math_ref_test, padding=0.0)
            math_ref_lp_test = torch.nested.to_padded_tensor(math_ref_lp_test, padding=0.0)

        actual_test = actual_test.to(dtype=torch.float32).contiguous()
        math_ref_test = math_ref_test.to(dtype=torch.float32).contiguous()
        math_ref_lp_test = math_ref_lp_test.to(dtype=torch.float32).contiguous()

        self.assertEqual(math_ref_test, math_ref_lp_test, atol=7e-3, rtol=7e-3)
        self.assertEqual(actual_test, math_ref_test, atol=5e-3, rtol=5e-3)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Flash Attention was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    @parametrize("is_causal", [True, False])
    def test_sdp_mem_efficient_grad_against_math(self, device, contiguous_inputs: bool, is_causal: bool):
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor((batch_size, seq_len, num_heads, head_dim))
        qkv_lp = qkv.detach().clone().to(torch.float32).requires_grad_()

        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_lp.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            query_lp = query_lp.contiguous()
            key_lp = key_lp.contiguous()
            value_lp = value_lp.contiguous()

        with sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, is_causal)

        with sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=False):
            out_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, None, 0.0, is_causal)

        rand_upward = torch.rand_like(out)
        rand_upward_lp = rand_upward.to(torch.float32)

        out.backward(rand_upward)
        out_lp.backward(rand_upward_lp)

        # Cast up and compare
        self.assertEqual(qkv.grad, qkv_lp.grad.to(torch.float64), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Flash Attention was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    @parametrize("is_causal", [True, False])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdp_flash_attention_grad_against_math(self, device, contiguous_inputs: bool, is_causal: bool, dtype: torch.dtype):
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor((batch_size, seq_len, num_heads, head_dim))
        qkv_lp = qkv.detach().clone().to(dtype).requires_grad_()

        query, key, value = qkv.chunk(3, dim=-1)
        query_lp, key_lp, value_lp = qkv_lp.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        query_lp = query_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key_lp = key_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value_lp = value_lp.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            query_lp = query_lp.contiguous()
            key_lp = key_lp.contiguous()
            value_lp = value_lp.contiguous()

        with sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, is_causal)

        with sdp_kernel(enable_math=False, enable_mem_efficient=False, enable_flash=True):
            out_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, None, 0.0, is_causal)

        rand_upward = torch.rand_like(out)
        rand_upward_lp = rand_upward.to(dtype)

        out.backward(rand_upward)
        out_lp.backward(rand_upward_lp)

        # Cast up and compare
        # Since we are doing the compute on fp16 we have to bump the tolerance
        # Bump down the tolearnce for blfoat16
        atol = 7e-4 if dtype == torch.float16 else 7e-3
        rtol = 7e-4 if dtype == torch.float16 else 7e-3
        self.assertEqual(qkv.grad, qkv_lp.grad.to(torch.float64), atol=atol, rtol=rtol)

    @parametrize("type", ["dense", "nested"])
    def test_fused_sdp_choice(self, device, type: str):
        if PLATFORM_SUPPORTS_FUSED_SDPA:
            batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
            shape = (batch_size, seq_len, num_heads, head_dim)
            make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float16, packed=True, requires_grad=True)

            qkv = make_tensor(shape, type=type)
            query, key, value = qkv.chunk(3, dim=-1)

            query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            if SM75OrLater and not type == "nested":
                assert torch._fused_sdp_choice(query, key, value) == SDPBackend.FLASH_ATTENTION
            else:
                assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION

            # Change dtype to float32 so that efficient attention should get chosen
            make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float32, packed=True)

            qkv = make_tensor(shape, type=type)
            query, key, value = qkv.chunk(3, dim=-1)

            query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Platform does not support fused SDPA")
    @parametrize("warn_only", [True, False])
    def test_sdp_choice_with_determinism(self, device, warn_only):
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = (batch_size, seq_len, num_heads, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float32, packed=False)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
                assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Platform does not support fused SDPA")
    @parametrize("warn_only", [True, False])
    def test_mem_eff_backwards_throws_determinism_warning(self, device, warn_only):
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = (batch_size, seq_len, num_heads, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float32, packed=False, requires_grad=True)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        warning_context = (
            self.assertWarnsRegex(
                UserWarning,
                "Memory Efficient attention defaults to a non-deterministic algorithm.",
            )
            if warn_only
            else contextlib.nullcontext()
        )
        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
                with warning_context:
                    torch.nn.functional.scaled_dot_product_attention(query, key, value).sum().backward()

    @unittest.skip("This test is not behaving deterministaclly non-deterministaclly on CI/CD")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Platform does not support fused SDPA")
    def test_mem_eff_backwards_determinism(self, device):
        # Need big seq_len to ensure that num_splits > 1
        dtype = torch.float32
        batch_size, seq_len, n_heads, head_dim = 1, 1024, 8, 64
        query = torch.rand(batch_size, n_heads, seq_len, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        with sdp_kernel(enable_mem_efficient=True, enable_math=False, enable_flash=False):
            # Run once to establish baseline
            out = F.scaled_dot_product_attention(query, key, value)
            upward_grad = torch.rand_like(out)
            out.backward(upward_grad)
            intial_query_grad = query.grad

            # Re-run the op with the same upward grad and check that the backward is
            # not deterministic
            diff_anwser_once = False
            for _ in range(100):
                query.grad = None
                out = F.scaled_dot_product_attention(query, key, value)
                out.backward(upward_grad)
                if not torch.equal(intial_query_grad, query.grad):
                    diff_anwser_once = True
                    break
            self.assertTrue(diff_anwser_once)

        with use_deterministic_algorithims(True, warn_only=False):
            query.grad = None
            out = F.scaled_dot_product_attention(query, key, value)
            upward_grad = torch.rand_like(out)
            out.backward(upward_grad)
            intial_query_grad = query.grad

            # Re-run the op with the same upward grad and check that the backward is
            # deterministic now that we have enforced it
            diff_anwser_once = False
            for _ in range(100):
                query.grad = None
                out = F.scaled_dot_product_attention(query, key, value)
                out.backward(upward_grad)
                if not torch.equal(intial_query_grad, query.grad):
                    diff_anwser_once = True
                    break
            self.assertFalse(diff_anwser_once)

    # verified passing successfully on H100
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support SDPA")
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [4, 8, 64, 128, 256, 512, 1024, 2048] if SM80OrLater else [4, 8, 64, 128, 256, 512])
    @parametrize("seq_len_k", [4, 8, 64, 128, 256, 512, 1024, 2048] if SM80OrLater else [4, 8, 64, 128, 256, 512])
    @parametrize("head_dim", [8, 16, 32, 64, 72, 96, 128] if SM80OrLater else [8, 16, 32, 64])
    @parametrize("is_causal", [False, True])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32] if
                 SM80OrLater else [torch.float16, torch.float32])
    @parametrize("scale", [None, "l1"])
    def test_mem_efficient_attention_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int, seq_len_k: int,
                                                       head_dim: int, is_causal: bool, dropout_p: float, dtype: torch.dtype,
                                                       scale: str):
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, p, seed, offset, device=device):
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, p, seed, offset)
            mask = (rand_uniform > p).to(torch.float32)
            return mask
        if max(seq_len_q, seq_len_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory < 40 * 2**30:
            unittest.skip("Reference implementation OOM")
            return
        seed = 42
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        # Run the math kernel on low precision references
        query_ref_lp, key_ref_lp, value_ref_lp = self.query_key_value_clones(query, key, value, dtype=dtype)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = self.query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        # Create real output
        with sdp_kernel(enable_mem_efficient=True, enable_flash=False, enable_math=False):
            # Set the seed and run the kernel
            torch.manual_seed(seed)
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        if dropout_p == 0.0:
            with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref,
                                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query_ref_lp, key_ref_lp, value_ref_lp,
                                                            dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        else:
            if seq_len_q > 1024:
                self.skipTest("Will call _fill_mem_eff_dropout_mask with too many threads!")
            # Create the dropout_mask
            torch.manual_seed(seed)
            dropout_mask = _get_mem_eff_drop_mask(batch_size, n_heads, seq_len_q, seq_len_k, dropout_p, seed, 0, device=device)
            # High Precision Math Reference
            out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal, scale=scale, dropout_mask=dropout_mask)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref_lp, key_ref_lp, value_ref_lp, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=dropout_mask)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        out.backward(upstream_grad)
        out_ref.backward(upstream_grad.to(out_ref.dtype))
        out_lp_ref.backward(upstream_grad.to(out_lp_ref.dtype))

        # [Note] Fused Tolerances
        # Establish the numerical error between the "true" high precision math output
        # and the low precision math reference. We use this reference for the atol
        # And we use the default rtol for the low precision type.
        # We then provide a fudge factor for gradients respectively to account
        # for the use of the fused kernel rather than the eager implemntation.
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)

        # Fudge Factor when dropout is enabled
        dropout_fudge_factor = 1.0 if dropout_p == 0.0 else 1.5

        query_fudge_factor = dropout_fudge_factor
        grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(query_ref.grad, query_ref_lp.grad, query_fudge_factor)

        # TODO: Investigate why grad_k needs larger tolerances
        key_fudge_factor = 8 * dropout_fudge_factor
        grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(key_ref.grad, key_ref_lp.grad, key_fudge_factor)

        value_fudge_factor = 7 if not SM80OrLater and dtype == torch.float16 else 1.0
        grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(value_ref.grad, value_ref_lp.grad, value_fudge_factor)

        self.assertEqual(out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol)
        self.assertEqual(query.grad, query_ref.grad.to(query.grad.dtype),
                         atol=grad_q_ref_atol, rtol=grad_q_ref_rtol)
        self.assertEqual(key.grad, key_ref.grad.to(key.grad.dtype),
                         atol=grad_k_ref_atol, rtol=grad_k_ref_rtol)
        self.assertEqual(value.grad, value_ref.grad.to(value.grad.dtype),
                         atol=grad_v_ref_atol, rtol=grad_v_ref_rtol)


    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support SDPA")
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [4, 8, 64, 128, 256, 312, 512, 1024, 2048] if SM80OrLater else [4, 8, 64, 128, 152, 256, 512])
    @parametrize("seq_len_k", [4, 8, 64, 65, 128, 256, 408, 512, 1024, 2048] if SM80OrLater else [4, 8, 37, 64, 128, 256, 512])
    @parametrize("head_dim", [8, 16, 32, 64, 72, 96, 128] if SM80OrLater else [8, 16, 32, 64])
    @parametrize("is_causal", [False])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32] if
                 SM80OrLater else [torch.float16, torch.float32])
    @parametrize("scale", [None, "l1"])
    def test_mem_efficient_attention_attn_mask_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int,
                                                                 seq_len_k: int, head_dim: int, is_causal: bool,
                                                                 dropout_p: float, dtype: torch.dtype,
                                                                 scale: str):
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, p, seed, offset, device=device):
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, p, seed, offset)
            mask = (rand_uniform > p).to(torch.float32)
            return mask
        if max(seq_len_q, seq_len_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory < 40 * 2**30:
            unittest.skip("Reference implementation OOM")
            return
        seed = 42
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        attn_mask = torch.rand(seq_len_q, seq_len_k, device=device, dtype=dtype, requires_grad=True)

        # Run the math kernel on low precision references
        query_ref_lp, key_ref_lp, value_ref_lp = self.query_key_value_clones(query, key, value, dtype=dtype)
        attn_mask_ref_lp = attn_mask.detach().to(dtype).requires_grad_(True)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = self.query_key_value_clones(query, key, value, dtype=higher_precision_dtype)
        attn_mask_ref = attn_mask.detach().to(higher_precision_dtype).requires_grad_(True)

        # Create real output
        with sdp_kernel(enable_mem_efficient=True, enable_flash=False, enable_math=False):
            # Set the seed and run the kernel
            torch.manual_seed(seed)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p=dropout_p,
                                                 is_causal=is_causal, scale=scale)

        if dropout_p == 0.0:
            with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref, attn_mask_ref,
                                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query_ref_lp, key_ref_lp, value_ref_lp, attn_mask_ref_lp,
                                                            dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        else:
            if seq_len_q > 1024:
                self.skipTest("Will call _fill_mem_eff_dropout_mask with too many threads!")
            # Create the dropout_mask
            torch.manual_seed(seed)
            dropout_mask = _get_mem_eff_drop_mask(batch_size, n_heads, seq_len_q,
                                                  seq_len_k, dropout_p, seed, 0, device=device)
            # High Precision Math Reference
            out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref, key_ref, value_ref, attn_mask_ref, dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, dropout_mask=dropout_mask)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref_lp, key_ref_lp, value_ref_lp, attn_mask_ref_lp,
                dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=dropout_mask)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        out.backward(upstream_grad)
        out_ref.backward(upstream_grad.to(out_ref.dtype))
        out_lp_ref.backward(upstream_grad.to(out_lp_ref.dtype))

        # [Note] Fused Tolerances
        # Establish the numerical error between the "true" high precision math output
        # and the low precision math reference. We use this reference for the atol
        # And we use the default rtol for the low precision type.
        # We then provide a fudge factor for gradients respectively to account
        # for the use of the fused kernel rather than the eager implemntation.
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)

        # Fudge Factor when dropout is enabled
        dropout_fudge_factor = 1.0 if dropout_p == 0.0 else 1.5
        mask_fudge_factor = 1.0 if attn_mask is None else 1.5

        query_fudge_factor = dropout_fudge_factor
        grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(query_ref.grad, query_ref_lp.grad, query_fudge_factor)

        # TODO: Investigate why grad_k needs larger tolerances
        key_fudge_factor = 8 * dropout_fudge_factor * mask_fudge_factor
        grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(key_ref.grad, key_ref_lp.grad, key_fudge_factor)

        value_fudge_factor = 7 if not SM80OrLater and dtype == torch.float16 else 1.0
        grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(value_ref.grad, value_ref_lp.grad, value_fudge_factor)

        mask_fudge_factor = 12 if attn_mask.numel() > 512 else 22
        grad_attn_mask_atol, grad_attn_mask_rtol = get_tolerances(
            attn_mask_ref.grad, attn_mask_ref_lp.grad, mask_fudge_factor)

        self.assertEqual(out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol)
        self.assertEqual(query.grad, query_ref.grad.to(query.grad.dtype),
                         atol=grad_q_ref_atol, rtol=grad_q_ref_rtol)
        self.assertEqual(key.grad, key_ref.grad.to(key.grad.dtype),
                         atol=grad_k_ref_atol, rtol=grad_k_ref_rtol)
        self.assertEqual(value.grad, value_ref.grad.to(value.grad.dtype),
                         atol=grad_v_ref_atol, rtol=grad_v_ref_rtol)

        self.assertEqual(attn_mask.grad, attn_mask_ref.grad.to(attn_mask.grad.dtype),
                         atol=grad_attn_mask_atol, rtol=grad_attn_mask_rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support SDPA or pre-SM80 hardware")
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [4, 8, 64, 128, 256, 512, 1024, 2048])
    @parametrize("seq_len_k", [4, 8, 64, 128, 256, 512, 1024, 2048])
    @parametrize("head_dim", [8, 16, 32, 64, 72, 96, 128])
    @parametrize("is_causal", [True, False])
    @parametrize("dropout_p", [0.0, 0.22, 0.48])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("scale", [None, "l1"])
    def test_flash_attention_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int, seq_len_k: int,
                                               head_dim: int, is_causal: bool, dropout_p: float, dtype: torch.dtype,
                                               scale: str):

        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        # Run the math kernel on low precision references
        query_ref_lp, key_ref_lp, value_ref_lp = self.query_key_value_clones(query, key, value, dtype=dtype)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = self.query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        is_dropout = dropout_p > 0.0

        # Create real output
        output_tuple = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale, return_debug_mask=True)
        out = output_tuple[0]
        dbug_mask = output_tuple[-1]

        query_padding_mask = torch.ones(
            1, seq_len_q, device=device, dtype=torch.bool)
        key_padding_mask = torch.ones(
            1, seq_len_k, device=device, dtype=torch.bool)

        softmax_mask = self.convert_flash_attn_S_to_softmax(
            dbug_mask, query_padding_mask, key_padding_mask, head_dim=head_dim, causal=is_causal)
        dropout_mask = softmax_mask >= 0

        if not is_dropout:
            with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(
                    query_ref, key_ref, value_ref, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(
                    query_ref_lp, key_ref_lp, value_ref_lp, is_causal=is_causal, scale=scale)
        else:
            # High Precision Math Reference
            out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal, scale=scale, dropout_mask=dropout_mask)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref_lp, key_ref_lp, value_ref_lp, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=dropout_mask)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        # backward for flash attention on sm86 and sm89 for headdim > 64 currently disabled
        if isSM86or89Device and head_dim in range(65, 129):
            self.assertRaises(RuntimeError, lambda: out.backward(upstream_grad))
            return
        out.backward(upstream_grad)
        out_ref.backward(upstream_grad.to(out_ref.dtype))
        out_lp_ref.backward(upstream_grad.to(out_lp_ref.dtype))

        # See [Note] Fused Tolerances above
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)

        # TODO: Investigate why grad_q needs larger tolerances
        query_fudge_factor = 4
        grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(query_ref.grad, query_ref_lp.grad, query_fudge_factor)

        grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(key_ref.grad, key_ref_lp.grad)

        grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(value_ref.grad, value_ref_lp.grad)

        self.assertEqual(out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol)
        self.assertEqual(query.grad, query_ref.grad.to(query.grad.dtype),
                         atol=grad_q_ref_atol, rtol=grad_q_ref_rtol)
        self.assertEqual(key.grad, key_ref.grad.to(key.grad.dtype),
                         atol=grad_k_ref_atol, rtol=grad_k_ref_rtol)
        self.assertEqual(value.grad, value_ref.grad.to(value.grad.dtype),
                         atol=grad_v_ref_atol, rtol=grad_v_ref_rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support SDPA or pre-SM80 hardware")
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [256, 512, 1024])
    @parametrize("seq_len_k", [256, 512, 1024])
    @parametrize("head_dim", [32, 64])
    @parametrize("is_causal", [True, False])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize("dtype", [torch.float16,])
    @parametrize("scale", [None, "l1"])
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_attention_vs_math_ref_grads_cudagraph(self, device, batch_size: int, seq_len_q: int, seq_len_k: int,
                                                         head_dim: int,
                                                         is_causal: bool,
                                                         dropout_p: float,
                                                         dtype: torch.dtype,
                                                         scale: str,
                                                         fused_kernel: SDPBackend):
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, dropout_p, seed, offset, device=device):
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, dropout_p, seed, offset)
            mask = (rand_uniform > dropout_p).to(torch.float32)
            return mask

        def get_dropout_mask(output, fused_kernel, batch_size, n_heads, q_len, kv_len, dropout_p, device=device):
            if fused_kernel == SDPBackend.EFFICIENT_ATTENTION:
                output_seed, output_offset = output_tuple[2], output_tuple[3]
                output_seed = output_seed.item()
                output_offset = output_offset.item()
                return _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len,
                                              dropout_p, output_seed, output_offset, device=device)
            else:
                dbug_mask = output[-1]
                query_padding_mask = torch.ones(
                    1, seq_len_q, device="cuda", dtype=torch.bool)
                key_padding_mask = torch.ones(
                    1, seq_len_k, device="cuda", dtype=torch.bool)

                softmax_mask = self.convert_flash_attn_S_to_softmax(
                    dbug_mask, query_padding_mask, key_padding_mask, head_dim=head_dim, causal=is_causal)
                dropout_mask = softmax_mask >= 0
                return dropout_mask

        seed = 42
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        fused_op = (torch.ops.aten._scaled_dot_product_efficient_attention
                    if fused_kernel == SDPBackend.EFFICIENT_ATTENTION else torch.ops.aten._scaled_dot_product_flash_attention)
        # Run the math kernel on low precision references
        query_ref_lp, key_ref_lp, value_ref_lp = self.query_key_value_clones(query, key, value, dtype=dtype)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = self.query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        # Set the global seed before capture
        torch.manual_seed(seed)
        kwargs = {"dropout_p": dropout_p, "is_causal": is_causal, "scale": scale}
        if fused_kernel == SDPBackend.EFFICIENT_ATTENTION:
            kwargs["compute_log_sumexp"] = True
            kwargs["attn_bias"] = None
        if fused_kernel == SDPBackend.FLASH_ATTENTION:
            kwargs['return_debug_mask'] = True
        with torch.cuda.stream(s):
            # Create real output
            output_tuple = fused_op(query, key, value, **kwargs)

        torch.cuda.current_stream().wait_stream(s)
        out = output_tuple[0]
        upstream_grad = torch.rand_like(out, requires_grad=False)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            out.backward(upstream_grad)
        for x in (query, key, value):
            x.grad = None
        g = torch.cuda.CUDAGraph()
        # Create real output
        with torch.cuda.graph(g):
            tmp = torch.rand_like(query, device=query.device)  # test non-zero intragraph offset
            # Create real output
            output_tuple = fused_op(query, key, value, **kwargs)
            assert all(not isinstance(o, torch.Tensor) or o.is_cuda for o in output_tuple)
        g.replay()
        out_first = output_tuple[0].clone()
        g.replay()
        out = output_tuple[0]
        if dropout_p == 0.0:
            self.assertEqual(out_first, out, atol=0, rtol=0)
        else:
            # replays produce different results
            self.assertNotEqual(out_first, out)

        with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            if dropout_p == 0.0:
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref,
                                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query_ref_lp, key_ref_lp, value_ref_lp,
                                                            dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            else:
                # Create the dropout_mask
                dropout_mask = get_dropout_mask(output_tuple, fused_kernel, batch_size,
                                                n_heads, seq_len_q, seq_len_k, dropout_p, device)
                # High Precision Math Reference
                out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal,
                    scale=scale, dropout_mask=dropout_mask)[0]
                # Low Precision Math Reference
                out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query_ref_lp, key_ref_lp, value_ref_lp, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                    dropout_mask=dropout_mask)[0]


        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            out.backward(upstream_grad)
        g1.replay()
        out_ref.backward(upstream_grad.to(out_ref.dtype))
        out_lp_ref.backward(upstream_grad.to(out_lp_ref.dtype))

        # [Note] Fused Tolerances
        # Establish the numerical error between the "true" high precision math output
        # and the low precision math reference. We use this reference for the atol
        # And we use the default rtol for the low precision type.
        # We then provide a fudge factor for gradients respectively to account
        # for the use of the fused kernel rather than the eager implemntation.
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)

        # Fudge Factor when dropout is enabled
        dropout_fudge_factor = 1.0 if dropout_p == 0.0 else 1.5

        query_fudge_factor = dropout_fudge_factor
        grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(query_ref.grad, query_ref_lp.grad, query_fudge_factor)

        # TODO: Investigate why grad_k needs larger tolerances
        key_fudge_factor = 8 * dropout_fudge_factor
        grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(key_ref.grad, key_ref_lp.grad, key_fudge_factor)

        value_fudge_factor = 7 if not SM80OrLater and dtype == torch.float16 else 1.0
        grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(value_ref.grad, value_ref_lp.grad, value_fudge_factor)

        self.assertEqual(out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol)
        self.assertEqual(query.grad, query_ref.grad.to(query.grad.dtype),
                         atol=grad_q_ref_atol, rtol=grad_q_ref_rtol)
        self.assertEqual(key.grad, key_ref.grad.to(key.grad.dtype),
                         atol=grad_k_ref_atol, rtol=grad_k_ref_rtol)
        self.assertEqual(value.grad, value_ref.grad.to(value.grad.dtype),
                         atol=grad_v_ref_atol, rtol=grad_v_ref_rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_kernels_seq_len_1_inputs(self, device, fused_kernel):
        if (not SM80OrLater) and fused_kernel == SDPBackend.FLASH_ATTENTION:
            return
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        batch, num_heads, head_dim = 32, 16, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        # make sure some seq_lens are 1
        num_ones = 10
        indices = torch.randint(low=0, high=batch, size=(num_ones,))
        seq_lens.scatter_(0, indices, 1)

        shape = (batch, seq_lens.tolist(), num_heads, head_dim)
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdp_kernel(**backend_map[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(torch.float16), atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_kernels_seq_len_0_inputs(self, device, fused_kernel):
        if (not SM80OrLater) and fused_kernel == SDPBackend.FLASH_ATTENTION:
            return
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        batch, num_heads, head_dim = 32, 16, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        # make sure some seq_lens are 0
        num_zeros = 10
        indices = torch.randint(low=0, high=batch, size=(num_zeros,))
        seq_lens.scatter_(0, indices, 0)

        shape = (batch, seq_lens.tolist(), num_heads, head_dim)
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdp_kernel(**backend_map[fused_kernel]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    @parametrize("expand_q_batch", [True, False])
    @parametrize("expand_k_batch", [True, False])
    @parametrize("expand_v_batch", [True, False])
    @parametrize("expand_q_num_heads", [True, False])
    @parametrize("expand_k_num_heads", [True, False])
    @parametrize("expand_v_num_heads", [True, False])
    def test_fused_kernels_nested_broadcasting(
        self,
        device,
        kernel,
        expand_q_batch,
        expand_k_batch,
        expand_v_batch,
        expand_q_num_heads,
        expand_k_num_heads,
        expand_v_num_heads,
    ):
        if (not SM80OrLater) and kernel == SDPBackend.FLASH_ATTENTION:
            return
        is_efficient = kernel == SDPBackend.EFFICIENT_ATTENTION
        dtype = torch.float32 if is_efficient else torch.float16
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=dtype)
        batch, num_heads, head_dim = 32, 8, 64
        head_dim_v = 32 if is_efficient else head_dim
        seq_lens_q = (torch.randint(low=1, high=5, size=(1,)).item()
                      if expand_q_batch
                      else torch.randint(low=1, high=32, size=(batch,)).tolist())
        seq_lens_kv = (torch.randint(low=1, high=5, size=(1,)).item()
                       if (expand_k_batch or expand_v_batch)
                       else torch.randint(low=1, high=32, size=(batch,)).tolist())

        batch_q = 1 if expand_q_batch else batch
        batch_k = 1 if expand_k_batch else batch
        batch_v = 1 if expand_v_batch else batch

        # handle case where all batch_sizes are 1
        batch = max(batch_q, batch_k, batch_v)

        num_heads_q = 1 if expand_q_num_heads else num_heads
        num_heads_k = 1 if expand_k_num_heads else num_heads
        num_heads_v = 1 if expand_v_num_heads else num_heads

        # handle case where all num_heads are 1
        num_heads = max(num_heads_q, num_heads_k, num_heads_v)

        q_shape = (batch_q, seq_lens_q, num_heads_q, head_dim)
        k_shape = (batch_k, seq_lens_kv, num_heads_k, head_dim)
        v_shape = (batch_v, seq_lens_kv, num_heads_v, head_dim_v)

        query = rand_nested_tensor(q_shape)
        key = rand_nested_tensor(k_shape)
        value = rand_nested_tensor(v_shape)

        def _broadcast(t, batch_broadcasted, num_heads_broadcasted):
            if batch_broadcasted and num_heads_broadcasted:
                # (1, seq_len, 1, head_dim) -> (batch, seq_len, num_heads, head_dim)
                result = torch.nested.nested_tensor(
                    [t[0].expand(-1, num_heads, t.size(-1)) for _ in range(batch)], dtype=torch.float32)
            elif batch_broadcasted:
                # (1, seq_len, num_heads, head_dim) -> (batch, seq_len, num_heads, head_dim)
                result = torch.nested.nested_tensor([t[0] for _ in range(batch)], dtype=torch.float32)
            elif num_heads_broadcasted:
                # (batch, seq_len, 1, head_dim) -> (batch, seq_len, num_heads, head_dim)
                result = torch.nested.nested_tensor([x.expand(-1, num_heads, t.size(-1))
                                                    for x in t.unbind()], dtype=torch.float32)
            else:
                result = t.to(torch.float32)
            return result

        query_expanded = _broadcast(query, expand_q_batch, expand_q_num_heads).transpose(1, 2)
        key_expanded = _broadcast(key, expand_k_batch, expand_k_num_heads).transpose(1, 2)
        value_expanded = _broadcast(value, expand_v_batch, expand_v_num_heads).transpose(1, 2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdp_kernel(**backend_map[kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query_expanded.contiguous(), key_expanded.contiguous(), value_expanded.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_query_dense(self, device):
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        batch, num_heads, head_dim, head_dim_v = 32, 16, 64, 96
        seq_lens = torch.randint(low=1, high=32, size=(batch,)).tolist()
        q_shape = (1, 1, num_heads, head_dim)
        k_shape = (batch, seq_lens, num_heads, head_dim)
        v_shape = (batch, seq_lens, 1, head_dim_v)

        # create a dense query
        query = torch.randn(q_shape, device=device, dtype=torch.float32)
        key = rand_nested_tensor(k_shape)
        value = rand_nested_tensor(v_shape)

        # (1, 1, num_heads, head_dim) -> (batch, 1, num_heads, head_dim)
        query_expanded = torch.nested.nested_tensor([query.squeeze(0) for _ in range(batch)]).transpose(1, 2)
        # (batch, seq_lens, 1, head_dim) -> (batch, seq_lens, num_heads, head_dim)
        value_expanded = torch.nested.nested_tensor(
            [t.expand(-1, num_heads, head_dim_v) for t in value.unbind()]).transpose(1, 2)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query_expanded.contiguous(), key.contiguous(), value_expanded.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=1e-3, rtol=1e-2)


device_types = ("cpu", "cuda")
instantiate_device_type_tests(TestTransformers, globals(), only_for=device_types)
instantiate_device_type_tests(TestSDPAFailureModes, globals(), only_for=device_types)
instantiate_device_type_tests(TestSDPA, globals(), only_for=device_types)
instantiate_device_type_tests(TestSDPACudaOnly, globals(), only_for=("cuda"))


if __name__ == '__main__':
    run_tests()
