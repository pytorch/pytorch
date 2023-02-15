# Owner(s): ["module: nn"]

import contextlib
from functools import partial
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

from typing import Tuple
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    TEST_FAIRSEQ,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
    freeze_rng_state,
    TEST_WITH_CROSSREF,
    slowTest,
    set_default_dtype,
    gradcheck
)


from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from torch.testing._internal.common_cuda import TEST_CUDA, SM80OrLater, PLATFORM_SUPPORTS_FUSED_SDPA

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
        yield{}
    except RuntimeError as err:
        raise err
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

isSM86Device = torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 6)


def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
    deviation = true_value - computed_value
    deviation = torch.abs(deviation / true_value)
    # Fill in the nans with the default rtol
    torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
    return deviation.max().item()

class TestTransformers(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    device_list = ['cpu']  # TODO: is there a way to do parametrize for this?
    if TEST_CUDA:
        device_list.append('cuda')

    @unittest.skip("4D mask not supported yet - activate when 4D mask supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")  # TODO: make this work for both cuda and cpu
    def test_self_attn_TxT_attn_mask(self):
        embed_dim = 16
        num_heads = 4
        batch_size = 10
        tgt_len = 16

        query = torch.rand(batch_size, tgt_len, embed_dim, device="cuda")  # [N, T, D]
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

    @parametrize("device", device_list)
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

    @parametrize("device", device_list)
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

    @parametrize("device", device_list)
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
    @parametrize("device", device_list)
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
    @parametrize("device", device_list)
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

            model = nn.TransformerEncoder(encoder_layer, 1).to(device)
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
            # If mask is not left aligned
            # We disable nested tensor
            model.enable_nested_tensor = enable_nested_tensor
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

    @unittest.skipIf(not TEST_FAIRSEQ, "Fairseq not found")
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
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

        class BetterDecoder(torch.nn.Module):
            """
            Only incremental decoder for now
            """

            def __init__(self, transformer, embedding, pad_idx):
                super().__init__()
                self.transformer = transformer
                self.embedding = embedding
                self.padding_idx = pad_idx

            def forward(
                self,
                x,
                src_mask=None,
                include_padding_mask=True,
                incr_key_lst=None,
                incr_value_lst=None,
                is_incremental_decoding=False,
            ):
                padding_mask = None
                if not x.is_nested and include_padding_mask:
                    padding_mask = x.eq(self.padding_idx)
                if(is_incremental_decoding):
                    x = x[:, -1:]  # only take the last token
                x = self.embedding(x)

                one_encoder_layer = self.transformer.layers[0]
                self_attn = one_encoder_layer.self_attn
                embed_dim = self_attn.embed_dim
                num_heads = self_attn.num_heads

                use_gelu = (
                    one_encoder_layer.activation_relu_or_gelu == 2
                )  # see torch/nn/modules/activation attention impl. 1 == relu, 2 == gelu
                assert (
                    one_encoder_layer.activation_relu_or_gelu != 0
                )  # 0 == not relu or gelu

                norm_first = one_encoder_layer.norm_first

                # TODO: make this a bit less janky. but for now we initialize with an empty tensor.
                if(not is_incremental_decoding):
                    assert len(incr_key_lst) == 0 or incr_key_lst[0] is None
                    assert len(incr_value_lst) == 0 or incr_value_lst[0] is None
                while len(incr_key_lst) <= len(self.transformer.layers):
                    if(is_incremental_decoding):
                        incr_key_lst.append(torch.Tensor([]).cuda().half())
                        incr_value_lst.append(torch.Tensor([]).cuda().half())
                    else:
                        incr_key_lst.append(None)
                        incr_value_lst.append(None)

                for i, layer in enumerate(self.transformer.layers):
                    incr_key = incr_key_lst[i]
                    incr_value = incr_value_lst[i]

                    x, incr_key, incr_value = torch._transformer_decoder_only_layer_fwd(
                        src=x,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv_weight=layer.self_attn.in_proj_weight,
                        qkv_bias=layer.self_attn.in_proj_bias,
                        proj_weight=layer.self_attn.out_proj.weight,
                        proj_bias=layer.self_attn.out_proj.bias,
                        use_gelu=use_gelu,
                        norm_first=norm_first,
                        # TODO: layer_norm_eps hardcoded to be same as nn.TransformerEncoder default.
                        # fix by pulling from self_attn.norm1
                        eps=1e-5,
                        norm_weight_1=layer.norm1.weight,
                        norm_bias_1=layer.norm1.bias,
                        norm_weight_2=layer.norm2.weight,
                        norm_bias_2=layer.norm2.bias,
                        ffn_weight_1=layer.linear1.weight,
                        ffn_bias_1=layer.linear1.bias,
                        ffn_weight_2=layer.linear2.weight,
                        ffn_bias_2=layer.linear2.bias,
                        mask=src_mask,
                        incr_key=incr_key,  # altered in place
                        incr_value=incr_value,
                    )

                    # not in-place
                    if(not is_incremental_decoding):
                        incr_key = None
                        incr_value = None
                    incr_key_lst[i] = incr_key
                    incr_value_lst[i] = incr_value

                return x, incr_key_lst, incr_value_lst

        def torch_to_fairseq(torch_encoder, fairseq_encoder):
            for src_layer, dst_layer in zip(torch_encoder.layers, fairseq_encoder.layers):
                w_q, w_k, w_v = src_layer.self_attn.in_proj_weight.chunk(3, dim=0)
                b_q, b_k, b_v = src_layer.self_attn.in_proj_bias.chunk(3, dim=0)

                dst_layer.self_attn.q_proj.weight = torch.nn.Parameter(w_q)
                dst_layer.self_attn.q_proj.bias = torch.nn.Parameter(b_q)
                dst_layer.self_attn.k_proj.weight = torch.nn.Parameter(w_k)
                dst_layer.self_attn.k_proj.bias = torch.nn.Parameter(b_k)
                dst_layer.self_attn.v_proj.weight = torch.nn.Parameter(w_v)
                dst_layer.self_attn.v_proj.bias = torch.nn.Parameter(b_v)

                dst_layer.self_attn.out_proj.weight = src_layer.self_attn.out_proj.weight
                dst_layer.self_attn.out_proj.bias = src_layer.self_attn.out_proj.bias

                dst_layer.fc1.weight = src_layer.linear1.weight
                dst_layer.fc1.bias = src_layer.linear1.bias

                # fairseq may use fusedlayernorm from nvidia apex - diff properties
                dst_layer.self_attn_layer_norm.load_state_dict(src_layer.norm1.state_dict())

                dst_layer.fc2.weight = src_layer.linear2.weight
                dst_layer.fc2.bias = src_layer.linear2.bias

                dst_layer.final_layer_norm.load_state_dict(src_layer.norm2.state_dict())

            return fairseq_encoder

        def set_weights_deterministic(model):
            for idx, p in enumerate(model.parameters()):
                x = p.data
                sz = x.view(-1).size(0)
                shape = x.shape
                x = torch.cos(torch.arange(0, sz).float().view(shape))
                p.data.copy_(x)

        D = 4  # d_model
        H = 2  # nhead
        FD = 16  # dim_feedforward
        V = 100  # vocab size
        L = 2  # num layers

        embedding_layer = torch.nn.Embedding(V, D, DEFAULT_PADDING_IDX)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=D,
            nhead=H,
            dim_feedforward=FD,
            batch_first=True,
            activation="gelu",
        )
        transformer = torch.nn.TransformerEncoder(
            layer,
            num_layers=L,
        ).eval().cuda().half()

        set_weights_deterministic(embedding_layer)
        set_weights_deterministic(transformer)

        better_decoder = (
            BetterDecoder(transformer, embedding_layer, DEFAULT_PADDING_IDX)
            .eval()
            .cuda()
            .half()
        )
        fairseq_decoder = (
            FairseqDecoder(
                D,
                H,
                FD,
                L,
                embedding_layer,
                dropout=0,
                normalize_before=False,
                torch_encoder=transformer,
                activation="gelu",
            )
            .eval()
            .cuda()
            .half()
        )

        tokens = torch.Tensor([
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ]).to(torch.int).cuda()
        lengths_tensor = torch.Tensor([2, 2]).to(torch.int).cuda()
        # bs = 2, seqlen = 4
        bs, seqlen = tokens.shape

        upper_triangle = torch.zeros(seqlen, seqlen)
        upper_triangle.fill_(-100000000)
        upper_triangle = torch.triu(upper_triangle, 1)
        upper_triangle = upper_triangle.cuda().half()
        upper_triangle_expanded = upper_triangle.unsqueeze(0).unsqueeze(0)
        upper_triangle_expanded = upper_triangle_expanded.expand(
            bs, H, -1, -1
        )

        # test forced decoding
        with torch.no_grad():
            result, _, _ = better_decoder(
                tokens,
                src_mask=upper_triangle_expanded,
                include_padding_mask=False,
                incr_key_lst=[],
                incr_value_lst=[],
                is_incremental_decoding=False,
            )
        ref_output = fairseq_decoder(tokens, lengths_tensor, with_triangle_mask=True)

        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=1e-3, rtol=1e-2)

        # test incremental decoding
        bs, seqlen = tokens.shape

        incr_state = {}
        ref_outputs = [fairseq_decoder(
            tokens[:, :i],
            src_lengths=None,
            with_triangle_mask=False,
            incremental_state=incr_state,
        ) for i in range(1, seqlen + 1)]
        ref_output = torch.stack(ref_outputs)

        incr_key_lst = []
        incr_value_lst = []
        results = []
        for i in range(1, seqlen + 1):
            res, incr_key_lst, incr_value_lst = better_decoder(
                tokens[:, :i],
                src_mask=None,
                include_padding_mask=False,
                incr_key_lst=incr_key_lst,
                incr_value_lst=incr_value_lst,
                is_incremental_decoding=True,
            )
            results.append(res)
        result = torch.stack(results)

        self.assertEqual(result.shape, ref_output.shape)
        torch.testing.assert_close(result, ref_output, atol=1e-3, rtol=1e-2)

    @parametrize("input_dim,attn_mask_dim,is_causal",
                 [(3, None, False), (3, 2, False), (3, 2, True), (3, 3, False), (3, 3, True),
                  (4, None, False), (4, 2, False), (4, 2, True), (4, 4, False), (4, 4, True)],
                 name_fn=lambda input_dim, attn_dim, is_causal: (
                     f"{input_dim}D_input_dim_" + (
                         f"{attn_dim}D_{'causal_' if is_causal else ''}attn_mask"
                         if attn_dim is not None else "no_attn_mask")))
    @parametrize("dropout_p", [0.0, 0.2, 0.5])
    @parametrize("device", device_list)
    @sdp_kernel(enable_flash=False)
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

    @parametrize("device", device_list)
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

        outputs = encoder(inputs, is_causal=True)

        loss = criterion(outputs[:, 0:2, :], inputs[:, 0:2, :])
        loss.backward()
        optimizer.step()

        # inference with is_causal
        t_qvk = torch.randn((S, L, E), device=device, dtype=torch.float32)
        mha = nn.MultiheadAttention(E, H).to(device)
        attn_out, _ = mha(t_qvk, t_qvk, t_qvk, is_causal=True)

        # Can't give both attn_mask AND is_causal
        attn_mask = torch.randint(0, 2, size=(L, L), device=device, dtype=torch.bool)
        with self.assertRaisesRegex(AssertionError, "Only allow causal mask or attn_mask"):
            _ = mha(t_qvk, t_qvk, t_qvk, attn_mask=attn_mask, is_causal=True)

        # # Passing a causal mask sets is_causal to 1
        causal_mask = torch.triu(
            torch.ones(L, L, device=inputs.device) * float('-inf'), diagonal=1
        ).to(torch.bool)

        mock_layer = MagicMock(torch.nn.MultiheadAttention(E, H), return_value=inputs)
        encoder.layers[0] = mock_layer
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

        for kernel in kernels:
            with torch.backends.cuda.sdp_kernel(
                enable_math=(kernel == 'math'),
                enable_flash=(kernel == 'flash'),
                enable_mem_efficient=(kernel == 'meff')
            ):
                actual, _ = mha(qkv, qkv, qkv, need_weights=False, is_causal=True)
                self.assertTrue(torch.equal(actual, expected))

                if kernel != 'math':
                    # fails with embedding size not multiple of 4
                    with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                        qkv_f, mha_f = ones_tensor(S, L, 2), nn.MultiheadAttention(2, H).to(device)
                        _ = mha_f(qkv_f, qkv_f, qkv_f, need_weights=False, is_causal=True)
                        torch.cuda.synchronize()

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "CUDA unavailable")
    def test_is_causal_gpu(self):
        device = 'cuda'
        self.is_causal_kernels(["math", "meff"], device)


class TestSDPA(NNTestCase):
    """ Used to test the functionality of scaled_dot_product_attention
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

    backend_map = {
        SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
        SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
    }

    def rand_tensor(self, shape: Tuple[int], device: str, dtype: torch.dtype,
                    type: str, requires_grad: bool = False, packed: bool = False) -> torch.Tensor:
        """Creates rand dense or nested tensor with given shape and type.

        Args:
            shape (Tuple[int]): _description_
            device (str): _description_
            dtype (torch.dtype): _description_
            type (str): _description_
            requires_grad (bool, optional): _description_. Defaults to False.
            packed (bool, optional): _description_. Defaults to False.

        Returns:
            torch.Tensor: _description_
        """
        batch, seq_len, num_heads, head_dim = shape
        if type == "nested":
            size = (seq_len, num_heads, head_dim) if not packed else (seq_len, 3 * num_heads * head_dim)
            return torch.nested.nested_tensor([
                torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
                for _ in range(batch)])
        else:
            size = (batch, seq_len, num_heads, head_dim) if not packed else (batch, seq_len, 3 * num_heads * head_dim)
            return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)

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

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("is_contiguous", [True, False])
    def test_scaled_dot_product_attention_fused_kernels(self, type: str, is_contiguous: bool):
        rand_tensor = partial(self.rand_tensor, type=type, device="cuda", dtype=torch.float16)

        batch, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = (batch, seq_len, num_heads, head_dim)

        query = rand_tensor(shape)
        key = rand_tensor(shape)
        value = rand_tensor(shape)

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
    def test_scaled_dot_product_attention_fused_kernels_packed(self, type: str, is_contiguous: bool):
        rand_tensor = partial(self.rand_tensor, type=type, device="cuda", dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = (batch_size, seq_len, num_heads, head_dim)

        # Test Packed
        qkv = rand_tensor(shape)
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
    @parametrize("fused_kernel", ["flash", "mem_efficient"])
    def test_scaled_dot_product_attention_fused_kernels_packed_accuracy(self, type: str, fused_kernel: str):
        if (not SM80OrLater) and fused_kernel == "flash":
            return

        def rand_nt(shape):
            batch, seq_len, num_heads, head_dim = shape
            tensors = [6 * torch.rand((seq_len, 3 * num_heads * head_dim), device="cuda", dtype=torch.float32) - 3
                       for _ in range(batch)]
            return (torch.nested.nested_tensor(tensors, device="cuda", dtype=torch.float32),
                    torch.nested.nested_tensor(tensors, device="cuda", dtype=torch.float16))

        def rand_tensor(shape):
            batch, seq_len, num_heads, head_dim = shape
            tensor = 6 * torch.rand((batch, seq_len, 3 * num_heads * head_dim), device="cuda", dtype=torch.float32) - 3
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

        if fused_kernel == "flash":
            with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                # TODO Flash for the nested path is currently not working due to cuda memory issues
                if type == "nested":
                    self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                        query_lp, key_lp, value_lp, attn_mask=None, dropout_p=0.0, is_causal=False))
                    return
                actual = torch.nn.functional.scaled_dot_product_attention(
                    query_lp, key_lp, value_lp, attn_mask=None, dropout_p=0.0, is_causal=False)
        elif fused_kernel == "mem_efficient":
            with sdp_kernel(enable_mem_efficient=True, enable_flash=False, enable_math=False):
                actual = torch.nn.functional.scaled_dot_product_attention(
                    query_lp, key_lp, value_lp, attn_mask=None, dropout_p=0.0, is_causal=False)

        with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            math_ref_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp.contiguous(), key_lp.contiguous(), value_lp.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
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

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Fused SDPA was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    def test_sdp_math_gradcheck(self, contiguous_inputs: bool):

        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        rand_tensor = partial(self.rand_tensor, type="dense", device="cuda",
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = rand_tensor((batch_size, seq_len, num_heads, head_dim))
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

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Flash Attention was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    @parametrize("is_causal", [True, False])
    def test_sdp_mem_efficient_grad_against_math(self, contiguous_inputs: bool, is_causal: bool):
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        rand_tensor = partial(self.rand_tensor, type="dense", device="cuda",
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = rand_tensor((batch_size, seq_len, num_heads, head_dim))
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
    def test_sdp_flash_attention_grad_against_math(self, contiguous_inputs: bool, is_causal: bool, dtype: torch.dtype):
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        rand_tensor = partial(self.rand_tensor, type="dense", device="cuda",
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = rand_tensor((batch_size, seq_len, num_heads, head_dim))
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
    def test_fused_sdp_choice(self, type: str):
        device = "cpu"
        # Test that cpu and nestedtensor cpu return MATH backend
        for dtype in floating_types_and_half():
            make_tensor = partial(self.rand_tensor, type=type, device=device, dtype=dtype)
            size = (2, 2, 3, 4)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            assert torch._fused_sdp_choice(q, k, v) == SDPBackend.MATH

        if PLATFORM_SUPPORTS_FUSED_SDPA:
            batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
            shape = (batch_size, seq_len, num_heads, head_dim)
            device = "cuda"
            make_tensor = partial(self.rand_tensor, device=device, dtype=torch.float16, packed=True)

            qkv = make_tensor(shape, type=type)
            query, key, value = qkv.chunk(3, dim=-1)

            query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            if SM80OrLater and not type == "nested":
                assert torch._fused_sdp_choice(query, key, value) == SDPBackend.FLASH_ATTENTION
            else:
                assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION

            # Change dtype to float32 so that efficient attention should get chosen
            make_tensor = partial(self.rand_tensor, device=device, dtype=torch.float32, packed=True)

            qkv = make_tensor(shape, type=type)
            query, key, value = qkv.chunk(3, dim=-1)

            query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

            assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "CUDA unavailable")
    @parametrize("warn_only", [True, False])
    def test_sdp_choice_with_determinism(self, warn_only):
        # If we are only warning we still expect that efficient_attention will still be called.
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = (batch_size, seq_len, num_heads, head_dim)
        make_tensor = partial(self.rand_tensor, type="dense", device="cuda", dtype=torch.float32, packed=False)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
                assert torch._fused_sdp_choice(query, key, value) == (
                    SDPBackend.EFFICIENT_ATTENTION if warn_only else SDPBackend.MATH)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not isSM86Device, "CUDA unavailable")
    def test_memory_efficeint_sm86_failure(self):
        device = 'cuda'
        dtype = torch.float16
        make_tensor = partial(self.rand_tensor, type="dense", device=device, dtype=dtype)
        # See check_gpu_sm86_head_dim_128 in pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.h
        size = (2, 2, 4, 128)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        with sdp_kernel(enable_mem_efficient=True, enable_flash=False, enable_math=False):
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    def test_dispatch_fails_no_backend(self):
        dtype = torch.float16
        device = "cuda"
        with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=False):
            size = (2, 3, 4)
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch._fused_sdp_choice(q, k, v))
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if SM80OrLater
        else [SDPBackend.EFFICIENT_ATTENTION],
    )
    def test_invalid_fused_inputs_dim_3(self, kernel: SDPBackend):
        with sdp_kernel(**self.backend_map[kernel]):
            # Dim is not 4
            device = "cuda"
            size = (2, 3, 8)
            dtype = torch.float16
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if SM80OrLater
        else [SDPBackend.EFFICIENT_ATTENTION],
    )
    def test_invalid_fused_inputs_broadcast(self, kernel: SDPBackend):
        with sdp_kernel(**self.backend_map[kernel]):
            #  Fused Kernels don't support broadcasting
            device = "cuda"
            dtype = torch.float16
            size = (2, 4, 3, 8)
            size_broadcast = (1, 4, 3, 8)
            q = torch.randn(size_broadcast, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "Does not support fused scaled dot product attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_fused_inputs_head_dim(self, kernel: SDPBackend):
        with sdp_kernel(**self.backend_map[kernel]):
            # The embed dim per head is not divisible by 8 for flash attention
            device = "cuda"
            dtype = torch.float16
            make_tensor = partial(self.rand_tensor, type="dense", device=device, dtype=dtype)
            size = (2, 2, 3, 9)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if SM80OrLater
        else [SDPBackend.EFFICIENT_ATTENTION],
    )
    def test_invalid_fused_inputs_invalid_dtype(self, kernel: SDPBackend):
        with sdp_kernel(**self.backend_map[kernel]):
            # Invalid dtype for both Flash Attention and Mem Efficient Attention
            device = "cuda"
            size = (2, 2, 3, 16)
            make_tensor = partial(self.rand_tensor, type="dense", device=device, dtype=torch.float64)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if SM80OrLater
        else [SDPBackend.EFFICIENT_ATTENTION],
    )
    def test_invalid_fused_inputs_attn_mask_present(self, kernel: SDPBackend):
        with sdp_kernel(**self.backend_map[kernel]):
            # Failures for unsupported SDP args
            device = "cuda"
            size = (2, 2, 3, 16)
            make_tensor = partial(self.rand_tensor, type="dense", device=device, dtype=torch.float16)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            # Non-None attention mask
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, torch.ones_like(q), 0.0, False))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "CUDA unavailable")
    def test_unaligned_tensors(self):
        # The alignment is depdent on arch so we specifiy SM80OrLater
        device = 'cuda'
        dtype = torch.float16
        shape = (2, 2, 8, 5)
        make_tensor = partial(self.rand_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False):
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "CUDA unavailable")
    def test_flash_fail_fp32(self):
        device = 'cuda'
        dtype = torch.float
        shape = (16, 16, 32, 32)
        make_tensor = partial(self.rand_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "CUDA unavailable")
    def test_flash_autocast_fp32_float16(self):
        device = 'cuda'
        dtype = torch.float
        shape = (16, 16, 32, 32)
        make_tensor = partial(self.rand_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "CUDA unavailable")
    def test_flash_autocast_fp32_bfloat16(self):
        device = 'cuda'
        dtype = torch.float
        shape = (16, 16, 32, 32)
        make_tensor = partial(self.rand_tensor, shape=shape, type=type, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    def test_incompatible_mask(self):
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

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "CUDA unavailable")
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [4, 8, 64, 128, 256, 512, 1024, 2048])
    @parametrize("seq_len_k", [4, 8, 64, 128, 256, 512, 1024, 2048])
    @parametrize("head_dim", [8, 16, 32, 64, 128])
    @parametrize("is_causal", [True, False])
    @parametrize("dropout_p", [0.0])  # mem_efficient_attention does not support dropout
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_mem_efficient_attention_vs_math_ref_grads(self, batch_size: int, seq_len_q: int, seq_len_k: int,
                                                       head_dim: int, is_causal: bool, dropout_p: float, dtype: torch.dtype):
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device="cuda", dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device="cuda",
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device="cuda", dtype=dtype, requires_grad=True)

        # Run the math kernel on low precision references
        query_ref_lp = query.clone().detach().requires_grad_(True)
        key_ref_lp = key.clone().detach().requires_grad_(True)
        value_ref_lp = value.clone().detach().requires_grad_(True)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32

        query_ref = query.clone().detach().to(higher_precision_dtype).requires_grad_(True)
        key_ref = key.clone().detach().to(higher_precision_dtype).requires_grad_(True)
        value_ref = value.clone().detach().to(higher_precision_dtype).requires_grad_(True)

        # Create real output
        with sdp_kernel(enable_mem_efficient=True, enable_flash=False, enable_math=False):
            # See check_gpu_sm86_head_dim_128 in pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.h
            if isSM86Device and head_dim == 128:
                self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value,
                                                                                       dropout_p=dropout_p, is_causal=is_causal))
                return
            else:
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

        with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            # High Precision Math Reference
            out_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref,
                                                     dropout_p=dropout_p, is_causal=is_causal)
            # Low Precision Math Reference
            out_lp_ref = F.scaled_dot_product_attention(query_ref_lp, key_ref_lp, value_ref_lp,
                                                        dropout_p=dropout_p, is_causal=is_causal)

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
        out_deviation = out_ref - out_lp_ref
        output_ref_atol = max(torch.abs(out_deviation).max().item(), default_atol[out.dtype])
        output_ref_rtol = max(get_rtol(out_ref, out_lp_ref), default_rtol[out.dtype])

        grad_q_deviation = query_ref.grad - query_ref_lp.grad
        grad_q_ref_atol = max(torch.abs(grad_q_deviation).max().item(), default_atol[out.dtype])
        grad_q_ref_rtol = max(get_rtol(query_ref.grad, query_ref_lp.grad), default_rtol[out.dtype])

        # TODO: Investigate why grad_k needs larger tolerances
        grad_k_deviation = key_ref.grad - key_ref_lp.grad
        grad_k_ref_atol = max(7 * torch.abs(grad_k_deviation).max().item(), 7 * default_atol[out.dtype])
        grad_k_ref_rtol = max(7 * get_rtol(key_ref.grad, key_ref_lp.grad), 7 * default_rtol[out.dtype])

        grad_v_deviation = value_ref.grad - value_ref_lp.grad
        grad_v_ref_atol = max(torch.abs(grad_v_deviation).max().item(), default_atol[out.dtype])
        grad_v_ref_rtol = max(get_rtol(value_ref.grad, value_ref_lp.grad), default_rtol[out.dtype])

        self.assertEqual(out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol)
        self.assertEqual(query.grad, query_ref.grad.to(query.grad.dtype),
                         atol=grad_q_ref_atol, rtol=grad_q_ref_rtol)
        self.assertEqual(key.grad, key_ref.grad.to(key.grad.dtype),
                         atol=grad_k_ref_atol, rtol=grad_k_ref_rtol)
        self.assertEqual(value.grad, value_ref.grad.to(value.grad.dtype),
                         atol=grad_v_ref_atol, rtol=grad_v_ref_rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_SDPA or not SM80OrLater, "CUDA unavailable")
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [4, 8, 64, 128, 256, 512, 1024, 2048])
    @parametrize("seq_len_k", [4, 8, 64, 128, 256, 512, 1024, 2048])
    @parametrize("head_dim", [8, 16, 32, 64])
    @parametrize("is_causal", [True, False])
    @parametrize("dropout_p", [0.0, 0.22, 0.48])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_flash_attention_vs_math_ref_grads(self, batch_size: int, seq_len_q: int, seq_len_k: int,
                                               head_dim: int, is_causal: bool, dropout_p: float, dtype: torch.dtype):
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device="cuda", dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device="cuda",
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device="cuda", dtype=dtype, requires_grad=True)

        # Run the math kernel on low precision references
        query_ref_lp = query.clone().detach().requires_grad_(True)
        key_ref_lp = key.clone().detach().requires_grad_(True)
        value_ref_lp = value.clone().detach().requires_grad_(True)

        query_ref = query.clone().detach().to(torch.float32).requires_grad_(True)
        key_ref = key.clone().detach().to(torch.float32).requires_grad_(True)
        value_ref = value.clone().detach().to(torch.float32).requires_grad_(True)

        is_dropout = dropout_p > 0.0

        # Create real output
        output_tuple = torch.ops.aten._scaled_dot_product_flash_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal, return_debug_mask=True)
        out = output_tuple[0]
        dbug_mask = output_tuple[-1]

        query_padding_mask = torch.ones(
            1, seq_len_q, device="cuda", dtype=torch.bool)
        key_padding_mask = torch.ones(
            1, seq_len_k, device="cuda", dtype=torch.bool)

        softmax_mask = self.convert_flash_attn_S_to_softmax(
            dbug_mask, query_padding_mask, key_padding_mask, head_dim=head_dim, causal=is_causal)
        dropout_mask = softmax_mask >= 0

        if not is_dropout:
            with sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(
                    query_ref, key_ref, value_ref, is_causal=is_causal)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(
                    query_ref_lp, key_ref_lp, value_ref_lp, is_causal=is_causal)
        else:
            # High Precision Math Reference
            out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal, dropout_mask=dropout_mask)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref_lp, key_ref_lp, value_ref_lp, dropout_p=dropout_p, is_causal=is_causal, dropout_mask=dropout_mask)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        out.backward(upstream_grad)
        out_ref.backward(upstream_grad.to(out_ref.dtype))
        out_lp_ref.backward(upstream_grad.to(out_lp_ref.dtype))

        # See [Note] Fused Tolerances above
        out_deviation = out_ref - out_lp_ref
        output_ref_atol = max(torch.abs(out_deviation).max().item(), default_atol[out.dtype])
        output_ref_rtol = max(get_rtol(out_ref, out_lp_ref), default_rtol[out.dtype])

        # TODO: Investigate why grad_q needs larger tolerances
        grad_q_deviation = query_ref.grad - query_ref_lp.grad
        grad_q_ref_atol = max(2 * torch.abs(grad_q_deviation).max().item(), default_atol[out.dtype])
        grad_q_ref_rtol = max(get_rtol(query_ref.grad, query_ref_lp.grad), default_rtol[out.dtype])

        grad_k_deviation = key_ref.grad - key_ref_lp.grad
        grad_k_ref_atol = max(torch.abs(grad_k_deviation).max().item(), default_atol[out.dtype])
        grad_k_ref_rtol = max(get_rtol(key_ref.grad, key_ref_lp.grad), default_rtol[out.dtype])

        grad_v_deviation = value_ref.grad - value_ref_lp.grad
        grad_v_ref_atol = max(torch.abs(grad_v_deviation).max().item(), default_atol[out.dtype])
        grad_v_ref_rtol = max(get_rtol(value_ref.grad, value_ref_lp.grad), default_rtol[out.dtype])

        self.assertEqual(out, out_ref.to(out.dtype), atol=output_ref_atol, rtol=output_ref_rtol)
        self.assertEqual(query.grad, query_ref.grad.to(query.grad.dtype),
                         atol=grad_q_ref_atol, rtol=grad_q_ref_rtol)
        self.assertEqual(key.grad, key_ref.grad.to(key.grad.dtype),
                         atol=grad_k_ref_atol, rtol=grad_k_ref_rtol)
        self.assertEqual(value.grad, value_ref.grad.to(value.grad.dtype),
                         atol=grad_v_ref_atol, rtol=grad_v_ref_rtol)

    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    @parametrize("device", ["cpu", "cuda"] if TEST_CUDA else ["cpu"])
    def test_invalid_inputs_different_datatypes(self, kernel: SDPBackend, device: str):
        with sdp_kernel(**self.backend_map[kernel]):
            # Different datatypes
            shape = (1, 4, 8, 16)
            query = torch.randn(shape, dtype=torch.float32, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    @parametrize("device", ["cpu", "cuda"] if TEST_CUDA else ["cpu"])
    def test_invalid_inputs_different_devices(self, kernel: SDPBackend, device: str):
        # Different devices
        shape = (1, 4, 8, 16)
        if device == "cuda":
            query = torch.randn(shape, dtype=torch.float32, device=device)
            key = torch.randn(shape, dtype=torch.float16, device='cpu')
            value = torch.randn(shape, dtype=torch.float16, device='cpu')
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    @parametrize("device", ["cpu", "cuda"] if TEST_CUDA else ["cpu"])
    def test_invalid_inputs_1_dimensional_inputs(self, kernel: SDPBackend, device: str):
        with sdp_kernel(**self.backend_map[kernel]):
            # 1 dimensional input
            shape = (1, 4)
            query = torch.randn(4, dtype=torch.float16, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

# TODO: Replace this with instantiate_device_type_tests() to take advantage of test framework support for
# cross device / dtype testing.
instantiate_parametrized_tests(TestTransformers)
instantiate_parametrized_tests(TestSDPA)

if __name__ == '__main__':
    run_tests()
