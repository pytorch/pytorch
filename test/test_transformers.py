# Owner(s): ["module: sdpa"]

import contextlib
from functools import partial
from collections import namedtuple
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.bias import CausalVariant, causal_lower_right, causal_upper_left
from torch.nn.parameter import Parameter
import unittest
from unittest.mock import patch, MagicMock, ANY
import math
import itertools
import torch.optim as optim
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA, onlyCPU
from typing import Optional
import torch.utils.cpp_extension
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    TEST_WITH_ROCM,
    skipIfRocm,
    skipIfTorchDynamo,
    TEST_FAIRSEQ,
    run_tests,
    parametrize,
    freeze_rng_state,
    TEST_WITH_CROSSREF,
    slowTest,
    set_default_dtype,
    gradcheck,
    make_tensor,
    NOTEST_CPU,
    IS_WINDOWS,
    TEST_WITH_TORCHDYNAMO,
)
from torch._dynamo.testing import CompileCounterWithBackend


from torch.testing._internal.common_methods_invocations import wrapper_set_seed
from torch.testing._internal.common_cuda import (
    IS_JETSON,
    SM80OrLater,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    SM90OrLater,
    tf32_on_and_off,
    tf32_enabled,
)

if TEST_FAIRSEQ:
    import fairseq.models.transformer as fairseq_transformer

SdpaShape = namedtuple('Sdpa_Shape', ['batch', 'num_heads', 'seq_len', 'head_dim'])
Tolerances = namedtuple('Tolerances', ['atol', 'rtol'])


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

isSM8XDevice = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(8, 6), (8, 7), (8, 9)]
isSM90Device = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
isSM120Device = torch.cuda.is_available() and torch.cuda.get_device_capability() in [(12, 0), (12, 1)]
isSM5xDevice = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 5
isLessThanSM80Device = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8

TEST_WITH_CK = TEST_WITH_ROCM and torch.backends.cuda.preferred_rocm_fa_library() == torch.backends.cuda._ROCmFABackends['ck']

def _check_equal(
    golden: torch.Tensor,
    reference: torch.Tensor,
    test: torch.Tensor,
    fudge_factor: float,
    tensor_name: Optional[str] = None
) -> None:
    """
    Compare test tensor against golden and reference tensors.
    Golden is the highest precision possible serving as the "ground truth"
    Refernce is the same precision as test and should also serve as less precisie ground truth.
    We calcculate the "reference error" by comparing the golden to reference and use this as the
    measruing stick for the test tensor.

    Raises ValueError if compiled error exceeds threshold.

    Args:
        golden (torch.Tensor): The golden tensor to compare against.
        reference (torch.Tensor): The reference tensor for error calculation.
        test (torch.Tensor): The test tensor to be evaluated.
        fudge_factor (float): A multiplier for the reference error to determine the threshold.
        tensor_name (Optional[str], optional): Name of the tensor for error reporting. Defaults to None.

    Raises:
        ValueError: If the test tensor contains NaN values while the reference does not,
                    or if the test error exceeds the calculated threshold.

    Notes:
        - For nested tensors, the function recursively calls itself on each nested element.
        - The error threshold is calculated as the maximum of a default tolerance for float32
          and the product of the reference error and the fudge_factor.
        - If the test error exceeds the threshold, a ValueError is raised with a detailed message.
    """
    if golden.is_nested and reference.is_nested and test.is_nested:
        for gold, ref, tst in zip(golden.unbind(), reference.unbind(), test.unbind()):
            _check_equal(gold, ref, tst, fudge_factor, tensor_name)
        return

    # Compute error between golden
    test_error = (golden - test).abs().max()
    ref_error = (golden - reference).abs().max()

    if torch.isnan(test_error).any() and not torch.isnan(ref_error).any():
        raise ValueError("Output/Grad with NaN")

    # Calculate the error threshold as the maximum of:
    # 1. A predefined default tolerance for float32
    # 2. The reference error multiplied by the fudge factor
    threshold = max(default_atol[torch.float32], ref_error * fudge_factor)
    if test_error > threshold:
        name = tensor_name or ""
        msg = f"{name} Test error {test_error} is greater than threshold {threshold}!"
        raise ValueError(msg)


def check_out_and_grad(
    out_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_query_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_key_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_value_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_attn_mask_tuple: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    fudge_factors: Optional[dict[str, float]] = None
) -> None:
    """
    Check output and gradients of attention mechanism tensors.
    Compares compiled results against reference and low-precision reference tensors.

    Args:
        out_tuple: Tuple of (ref, lp_ref, compiled) for output tensor
        grad_query_tuple: Tuple of (ref, lp_ref, compiled) for query gradient
        grad_key_tuple: Tuple of (ref, lp_ref, compiled) for key gradient
        grad_value_tuple: Tuple of (ref, lp_ref, compiled) for value gradient
        grad_attn_mask_tuple: Optional tuple of (ref, lp_ref, compiled) for attention mask gradient
        fudge_factors: Dictionary of fudge factors for each tensor type (default uses 5.0 for all)
    """
    default_fudge_factor = 5.0
    if fudge_factors is None:
        fudge_factors = {}

    out_ref, out_lp_ref, out = out_tuple

    with torch.no_grad():
        _check_equal(out_ref, out_lp_ref, out, fudge_factors.get('out', default_fudge_factor), "out")

        grad_checks = [
            (grad_query_tuple, "grad_query"),
            (grad_key_tuple, "grad_key"),
            (grad_value_tuple, "grad_value")
        ]

        for grad_tuple, name in grad_checks:
            ref_grad, lp_ref_grad, comp_grad = grad_tuple
            _check_equal(ref_grad, lp_ref_grad, comp_grad, fudge_factors.get(name, default_fudge_factor), name)

        if grad_attn_mask_tuple:
            attn_mask_ref_grad, attn_mask_ref_lp_grad, attn_mask_grad = grad_attn_mask_tuple
            _check_equal(
                attn_mask_ref_grad,
                attn_mask_ref_lp_grad,
                attn_mask_grad,
                fudge_factors.get("grad_attn_mask", default_fudge_factor),
                "grad_attn_mask",
            )


def query_key_value_clones(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dtype: torch.dtype = None):
    """ Clones the query, key, and value tensors and moves them to the specified dtype. """
    if dtype is None:
        dtype = query.dtype
    query_ref = query.detach().clone().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.detach().clone().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.detach().clone().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref

def get_platform_specific_sdpa():
    ret = []
    if PLATFORM_SUPPORTS_FLASH_ATTENTION:
        ret.append(SDPBackend.FLASH_ATTENTION)
    if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
        ret.append(SDPBackend.CUDNN_ATTENTION)
    if not ret:
        # Add a placeholder, an empty list causes "An empty arg_values was passed to @parametrize"
        ret.append(SDPBackend.EFFICIENT_ATTENTION)
    return ret

PLATFORM_SPECIFIC_SDPA = get_platform_specific_sdpa()
# Indicate the Efficient attention backend can support:
# 1. sequence longher than 512
# 2. head dimsion larger than 64
MEM_EFF_CAPABILITY_MATCHES_SM80 = SM80OrLater or TEST_WITH_ROCM

def rand_sdpa_tensor(shape: SdpaShape, device: str, dtype: torch.dtype, type: str,
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
    batch, num_heads, seq_len, head_dim = shape.batch, shape.num_heads, shape.seq_len, shape.head_dim
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
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, 0.0)

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
                e = None
                try:
                    encoder(test, src_key_padding_mask=pad_mask.to(torch.uint8))
                except AssertionError:
                    continue
                self.assertFalse(e, "Failed to catch unsupported uint8 type exception")

                test_train_bool = encoder(test, src_key_padding_mask=pad_mask)
                encoder.eval()

                # Expect long type not supported
                e = None
                try:
                    encoder(test, src_key_padding_mask=pad_mask.to(torch.int64))
                except AssertionError as e:
                    continue
                self.assertFalse(e, "Failed to catch unsupported Long type exception")

                test_eval_bool = encoder(test, src_key_padding_mask=pad_mask)
                l1_bool = nn.L1Loss()(test_train_bool[:, 0:2, :], test_eval_bool[:, 0:2, :]).item()
                self.assertTrue(l1_bool < 1e-4, "Eval/Train difference in pad_mask BOOL")

    @tf32_on_and_off(0.001)
    @parametrize("attn_mask_dim", [2, 3, None])
    @parametrize("key_padding_mask_dim", [2, None])
    @parametrize("mask_dtype", [torch.bool, torch.float32])
    def test_multiheadattention_fastpath_attn_mask(self, device, attn_mask_dim, key_padding_mask_dim, mask_dtype):
        if TEST_WITH_ROCM:
            if attn_mask_dim is not None and mask_dtype == torch.bool:
                self.skipTest("boolean mask is not fully supported on ROCm yet.")
        # MHA converts all
        with torch.no_grad():
            B = 2
            L = 4
            D = 8
            H = 4

            if attn_mask_dim == 2:
                attn_mask = make_tensor((L, L), dtype=mask_dtype, device=device)
            elif attn_mask_dim == 3:
                attn_mask = make_tensor((B, 1, L, L), dtype=mask_dtype, device=device).expand(B, H, L, L).reshape(B * H, L, L)
            elif attn_mask_dim is None:
                attn_mask = None

            if key_padding_mask_dim == 2:
                key_padding_mask = make_tensor((B, L), dtype=mask_dtype, device=device)
            elif key_padding_mask_dim is None:
                key_padding_mask = None

            mha = nn.MultiheadAttention(D, H, batch_first=True, device=device)
            X = torch.randn(B, L, D, device=device)

            mha.train()  # disable fast path
            out, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            mha.eval()  # enable fast path
            out_fp, _ = mha(X, X, X, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
            # The FP kernel will return NaNs while the sdpa kernel which is ran when the fast path is turned off returns 0 instead
            # of NaNs for fully masked rows
            self.assertEqual(out, out_fp.nan_to_num())

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

    @parametrize("nhead", [3, 4])
    def test_transformerencoderlayer_no_fastpath_with_hooks(self, device, nhead):
        batch_size = 2
        seqlen = 4
        d_model = 12

        model = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model,
            batch_first=True).to(device).eval()
        src = torch.rand(batch_size, seqlen, d_model).to(device)  # bs, seqlen, d_model

        cache = []

        # forward hook to save output
        def hook(module, inputs, output):
            cache.append(output[0].detach())

        # register hook to get the output of the self-attention layer
        handle = model.self_attn.register_forward_hook(hook)

        # forward pass
        with torch.inference_mode():
            model(src)

        # output of the self-attention layer
        assert len(cache) == 1, f"Expected 1 output, got {len(cache)}"

        # remove hook
        handle.remove()

    @skipIfRocm
    @tf32_on_and_off(0.001)
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
                self.assertEqual(fastpath_output_expanded, slowpath_output)

    @tf32_on_and_off(0.001)
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
        self.assertEqual(result, ref_output)

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

        with (self.assertNoLogs(None) if not TEST_WITH_TORCHDYNAMO else contextlib.nullcontext()):
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
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        is_causal_output = layer(x, src_mask=mask, is_causal=True)
        masked_output = layer(x, src_mask=mask)

        self.assertEqual(masked_output, is_causal_output)

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not supposrt pre-SM80 hardware"
    )
    def test_math_backend_high_precision(self):
        xq = torch.rand([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16) * 5
        xk = torch.rand([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16) * 5
        xv = torch.randn([1, 128, 2, 80], device="cuda", dtype=torch.bfloat16)
        mask = None

        def scaled_dot_product_attention(
            xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, mask: Optional[torch.Tensor], backend: SDPBackend
        ) -> torch.Tensor:
            n_rep = 1
            xq, xk, xv = (tensor.transpose(1, 2) for tensor in (xq, xk, xv))
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)

            with sdpa_kernel(backends=[backend]):
                attn_output = F.scaled_dot_product_attention(
                    xq, xk, xv, attn_mask=mask, dropout_p=0.0
                )
            return attn_output.transpose(1, 2)

        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        sdp_math_low_prec_out = scaled_dot_product_attention(xq, xk, xv, mask, SDPBackend.MATH)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        sdp_math_high_prec_out = scaled_dot_product_attention(xq, xk, xv, mask, SDPBackend.MATH)

        sdp_math_fp64_out_ref = scaled_dot_product_attention(
            xq.double(), xk.double(), xv.double(), mask, SDPBackend.MATH
        ).bfloat16()

        torch.testing.assert_close(sdp_math_high_prec_out, sdp_math_fp64_out_ref, atol=1e-2, rtol=1e-2)

        with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close"):
            torch.testing.assert_close(sdp_math_low_prec_out, sdp_math_fp64_out_ref, atol=1e-2, rtol=1e-2)

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
                    self.decoder = torch_to_fairseq(torch_encoder, self.decoder)  # noqa: F821
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

    @tf32_on_and_off(0.003)
    @parametrize("input_dim,attn_mask_dim,is_causal",
                 [(3, None, False), (3, 2, False), (3, 2, True), (3, 3, False), (3, 3, True),
                  (4, None, False), (4, 2, False), (4, 2, True), (4, 4, False), (4, 4, True)],
                 name_fn=lambda input_dim, attn_dim, is_causal: (
                     f"{input_dim}D_input_dim_" + (
                         f"{attn_dim}D_{'causal_' if is_causal else ''}attn_mask"
                         if attn_dim is not None else "no_attn_mask")))
    @parametrize("dropout_p", [0.0, 0.2, 0.5])
    @sdpa_kernel(backends=[SDPBackend.MATH])
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
                    # This test the fully masked out rows case
                if torch.isnan(expected).any():
                    row_sums = attn_mask.sum(dim=-1)
                    masked_out_rows = (row_sums == 0)

                    for _ in range((input_dim - attn_mask_dim) - 1):
                        masked_out_rows = masked_out_rows.unsqueeze(0)

                    masked_out_rows = masked_out_rows.expand(expected.shape[:-1])
                    # Slice out the fully masked rows from expected and actual
                    expected_masked_out = expected[masked_out_rows]
                    actual_masked_out = actual[masked_out_rows]

                    expected_all_nan = torch.isnan(expected_masked_out).all()
                    actual_all_zero = (actual_masked_out.abs().sum() == 0)

                    self.assertTrue(expected_all_nan)
                    self.assertTrue(actual_all_zero)
                    return

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

    def test_transformer_bias_is_none(self, device):
        batch_size = 2
        seqlen = 3
        d_model = 8
        nhead = 4

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, bias=False, batch_first=True, device=device)
        encoder_layer.eval()
        x = torch.randn(batch_size, seqlen, d_model, device=device)
        # runs without error
        encoder_layer(x)

        with self.assertWarnsRegex(UserWarning, "encoder_layer.self_attn was passed bias=False"):
            encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1).eval()
            encoder(x)

        with self.assertWarnsRegex(UserWarning, "self_attn was passed bias=False"):
            transformer = torch.nn.Transformer(
                d_model=d_model, nhead=nhead, bias=False, batch_first=True, device=device
            ).eval()
            transformer(x, x)

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
        with self.assertRaises(RuntimeError):
            mha(t_qvk, t_qvk, t_qvk, is_causal=True)

        # # Passing a causal mask sets is_causal to 1
        causal_mask = torch.triu(
            torch.ones(L, L, device=inputs.device) * float('-inf'), diagonal=1
        ).to(torch.bool)

        mock_layer = MagicMock(torch.nn.MultiheadAttention(E, H), return_value=inputs)
        encoder.layers[1] = mock_layer
        outputs = encoder(inputs, mask=causal_mask)
        mock_layer.assert_called_with(ANY, src_mask=ANY, is_causal=True, src_key_padding_mask=ANY)

        # check expected numerical values with all kernels
        self.is_causal_kernels([SDPBackend.MATH], device)

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
            with sdpa_kernel(backends=[kernel]):
                actual, _ = mha(qkv, qkv, qkv, attn_mask=mask, need_weights=False, is_causal=True)
                self.assertTrue(torch.equal(actual, expected))

                if kernel != SDPBackend.MATH:
                    # fails with embedding size not multiple of 4
                    with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                        qkv_f, mha_f = ones_tensor(S, L, 2), nn.MultiheadAttention(2, H).to(device)
                        mask = torch.nn.Transformer.generate_square_subsequent_mask(
                            qkv_f.size(1), device=device
                        )
                        _ = mha_f(qkv_f, qkv_f, qkv_f, attn_mask=mask, need_weights=False, is_causal=True)
                        torch.cuda.synchronize()

    @skipIfRocm  # Missing EFFICIENT_ATTENTION
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not supposrt fused SDPA or pre-SM80 hardware"
    )
    def test_is_causal_gpu(self):
        device = 'cuda'
        self.is_causal_kernels([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION], device)

    def test_script_mha_in_proj_weight_none(self):
        mha = torch.nn.MultiheadAttention(
            embed_dim=128, num_heads=8, kdim=256, vdim=256
        ).eval()

        torch.jit.script(mha)

    @unittest.skipIf(TEST_WITH_CROSSREF, 'Fastpath not available with crossref')
    @torch.no_grad()
    def test_disable_fastpath(self, device):
        def _test_te_fastpath_called(model, args, kwargs=None, return_value=None, is_called=True):
            if kwargs is None:
                kwargs = {}
            with patch('torch._transformer_encoder_layer_fwd') as fastpath_mock:
                fastpath_mock.return_value = return_value
                model(*args, **kwargs)
                self.assertTrue(fastpath_mock.called == is_called)

        def _test_mha_fastpath_called(model, args, kwargs=None, return_value=None, is_called=True):
            if kwargs is None:
                kwargs = {}
            with patch('torch._native_multi_head_attention') as fastpath_mock:
                fastpath_mock.return_value = return_value
                model(*args, **kwargs)
                self.assertTrue(fastpath_mock.called == is_called)

        inp = torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32, device=device)
        src_key_padding_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool, device=device)
        te_return_value = torch.ones((1, 3, 2), dtype=torch.float32)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=2, nhead=2, dim_feedforward=8, batch_first=True)
        te = torch.nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=True, mask_check=True)
        te = te.to(device).eval()

        t = torch.nn.Transformer(d_model=2, nhead=2, batch_first=True, device=device).eval()
        src = torch.tensor([[[0, 1], [2, 3], [4, 5]]], dtype=torch.float32, device=device)
        tgt = torch.tensor([[[0, 1], [2, 3], [4, 5], [6, 7]]], dtype=torch.float32, device=device)
        t_return_value = torch.ones((1, 3, 2), dtype=torch.float32, device=device)

        mha = nn.MultiheadAttention(2, 2, batch_first=True, device=device).eval()
        q = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.float32, device=device)
        mha_return_value = torch.ones((1, 3, 2), dtype=torch.float32, device=device)

        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=True
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=True)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=True)

        torch.backends.mha.set_fastpath_enabled(False)
        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=False
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=False)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=False)

        torch.backends.mha.set_fastpath_enabled(True)
        _test_te_fastpath_called(
            te, (inp,), kwargs={'src_key_padding_mask': src_key_padding_mask},
            return_value=te_return_value, is_called=True
        )
        _test_te_fastpath_called(t, (src, tgt), return_value=t_return_value, is_called=True)
        _test_mha_fastpath_called(mha, (q, q, q,), return_value=mha_return_value, is_called=True)


class TestSDPAFailureModes(NNTestCase):
    """ Used to test the failure modes of scaled_dot_product_attention
    """
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @onlyCUDA
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION or not isSM8XDevice or not isSM120Device,
        "Does not support fused SDPA or not SM86+ hardware",
    )
    @parametrize("head_dim", [193, 256])
    @parametrize("dropout_p", [0.0, 0.2])
    def test_flash_backward_failure_sm86plus(self, device, head_dim: int, dropout_p: float):
        dtype = torch.float16
        make_tensor = partial(torch.rand, device=device, dtype=dtype)
        # See check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89 in
        # pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.h
        size = (2, 2, 4, head_dim)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            # Should not fail because inputs don't require grad
            flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

            self.assertEqual(math_ref, flash_ref, atol=1e-3, rtol=1e-3)

            # Should fail because inputs require grad
            q = make_tensor(size, requires_grad=True)
            k = make_tensor(size, requires_grad=True)
            v = make_tensor(size, requires_grad=True)
            if 192 < head_dim <= 224 or (head_dim > 224 and dropout_p != 0.0):
                self.assertRaises(
                    RuntimeError,
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, None, dropout_p, False
                    ),
                )
            else:
                flash_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, None, dropout_p, False)

    @onlyCUDA
    def test_dispatch_fails_no_backend(self, device):
        dtype = torch.float16
        with sdpa_kernel(backends=[SDPBackend.ERROR]):
            size = (2, 3, 4)
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch._fused_sdp_choice(q, k, v))
            self.assertRaisesRegex(RuntimeError, "No viable backend for scaled_dot_product_attention was found.",
                                   lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_dim_3(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Dim is not 4
            size = (2, 3, 8)
            dtype = torch.float16
            q = torch.randn(size, device=device, dtype=dtype)
            k = torch.randn(size, device=device, dtype=dtype)
            v = torch.randn(size, device=device, dtype=dtype)
            with self.assertWarnsRegex(UserWarning, "All fused kernels requires query, key and value to be 4 dimensional"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_broadcast(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
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
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)
    def test_invalid_sequence_lengths(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Passing in a q,k,v with 0 length sequences will error
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 0, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            with self.assertWarnsRegex(UserWarning, "All fused kernels do not support zero seq_len_q or seq_len_kv."):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)
    def test_invalid_last_dim_stride(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Passing in a q,k,v with last dim stride not equal to 1 will error
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 8, 8)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            q.as_strided_(size, [2, 2, 2, 2])
            with self.assertWarnsRegex(UserWarning, "All fused kernels require the last dimension of the input to have stride 1."):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    @parametrize("fused_kernel", [SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_sdpa_kernel_grouped_query_attention_cuda(self, device, fused_kernel):
        rand_query = torch.rand(8, 8, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        rand_key = torch.rand(8, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        rand_value = torch.rand(8, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)

        with sdpa_kernel(fused_kernel):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                with self.assertWarnsRegex(UserWarning, "For dense inputs, both fused kernels require query, "
                                           "key and value to have"):
                    F.scaled_dot_product_attention(rand_query, rand_key, rand_value, dropout_p=0.0,
                                                   is_causal=False, enable_gqa=True)

    @onlyCPU
    def test_invalid_sdpa_kernel_grouped_query_attention_cpu(self, device):
        rand_query = torch.rand(8, 8, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        rand_key = torch.rand(8, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)
        rand_value = torch.rand(8, 4, 64, 64, device=device, dtype=torch.float16, requires_grad=True)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                with self.assertWarnsRegex(UserWarning, "For dense inputs, both fused kernels require query, "
                                           "key and value to have"):
                    F.scaled_dot_product_attention(rand_query, rand_key, rand_value, dropout_p=0.0,
                                                   is_causal=False, enable_gqa=True)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not flash_attention fused scaled dot product attention")
    @parametrize("kernel", PLATFORM_SPECIFIC_SDPA)
    def test_invalid_fused_inputs_head_dim(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # The embed dim per head is not divisible by 8 for flash attention
            dtype = torch.float16
            make_tensor = partial(torch.rand, device=device, dtype=dtype)
            size = SdpaShape(2, 2, 3, 9) if kernel == SDPBackend.EFFICIENT_ATTENTION else SdpaShape(2, 2, 3, 257)
            if TEST_WITH_ROCM:  # On ROCM, FA and EA share the backend GPU kernels
                size = SdpaShape(2, 2, 3, 513)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Does not support fused scaled dot product attention")
    @parametrize(
        "kernel",
        PLATFORM_SPECIFIC_SDPA,
    )
    def test_invalid_fused_inputs_invalid_dtype(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Invalid dtype for both Flash Attention and Mem Efficient Attention
            size = SdpaShape(2, 2, 3, 16)
            make_tensor = partial(torch.rand, device=device, dtype=torch.float64)
            q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION])
    def test_invalid_fused_inputs_attn_mask_present(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
            # Failures for unsupported SDP args
            size = SdpaShape(2, 2, 3, 16)
            make_tensor = partial(torch.rand, size, device=device, dtype=torch.float16)
            q, k, v = make_tensor(), make_tensor(), make_tensor()
            # Non-None attention mask
            mask = torch.ones((2, 2, 3, 3), device=device, dtype=q.dtype)
            self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, mask, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_unaligned_tensors(self, device):
        # The alignment is depdent on arch so we specifiy SM80OrLater
        dtype = torch.float16
        size = SdpaShape(2, 2, 8, 5)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            ctxmgr = self.assertRaises(RuntimeError) if not TEST_WITH_ROCM else contextlib.nullcontext()
            with ctxmgr:
                torch.nn.functional.scaled_dot_product_attention(q, k, v, None, 0.0, False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support fused SDPA or pre-SM80 hardware")
    def test_flash_fail_fp32(self, device):
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, BFloat16}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_float16(self, device):
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    def test_flash_autocast_fp32_bfloat16(self, device):
        dtype = torch.float
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False)

    # Note: do not truncate the list according to platforms. These tests should always raise errors.
    @parametrize("kernel", [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    def test_invalid_inputs_different_datatypes(self, device, kernel: SDPBackend):
        with sdpa_kernel(backends=[kernel]):
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
        with sdpa_kernel(backends=[kernel]):
            # 1 dimensional input
            shape = (1, 4)
            query = torch.randn(4, dtype=torch.float16, device=device)
            key = torch.randn(shape, dtype=torch.float16, device=device)
            value = torch.randn(shape, dtype=torch.float16, device=device)
            self.assertRaises(RuntimeError, lambda: F.scaled_dot_product_attention(query, key, value))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_error_cases(self, device):
        # one of k,v needs to be broadcasted and other has non consistent seq_len dim
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        batch, num_heads, head_dim = 32, 8, 64
        seq_lens_q = torch.randint(low=1, high=32, size=(batch,)).tolist()
        seq_lens_v = torch.randint(low=1, high=32, size=(batch,)).tolist()

        q_shape = SdpaShape(batch, num_heads, seq_lens_q, head_dim)
        k_shape = SdpaShape(1, num_heads, 1, head_dim)
        v_shape = SdpaShape(batch, num_heads, seq_lens_v, head_dim)

        query = rand_nested_tensor(q_shape).transpose(1, 2)
        key = rand_nested_tensor(k_shape).transpose(1, 2)
        value = rand_nested_tensor(v_shape).transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_nested_fails_on_padding_head_dim(self, device):
        dtype = torch.bfloat16
        seq_len_list = [2, 4, 5, 6, 7]
        shape = SdpaShape(5, 8, seq_len_list, 57)
        make_tensor = partial(rand_sdpa_tensor, shape=shape, type="nested", device=device, dtype=dtype)
        q, k, v = make_tensor().transpose(1, 2), make_tensor().transpose(1, 2), make_tensor().transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "For NestedTensor inputs, Flash attention requires"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION or not isLessThanSM80Device,
                     "Current platform does not support fused SDPA or is an SM80+ device.")
    def test_mem_efficient_fail_bfloat16_less_than_sm80(self, device):
        dtype = torch.bfloat16
        size = SdpaShape(16, 16, 32, 32)
        make_tensor = partial(torch.rand, size, device=device, dtype=dtype)
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Expected query, key and value to all be of dtype: {Half, Float}"):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, False))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    def test_flash_atteention_large_bf16_nan_values(self, device):
        query = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")
        key = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")
        value = torch.full((1, 1, 1, 64), 133120.0, dtype=torch.bfloat16, device="cuda")

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        self.assertFalse(torch.isnan(out).any(), "Output should not contain NaNs!")

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_kernels_seq_len_0_inputs(self, device, fused_kernel):
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        batch, num_heads, head_dim = 32, 16, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        # make sure some seq_lens are 0
        num_zeros = 10
        indices = torch.randint(low=0, high=batch, size=(num_zeros,))
        seq_lens.scatter_(0, indices, 0)

        shape = SdpaShape(batch, num_heads, seq_lens.tolist(), head_dim)
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[fused_kernel]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_requires_grad_failure(self, device):
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16, requires_grad=True)
        batch, num_heads, head_dim, head_dim_v = 32, 16, 64, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,)).tolist()
        q_shape = SdpaShape(1, num_heads, 1, head_dim)
        k_shape = SdpaShape(batch, num_heads, seq_lens, head_dim)
        v_shape = SdpaShape(batch, 1, seq_lens, head_dim_v)

        # create a dense query
        query = torch.randn(q_shape, device=device, dtype=torch.float16, requires_grad=True)
        key = rand_nested_tensor(k_shape)
        value = rand_nested_tensor(v_shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, "Both fused kernels do not support training with broadcasted NT inputs"):
                with self.assertRaisesRegex(RuntimeError, "No available kernel"):
                    torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support flash attention")
    def test_flash_attention_fail_with_non_square_causal_attention(self, device):
        dtype = torch.bfloat16
        q_shape = SdpaShape(1, 1, 8, 16)
        kv_shape = SdpaShape(1, 1, 12, 16)
        make_q = partial(torch.rand, q_shape, device=device, dtype=dtype)
        make_kv = partial(torch.rand, kv_shape, device=device, dtype=dtype)
        q, k, v = make_q(), make_kv(), make_kv()
        warning_str = "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k."
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            with self.assertWarnsRegex(UserWarning, warning_str):
                self.assertRaises(RuntimeError, lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, None, 0.0, is_causal=True))

def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    # This should match the block sizes in the CUDA kernel
    assert head_dim <= 256
    major, minor = torch.cuda.get_device_capability(device)
    is_sm8x = major == 8 and minor > 0  # Only include sm86 and sm89, exclude sm80 (A100)
    if head_dim <= 32:
        return 128
    if head_dim <= 64:
        return 128 if not is_dropout else 64
    elif head_dim <= 96:
        return 64
    elif head_dim <= 128:
        if is_sm8x:
            return 64 if (not is_dropout and is_causal) else 32
        else:
            return 64 if not is_dropout else 32
    elif head_dim <= 160:
        if is_sm8x:
            return 64
        else:
            return 32
    elif head_dim <= 192:
        return 64
    elif head_dim <= 224:
        return 64
    elif head_dim <= 256:
        return 64


def pad_last_dim(input_tensor, alignment_size, slice: bool = False):
    last_dim_size = input_tensor.size(-1)
    if (last_dim_size % alignment_size == 0):
        return input_tensor, last_dim_size
    pad_count = alignment_size - (last_dim_size % alignment_size)
    padded_tensor = F.pad(input_tensor, (0, pad_count))
    if slice:
        return padded_tensor[..., :last_dim_size], last_dim_size
    return padded_tensor, last_dim_size


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
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if contiguous_inputs:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            assert gradcheck(lambda *args, **kwargs:
                             wrapper_set_seed(torch.nn.functional.scaled_dot_product_attention, *args, **kwargs),
                             (query, key, value, None, 0.0, False)
                             )

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
        with sdpa_kernel(backends=[kernel]):
            sdp_math = torch.nn.functional.scaled_dot_product_attention(x, x, x, scale=-1.0 / 0.0001)
        self.assertEqual(ref_result, sdp_math)


class TestSDPACpuOnly(NNTestCase):
    """ Used to test CPU only functionality of scaled_dot_product_attention """

    @parametrize("type", ["dense", "nested"])
    @parametrize("dropout", [0.0, 0.7])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.half])
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_cpu(self, device, type: str, dropout: float, dtype: torch.dtype):
        # Test that cpu and nestedtensor cpu return MATH backend
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype)
        size = SdpaShape(2, 8, 128, 64)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        if type == "nested" \
                or dropout > 0.0 \
                or dtype not in [torch.float32, torch.float64, torch.bfloat16, torch.float16]:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.MATH.value
        else:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.FLASH_ATTENTION.value

    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
    @parametrize("batch_size", [2, 12])
    @parametrize("q_seq_len", [11, 514, 1030])
    @parametrize("kv_seq_len", [17, 514])
    @parametrize("n_head", [1, 3])
    @parametrize("head_dim", [8])
    @parametrize("mask_dim", [2, 4])
    @parametrize("bool_mask", [False, True])
    @parametrize("train", [True, False])
    @parametrize("casual", [True, False])
    @parametrize("set_attn_mask", [True, False])
    def test_scaled_dot_product_fused_attention_mask_vs_math_cpu(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        q_seq_len,
        kv_seq_len,
        n_head,
        head_dim,
        mask_dim,
        bool_mask,
        train,
        casual,
        set_attn_mask,
    ):
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        for mask_shape in itertools.product(
            [q_seq_len, 1], [kv_seq_len, 1]
        ) if mask_dim == 2 else itertools.product(
            [batch_size, 1], [n_head, 1], [q_seq_len, 1], [kv_seq_len, 1]
        ):
            make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
            q_shape = SdpaShape(batch_size, n_head, q_seq_len, head_dim)
            kv_shape = SdpaShape(batch_size, n_head, kv_seq_len, head_dim)
            q = make_tensor(q_shape)
            k = make_tensor(kv_shape)
            v = make_tensor(kv_shape)
            q2, k2, v2 = q.clone(), k.clone(), v.clone()

            if train:
                q.requires_grad_(True)
                k.requires_grad_(True)
                v.requires_grad_(True)
                q2.requires_grad_(True)
                k2.requires_grad_(True)
                v2.requires_grad_(True)

            if dtype in [torch.bfloat16, torch.float16]:
                q2, k2, v2 = q2.float(), k2.float(), v2.float()
            # (B, nh, T, hs)
            q = q.view(batch_size, q_seq_len, n_head, head_dim).transpose(1, 2)
            k = k.view(batch_size, kv_seq_len, n_head, head_dim).transpose(1, 2)
            v = v.view(batch_size, kv_seq_len, n_head, head_dim).transpose(1, 2)
            if set_attn_mask and not casual:
                if bool_mask:
                    attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
                else:
                    attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)
            else:
                attn_mask = None
            q2 = q2.view(batch_size, q_seq_len, n_head, head_dim).transpose(1, 2)
            k2 = k2.view(batch_size, kv_seq_len, n_head, head_dim).transpose(1, 2)
            v2 = v2.view(batch_size, kv_seq_len, n_head, head_dim).transpose(1, 2)

            with sdpa_kernel(backends=[fused_kernel]):
                actual = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=casual)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                if not bool_mask and dtype in [torch.bfloat16, torch.float16] and attn_mask is not None:
                    attn_mask = attn_mask.float()
                math_ref = torch.nn.functional.scaled_dot_product_attention(
                    q2, k2, v2, attn_mask=attn_mask, dropout_p=0.0, is_causal=casual)

            if dtype in [torch.bfloat16, torch.float16]:
                math_ref = math_ref.to(dtype)

            self.assertFalse(torch.isnan(math_ref).any())
            self.assertFalse(torch.isnan(actual).any())

            self.assertEqual(actual, math_ref, atol=tol.atol, rtol=tol.rtol)

            if train:
                actual.sum().backward()
                math_ref.sum().backward()

                grad_q_actual, grad_k_actual, grad_v_actual = q.grad, k.grad, v.grad
                grad_q_ref, grad_k_ref, grad_v_ref = q2.grad, k2.grad, v2.grad

                self.assertEqual(grad_q_actual, grad_q_ref, atol=tol.atol, rtol=tol.rtol)
                self.assertEqual(grad_k_actual, grad_k_ref, atol=tol.atol, rtol=tol.rtol)
                self.assertEqual(grad_v_actual, grad_v_ref, atol=tol.atol, rtol=tol.rtol)

    def test_sdpa_with_inf(self, device):
        # https://github.com/pytorch/pytorch/issues/127055.
        full = torch.full((600, 600), float("-inf"), device=device)
        mask = torch.triu(full, diagonal=1) + torch.tril(full, diagonal=-10)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float32, requires_grad=False)
        input_shape = SdpaShape(1, 600, 2, 8)
        q = make_tensor(input_shape)
        k = make_tensor(input_shape)
        v = make_tensor(input_shape)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        self.assertEqual(math_ref, actual)

    def test_sdpa_backward_with_gradient(self, device):
        # https://github.com/pytorch/pytorch/issues/133671.
        def sdpa_helper():
            torch.manual_seed(777)
            query = (
                torch.empty(size=[2, 2, 49, 32], dtype=torch.float32, device=device)
                .uniform_(-1, 1)
                .requires_grad_(True)
            )
            key = (
                torch.empty(size=[2, 2, 49, 32], dtype=torch.float32, device=device)
                .uniform_(-1, 1)
                .requires_grad_(True)
            )
            value = (
                torch.empty(size=[2, 2, 49, 32], dtype=torch.float32, device=device)
                .uniform_(-1, 1)
                .requires_grad_(True)
            )
            res = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, None, 0.0, False
            )
            res_grad = (
                torch.empty_like(res, device=device)
                .uniform_(-1, 1)
            )
            res.backward(res_grad, retain_graph=True)
            return res, query.grad, key.grad, value.grad
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            res_ref, query_grad_ref, key_grad_ref, value_grad_ref = sdpa_helper()
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            res_actual, query_grad_actual, key_grad_actual, value_grad_actual = sdpa_helper()
        self.assertEqual(res_ref, res_actual)
        self.assertEqual(query_grad_ref, query_grad_actual)
        self.assertEqual(key_grad_ref, key_grad_actual)
        self.assertEqual(value_grad_ref, value_grad_actual)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("backend", [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION])
    @parametrize("seq_len", [32, 64, 128])
    @parametrize("head_dim", [16, 32])
    @parametrize("dtype", [torch.float32, torch.float16])
    def test_fully_masked_out_rows(self, backend, device, seq_len, head_dim, dtype):
        def attention_inputs(seq_len, head_dim, device, dtype, mask_every_n_rows=4):
            query = torch.rand(1, 1, seq_len, head_dim, requires_grad=True, device=device, dtype=dtype)
            key = torch.rand(1, 1, seq_len, head_dim, requires_grad=True, device=device, dtype=dtype)
            value = torch.rand(1, 1, seq_len, head_dim, requires_grad=True, device=device, dtype=dtype)

            # Create a mask with deterministic row masking
            mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)

            # Mask every nth row
            mask[0, 0, ::mask_every_n_rows, :] = False

            # Create a fixed pattern for element-wise masking
            element_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
            element_mask[torch.arange(seq_len)[:, None] % 5 == torch.arange(seq_len) % 5] = True

            # Combine row masking and element-wise masking
            mask = mask & element_mask.unsqueeze(0).unsqueeze(0)

            return query, key, value, mask

        def compute_output_and_grads(query, key, value, mask, backend):
            with sdpa_kernel(backend):
                masked_out = scaled_dot_product_attention(query, key, value, attn_mask=mask)
                loss = masked_out.sum()
            grads = torch.autograd.grad(loss, [query, key, value])
            return masked_out, grads

        if backend == SDPBackend.FLASH_ATTENTION and "cuda" in str(device):
            unittest.skip("FlashAttention does not support masks on cuda")
            return
        if backend == SDPBackend.EFFICIENT_ATTENTION and "cpu" in str(device):
            unittest.skip("EfficientAttention does not support masks on cpu")
            return
        query, key, value, mask = attention_inputs(seq_len, head_dim, device, dtype)

        # Compute results for the tested backend
        backend_out, backend_grads = compute_output_and_grads(query, key, value, mask, backend)

        # Compute results for the Math backend
        math_out, math_grads = compute_output_and_grads(query, key, value, mask, SDPBackend.MATH)

        # Compare outputs
        torch.testing.assert_close(backend_out, math_out, atol=5e-3, rtol=0)
        self.assertFalse(backend_out.isnan().any())
        self.assertFalse(math_out.isnan().any())
        # Compare gradients
        for bg, mg in zip(backend_grads, math_grads):
            torch.testing.assert_close(bg, mg, atol=3e-3, rtol=0)
            self.assertFalse(bg.isnan().any())
            self.assertFalse(mg.isnan().any())

        # Check if masked rows are zero in output
        mask_sum = mask.sum(dim=-1, keepdim=True)
        masked_rows = (mask_sum == 0).expand_as(backend_out)
        self.assertTrue((mask_sum == 0).sum() > 0, "No fully masked out rows found")
        assert torch.all(backend_out[masked_rows] == 0), \
            f"Non-zero values in fully masked rows for {backend=}"

        # Check if gradients for masked rows are zero
        grad_query = backend_grads[0]
        assert torch.all(grad_query[masked_rows] == 0), f"Non-zero gradients in fully masked rows for {backend=}"

    @parametrize("dtype", [torch.float32, torch.float16])
    @parametrize("fill_val", [float("inf")])
    def test_non_masked_rows_nan_props(self, device, dtype, fill_val):
        query = torch.randn(1, 2, 4, 16, device=device, dtype=dtype)
        # a single NaN in the query input
        query[0, 1, 2, 3] = fill_val
        query = query.detach().requires_grad_(True)
        key = torch.randn(1, 2, 4, 16, device=device, dtype=dtype, requires_grad=True)
        value = torch.randn(1, 2, 4, 16, device=device, dtype=dtype, requires_grad=True)

        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        self.assertTrue(torch.isnan(out).any())
        out.sum().backward()
        self.assertTrue(torch.isnan(query.grad).any())

    @parametrize("dtype", [torch.float32, torch.float16])
    def test_cpu_flash_attn_nan_propagation(self, dtype):
        # Setup tensors
        query = torch.full((1, 1, 16, 16), torch.nan, dtype=dtype)
        key = torch.randn(1, 1, 16, 16, dtype=dtype)
        value = torch.randn(1, 1, 16, 16, dtype=dtype)

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )

            # Check that output contains NaN
            self.assertTrue(torch.isnan(out).all())

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
        with sdpa_kernel(backends=[kernel]):
            sdp_math = torch.nn.functional.scaled_dot_product_attention(x, x, x, scale=-1.0 / 0.0001)
        self.assertEqual(ref_result, sdp_math)

class TestSDPACudaOnly(NNTestCase):
    """ Used to test CUDA only functionality of scaled_dot_product_attention
    Quarks:
        There is some trickiness with this function. Its runtime behavior
        is dependent on the CUDA architecture you are testing it on. See
        `PLATFORM_SUPPORTS_FUSED_ATTENTION` at the top of the file.
        Summary:
            Math: always supported
            FlashAttention: Supported on sm80 or newer hardware
            MemEfficientAttention: Supported on sm50 or newer hardware
    """
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    # TODO USED FOR TESTING THE SCORES, e.g. testing ALIBI we don't need this now
    def normalize_flash_attn_S(
        self,
        attn_unnorm,
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        is_dropout=False,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
        scale=None,
    ):
        """
        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim)
            k, v: (batch_size, seqlen_k, nheads, head_dim)
            key_padding_mask: (batch_size, seqlen_q)
            attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        Output:
            softmax_lse: (batch_size, nheads, seqlen_q)
            softmax_max: (batch_size, nheads, seqlen_q)
        """
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if causal:
            window_size = (window_size[0], 0)
        q, k, v = q.float(), k.float(), v.float()
        _, seqlen_q, _, head_dim = q.shape
        seqlen_k = k.shape[1]
        b = q.shape[0]
        from torch.nn.attention.bias import _calculate_scale
        scale = _calculate_scale(head_dim, scale)
        scores = torch.matmul(q.transpose(1, 2) * scale, k.permute(0, 2, 3, 1))
        if key_padding_mask is not None:
            scores.masked_fill_(~key_padding_mask.view(b, 1, 1, -1), float("-inf"))
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self.construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                q.device,
            )
            scores.masked_fill_(local_mask, float("-inf"))
        if attn_bias is not None:
            scores = scores + attn_bias.to(dtype=scores.dtype)
        block_size_n = _get_block_size_n(scores.device, head_dim, is_dropout, causal)
        scores_block = scores.split(block_size_n, dim=-1)
        lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
        lse = torch.logsumexp(lse_block, dim=-1)
        # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
        # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
        lse[lse == float("-inf")] = float("inf")
        scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
        cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
        attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
        attn_norm = torch.cat(
            [
                a * (torch.exp(m - lse)).unsqueeze(-1)
                for a, m in zip(attn_unnorm_block, cummax_block)
            ],
            dim=-1,
        )
        if query_padding_mask is not None:
            attn_norm.masked_fill_(~query_padding_mask.view(b, 1, -1, 1), 0.0)
            # attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
        return attn_norm.to(dtype=attn_unnorm.dtype)

    def construct_local_mask(self, seqlen_q, seqlen_k, window_size, query_padding_mask, key_padding_mask, device):
        # row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        row_idx = torch.arange(seqlen_q, device=device, dtype=torch.long).view(-1, 1)
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else key_padding_mask.sum(-1).view(-1, 1, 1, 1)
            # else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else query_padding_mask.sum(-1).view(-1, 1, 1, 1)
            # else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        if window_size[0] < 0:
            return col_idx > row_idx + sk - sq + window_size[1]
        else:
            sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
            return torch.logical_or(
                col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
                col_idx < row_idx + sk - sq - window_size[0],
            )

    def convert_flash_attn_S_to_softmax(
        self,
        S,
        seqlen_q,
        seqlen_k,
        query_padding_mask,
        key_padding_mask,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
    ):
        """FlashAttention stores the S matrix in a different way.
        Arguments:
            S: (batch_size, nheads, seqlen_q, seqlen_k)
            query_padding_mask: (batch_size, seqlen_q)
            key_padding_mask: (batch_size, seqlen_k)
        """
        if TEST_WITH_ROCM:
            return S
        b = S.shape[0]

        if causal:
            window_size = (window_size[0], 0)
        seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
        S_converted = S
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self.construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                S.device,
            )
            local_mask = F.pad(
                local_mask,
                (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
                value=True,
            )
            S_converted = S_converted.masked_fill(local_mask, 0.0)

        # Need to zero out things not in attention_mask in case S was initialized with random values
        # and some of those values aren't overwritten.
        seqlen_q_og = (
            query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
        )
        if query_padding_mask is not None:
            query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
            # S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
            S_converted = S_converted.masked_fill(~query_padding_mask.view(b, 1, -1, 1), 0.0)
        seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
            S_converted = S_converted.masked_fill(~key_padding_mask.view(b, 1, 1, -1), 0.0)
            # S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
        S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
        S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
        return S_converted[:, :, :seqlen_q, :seqlen_k]

    @skipIfRocm  # No cuDNN Attention
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_different_dk_dv(self, device):
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        seq_len = 640
        q_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, seq_len, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    @skipIfRocm  # No cuDNN Attention
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_gqa(self, device):
        batch = 4
        seq_len_q = 512
        seq_len_kv = 1024
        D = 128
        # Sample call to SDPA - GQ
        query = torch.rand(batch, 32, seq_len_q, D, device='cuda', dtype=torch.bfloat16)
        key = torch.rand(batch, 8, seq_len_kv, D, device='cuda', dtype=torch.bfloat16)
        # cuDNN supports h_k != h_v
        value = torch.rand(batch, 4, seq_len_kv, D, device='cuda', dtype=torch.bfloat16)
        with sdpa_kernel([SDPBackend.MATH]):
            output_math = scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)

        with self.assertRaisesRegex(RuntimeError, "No available kernel."):
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                output_cudnn = scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=False)

        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            output_cudnn = scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)

        self.assertEqual(output_math, output_cudnn)

    @skipIfRocm  # No cuDNN Attention
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_d256_heuristic(self, device):
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 256, 64
        seq_len = 640
        q_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, seq_len, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH], set_priority=True):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
            actual.backward(torch.randn_like(actual))
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    @skipIfRocm(msg="No cuDNN on ROCm")
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_fused_attention_different_dk_dv(self, device):
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        q_shape = SdpaShape(batch, num_heads, 1, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, 2, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, 2, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        # test that we do not dispatch to cuDNN for an unsupported case
        actual = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)


    @skipIfRocm  # No cuDNN Attention
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    def test_cudnn_attention_fail_d128(self, device):
        # Test that cuDNN attention dispatching correctly bails out on d > 128
        b, h = 1, 2
        s_q, s_kv = 128, 128
        d_qk, d_v = 128, 144

        q = torch.randn(b, h, s_q, d_qk, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, h, s_kv, d_qk, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, h, s_kv, d_v, device=device, dtype=torch.bfloat16)

        device_cap = torch.cuda.get_device_capability()
        ISSM90 = device_cap == (9, 0)
        ISSM100 = device_cap == (10, 0)
        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            # SM90/100 support d <= 256 as of cuDNN 9.5.1+
            if (ISSM90 or ISSM100) and torch.backends.cudnn.version() >= 90501:
                torch.nn.functional.scaled_dot_product_attention(q, k, v)
            else:
                with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                    torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @skipIfRocm(msg="No cuDNN on ROCm")
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_trivial_output_transpose(self, device):
        # see also: https://github.com/pytorch/pytorch/issues/134001
        x = torch.randn(2, 4, 1, 64, device='cuda', dtype=torch.float16, requires_grad=True)
        x2 = x.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            o = torch.nn.functional.scaled_dot_product_attention(x2, x2, x2).transpose(1, 2).reshape(2, 64, 4)
        o.backward(o)
        x_cpu = x.clone().cpu().detach()
        x_cpu.requires_grad = True
        x2_cpu = x_cpu.transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(x2_cpu, x2_cpu, x2_cpu).transpose(1, 2).reshape(2, 64, 4)
        o.backward(o)
        torch.testing.assert_close(x.grad, x_cpu.grad.cuda(), atol=7e-3, rtol=7e-3)

    @skipIfRocm  # No cuDNN Attention
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_nonmodulo64seqlen(self, device):
        # see also: https://github.com/pytorch/pytorch/issues/137347
        mask = torch.randint(0, 2, (2, 1, 157, 6404)).to(device="cuda", dtype=torch.bool)
        q = torch.randn(2, 32, 157, 128, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(2, 32, 6404, 128, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(2, 32, 6404, 128, device='cuda', dtype=torch.float16, requires_grad=True)
        q_cpu = q.detach().clone().cpu()
        k_cpu = k.detach().clone().cpu()
        v_cpu = v.detach().clone().cpu()
        q_cpu.requires_grad = True
        k_cpu.requires_grad = True
        v_cpu.requires_grad = True
        mask_cpu = mask.detach().clone().cpu()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
            out = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        out_cpu = nn.functional.scaled_dot_product_attention(
            q_cpu,
            k_cpu,
            v_cpu,
            attn_mask=mask_cpu,
            dropout_p=0.0,
            is_causal=False,
        )

        out.sum().backward()
        out_cpu.sum().backward()

        torch.testing.assert_close(q.grad, q_cpu.grad.cuda(), atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(k.grad, k_cpu.grad.cuda(), atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(v.grad, v_cpu.grad.cuda(), atol=3e-3, rtol=2e-3)

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cudnn Attention is not supported on this system")
    def test_cudnn_attention_preserves_query_layout(self, device):

        def test_attention(backend: SDPBackend, permute_order: list[list[int]]):
            BHSqD = [4, 16, 256, 64]
            BHSkvD = [4, 16, 512, 64]

            shape_q = [BHSqD[idx] for idx in permute_order]
            shape_kv = [BHSkvD[idx] for idx in permute_order]
            reverse = [permute_order.index(idx) for idx in range(4)]
            q = torch.randn(*shape_q, dtype=torch.bfloat16, device='cuda', requires_grad=True).permute(reverse)
            k = torch.randn(*shape_kv, dtype=torch.bfloat16, device='cuda', requires_grad=True).permute(reverse)
            v = torch.randn(*shape_kv, dtype=torch.bfloat16, device='cuda', requires_grad=True).permute(reverse)
            self.assertEqual(q.shape, BHSqD)
            self.assertEqual(k.shape, BHSkvD)
            self.assertEqual(v.shape, BHSkvD)

            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(q, k, v)
                self.assertTrue(out.permute(permute_order).is_contiguous())
                out.sum().backward()

        permute_orders = list()
        permutable = [0, 1, 2]
        permute_orders = itertools.permutations(permutable)

        for permute_order in permute_orders:
            test_attention(SDPBackend.CUDNN_ATTENTION, list(permute_order) + [3])

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("mask_dim", [1, 2, 3, 4])
    def test_mem_efficient_attention_mask_variants(self, device, mask_dim: list[int]):
        dtype = torch.float16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 8, 8, 64
        seq_len_q, seq_len_kv = 64, 15
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)

        if mask_dim == 1:
            mask = torch.randn((seq_len_kv,), device=device, dtype=dtype)
        elif mask_dim == 2:
            mask = torch.randn((seq_len_q, seq_len_kv), device=device, dtype=dtype)
        elif mask_dim == 3:
            mask = torch.randn((num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        elif mask_dim == 4:
            mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("dtype", [torch.float, torch.float16])
    def test_mem_eff_attention_non_contiguous_mask(self, device, dtype):
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 8, 8, 64
        seq_len_q, seq_len_kv = 64, 16
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        mask = torch.as_strided(mask, (batch, num_heads, seq_len_q, seq_len_kv), (0, 0, 0, 1))
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("dtype", [torch.float, torch.float16])
    def test_mem_eff_attention_long_sequence_mask(self, device, dtype):
        if torch.cuda.get_device_properties('cuda').total_memory < 80 * 2**30:
            unittest.skip("This test requires substatnial GPU memory.")
            return
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim = 1, 32, 64
        seq_len_q, seq_len_kv = 8192, 8192
        query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
        kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
        key, value = make_tensor(kv_shape), make_tensor(kv_shape)
        mask = torch.randn((batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, mask)
        out.sum().backward()

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_mem_eff_attention_non_contig_mask_bug(self, device):
        # Without the fix this produces `AssertionError: assert 0.07352933287620544 < 1e-07`
        # Shapes taken from repro
        query_size = (3, 16, 1, 128)
        query_strides = (2304, 128, 2048, 1)
        key_size = (3, 16, 14, 128)
        key_strides = (3584, 0, 256, 1)
        value_size = (3, 16, 14, 128)
        value_strides = (3584, 0, 256, 1)
        attention_mask_size = (3, 1, 1, 14)
        attn_mask_strides = (14, 14, 14, 1)

        # Calculate the number of elements needed for each tensor
        query_num_elements = max(size * stride for size, stride in zip(query_size, query_strides))
        key_num_elements = max(size * stride for size, stride in zip(key_size, key_strides))
        value_num_elements = max(size * stride for size, stride in zip(value_size, value_strides))
        attention_mask_num_elements = max(size * stride for size, stride in zip(attention_mask_size, attn_mask_strides))

        # Create the tensors with the specified sizes and strides
        query = torch.randn(query_num_elements, device=device).as_strided(query_size, query_strides)
        key = torch.randn(key_num_elements, device=device).as_strided(key_size, key_strides)
        value = torch.randn(value_num_elements, device=device).as_strided(value_size, value_strides)
        bias = torch.randn(attention_mask_num_elements, device=device).as_strided(attention_mask_size, attn_mask_strides)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F.scaled_dot_product_attention(query, key, value, bias)
            out_contig = F.scaled_dot_product_attention(query, key, value, bias.contiguous())

        max_diff = (out - out_contig).abs().mean()
        self.assertTrue(max_diff.item() < 1e-7)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Fused SDPA was not built for this system")
    def test_singelton_head_dim_stride_ne_1(self, device):
        query = torch.tensor([[[[1, 2]]]], dtype=torch.float16, device=device)
        query = query.transpose(-1, -2)
        key = torch.tensor([[[[1]]]], dtype=torch.float16, device=device)
        value = torch.tensor([[[[1]]]], dtype=torch.float16, device=device)

        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
            scaled_dot_product_attention(query, key, value)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("is_contiguous", [True, False])
    def test_scaled_dot_product_attention_fused_kernels_packed(self, device, type: str, is_contiguous: bool):
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

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

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "Fused SDPA was not built for this system")
    @unittest.skipIf("TORCH_CUDNN_SDPA_NESTED_TENSOR_ENABLED" not in os.environ, "cuDNN Nested Tensor support not enabled")
    @parametrize("type", ["nested"])
    @parametrize("is_contiguous", [True])
    def test_scaled_dot_product_attention_cudnn_nested(self, device, type: str, is_contiguous: bool):
        if TEST_WITH_ROCM and type == 'nested':
            self.skipTest("ROCM does not support efficient attention on nested tensors, for now")
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 8, 64, 16, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

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

        with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)
        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("type", ["dense", "nested"])
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    def test_scaled_dot_product_attention_fused_kernels_packed_accuracy(self, device, type: str, fused_kernel: str):
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

        with sdpa_kernel(backends=[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, attn_mask=None, dropout_p=0.0, is_causal=False)

        with sdpa_kernel(backends=[SDPBackend.MATH]):
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

        self.assertEqual(math_ref_test, math_ref_lp_test, atol=8e-3, rtol=7e-3)
        self.assertEqual(actual_test, math_ref_test, atol=7e-3, rtol=7e-3)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Efficient Attention was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    @parametrize("is_causal", [True, False])
    def test_sdp_mem_efficient_grad_against_math(self, device, contiguous_inputs: bool, is_causal: bool):
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor(SdpaShape(batch_size, num_heads, seq_len, head_dim))
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

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, is_causal)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out_lp = torch.nn.functional.scaled_dot_product_attention(
                query_lp, key_lp, value_lp, None, 0.0, is_causal)

        rand_upward = torch.rand_like(out)
        rand_upward_lp = rand_upward.to(torch.float32)

        out.backward(rand_upward)
        out_lp.backward(rand_upward_lp)

        # Cast up and compare
        self.assertEqual(qkv.grad, qkv_lp.grad.to(torch.float64), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention was not built for this system")
    @parametrize("contiguous_inputs", [True, False])
    @parametrize("is_causal", [True, False])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdp_flash_attention_grad_against_math(self, device, contiguous_inputs: bool, is_causal: bool, dtype: torch.dtype):
        batch_size, seq_len, num_heads, head_dim = 4, 4, 2, 16
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device,
                              dtype=torch.float64, requires_grad=True, packed=True)

        qkv = make_tensor(SdpaShape(batch_size, num_heads, seq_len, head_dim))
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

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            out = torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, is_causal)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
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
        if TEST_WITH_ROCM:
            atol = 9e-4 if dtype == torch.float16 else 9e-3
        self.assertEqual(qkv.grad, qkv_lp.grad.to(torch.float64), atol=atol, rtol=rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Platform does not support fused SDPA")
    @parametrize("type", ["dense", "nested"])
    def test_fused_sdp_choice(self, device, type: str):
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float16, packed=True, requires_grad=True)

        qkv = make_tensor(shape, type=type)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # TODO we are currently disabling this by default, lets assert that this returns
        # FlashAttention, we need to change when we make remove opt-in for cudnn
        if type != "nested" and PLATFORM_SUPPORTS_CUDNN_ATTENTION and SM90OrLater:
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.FLASH_ATTENTION.value)
            with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
                self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.CUDNN_ATTENTION.value)
        elif PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.FLASH_ATTENTION.value)
        elif type != "nested" and PLATFORM_SUPPORTS_CUDNN_ATTENTION:  # e.g., we're on Windows
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.EFFICIENT_ATTENTION.value)
            with sdpa_kernel(backends=[SDPBackend.CUDNN_ATTENTION]):
                self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.CUDNN_ATTENTION.value)
        else:
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.EFFICIENT_ATTENTION.value)

        # Change dtype to float32 so that efficient attention should get chosen
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float32, packed=True)

        qkv = make_tensor(shape, type=type)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION.value

    @skipIfRocm  # Missing triton.float32 ("triton" prefix is to locate skipped UTs), and deterministic algo
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Platform does not support fused SDPA")
    @parametrize("warn_only", [True, False])
    def test_sdp_choice_with_determinism(self, device, warn_only):
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float32, packed=False)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                assert torch._fused_sdp_choice(query, key, value) == SDPBackend.EFFICIENT_ATTENTION.value

    @skipIfRocm
    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_CUDNN_ATTENTION, "cuDNN Attention is not supported on this system")
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Platform does not support fused SDPA")
    @parametrize("use_compile", [True, False])
    def test_fused_sdp_priority_order(self, device, use_compile):
        @torch.compile
        def compiled_func(order):
            with sdpa_kernel(order, set_priority=True):
                out = scaled_dot_product_attention(q, q, q)
            return out

        q = torch.randn(64, 8, 1024, 64, dtype=torch.half, device='cuda')
        default_order = torch._C._get_sdp_priority_order()
        orders = [[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION],
                  [SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
                  [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH],
                  [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]]
        import time
        times = list()
        for order in orders:
            if use_compile:
                compiled_func(order)
            else:
                with sdpa_kernel(order, set_priority=True):
                    scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_compile:
                compiled_func(order)
            else:
                with sdpa_kernel(order, set_priority=True):
                    scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        self.assertTrue(times[0] < times[1], "expected cuDNN SDPA to be faster than Math backend.")
        self.assertTrue(times[1] > times[2], "expected Eff Attn backend to faster than Math backend.")
        self.assertTrue(times[3] < times[2], "expected Flash Attn backend to faster than Math backend.")
        self.assertTrue(times[0] < times[2], "expected cuDNN Attn backend to faster than Eff Attn backend.")
        reset_order = torch._C._get_sdp_priority_order()
        self.assertEqual(default_order, reset_order, "expected SDPA context manager to reset priority order.")

    @skipIfRocm  # Missing deterministic algo
    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", PLATFORM_SPECIFIC_SDPA)
    @parametrize("warn_only", [True, False])
    def test_fused_backwards_throws_determinism_warning(self, device, warn_only, fused_kernel):
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float16, packed=False, requires_grad=True)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        kernel_name = "Memory Efficient attention" if fused_kernel == SDPBackend.EFFICIENT_ATTENTION else \
            "Flash Attention" if fused_kernel == SDPBackend.FLASH_ATTENTION else "cuDNN Attention"
        warning_context = (
            self.assertWarnsRegex(
                UserWarning,
                f"{kernel_name} defaults to a non-deterministic algorithm.",
            )
            if warn_only
            else contextlib.nullcontext()
        )
        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdpa_kernel(backends=[fused_kernel]):
                with warning_context:
                    if warn_only or fused_kernel != SDPBackend.CUDNN_ATTENTION:
                        torch.nn.functional.scaled_dot_product_attention(query, key, value).sum().backward()
                    else:
                        # cuDNN attention has no deterministic fallback
                        self.assertRaises(RuntimeError, lambda:
                                          torch.nn.functional.scaled_dot_product_attention(query, key, value).sum().backward())

    @unittest.skip("This test is not behaving deterministaclly non-deterministaclly on CI/CD")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Platform does not support fused SDPA")
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

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
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
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA")
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    @parametrize("batch_size", [1, 8])
    @parametrize(
        "seq_len_q",
        [8, 103, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80 else [4, 8, 256, 512],
    )
    @parametrize(
        "seq_len_k",
        [8, 103, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80 else [4, 8, 256, 512],
    )
    @parametrize(
        "head_dim",
        [8, 16, 96, 128] if MEM_EFF_CAPABILITY_MATCHES_SM80 and not isSM120Device else [8, 16, 32, 64],
    )
    @parametrize("is_causal", [False, True])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize(
        "dtype",
        (
            [torch.float16, torch.bfloat16, torch.float32]
            if MEM_EFF_CAPABILITY_MATCHES_SM80
            else [torch.float16, torch.float32]
        ),
    )
    @parametrize("scale", [None, "l1"])
    @tf32_enabled()
    def test_mem_efficient_attention_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int, seq_len_k: int,
                                                       head_dim: int, is_causal: bool, dropout_p: float, dtype: torch.dtype,
                                                       scale: str):
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, p, seed, offset, device=device):
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, p, seed, offset)
            # On ROCM _fill_mem_eff_dropout_mask fills 0.5 if (prng > p) otherwise -0.5 to the tensor
            tester_p = p if not TEST_WITH_ROCM else 0.0
            mask = (rand_uniform > tester_p).to(torch.float32)
            return mask
        if max(seq_len_q, seq_len_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory < 40 * 2**30:
            unittest.skip("Reference implementation OOM")
            return
        if TEST_WITH_ROCM and seq_len_q * seq_len_k * head_dim * batch_size > 1024 * 1024 * 128:
            torch.cuda.empty_cache()  # Prevent memory fragmentation
        seed = 42
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        higher_precision_dtype = torch.float64
        query_ref, key_ref, value_ref = query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        # Create real output
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # Set the seed and run the kernel
            torch.manual_seed(seed)
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        if dropout_p == 0.0:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref,
                                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query, key, value,
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
                query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=dropout_mask)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        grads = torch.autograd.grad(out, (query, key, value), upstream_grad)
        grads_ref_lp = torch.autograd.grad(out_lp_ref, (query, key, value), upstream_grad)
        grads_ref = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref), upstream_grad)

        fudge_factors = {
            'out': 3.0 ,
            'grad_query': 150.0 ,
            'grad_key': 25.0,
            'grad_value': 8.5,
        }
        if TEST_WITH_ROCM:
            fudge_factors['grad_key'] = 45.0
            fudge_factors['grad_query'] = 360.0
            if seq_len_k >= 1024:
                fudge_factors['grad_key'] = 70.0
            if seq_len_k >= 2048:
                fudge_factors['grad_key'] = 160.0
                fudge_factors['grad_query'] = 650.0
            if dtype == torch.float32:
                fudge_factors['grad_key'] = 90.0

        check_out_and_grad(
            (out_ref, out_lp_ref, out),
            *zip(grads_ref, grads_ref_lp, grads),
            fudge_factors=fudge_factors,
        )

    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Does not support SDPA")
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    @parametrize("batch_size", [1, 8])
    @parametrize(
        "seq_len_q",
        [8, 312, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80 else [8, 152, 512],
    )
    @parametrize(
        "seq_len_k",
        [8, 408, 1024, 2048] if MEM_EFF_CAPABILITY_MATCHES_SM80 else [8, 37, 512],
    )
    @parametrize(
        "head_dim",
        [8, 16, 96, 128] if MEM_EFF_CAPABILITY_MATCHES_SM80 and not isSM120Device else [8, 16, 32, 64],
    )
    @parametrize("is_causal", [False])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize(
        "dtype",
        (
            [torch.float16, torch.bfloat16, torch.float32]
            if MEM_EFF_CAPABILITY_MATCHES_SM80
            else [torch.float16, torch.float32]
        ),
    )
    @parametrize("scale", [None, "l1"])
    @tf32_enabled()
    def test_mem_efficient_attention_attn_mask_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int,
                                                                 seq_len_k: int, head_dim: int, is_causal: bool,
                                                                 dropout_p: float, dtype: torch.dtype,
                                                                 scale: str):
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, p, seed, offset, device=device):
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, p, seed, offset)
            # On ROCM _fill_mem_eff_dropout_mask fills 0.5 if (prng > p) otherwise -0.5 to the tensor
            tester_p = p if not TEST_WITH_ROCM else 0.0
            mask = (rand_uniform > tester_p).to(torch.float32)
            return mask
        if max(seq_len_q, seq_len_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory < 40 * 2**30:
            unittest.skip("Reference implementation OOM")
            return
        if TEST_WITH_ROCM and seq_len_q * seq_len_k * head_dim * batch_size > 1024 * 1024 * 128:
            torch.cuda.empty_cache()  # Prevent memory fragmentation
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

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = query_key_value_clones(query, key, value, dtype=higher_precision_dtype)
        attn_mask_ref = attn_mask.detach().to(higher_precision_dtype).requires_grad_(True)

        # Create real output
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            # Set the seed and run the kernel
            torch.manual_seed(seed)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask, dropout_p=dropout_p,
                                                 is_causal=is_causal, scale=scale)

        if dropout_p == 0.0:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref, attn_mask_ref,
                                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query, key, value, attn_mask,
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
                query, key, value, attn_mask,
                dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=dropout_mask)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        grads = torch.autograd.grad(out, (query, key, value, attn_mask), upstream_grad)
        grads_ref_lp = torch.autograd.grad(out_lp_ref, (query, key, value, attn_mask), upstream_grad)
        grads_ref = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref, attn_mask_ref), upstream_grad)

        fudge_factors = {
            "out": 4,
            "grad_query": 160.0,
            "grad_key": 25.0,
            "grad_value": 8.0,
            "grad_attn_mask": 45.0,
        }
        if TEST_WITH_ROCM:
            fudge_factors['grad_key'] = 45.0
            fudge_factors['grad_query'] = 360.0
            if seq_len_k >= 1024:
                fudge_factors['grad_key'] = 70.0
            if seq_len_k >= 2048:
                fudge_factors['grad_key'] = 160.0
                fudge_factors['grad_query'] = 650.0
            if dtype == torch.float32:
                fudge_factors['grad_key'] = 90.0

        check_out_and_grad(
            (out_ref, out_lp_ref, out),
            *zip(grads_ref, grads_ref_lp, grads),
            fudge_factors=fudge_factors,
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    @unittest.skipIf(IS_JETSON, "causing sigkill on Jetson")
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [4, 143, 2048])
    @parametrize("seq_len_k", [4, 127, 579, 2048])
    @parametrize("head_dim", [8, 203, 256])
    @parametrize("is_causal", [True, False])
    @parametrize("dropout_p", [0.0, 0.22, 0.48])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("scale", [None, "l1"])
    @parametrize("enable_gqa", [True, False])
    @parametrize("n_heads", [[16, 8], [10, 2]])
    @tf32_enabled()
    def test_flash_attention_vs_math_ref_grads(self, device, batch_size: int, seq_len_q: int, seq_len_k: int,
                                               head_dim: int, is_causal: bool, dropout_p: float, dtype: torch.dtype,
                                               scale: str, enable_gqa: bool, n_heads: list[int]):
        if isSM8XDevice or isSM120Device and head_dim in range(193, 256 + 1):
            self.skipTest("Flash attention on sm86, sm87, and sm89 for headdim > 192 currently disabled")
        if is_causal and seq_len_q != seq_len_k:
            self.skipTest("Flash V2 does not accept is_casual when seq_len_q != seq_len_k")
        if TEST_WITH_ROCM and seq_len_q >= 1024 and seq_len_k >= 1024 and batch_size > 1:
            torch.cuda.empty_cache()  # Prevent memory fragmentation
        if max(seq_len_q, seq_len_k) >= 2048 and torch.cuda.get_device_properties('cuda').total_memory < 40 * 2**30:
            unittest.skip("Reference implementation OOM")
            return
        if TEST_WITH_CK and dropout_p != 0:
            self.skipTest("CK does not support tensor format dropout masks")
        if TEST_WITH_CK and head_dim > 128:
            self.skipTest("CK does not support head dims over 128")

        scale = scale if scale is None else (1 / head_dim)
        num_heads_q = num_heads_kv = 4
        if enable_gqa:
            num_heads_q = n_heads[0]
            num_heads_kv = n_heads[1]

        query = torch.rand(batch_size, num_heads_q, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, num_heads_kv, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, num_heads_kv, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        is_dropout = dropout_p > 0.0

        if not is_dropout:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(
                    query_ref, key_ref, value_ref, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(
                    query, key, value, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        else:
            # Problem: We pad sizes in the composite region of the top level SDPA. But we need the
            # Debug mask when have dropout. So I am going to manualy pad up here when testing dropout
            q_padded, q_og_size = pad_last_dim(query, 8)
            k_padded, k_og_size = pad_last_dim(key, 8)
            v_padded, v_og_size = pad_last_dim(value, 8)
            # scale needs to be calculated on the og head_size
            if scale is None:
                scale = 1 / math.sqrt(q_og_size)
            output_tuple = torch.ops.aten._scaled_dot_product_flash_attention(
                q_padded, k_padded, v_padded, dropout_p=dropout_p, is_causal=is_causal, scale=scale, return_debug_mask=is_dropout)
            out = output_tuple[0]
            out = out[..., :v_og_size]
            # Build dropout_mask
            dbug_mask = output_tuple[-1]
            query_padding_mask = torch.ones(
                batch_size, seq_len_q, device=device, dtype=torch.bool)
            key_padding_mask = torch.ones(
                batch_size, seq_len_k, device=device, dtype=torch.bool)

            softmax_mask = self.convert_flash_attn_S_to_softmax(
                dbug_mask, seq_len_q, seq_len_k, query_padding_mask, key_padding_mask,
                causal=is_causal)[:, :, :seq_len_q, :seq_len_k]
            dropout_mask = softmax_mask >= 0
            # High Precision Math Reference
            out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, dropout_mask=dropout_mask, enable_gqa=enable_gqa)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=dropout_mask, enable_gqa=enable_gqa)[0]

        upstream_grad = torch.rand_like(out, requires_grad=False)

        # backward for flash attention on sm86, sm87, and sm89 for headdim >= 193 currently disabled
        if isSM8XDevice or isSM120Device and head_dim in range(193, 256):
            self.assertRaises(RuntimeError, lambda: out.backward(upstream_grad))
            return

        grads = torch.autograd.grad(out, (query, key, value), upstream_grad)
        grads_ref_lp = torch.autograd.grad(out_lp_ref, (query, key, value), upstream_grad)
        grads_ref = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref), upstream_grad)

        fudge_factors = {
            'out': 4,
            'grad_query': 180.0,
            'grad_key': 16,
            'grad_value': 4,
        }
        if TEST_WITH_ROCM:
            fudge_factors['grad_key'] = 45.0
            fudge_factors['grad_query'] = 360.0
            if seq_len_k >= 1024:
                fudge_factors['grad_key'] = 70.0
            if seq_len_k >= 2048:
                fudge_factors['grad_key'] = 190.0
                fudge_factors['grad_query'] = 650.0
                if seq_len_q >= 2048:
                    fudge_factors['grad_query'] = 1100.0
            if dtype == torch.float32:
                fudge_factors['grad_key'] = 90.0

        check_out_and_grad(
            (out_ref, out_lp_ref, out),
            *zip(grads_ref, grads_ref_lp, grads),
            fudge_factors=fudge_factors,
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Does not support SDPA or pre-SM80 hardware",
    )
    @parametrize("batch_size", [1, 8])
    @parametrize("seq_len_q", [256, 1024])
    @parametrize("seq_len_k", [256, 1024])
    @parametrize("head_dim", [32, 64])
    @parametrize("is_causal", [True, False])
    @parametrize("dropout_p", [0.0, 0.22])
    @parametrize("dtype", [torch.float16])
    @parametrize("scale", [None, "l1"])
    @parametrize("fused_kernel", PLATFORM_SPECIFIC_SDPA)
    @tf32_enabled()
    def test_fused_attention_vs_math_ref_grads_cudagraph(self, device, batch_size: int,
                                                         seq_len_q: int, seq_len_k: int,
                                                         head_dim: int,
                                                         is_causal: bool,
                                                         dropout_p: float,
                                                         dtype: torch.dtype,
                                                         scale: str,
                                                         fused_kernel: SDPBackend):
        def _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len, dropout_p, seed, offset, device=device):
            mask = torch.empty((batch_size, n_heads, q_len, kv_len), device=device, dtype=torch.float32)
            rand_uniform = torch._fill_mem_eff_dropout_mask_(mask, dropout_p, seed, offset)
            # On ROCM _fill_mem_eff_dropout_mask fills 0.5 if (prng > p) otherwise -0.5 to the tensor
            tester_p = dropout_p if not TEST_WITH_ROCM else 0.0
            mask = (rand_uniform > tester_p).to(torch.float32)
            return mask

        def get_dropout_mask(output, fused_kernel, batch_size, n_heads, q_len, kv_len, dropout_p, device=device):
            if fused_kernel == SDPBackend.EFFICIENT_ATTENTION:
                output_seed, output_offset = output_tuple[2], output_tuple[3]
                output_seed = output_seed.item()
                output_offset = output_offset.item()
                return _get_mem_eff_drop_mask(batch_size, n_heads, q_len, kv_len,
                                              dropout_p, output_seed, output_offset, device=device)
            else:
                # Build dropout_mask
                dbug_mask = output_tuple[-1]
                query_padding_mask = torch.ones(
                    batch_size, seq_len_q, device=device, dtype=torch.bool)
                key_padding_mask = torch.ones(
                    batch_size, seq_len_k, device=device, dtype=torch.bool)

                softmax_mask = self.convert_flash_attn_S_to_softmax(
                    dbug_mask, seq_len_q, seq_len_k, query_padding_mask, key_padding_mask,
                    causal=is_causal)[:, :, :seq_len_q, :seq_len_k]
                dropout_mask = softmax_mask >= 0
                return dropout_mask

        if fused_kernel == SDPBackend.FLASH_ATTENTION and is_causal and seq_len_q != seq_len_k:
            self.skipTest("Flash V2 does not accept is_casual when seq_len_q != seq_len_k")

        seed = 42
        n_heads = 4
        query = torch.rand(batch_size, n_heads, seq_len_q, head_dim,
                           device=device, dtype=dtype, requires_grad=True)
        key = torch.rand(batch_size, n_heads, seq_len_k, head_dim, device=device,
                         dtype=dtype, requires_grad=True)
        value = torch.rand(batch_size, n_heads, seq_len_k, head_dim,
                           device=device, dtype=dtype, requires_grad=True)

        fused_op = (torch.ops.aten._scaled_dot_product_efficient_attention
                    if fused_kernel == SDPBackend.EFFICIENT_ATTENTION else torch.ops.aten._scaled_dot_product_flash_attention
                    if fused_kernel == SDPBackend.FLASH_ATTENTION else torch.ops.aten._scaled_dot_product_cudnn_attention)

        higher_precision_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        query_ref, key_ref, value_ref = query_key_value_clones(query, key, value, dtype=higher_precision_dtype)

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        # Set the global seed before capture
        torch.manual_seed(seed)
        kwargs = {"dropout_p": dropout_p, "is_causal": is_causal}
        if fused_kernel == SDPBackend.EFFICIENT_ATTENTION:
            kwargs["compute_log_sumexp"] = True
            kwargs["attn_bias"] = None
        if fused_kernel == SDPBackend.FLASH_ATTENTION:
            kwargs['return_debug_mask'] = dropout_p > 0.0
        if fused_kernel == SDPBackend.CUDNN_ATTENTION:
            kwargs["compute_log_sumexp"] = True
            kwargs["attn_bias"] = None
            if "return_debug_mask" in kwargs:
                kwargs.pop("return_debug_mask")
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
            torch.rand_like(query, device=query.device)  # test non-zero intragraph offset
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

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            if dropout_p == 0.0:
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(query_ref, key_ref, value_ref,
                                                         dropout_p=dropout_p, is_causal=is_causal)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(query, key, value,
                                                            dropout_p=dropout_p, is_causal=is_causal)
            # cuDNN attention doesn't support returning dropout mask
            elif fused_kernel != SDPBackend.CUDNN_ATTENTION:
                # Create the dropout_mask
                dropout_mask = get_dropout_mask(output_tuple, fused_kernel, batch_size,
                                                n_heads, seq_len_q, seq_len_k, dropout_p, device)
                # High Precision Math Reference
                out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query_ref, key_ref, value_ref, dropout_p=dropout_p, is_causal=is_causal,
                    dropout_mask=dropout_mask)[0]
                # Low Precision Math Reference
                out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                    query, key, value, dropout_p=dropout_p, is_causal=is_causal,
                    dropout_mask=dropout_mask)[0]

        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1):
            grads = torch.autograd.grad(out, (query, key, value), upstream_grad)
        g1.replay()
        if fused_kernel != SDPBackend.CUDNN_ATTENTION or dropout_p == 0.0:
            grads_ref_lp = torch.autograd.grad(out_lp_ref, (query, key, value), upstream_grad)
            grads_ref = torch.autograd.grad(out_ref, (query_ref, key_ref, value_ref), upstream_grad)

            check_out_and_grad(
                (out_ref, out_lp_ref, out),
                *zip(grads_ref, grads_ref_lp, grads),
                fudge_factors={
                    'out': 3.0,
                    'grad_query': 110.0,
                    'grad_key': 8.0,
                    'grad_value': 3.0,
                }
            )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("fused_kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
    def test_fused_kernels_seq_len_1_inputs(self, device, fused_kernel):
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float16)
        batch, num_heads, head_dim = 32, 16, 64
        seq_lens = torch.randint(low=1, high=32, size=(batch,))
        # make sure some seq_lens are 1
        num_ones = 10
        indices = torch.randint(low=0, high=batch, size=(num_ones,))
        seq_lens.scatter_(0, indices, 1)

        shape = SdpaShape(batch, num_heads, seq_lens.tolist(), head_dim)
        query = rand_nested_tensor(shape)
        key = rand_nested_tensor(shape)
        value = rand_nested_tensor(shape)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        with sdpa_kernel(backends=[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(torch.float16), atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FUSED_ATTENTION, "Fused SDPA was not built for this system")
    @parametrize("kernel", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] if
                 PLATFORM_SUPPORTS_FLASH_ATTENTION else [SDPBackend.EFFICIENT_ATTENTION])
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
        is_efficient = kernel == SDPBackend.EFFICIENT_ATTENTION
        dtype = torch.float32 if is_efficient else torch.float16
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=dtype)
        batch, num_heads, head_dim = 32, 8, 64
        head_dim_v = 32 if is_efficient else head_dim
        if TEST_WITH_ROCM and head_dim != head_dim_v:
            self.skipTest("head_dim != head_dim_v unsupported on ROCm for now")
            return
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

        q_shape = SdpaShape(batch_q, num_heads_q, seq_lens_q, head_dim)
        k_shape = SdpaShape(batch_k, num_heads_k, seq_lens_kv, head_dim)
        v_shape = SdpaShape(batch_v, num_heads_v, seq_lens_kv, head_dim_v)

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

        with sdpa_kernel(backends=[kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query_expanded.contiguous(), key_expanded.contiguous(), value_expanded.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1.5e-3, rtol=1e-2)

    @skipIfRocm(msg="Efficient Attention on ROCM does not support head_dim != head_dim_v for now.")
    @unittest.skipIf(not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION, "Fused SDPA was not built for this system")
    def test_fused_kernels_nested_broadcasting_query_dense(self, device):
        rand_nested_tensor = partial(rand_sdpa_tensor, type="nested", device=device, dtype=torch.float32)
        batch, num_heads, head_dim, head_dim_v = 32, 16, 64, 96
        seq_lens = torch.randint(low=1, high=32, size=(batch,)).tolist()
        q_shape = (1, 1, num_heads, head_dim)
        k_shape = SdpaShape(batch, num_heads, seq_lens, head_dim)
        v_shape = SdpaShape(batch, 1, seq_lens, head_dim_v)

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

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query_expanded.contiguous(), key.contiguous(), value_expanded.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=1e-3, rtol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Does not support SDPA or pre-SM80 hardware")
    @parametrize("batch_size", [8, 32])
    @parametrize("max_seq_len_q", [32, 256])
    @parametrize("max_seq_len_kv", [32, 256])
    @parametrize("head_dim", [8, 64])
    @parametrize("dropout_p", [0.0, 0.1])
    @parametrize("dtype", [torch.float16])
    @parametrize("scale", [None, "l1"])
    @parametrize("is_causal", [True, False])
    def test_flash_attention_vs_math_ref_grads_nestedtensor(self, device, batch_size: int, max_seq_len_q: int, max_seq_len_kv: int,
                                                            head_dim: int, dropout_p: float, dtype: torch.dtype,
                                                            scale: str, is_causal: bool):
        if is_causal:
            # TODO we should support this
            self.assertRaisesRegex(RuntimeError, "Nested tensors for query / key are not supported when is_causal=True")
            return
        scale = scale if scale is None else (1 / head_dim)
        n_heads = 4
        seq_lens_q = torch.randint(low=1, high=max_seq_len_q, size=(batch_size,))
        # Set one entry to max length
        seq_lens_q[torch.randint(0, batch_size, size=(1,))] = max_seq_len_q
        seq_lens_kv = torch.randint(low=1, high=max_seq_len_kv, size=(batch_size,))
        seq_lens_kv[torch.randint(0, batch_size, size=(1,))] = max_seq_len_kv

        def rand_nt(sequence_list, num_heads, head_dim):
            tensors = [torch.rand((num_heads, seq_len, head_dim)) for seq_len in sequence_list]
            return torch.nested.nested_tensor(tensors, requires_grad=True, device=device, dtype=dtype)

        query = rand_nt(seq_lens_q, n_heads, head_dim)
        key = rand_nt(seq_lens_kv, n_heads, head_dim)
        value = rand_nt(seq_lens_kv, n_heads, head_dim)

        # Run the math kernel on low precision references
        query_ref_lp = query.detach().clone().requires_grad_(True)
        key_ref_lp = key.detach().clone().requires_grad_(True)
        value_ref_lp = value.detach().clone().requires_grad_(True)

        query_ref = query.detach().clone().to(torch.float32).requires_grad_(True)
        key_ref = key.detach().clone().to(torch.float32).requires_grad_(True)
        value_ref = value.detach().clone().to(torch.float32).requires_grad_(True)

        is_dropout = dropout_p > 0.0

        if not is_dropout:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                # High Precision Math Reference
                out_ref = F.scaled_dot_product_attention(
                    query_ref, key_ref, value_ref, is_causal=is_causal, scale=scale)
                # Low Precision Math Reference
                out_lp_ref = F.scaled_dot_product_attention(
                    query_ref_lp, key_ref_lp, value_ref_lp, is_causal=is_causal, scale=scale)
        else:
            # Create real output
            output_tuple = torch.ops.aten._scaled_dot_product_flash_attention(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal,
                scale=scale, return_debug_mask=is_dropout)
            out = output_tuple[0]
            dbug_mask = output_tuple[-1]

            query_padding_mask = torch.arange(max_seq_len_q).unsqueeze(0).expand(
                batch_size, max_seq_len_q
            ) < seq_lens_q.unsqueeze(-1)
            query_padding_mask = query_padding_mask.to("cuda")

            key_padding_mask = torch.arange(max_seq_len_kv).unsqueeze(0).expand(
                batch_size, max_seq_len_kv
            ) < seq_lens_kv.unsqueeze(-1)
            key_padding_mask = key_padding_mask.to("cuda")

            softmax_mask = self.convert_flash_attn_S_to_softmax(
                dbug_mask, max_seq_len_q, max_seq_len_kv, query_padding_mask, key_padding_mask, causal=is_causal)
            dropout_mask = softmax_mask >= 0
            nt_stack = []
            for tensor_component in range(batch_size):
                batch_stack = []
                for head in range(n_heads):
                    batch_stack.append(dropout_mask[tensor_component, head,
                                                    0:seq_lens_q[tensor_component],
                                                    0:seq_lens_kv[tensor_component]].unsqueeze(0))
                nt_stack.append(torch.cat(batch_stack))
            nested_dropout_mask = torch.nested.nested_tensor(nt_stack)
            # High Precision Math Reference
            out_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref, key_ref, value_ref, dropout_p=dropout_p,
                is_causal=is_causal, scale=scale, dropout_mask=nested_dropout_mask)[0]
            # Low Precision Math Reference
            out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
                query_ref_lp, key_ref_lp, value_ref_lp, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                dropout_mask=nested_dropout_mask)[0]

        upstream_grad = out.detach().clone().contiguous()

        out.backward(upstream_grad)
        out_ref.backward(upstream_grad.to(out_ref.dtype))
        out_lp_ref.backward(upstream_grad.to(out_lp_ref.dtype))

        dropout_fudge_factor = 1.0 if dropout_p == 0.0 else 2.0
        check_out_and_grad(
            (out_ref, out_lp_ref, out),
            (query_ref, query_ref_lp, query),
            (key_ref, key_ref_lp, key),
            (value_ref, value_ref_lp, value),
            fudge_factors={
                'out': 1.5 * dropout_fudge_factor,
                'grad_query': 12.0 * dropout_fudge_factor,
                'grad_key': 1.5 * dropout_fudge_factor,
                'grad_value': 2.0 * dropout_fudge_factor,
            }
        )

class TestSDPAXpuOnly(NNTestCase):
    """ Used to test XPU only functionality of scaled_dot_product_attention
    Mostly migrate from TestSDPACudaOnly in test/test_transformers.py

    Note that as SDPBackend.OVERRIDEABLE is not managed by sdpa_kernel so that
    math ref has to be called explicitly via torch.ops.aten._scaled_dot_product_attention_math.
    """

    @parametrize("type", ["dense"])
    @parametrize("dropout", [0.0, 0.7])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.half])
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_xpu(self, device, type: str, dropout: float, dtype: torch.dtype):
        # Migrate from test_fused_sdp_choice_cpu
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype)
        size = SdpaShape(2, 8, 128, 64)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        if dropout > 0.0 or dtype not in [torch.float32, torch.bfloat16, torch.float16]:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.MATH.value
        else:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.OVERRIDEABLE.value

    def test_fused_attention_different_dk_dv(self, device):
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        q_shape = SdpaShape(batch, num_heads, 1, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, 2, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, 2, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        # test that we do not dispatch to onednn for an unsupported case
        actual = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

        math_ref = torch.ops.aten._scaled_dot_product_attention_math(
            query.float(), key.float(), value.float(), attn_mask=None, dropout_p=0.0, is_causal=False)[0]

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    def test_onednn_attention_fail_d256(self, device):
        # Test that onednn graph attention dispatching correctly bails out on d > 256
        b, h = 1, 2
        s_q, s_kv = 128, 128
        d_qk, d_v = 512, 512

        q = torch.randn(b, h, s_q, d_qk, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, h, s_kv, d_qk, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, h, s_kv, d_v, device=device, dtype=torch.bfloat16)

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                _ = F.scaled_dot_product_attention(q, k, v)

    @parametrize("type", ["dense"])
    @parametrize("is_contiguous", [True, False])
    def test_scaled_dot_product_attention_fused_kernels_packed(self, device, type: str, is_contiguous: bool):
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

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

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        math_ref = torch.ops.aten._scaled_dot_product_attention_math(
            query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=None, dropout_p=0.0, is_causal=False)[0]

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @parametrize("fused_kernel", [SDPBackend.MATH, SDPBackend.OVERRIDEABLE])
    @parametrize("dtype", [torch.half, torch.bfloat16, torch.float32])
    @parametrize("batch_size,n_head,q_size,kv_size,head_dim", [
        (2, 5, 9216, 9216, 64),
        (2, 5, 9216, 77, 64),
        (2, 10, 2304, 2304, 64),
        (2, 10, 2304, 77, 64),
        (2, 20, 576, 576, 64),
        (2, 20, 576, 77, 64),
        (2, 20, 144, 144, 64),
        (2, 20, 144, 77, 64),
        (1, 32, 1, 32, 128),
        (4, 32, 1, 32, 128),
        (1, 32, 32, 32, 128),
        (4, 32, 32, 32, 128),
        (1, 32, 2016, 2016, 128),
        (4, 32, 2016, 2016, 128),
    ])
    @parametrize("mask_type", ["float", "causal"])
    @parametrize("train", [False])
    def test_scaled_dot_product_fused_attention_mask_vs_math(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        q_size,
        kv_size,
        n_head,
        head_dim,
        mask_type,
        train,
    ):
        # Migrate from TestSDPACpuOnly
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        mask_shape = [batch_size, 1, 1, kv_size]
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
        q_shape = SdpaShape(batch_size, n_head, q_size, head_dim)
        kv_shape = SdpaShape(batch_size, n_head, kv_size, head_dim)
        q = make_tensor(q_shape)
        k = make_tensor(kv_shape)
        v = make_tensor(kv_shape)
        q2, k2, v2 = q.clone(), k.clone(), v.clone()

        if train:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            q2.requires_grad_(True)
            k2.requires_grad_(True)
            v2.requires_grad_(True)

        # (B, nh, T, hs)
        q = q.view(batch_size, q_size, n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        attn_mask = None
        is_causal = False
        if mask_type == "bool":
            attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
        elif mask_type == "float":
            attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)
        elif mask_type == "causal":
            is_causal = True

        q2, k2, v2 = q2.float(), k2.float(), v2.float()
        q2 = q2.view(batch_size, q_size, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        attn_mask2 = attn_mask.float() if attn_mask is not None else None

        if fused_kernel == SDPBackend.MATH:
            actual = torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)[0]
        elif fused_kernel == SDPBackend.OVERRIDEABLE:
            actual = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
                q, k, v, attn_bias=attn_mask, dropout_p=0.0, is_causal=is_causal)[0]

        math_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q2, k2, v2, attn_mask=attn_mask2, dropout_p=0.0, is_causal=is_causal)[0]

        self.assertEqual(actual.float(), math_ref, atol=tol.atol, rtol=tol.rtol)


class TestAttnBias(NNTestCase):

    def run_test(
        self,
        device,
        make_q,
        make_kv,
        attn_bias=None,
        forw_tolerances: Optional[Tolerances] = None,
        grad_tolerances: Optional[Tolerances] = None,
        backend=None,
        causal_variant=None,
    ):
        if backend is not None:
            torch._dynamo.reset()

        query, key, value = make_q(), make_kv(), make_kv()
        query_prototype, key_prototype, value_prototype = query_key_value_clones(query, key, value)

        realized = attn_bias._materialize(device) if attn_bias is not None else None
        pytorch_output = scaled_dot_product_attention(
            query, key, value, attn_mask=realized, dropout_p=0.0, is_causal=False
        )

        sdpa_op = (
            torch.compile(scaled_dot_product_attention, backend=backend)
            if backend is not None
            else scaled_dot_product_attention
        )
        sdpa_output = sdpa_op(
            query_prototype,
            key_prototype,
            value_prototype,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        )

        dOut = torch.randn_like(pytorch_output)
        pytorch_output.backward(dOut)
        sdpa_output.backward(dOut)

        # Use default assert_close tolerances for dtypes
        if forw_tolerances is None:
            forw_tolerances = Tolerances(atol=None, rtol=None)
        if grad_tolerances is None:
            grad_tolerances = Tolerances(atol=None, rtol=None)

        torch.testing.assert_close(pytorch_output, sdpa_output, rtol=forw_tolerances.rtol, atol=forw_tolerances.atol)
        torch.testing.assert_close(query.grad, query_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)
        torch.testing.assert_close(key.grad, key_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)
        torch.testing.assert_close(value.grad, value_prototype.grad, rtol=grad_tolerances.rtol, atol=grad_tolerances.atol)

    @parametrize("causal_variant", [CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT])
    @parametrize(
        "shape",
        [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)],
    )
    def test_causal_variants(self, device, causal_variant: CausalVariant, shape: list[tuple[int]]):
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )
        if TEST_WITH_ROCM and causal_variant == CausalVariant.LOWER_RIGHT:
            self.skipTest("No support for LOWER_RIGHT variant for now")
            return

        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))
        if causal_variant == CausalVariant.LOWER_RIGHT and seq_len_q > seq_len_kv:
            self.skipTest(
                "Lower right causal mask will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )

        forw_tol = Tolerances(1e-3, 1e-3)
        grad_tol = Tolerances(5e-3, 5e-3)

        if causal_variant == CausalVariant.UPPER_LEFT:
            attn_bias = causal_upper_left(seq_len_q, seq_len_kv)
        else:
            attn_bias = causal_lower_right(seq_len_q, seq_len_kv)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION,
                                   SDPBackend.FLASH_ATTENTION,
                                   SDPBackend.MATH,
                                   SDPBackend.CUDNN_ATTENTION]):
            self.run_test(device, make_q_tensor, make_kv_tensor, attn_bias, forw_tol, grad_tol, backend=None)

    @parametrize("causal_variant", [CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT])
    @parametrize(
        "shape",
        [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)],
    )
    @unittest.skipIf(IS_WINDOWS, "torch.compile is not supported on windows")
    @skipIfTorchDynamo("This function already calls torch.compile.")
    def test_causal_variants_compile(self, device, causal_variant: CausalVariant, shape: list[tuple[int]]):
        if TEST_WITH_ROCM and causal_variant == CausalVariant.LOWER_RIGHT:
            self.skipTest("No support for LOWER_RIGHT variant for now")
            return

        cnts = CompileCounterWithBackend("aot_eager")
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )

        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))
        if causal_variant == CausalVariant.LOWER_RIGHT and seq_len_q > seq_len_kv:
            self.skipTest(
                "Lower right causal mask will produce NaNs in the output when seq_len_q > seq_len_kv!"
            )
        forw_tol = Tolerances(1e-3, 1e-3)
        grad_tol = Tolerances(5e-3, 5e-3)

        if causal_variant == CausalVariant.UPPER_LEFT:
            attn_bias = causal_upper_left(seq_len_q, seq_len_kv)
        else:
            attn_bias = causal_lower_right(seq_len_q, seq_len_kv)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION,
                                   SDPBackend.FLASH_ATTENTION,
                                   SDPBackend.MATH,
                                   SDPBackend.CUDNN_ATTENTION]):
            self.run_test(device, make_q_tensor, make_kv_tensor, attn_bias, forw_tol, grad_tol, backend=cnts)
        self.assertEqual(cnts.frame_count, 1, "Compiled graph should have 1 frame!")

    @parametrize("shape", [(16, 16, 128, 128, 16), (16, 16, 128, 256, 32), (16, 16, 256, 128, 32), (1, 1, 23, 56, 15)])
    def test_is_causal_equals_upper_left(self, device, shape: list[tuple[int]]):
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )

        bsz, num_heads, seq_len_q, seq_len_kv, head_dim = shape
        make_q_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim))
        make_kv_tensor = partial(make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim))

        forw_tol = Tolerances(1e-3, 1e-3)

        query = make_q_tensor()
        key = make_kv_tensor()
        value = make_kv_tensor()
        attn_bias = causal_upper_left(seq_len_q, seq_len_kv)

        out_attn_bias = scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, dropout_p=0.0)
        out_is_causal = scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0.0)
        torch.testing.assert_close(out_attn_bias, out_is_causal, rtol=forw_tol.rtol, atol=forw_tol.atol)

    def test_is_causal_and_mask_fails(self, device):
        make_tensor = partial(
            torch.rand, device=device, dtype=torch.float16, requires_grad=True
        )
        make_q_tensor = partial(make_tensor, SdpaShape(16, 16, 128, 16))
        make_kv_tensor = partial(make_tensor, SdpaShape(16, 16, 128, 16))

        query = make_q_tensor()
        key = make_kv_tensor()
        value = make_kv_tensor()
        attn_bias = causal_upper_left(128, 128)

        with self.assertRaisesRegex(ValueError, "CausalBias should not be used with causal=True"):
            scaled_dot_product_attention(query, key, value, attn_mask=attn_bias, is_causal=True, dropout_p=0.0)

if NOTEST_CPU:
    device_types = ("cuda", )
else:
    device_types = ("cpu", "cuda")

instantiate_device_type_tests(TestTransformers, globals(), only_for=device_types)
instantiate_device_type_tests(TestSDPAFailureModes, globals(), only_for=device_types)
instantiate_device_type_tests(TestSDPA, globals(), only_for=device_types)
instantiate_device_type_tests(TestSDPACudaOnly, globals(), only_for=("cuda"))
instantiate_device_type_tests(TestSDPACpuOnly, globals(), only_for=("cpu"))
instantiate_device_type_tests(TestAttnBias, globals(), only_for=device_types)
instantiate_device_type_tests(TestSDPAXpuOnly, globals(), only_for="xpu", allow_xpu=True)

if __name__ == '__main__':
    run_tests()
