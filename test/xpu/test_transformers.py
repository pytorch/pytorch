# Owner(s): ["module: intel"]

import contextlib
from functools import partial
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.attention.bias import CausalVariant, causal_lower_right, causal_upper_left
from torch.nn.parameter import Parameter
import unittest
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
    # make_tensor,
    NOTEST_CPU,
    IS_WINDOWS,
    TEST_WITH_TORCHDYNAMO,
    TEST_XPU,
)

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


PLATFORM_SPECIFIC_SDPA = [SDPBackend.EFFICIENT_ATTENTION]
# Indicate the Efficient attention backend can support:
# 1. sequence longher than 512
# 2. head dimsion larger than 64
MEM_EFF_CAPABILITY_MATCHES_SM80 = True


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


class TestSDPAXpuOnly(NNTestCase):
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

    @unittest.skip("OneDNN Graph does not support different dk dv")
    def test_onednn_attention_different_dk_dv(self, device):
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        seq_len = 640
        q_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, seq_len, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, seq_len, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous().to(torch.float32),
                key.contiguous().to(torch.float32),
                value.contiguous().to(torch.float32),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    def test_fused_attention_different_dk_dv(self, device):
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        seq_len = 640
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

    def test_cudnn_attention_fail_d128(self, device):
        # Test that cuDNN attention dispatching correctly bails out on d > 128
        b, h = 1, 2
        s_q, s_kv = 128, 128
        d_qk, d_v = 128, 144

        q = torch.randn(b, h, s_q, d_qk, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, h, s_kv, d_qk, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, h, s_kv, d_v, device=device, dtype=torch.bfloat16)

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                o = torch.nn.functional.scaled_dot_product_attention(q, k, v)

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

        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = torch.nn.functional.scaled_dot_product_attention(
                query.contiguous(), key.contiguous(), value.contiguous(),
                attn_mask=None, dropout_p=0.0, is_causal=False)

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @parametrize("type", ["dense"])
    @parametrize("fused_kernel", [SDPBackend.EFFICIENT_ATTENTION])
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

    @parametrize("type", ["dense"])
    def test_fused_sdp_choice(self, device, type: str):
        batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float16, packed=True,
                              requires_grad=False)  # set requires_grad to False for onednn graph

        qkv = make_tensor(shape, type=type)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        # TODO we are currently disabling this by default, lets assert that this returns
        # FlashAttention, we need to change when we make remove opt-in for cudnn
        if type != "nested":
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.OVERRIDEABLE.value)
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.OVERRIDEABLE.value)
        else:
            self.assertEqual(torch._fused_sdp_choice(query, key, value), SDPBackend.MATH.value)

        # Change dtype to float32 so that efficient attention should get chosen
        make_tensor = partial(rand_sdpa_tensor, device=device, dtype=torch.float32, packed=True)

        qkv = make_tensor(shape, type=type)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        assert torch._fused_sdp_choice(query, key, value) == SDPBackend.OVERRIDEABLE.value

    @parametrize("warn_only", [True, False])
    def test_sdp_choice_with_determinism(self, device, warn_only):
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=torch.float32, packed=False)
        query, key, value = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        with use_deterministic_algorithims(True, warn_only=warn_only):
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                assert torch._fused_sdp_choice(query, key, value) == SDPBackend.OVERRIDEABLE.value

    @parametrize("fused_kernel", [SDPBackend.MATH])
    @parametrize("dtype", [torch.half])
    @parametrize("batch_size,n_head,q_size,kv_size,head_dim", [
        # (2, 5, 9216, 9216, 64),
        # (2, 5, 9216, 77, 64),
        # (2, 10, 2304, 2304, 64),
        # (2, 10, 2304, 77, 64),
        # (2, 20, 576, 576, 64),
        # (2, 20, 576, 77, 64),
        # (2, 20, 144, 144, 64),
        (2, 20, 144, 77, 64),
        # (1, 32, 1, 32, 128),
        # (4, 32, 1, 32, 128),
        # (1, 32, 32, 32, 128),
        # (4, 32, 32, 32, 128),
        # (1, 32, 2016, 2016, 128),
        # (4, 32, 2016, 2016, 128),
    ])
    @parametrize("mask_type", ["float"])
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

        q2, k2, v2 = q2.float(), k2.float(), v2.float()
        # (B, nh, T, hs)
        q = q.view(batch_size, q_size, n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        attn_mask = None
        if mask_type == "bool":
            attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
        elif mask_type == "float":
            attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)

        q2 = q2.view(batch_size, q_size, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)

        with sdpa_kernel(backends=[fused_kernel]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        math_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q2, k2, v2, attn_mask=attn_mask.float(), dropout_p=0.0, is_causal=False)[0]

        math_ref = math_ref.to(dtype)

        self.assertEqual(actual.float(), math_ref, atol=tol.atol, rtol=tol.rtol)

        iter_n = 100
        with sdpa_kernel(backends=[fused_kernel]), torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU],
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=iter_n,
                    active=iter_n),
                on_trace_ready=self.trace_handler()) as prof:
            for _ in range(iter_n * 2 + 2):
                # actual = torch.ops.aten._scaled_dot_product_attention_math(
                #     q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
                actual = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
                # actual = torch._scaled_dot_product_efficient_attention(
                #     q, k, v, attn_bias=attn_mask, dropout_p=0.0, is_causal=False, compute_log_sumexp=False)
                # actual = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
                #     q, k, v, attn_bias=attn_mask, dropout_p=0.0, is_causal=False)
                torch.xpu.synchronize()
                prof.step()

    def trace_handler(self):
        test_name = self.id()

        def h(p):
            print()
            print(p.key_averages().table(sort_by="self_xpu_time_total", max_name_column_width=60, row_limit=999))
            from datetime import datetime
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            p.export_chrome_trace(f"trace_{test_name}_{p.step_num}_{now}.json")
        return h


instantiate_device_type_tests(
    TestSDPAXpuOnly, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
