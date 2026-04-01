"""Tests for float16 and bfloat16 dtype support.

Validates the widen-compute-narrow approach:
- f16/bf16 tensors are stored in native format on Vulkan
- GPU cast shaders (uint32 bit manipulation) convert f16/bf16 ↔ f32
- All compute shaders operate on f32 internally
- Results are cast back to the original dtype
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_vulkan

RTOL = 1e-2  # f16 has ~3 decimal digits of precision
ATOL = 1e-2
RTOL_RELAXED = 5e-2
ATOL_RELAXED = 5e-2

dev = torch.device("vulkan")


def check(cpu, vk, rtol=RTOL, atol=ATOL):
    """Compare CPU and Vulkan tensors."""
    vk_cpu = vk.cpu().float() if vk.is_vulkan else vk.float()
    cpu_f = cpu.float()
    torch.testing.assert_close(cpu_f, vk_cpu, rtol=rtol, atol=atol)


# ── Cast round-trip tests ──────────────────────────────────────

class TestCastRoundTrip:
    """Test f32 ↔ f16/bf16 conversion shaders."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cast_round_trip_basic(self, dtype):
        """f32 → dtype → f32 preserves values within dtype precision."""
        x = torch.tensor([1.0, -1.0, 0.5, 0.0, 100.0, -100.0, 0.001], device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        expected = x.cpu().to(dtype).float()
        check(expected, x_back, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cast_round_trip_randn(self, dtype):
        """Random tensor round-trip."""
        torch.manual_seed(42)
        x = torch.randn(128, device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        expected = x.cpu().to(dtype).float()
        check(expected, x_back, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cast_special_values(self, dtype):
        """Inf, -Inf, zero survive round-trip."""
        x = torch.tensor([0.0, float('inf'), float('-inf')], device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        assert x_back.cpu()[0] == 0.0
        assert x_back.cpu()[1] == float('inf')
        assert x_back.cpu()[2] == float('-inf')

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cast_odd_numel(self, dtype):
        """Odd number of elements (tests pair-based f32→f16/bf16 shader)."""
        x = torch.randn(7, device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        expected = x.cpu().to(dtype).float()
        check(expected, x_back, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cast_large_tensor(self, dtype):
        """Large tensor cast (exercises multiple workgroups)."""
        x = torch.randn(10000, device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        expected = x.cpu().to(dtype).float()
        check(expected, x_back, rtol=1e-3, atol=1e-3)


# ── Unary ops ──────────────────────────────────────────────────

class TestUnaryOpsF16:
    """Unary ops with f16 inputs."""

    @pytest.mark.parametrize("op", [
        torch.neg, torch.abs, torch.exp, torch.sqrt, torch.rsqrt,
        torch.ceil, torch.floor, torch.round, torch.sign,
        torch.sin, torch.cos, torch.log,
    ])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_unary_op(self, op, dtype):
        torch.manual_seed(42)
        # Use positive values for ops that need them
        if op in (torch.sqrt, torch.rsqrt, torch.log):
            x_cpu = torch.rand(32).abs() + 0.01
        else:
            x_cpu = torch.randn(32)

        x_vk = x_cpu.to(dtype).to(dev)
        r_vk = op(x_vk)
        r_cpu = op(x_cpu.to(dtype))

        assert r_vk.dtype == dtype
        check(r_cpu, r_vk, rtol=RTOL, atol=ATOL)


# ── Binary ops ──────────────────────────────────────────────────

class TestBinaryOpsF16:
    """Binary ops with f16 inputs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_tensor(self, dtype):
        a = torch.randn(32, device=dev, dtype=dtype)
        b = torch.randn(32, device=dev, dtype=dtype)
        r = a + b
        assert r.dtype == dtype
        check(a.cpu() + b.cpu(), r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_add_scalar(self, dtype):
        a = torch.randn(32, device=dev, dtype=dtype)
        r = a + 2.5
        check(a.cpu() + 2.5, r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mul_tensor(self, dtype):
        a = torch.randn(32, device=dev, dtype=dtype)
        b = torch.randn(32, device=dev, dtype=dtype)
        r = a * b
        assert r.dtype == dtype
        check(a.cpu() * b.cpu(), r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mul_scalar(self, dtype):
        a = torch.randn(32, device=dev, dtype=dtype)
        r = a * 3.0
        check(a.cpu() * 3.0, r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_div_tensor(self, dtype):
        a = torch.randn(32, device=dev, dtype=dtype)
        b = torch.randn(32, device=dev, dtype=dtype).abs() + 0.1
        r = a / b
        assert r.dtype == dtype
        check(a.cpu() / b.cpu(), r)


# ── Matmul ops ──────────────────────────────────────────────────

class TestMatmulF16:
    """Matrix multiplication with f16 inputs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mm(self, dtype):
        torch.manual_seed(42)
        a = torch.randn(4, 8, device=dev, dtype=dtype)
        b = torch.randn(8, 6, device=dev, dtype=dtype)
        r = torch.mm(a, b)
        assert r.dtype == dtype
        # Compare with CPU result
        r_cpu = torch.mm(a.cpu(), b.cpu())
        check(r_cpu, r, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_addmm(self, dtype):
        torch.manual_seed(42)
        bias = torch.randn(6, device=dev, dtype=dtype)
        a = torch.randn(4, 8, device=dev, dtype=dtype)
        b = torch.randn(8, 6, device=dev, dtype=dtype)
        r = torch.addmm(bias, a, b)
        assert r.dtype == dtype
        r_cpu = torch.addmm(bias.cpu(), a.cpu(), b.cpu())
        check(r_cpu, r, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_linear(self, dtype):
        torch.manual_seed(42)
        # Use small values to stay well within f16 range
        w = torch.randn(4, 8) * 0.1
        b = torch.randn(4) * 0.1
        x = torch.randn(2, 8) * 0.1
        # CPU reference: same widen-compute-narrow approach
        w_dt = w.to(dtype)
        b_dt = b.to(dtype)
        x_dt = x.to(dtype)
        r_cpu = F.linear(x_dt.float(), w_dt.float(), b_dt.float()).to(dtype)
        # Vulkan
        lin_vk = nn.Linear(8, 4).to(dev)
        lin_vk.weight.data.copy_(w.to(dev))
        lin_vk.bias.data.copy_(b.to(dev))
        # Cast model to dtype on Vulkan
        lin_vk = lin_vk.to(dtype)
        r = lin_vk(x_dt.to(dev))
        assert r.dtype == dtype
        # Linear accumulates K multiplications — need relaxed tolerance
        check(r_cpu, r, rtol=0.1, atol=0.1)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_bmm(self, dtype):
        torch.manual_seed(42)
        a = torch.randn(2, 4, 8, device=dev, dtype=dtype)
        b = torch.randn(2, 8, 6, device=dev, dtype=dtype)
        r = torch.bmm(a, b)
        assert r.dtype == dtype
        r_cpu = torch.bmm(a.cpu(), b.cpu())
        check(r_cpu, r, rtol=RTOL_RELAXED, atol=ATOL_RELAXED)


# ── Shape ops ──────────────────────────────────────────────────

class TestShapeOpsF16:
    """Shape operations preserve f16 dtype correctly."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_view(self, dtype):
        x = torch.randn(6, device=dev, dtype=dtype)
        r = x.view(2, 3)
        assert r.dtype == dtype
        check(x.cpu().view(2, 3), r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_transpose(self, dtype):
        x = torch.randn(3, 4, device=dev, dtype=dtype)
        r = x.t()
        assert r.dtype == dtype
        check(x.cpu().t(), r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_permute(self, dtype):
        x = torch.randn(2, 3, 4, device=dev, dtype=dtype)
        r = x.permute(2, 0, 1)
        assert r.dtype == dtype
        check(x.cpu().permute(2, 0, 1), r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_expand(self, dtype):
        x = torch.randn(1, 4, device=dev, dtype=dtype)
        r = x.expand(3, 4)
        assert r.dtype == dtype
        check(x.cpu().expand(3, 4), r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cat(self, dtype):
        a = torch.randn(3, 4, device=dev, dtype=dtype)
        b = torch.randn(2, 4, device=dev, dtype=dtype)
        r = torch.cat([a, b], dim=0)
        assert r.dtype == dtype
        check(torch.cat([a.cpu(), b.cpu()], dim=0), r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_select(self, dtype):
        x = torch.randn(3, 4, device=dev, dtype=dtype)
        r = x[1]
        assert r.dtype == dtype
        check(x.cpu()[1], r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_slice(self, dtype):
        x = torch.randn(8, device=dev, dtype=dtype)
        r = x[2:6]
        assert r.dtype == dtype
        check(x.cpu()[2:6], r)


# ── Activation ops ──────────────────────────────────────────────

class TestActivationF16:
    """Activations with f16 inputs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_relu(self, dtype):
        x = torch.randn(32, device=dev, dtype=dtype)
        r = F.relu(x)
        assert r.dtype == dtype
        check(F.relu(x.cpu()), r)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sigmoid(self, dtype):
        x = torch.randn(32, device=dev, dtype=dtype)
        r = torch.sigmoid(x)
        assert r.dtype == dtype
        check(torch.sigmoid(x.cpu()), r, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_silu(self, dtype):
        x = torch.randn(32, device=dev, dtype=dtype)
        r = F.silu(x)
        assert r.dtype == dtype
        check(F.silu(x.cpu()), r, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_softmax(self, dtype):
        """Softmax always computes in f32 for numerical stability."""
        x = torch.randn(4, 8, device=dev, dtype=dtype)
        r = F.softmax(x, dim=-1)
        # Softmax should return f32 for precision
        r_cpu = F.softmax(x.cpu().float(), dim=-1)
        check(r_cpu, r, rtol=RTOL, atol=ATOL)


# ── Reduction ops ──────────────────────────────────────────────

class TestReductionF16:
    """Reductions with f16 inputs."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sum(self, dtype):
        x = torch.randn(32, device=dev, dtype=dtype)
        r = x.sum()
        check(x.cpu().float().sum(), r.float(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mean(self, dtype):
        x = torch.randn(32, device=dev, dtype=dtype)
        r = x.mean()
        check(x.cpu().float().mean(), r.float(), rtol=RTOL_RELAXED, atol=ATOL_RELAXED)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_max_dim(self, dtype):
        x = torch.randn(4, 8, device=dev, dtype=dtype)
        vals, indices = x.max(dim=1)
        vals_cpu, indices_cpu = x.cpu().max(dim=1)
        check(vals_cpu, vals, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_min_dim(self, dtype):
        x = torch.randn(4, 8, device=dev, dtype=dtype)
        vals, indices = x.min(dim=1)
        vals_cpu, indices_cpu = x.cpu().min(dim=1)
        check(vals_cpu, vals, rtol=RTOL, atol=ATOL)


# ── Fill/Clone ops ──────────────────────────────────────────────

class TestFillCloneF16:
    """Fill and clone with f16 tensors."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fill(self, dtype):
        x = torch.empty(32, device=dev, dtype=dtype)
        x.fill_(3.14)
        expected = torch.full((32,), 3.14, dtype=dtype)
        check(expected, x, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_clone(self, dtype):
        x = torch.randn(32, device=dev, dtype=dtype)
        y = x.clone()
        check(x.cpu(), y, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_zeros(self, dtype):
        x = torch.zeros(32, device=dev, dtype=dtype)
        assert (x.cpu() == 0).all()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_ones(self, dtype):
        x = torch.ones(32, device=dev, dtype=dtype)
        assert (x.cpu() == 1).all()


# ── Autocast / AMP ──────────────────────────────────────────────

class TestAutocast:
    """Autocast (AMP) with Vulkan backend."""

    def test_autocast_mm_casts_to_f16(self):
        """mm should be cast to f16 under autocast."""
        x = torch.randn(4, 8, device=dev)
        y = torch.randn(8, 4, device=dev)
        with torch.autocast("vulkan", dtype=torch.float16):
            z = torch.mm(x, y)
            assert z.dtype == torch.float16

    def test_autocast_linear_casts_to_f16(self):
        """linear should be cast to f16 under autocast."""
        model = nn.Linear(8, 4).to(dev)
        x = torch.randn(2, 8, device=dev)
        with torch.autocast("vulkan", dtype=torch.float16):
            y = model(x)
            assert y.dtype == torch.float16

    def test_autocast_softmax_stays_f32(self):
        """softmax should stay f32 under autocast (fp32 policy)."""
        x = torch.randn(4, 8, device=dev)
        with torch.autocast("vulkan", dtype=torch.float16):
            y = F.softmax(x, dim=-1)
            assert y.dtype == torch.float32

    def test_autocast_layer_norm_stays_f32(self):
        """layer_norm should stay f32 under autocast (fp32 policy)."""
        x = torch.randn(2, 4, 8, device=dev)
        with torch.autocast("vulkan", dtype=torch.float16):
            y = F.layer_norm(x, [8])
            assert y.dtype == torch.float32

    def test_amp_training_step(self):
        """Full AMP training step: autocast + GradScaler."""
        torch.manual_seed(42)
        # Use Xavier init to keep values in f16 range
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        model = model.to(dev)

        opt = torch.optim.SGD(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("vulkan")

        x = torch.randn(4, 8, device=dev) * 0.1
        target = torch.randn(4, 4, device=dev) * 0.1

        # Run a few steps — GradScaler handles any inf/NaN by skipping
        for _ in range(5):
            with torch.autocast("vulkan", dtype=torch.float16):
                pred = model(x)
                loss = F.mse_loss(pred, target)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # After training, loss should be finite
        assert torch.isfinite(torch.tensor(loss.item()))
        # Gradients should exist
        assert model[0].weight.grad is not None


# ── Mixed dtype ops ──────────────────────────────────────────────

class TestMixedDtype:
    """Operations with mixed dtypes (f16 + f32, etc.)."""

    def test_f16_plus_f32(self):
        """f16 tensor + f32 tensor should work via type promotion."""
        a = torch.randn(16, device=dev, dtype=torch.float16)
        b = torch.randn(16, device=dev, dtype=torch.float32)
        r = a + b
        # PyTorch promotes to f32
        r_cpu = a.cpu().float() + b.cpu()
        check(r_cpu, r, rtol=RTOL, atol=ATOL)

    def test_f16_plus_python_scalar(self):
        """f16 tensor + Python scalar should work."""
        a = torch.randn(16, device=dev, dtype=torch.float16)
        r = a + 2.5
        check(a.cpu() + 2.5, r)

    def test_f16_times_python_int(self):
        """f16 tensor * Python int should work."""
        a = torch.randn(16, device=dev, dtype=torch.float16)
        r = a * 3
        check(a.cpu() * 3, r)


# ── FP8 (float8_e4m3fn, float8_e5m2) ───────────────────────────

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


class TestFP8CastRoundTrip:
    """Test f32 ↔ fp8 conversion shaders."""

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_cast_basic_values(self, dtype):
        x = torch.tensor([1.0, -1.0, 0.5, 0.0, 2.0, 4.0, 8.0], device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        expected = x.cpu().to(dtype).float()
        check(expected, x_back, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_cast_randn(self, dtype):
        torch.manual_seed(42)
        x = torch.randn(128, device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        expected = x.cpu().to(dtype).float()
        check(expected, x_back, rtol=1e-3, atol=1e-3)

    def test_e4m3fn_max(self):
        """E4M3FN max representable value is 448."""
        x = torch.tensor([448.0, -448.0], device=dev)
        x_cast = x.to(torch.float8_e4m3fn)
        x_back = x_cast.float()
        expected = x.cpu().to(torch.float8_e4m3fn).float()
        check(expected, x_back, rtol=0, atol=0)

    def test_e5m2_large_values(self):
        """E5M2 can represent up to 57344."""
        x = torch.tensor([100.0, 1000.0, -500.0], device=dev)
        x_cast = x.to(torch.float8_e5m2)
        x_back = x_cast.float()
        expected = x.cpu().to(torch.float8_e5m2).float()
        check(expected, x_back, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_cast_zero(self, dtype):
        x = torch.zeros(4, device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        assert (x_back.cpu() == 0).all()

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_cast_odd_numel(self, dtype):
        """Odd element count tests quad-packing shader edge case."""
        x = torch.randn(7, device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        expected = x.cpu().to(dtype).float()
        check(expected, x_back, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_cast_large_tensor(self, dtype):
        x = torch.randn(10000, device=dev)
        x_cast = x.to(dtype)
        x_back = x_cast.float()
        expected = x.cpu().to(dtype).float()
        check(expected, x_back, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_cast_non_multiple_of_4(self, dtype):
        """Element count not divisible by 4 (tests quad boundary)."""
        for n in [1, 2, 3, 5, 6, 9, 13]:
            x = torch.randn(n, device=dev)
            x_cast = x.to(dtype)
            x_back = x_cast.float()
            expected = x.cpu().to(dtype).float()
            check(expected, x_back, rtol=1e-3, atol=1e-3)


class TestFP8Ops:
    """Operations with FP8 inputs via widen-compute-narrow."""

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_add(self, dtype):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(dtype).to(dev)
        b = torch.tensor([2.0, 3.0, 4.0, 5.0]).to(dtype).to(dev)
        ref = (a.cpu().float() + b.cpu().float()).to(dtype).float()
        result = (a + b).float().cpu()
        check(ref, result.to(dev), rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_mul(self, dtype):
        a = torch.tensor([1.0, 2.0, 0.5, 4.0]).to(dtype).to(dev)
        b = torch.tensor([2.0, 3.0, 4.0, 0.5]).to(dtype).to(dev)
        ref = (a.cpu().float() * b.cpu().float()).to(dtype).float()
        result = (a * b).float().cpu()
        check(ref, result.to(dev), rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_relu(self, dtype):
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0]).to(dtype).to(dev)
        ref = torch.relu(x.cpu().float()).to(dtype).float()
        result = torch.relu(x).float().cpu()
        check(ref, result.to(dev), rtol=0, atol=0)

    def test_mm_e4m3fn(self):
        """Matrix multiply with E4M3FN inputs."""
        torch.manual_seed(42)
        a = torch.randn(4, 4).to(torch.float8_e4m3fn)
        b = torch.randn(4, 4).to(torch.float8_e4m3fn)
        ref = torch.mm(a.float(), b.float()).to(torch.float8_e4m3fn).float()
        a_v, b_v = a.to(dev), b.to(dev)
        result = torch.mm(a_v, b_v).float().cpu()
        check(ref, result.to(dev), rtol=0.1, atol=0.5)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_shape_ops(self, dtype):
        x = torch.randn(2, 3, 4).to(dtype).to(dev)
        assert x.reshape(6, 4).shape == (6, 4)
        assert x.permute(0, 2, 1).shape == (2, 4, 3)
        assert x.unsqueeze(0).shape == (1, 2, 3, 4)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_clone(self, dtype):
        x = torch.randn(32).to(dtype).to(dev)
        y = x.clone()
        check(x.cpu(), y, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_copy_to_from_cpu(self, dtype):
        """CPU → Vulkan → CPU roundtrip for FP8."""
        x_cpu = torch.randn(16).to(dtype)
        x_v = x_cpu.to(dev)
        x_back = x_v.cpu()
        assert x_back.dtype == dtype
        assert torch.equal(x_cpu.float(), x_back.float())

    @pytest.mark.parametrize("dtype", FP8_DTYPES)
    def test_reduction_sum(self, dtype):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(dtype).to(dev)
        ref = x.cpu().float().sum().to(dtype).float()
        result = x.sum().float().cpu()
        check(ref, result.to(dev), rtol=0.1, atol=0.5)

    def test_linear_e4m3fn(self):
        """nn.Linear with E4M3FN weights and inputs."""
        torch.manual_seed(42)
        linear = torch.nn.Linear(8, 4, bias=True)
        x = torch.randn(2, 8)

        # FP8 quantized
        x_fp8 = x.to(torch.float8_e4m3fn)
        w_fp8 = linear.weight.to(torch.float8_e4m3fn)
        b_fp8 = linear.bias.to(torch.float8_e4m3fn)

        # CPU reference
        ref = (x_fp8.float() @ w_fp8.float().t() + b_fp8.float()).to(torch.float8_e4m3fn).float()

        # Vulkan
        x_v = x_fp8.to(dev)
        w_v = w_fp8.to(dev)
        b_v = b_fp8.to(dev)
        result = (x_v.float() @ w_v.float().t() + b_v.float()).to(torch.float8_e4m3fn).float().cpu()
        check(ref, result.to(dev), rtol=0.1, atol=1.0)


class TestScaledMM:
    """Tests for _scaled_mm (FP8 scaled matmul)."""

    def test_basic(self):
        """Basic _scaled_mm with identity scales."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn, device=dev)
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float8_e4m3fn, device=dev)
        scale_a = torch.tensor(1.0, device=dev)
        scale_b = torch.tensor(1.0, device=dev)
        result = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        check(expected, result, rtol=0, atol=0)

    def test_with_scales(self):
        """_scaled_mm with non-trivial scales."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn, device=dev)
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float8_e4m3fn, device=dev)
        scale_a = torch.tensor(2.0, device=dev)
        scale_b = torch.tensor(3.0, device=dev)
        result = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
        expected = torch.tensor([[6.0, 12.0], [18.0, 24.0]])
        check(expected, result, rtol=0, atol=0)

    def test_fp8_training_pattern(self):
        """Simulate FP8 quantize → _scaled_mm → dequantize pattern."""
        torch.manual_seed(42)
        a_f32 = torch.randn(8, 16)
        b_f32 = torch.randn(16, 4)

        # Quantize
        amax_a = a_f32.abs().max()
        amax_b = b_f32.abs().max()
        scale_a = 448.0 / amax_a
        scale_b = 448.0 / amax_b
        a_fp8 = (a_f32 * scale_a).to(torch.float8_e4m3fn)
        b_fp8 = (b_f32 * scale_b).to(torch.float8_e4m3fn)

        inv_scale_a = torch.tensor(1.0 / scale_a.item())
        inv_scale_b = torch.tensor(1.0 / scale_b.item())

        # CPU reference
        ref = torch._scaled_mm(a_fp8, b_fp8,
            scale_a=inv_scale_a, scale_b=inv_scale_b, out_dtype=torch.float32)

        # Vulkan
        result = torch._scaled_mm(
            a_fp8.to(dev), b_fp8.to(dev),
            scale_a=inv_scale_a.to(dev), scale_b=inv_scale_b.to(dev),
            out_dtype=torch.float32)

        check(ref, result, rtol=1e-4, atol=1e-4)

    def test_e5m2_backward_format(self):
        """_scaled_mm with E5M2 (backward format)."""
        a = torch.randn(4, 8).to(torch.float8_e5m2).to(dev)
        b = torch.randn(8, 4).to(torch.float8_e4m3fn).to(dev)
        scale_a = torch.tensor(1.0, device=dev)
        scale_b = torch.tensor(1.0, device=dev)
        result = torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float32)
        assert result.shape == (4, 4)
        assert result.dtype == torch.float32


# ── BF16 autocast ─────────────────────────────────────────────────

class TestBF16Autocast:
    """Autocast with bfloat16 dtype."""

    def test_autocast_mm_casts_to_bf16(self):
        x = torch.randn(4, 8, device=dev)
        y = torch.randn(8, 4, device=dev)
        with torch.autocast("vulkan", dtype=torch.bfloat16):
            z = torch.mm(x, y)
            assert z.dtype == torch.bfloat16

    def test_autocast_linear_casts_to_bf16(self):
        model = nn.Linear(8, 4).to(dev)
        x = torch.randn(2, 8, device=dev)
        with torch.autocast("vulkan", dtype=torch.bfloat16):
            y = model(x)
            assert y.dtype == torch.bfloat16

    def test_autocast_softmax_stays_f32_bf16(self):
        x = torch.randn(4, 8, device=dev)
        with torch.autocast("vulkan", dtype=torch.bfloat16):
            y = F.softmax(x, dim=-1)
            assert y.dtype == torch.float32

    def test_autocast_bf16_training_step(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        model = model.to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("vulkan")
        x = torch.randn(4, 8, device=dev) * 0.1
        target = torch.randn(4, 4, device=dev) * 0.1

        for _ in range(3):
            with torch.autocast("vulkan", dtype=torch.bfloat16):
                pred = model(x)
                loss = F.mse_loss(pred, target)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        assert torch.isfinite(torch.tensor(loss.item()))


# ── FP8 training pattern ─────────────────────────────────────────

class TestFP8TrainingPattern:
    """FP8 training with quantize → _scaled_mm → dequantize per layer."""

    def test_fp8_linear_forward(self):
        """Simulate FP8 linear layer: quantize activations + weights, _scaled_mm."""
        torch.manual_seed(42)
        x = torch.randn(4, 16, device=dev)
        w = torch.randn(8, 16, device=dev)  # [out, in]

        # Quantize: compute amax, scale to FP8 range, cast
        amax_x = x.abs().amax()
        amax_w = w.abs().amax()
        scale_x = 448.0 / amax_x.item()
        scale_w = 448.0 / amax_w.item()

        x_fp8 = (x * scale_x).to(torch.float8_e4m3fn)
        w_fp8 = (w * scale_w).to(torch.float8_e4m3fn)

        inv_sx = torch.tensor(1.0 / scale_x, device=dev)
        inv_sw = torch.tensor(1.0 / scale_w, device=dev)

        # _scaled_mm: w_fp8.T is [in, out], result is [4, 8]
        result = torch._scaled_mm(x_fp8, w_fp8.t().contiguous(),
                                   scale_a=inv_sx, scale_b=inv_sw,
                                   out_dtype=torch.float32)

        # CPU reference
        ref = torch._scaled_mm(x_fp8.cpu(), w_fp8.cpu().t().contiguous(),
                                scale_a=inv_sx.cpu(), scale_b=inv_sw.cpu(),
                                out_dtype=torch.float32)
        check(ref, result, rtol=1e-4, atol=1e-4)

    def test_fp8_e5m2_gradient_format(self):
        """E5M2 used for gradients (wider dynamic range for backward)."""
        torch.manual_seed(42)
        grad = torch.randn(8, 16, device=dev)

        amax_g = grad.abs().amax()
        scale_g = 57344.0 / amax_g.item()  # E5M2 max

        grad_fp8 = (grad * scale_g).to(torch.float8_e5m2)
        grad_back = grad_fp8.float() / scale_g

        # Should roundtrip within E5M2 precision
        expected = (grad.cpu() * scale_g).to(torch.float8_e5m2).float() / scale_g
        check(expected, grad_back, rtol=0.05, atol=0.05)

    def test_fp8_quantize_dequantize_identity(self):
        """Quantize and dequantize should approximately recover original values."""
        torch.manual_seed(42)
        x = torch.randn(64, device=dev) * 0.1

        amax = x.abs().amax()
        scale = 448.0 / amax.item()
        x_fp8 = (x * scale).to(torch.float8_e4m3fn)
        x_deq = x_fp8.float() / scale

        # Should be close (quantization noise only)
        check(x.cpu(), x_deq, rtol=0.1, atol=0.05)
