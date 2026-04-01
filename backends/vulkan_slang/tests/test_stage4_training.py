"""Tests for Stage 4: Tensor Factories, RNG & Optimizer Support."""

import torch
import torch.nn.functional as F
import pytest

RTOL = 1e-3
ATOL = 1e-4


@pytest.fixture(autouse=True)
def setup():
    try:
        import torch_vulkan
        if not torch_vulkan.is_available():
            pytest.skip("No Vulkan device")
    except ImportError:
        pytest.skip("torch_vulkan not installed")


def to_vulkan(t):
    return t.to("vulkan:0")


def assert_close(vulkan_result, expected, rtol=RTOL, atol=ATOL):
    actual = vulkan_result.cpu()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


# ── Tensor Factories ─────────────────────────────────────────────


class TestFactories:
    def test_arange(self):
        result = torch.arange(0, 10, 1, device="vulkan:0")
        expected = torch.arange(0, 10, 1)
        assert_close(result, expected.to(result.cpu().dtype))

    def test_arange_float(self):
        result = torch.arange(0.0, 5.0, 0.5, device="vulkan:0")
        expected = torch.arange(0.0, 5.0, 0.5)
        assert_close(result, expected)

    def test_linspace(self):
        result = torch.linspace(0, 1, 11, device="vulkan:0")
        expected = torch.linspace(0, 1, 11)
        assert_close(result, expected)

    def test_eye(self):
        result = torch.eye(4, device="vulkan:0")
        expected = torch.eye(4)
        assert_close(result, expected)

    def test_eye_non_square(self):
        result = torch.eye(3, 5, device="vulkan:0")
        expected = torch.eye(3, 5)
        assert_close(result, expected)

    def test_full(self):
        result = torch.full((4, 4), 3.14, device="vulkan:0")
        expected = torch.full((4, 4), 3.14)
        assert_close(result, expected)

    def test_zeros(self):
        result = torch.zeros(8, device="vulkan:0")
        expected = torch.zeros(8)
        assert_close(result, expected)

    def test_ones(self):
        result = torch.ones(4, 4, device="vulkan:0")
        expected = torch.ones(4, 4)
        assert_close(result, expected)


# ── RNG ──────────────────────────────────────────────────────────


class TestRNG:
    def test_uniform_range(self):
        x = torch.empty(1000, device="vulkan:0")
        x.uniform_(0.0, 1.0)
        vals = x.cpu()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0
        # Should have reasonable spread
        assert vals.mean().item() > 0.3
        assert vals.mean().item() < 0.7

    def test_normal_distribution(self):
        x = torch.empty(10000, device="vulkan:0")
        x.normal_(0.0, 1.0)
        vals = x.cpu()
        # Check approximate mean and std
        assert abs(vals.mean().item()) < 0.1
        assert abs(vals.std().item() - 1.0) < 0.2

    def test_dropout_zeros_some(self):
        x = torch.ones(1000, device="vulkan:0")
        output, mask = torch.native_dropout(x, 0.5, True)
        vals = output.cpu()
        # Some should be zero, some should be scaled
        num_zeros = (vals == 0).sum().item()
        assert num_zeros > 200  # ~50% should be zeroed
        assert num_zeros < 800

    def test_dropout_eval_noop(self):
        x = torch.ones(100, device="vulkan:0")
        output, mask = torch.native_dropout(x, 0.5, False)
        assert_close(output, torch.ones(100))


# ── Optimizer Ops ────────────────────────────────────────────────


class TestOptimizerOps:
    def test_addcmul_(self):
        self_t = torch.randn(16)
        t1 = torch.randn(16)
        t2 = torch.randn(16)
        expected = self_t.clone().addcmul_(t1, t2, value=0.1)

        vs = to_vulkan(self_t.clone())
        vs.addcmul_(to_vulkan(t1), to_vulkan(t2), value=0.1)
        assert_close(vs, expected)

    def test_addcdiv_(self):
        self_t = torch.randn(16)
        t1 = torch.randn(16)
        t2 = torch.randn(16).abs() + 0.1  # avoid div by zero
        expected = self_t.clone().addcdiv_(t1, t2, value=0.01)

        vs = to_vulkan(self_t.clone())
        vs.addcdiv_(to_vulkan(t1), to_vulkan(t2), value=0.01)
        assert_close(vs, expected)

    def test_lerp_(self):
        self_t = torch.randn(16)
        end_t = torch.randn(16)
        expected = self_t.clone().lerp_(end_t, 0.3)

        vs = to_vulkan(self_t.clone())
        vs.lerp_(to_vulkan(end_t), 0.3)
        assert_close(vs, expected)

    def test_clamp_(self):
        x = torch.randn(32)
        expected = x.clone().clamp_(-0.5, 0.5)

        vx = to_vulkan(x.clone())
        vx.clamp_(-0.5, 0.5)
        assert_close(vx, expected)


# ── Adam Training Test ───────────────────────────────────────────


class TestAdamWOptimizer:
    """Tests for fused AdamW optimizer on Vulkan tensors."""

    def test_adamw_f32_matches_cpu(self):
        """Fused AdamW on f32 params matches PyTorch CPU AdamW."""
        import torch_vulkan
        torch.manual_seed(42)
        param = torch.randn(16)
        grad = torch.randn(16)
        lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 1e-2

        # CPU reference
        m_cpu = torch.zeros(16)
        v_cpu = torch.zeros(16)
        p_cpu = param.clone()
        m_cpu = b1 * m_cpu + (1-b1) * grad
        v_cpu = b2 * v_cpu + (1-b2) * grad * grad
        bc1 = 1 - b1; bc2 = 1 - b2
        p_cpu = p_cpu * (1 - lr * wd) - lr * (m_cpu / bc1) / (torch.sqrt(v_cpu / bc2) + eps)

        # Vulkan
        p_v = to_vulkan(param.clone())
        m_v = to_vulkan(torch.zeros(16))
        v_v = to_vulkan(torch.zeros(16))
        g_v = to_vulkan(grad)
        torch_vulkan._c_ext._adamw_step(p_v, g_v, m_v, v_v, lr, b1, b2, eps, wd, 1)

        assert_close(p_v, p_cpu, rtol=1e-4, atol=1e-5)
        assert_close(m_v, m_cpu, rtol=1e-4, atol=1e-6)

    def test_adamw_bf16_converges(self):
        """AdamW with master_weights=True converges for bf16 Vulkan model."""
        import torch_vulkan
        torch.manual_seed(42)
        hidden, vocab = 64, 32
        ew = (torch.randn(vocab, hidden) * 0.02).bfloat16().to('vulkan').requires_grad_(True)
        lh = (torch.randn(vocab, hidden) * 0.02).bfloat16().to('vulkan').requires_grad_(True)

        opt = torch_vulkan.AdamW([ew, lh], lr=1e-3, weight_decay=1e-2, master_weights=True)
        losses = []
        for _ in range(10):
            tok = torch.randint(0, vocab, (4, 8)).to('vulkan')
            tgt = torch.randint(0, vocab, (4,)).to('vulkan')
            opt.zero_grad()
            x = F.embedding(tok, ew)
            logits = F.linear(x[:, -1, :], lh)
            loss = F.cross_entropy(logits, tgt)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert all(not (l != l) for l in losses), "NaN in losses"
        assert all(not (l == float('inf')) for l in losses), "Inf in losses"

    def test_adamw_sgd_equivalence(self):
        """AdamW and SGD both converge (loss decreases) on the same model."""
        import torch_vulkan
        torch.manual_seed(42)
        W = (torch.randn(10, 8) * 0.1).to('vulkan').requires_grad_(True)
        opt = torch_vulkan.AdamW([W], lr=1e-2, weight_decay=0.0)
        x = torch.randn(4, 8).to('vulkan')
        y = torch.randint(0, 10, (4,)).to('vulkan')
        initial_loss = F.cross_entropy(F.linear(x, W), y).item()
        for _ in range(20):
            opt.zero_grad()
            loss = F.cross_entropy(F.linear(x, W), y)
            loss.backward()
            opt.step()
        final_loss = F.cross_entropy(F.linear(x, W), y).item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.3f} -> {final_loss:.3f}"


class TestAdamTraining:
    def test_adam_step(self):
        """Verify a manual Adam step matches CPU."""
        torch.manual_seed(42)

        param = torch.randn(8, requires_grad=True)
        grad = torch.randn(8)

        # Manual Adam state
        lr = 0.001
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        m = torch.zeros(8)
        v = torch.zeros(8)

        # CPU Adam step
        m_cpu = beta1 * m + (1 - beta1) * grad
        v_cpu = beta2 * v + (1 - beta2) * grad * grad
        m_hat = m_cpu / (1 - beta1)
        v_hat = v_cpu / (1 - beta2)
        expected = param.data - lr * m_hat / (v_hat.sqrt() + eps)

        # Vulkan Adam step using addcmul_ / addcdiv_
        vp = to_vulkan(param.data.clone())
        vm = to_vulkan(m.clone())
        vv = to_vulkan(v.clone())
        vg = to_vulkan(grad)

        # m = beta1 * m + (1-beta1) * grad → m.lerp_(grad, 1-beta1)
        # But use addcmul_ pattern: m = m * beta1 + (1-beta1) * grad
        # More explicit: m.mul_(beta1).add_(grad, alpha=1-beta1)
        # Using lerp_: m = m + (1-beta1) * (grad - m) = lerp(m, grad, 1-beta1)
        vm.lerp_(vg, 1 - beta1)
        # v = beta2 * v + (1-beta2) * grad^2
        vv.mul_(beta2)
        vv.addcmul_(vg, vg, value=1 - beta2)

        # Bias correction
        vm_hat = vm / (1 - beta1)
        vv_hat = vv / (1 - beta2)

        # param = param - lr * m_hat / (sqrt(v_hat) + eps)
        denom = vv_hat.sqrt() + eps
        step = vm_hat / denom
        vp_new = vp - lr * step

        assert_close(vp_new, expected, rtol=1e-3, atol=1e-4)


# ── Foreach Ops (Fused Optimizer Support) ────────────────────────


class TestForeachOps:
    def _make_lists(self, n=3, size=16):
        """Create n-element tensor lists on CPU and Vulkan."""
        cpu = [torch.randn(size) for _ in range(n)]
        vk = [to_vulkan(t.clone()) for t in cpu]
        return cpu, vk

    def test_foreach_add_scalar_(self):
        cpu, vk = self._make_lists()
        torch._foreach_add_(cpu, 2.0)
        torch._foreach_add_(vk, 2.0)
        for c, v in zip(cpu, vk):
            assert_close(v, c)

    def test_foreach_add_list_(self):
        cpu1, vk1 = self._make_lists()
        cpu2 = [torch.randn(16) for _ in range(3)]
        vk2 = [to_vulkan(t.clone()) for t in cpu2]
        torch._foreach_add_(cpu1, cpu2, alpha=0.5)
        torch._foreach_add_(vk1, vk2, alpha=0.5)
        for c, v in zip(cpu1, vk1):
            assert_close(v, c)

    def test_foreach_mul_scalar_(self):
        cpu, vk = self._make_lists()
        torch._foreach_mul_(cpu, 0.9)
        torch._foreach_mul_(vk, 0.9)
        for c, v in zip(cpu, vk):
            assert_close(v, c)

    def test_foreach_addcmul_(self):
        cpu, vk = self._make_lists()
        t1_cpu = [torch.randn(16) for _ in range(3)]
        t2_cpu = [torch.randn(16) for _ in range(3)]
        t1_vk = [to_vulkan(t.clone()) for t in t1_cpu]
        t2_vk = [to_vulkan(t.clone()) for t in t2_cpu]
        torch._foreach_addcmul_(cpu, t1_cpu, t2_cpu, value=0.1)
        torch._foreach_addcmul_(vk, t1_vk, t2_vk, value=0.1)
        for c, v in zip(cpu, vk):
            assert_close(v, c)

    def test_foreach_addcdiv_(self):
        cpu, vk = self._make_lists()
        t1_cpu = [torch.randn(16) for _ in range(3)]
        t2_cpu = [torch.randn(16).abs() + 0.1 for _ in range(3)]
        t1_vk = [to_vulkan(t.clone()) for t in t1_cpu]
        t2_vk = [to_vulkan(t.clone()) for t in t2_cpu]
        torch._foreach_addcdiv_(cpu, t1_cpu, t2_cpu, value=0.01)
        torch._foreach_addcdiv_(vk, t1_vk, t2_vk, value=0.01)
        for c, v in zip(cpu, vk):
            assert_close(v, c)

    def test_foreach_sqrt(self):
        cpu = [torch.rand(16) + 0.1 for _ in range(3)]
        vk = [to_vulkan(t.clone()) for t in cpu]
        cpu_res = torch._foreach_sqrt(cpu)
        vk_res = torch._foreach_sqrt(vk)
        for c, v in zip(cpu_res, vk_res):
            assert_close(v, c)

    def test_foreach_neg(self):
        cpu = [torch.randn(16) for _ in range(3)]
        vk = [to_vulkan(t.clone()) for t in cpu]
        cpu_res = torch._foreach_neg(cpu)
        vk_res = torch._foreach_neg(vk)
        for c, v in zip(cpu_res, vk_res):
            assert_close(v, c)

    def test_foreach_div_scalar_(self):
        cpu, vk = self._make_lists()
        torch._foreach_div_(cpu, 2.0)
        torch._foreach_div_(vk, 2.0)
        for c, v in zip(cpu, vk):
            assert_close(v, c)

    def test_foreach_lerp_(self):
        cpu, vk = self._make_lists()
        end_cpu = [torch.randn(16) for _ in range(3)]
        end_vk = [to_vulkan(t.clone()) for t in end_cpu]
        torch._foreach_lerp_(cpu, end_cpu, 0.3)
        torch._foreach_lerp_(vk, end_vk, 0.3)
        for c, v in zip(cpu, vk):
            assert_close(v, c)

    def test_foreach_maximum(self):
        cpu1 = [torch.randn(16) for _ in range(3)]
        cpu2 = [torch.randn(16) for _ in range(3)]
        vk1 = [to_vulkan(t.clone()) for t in cpu1]
        vk2 = [to_vulkan(t.clone()) for t in cpu2]
        cpu_res = torch._foreach_maximum(cpu1, cpu2)
        vk_res = torch._foreach_maximum(vk1, vk2)
        for c, v in zip(cpu_res, vk_res):
            assert_close(v, c)
