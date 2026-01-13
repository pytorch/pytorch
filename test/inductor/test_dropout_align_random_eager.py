# Owner(s): ["module: inductor"]

import struct
import time

import pytest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    requires_gpu,
)


# ───────────────────────────────────────────────────────────────
# Global config
# ───────────────────────────────────────────────────────────────
BASE_SEED = 1234
DROPOUT_P = 0.3
FFN_DIM = 3072
HIDDEN_DIM = 1024
BATCH = 3
SEQ_LEN = 512


# ───────────────────────────────────────────────────────────────
# Model under test
# ───────────────────────────────────────────────────────────────
class LinearBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = DROPOUT_P):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ffn_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class MultiDropoutBlock(torch.nn.Module):
    """Block with multiple Dropout ops to stress RNG alignment."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = DROPOUT_P):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ffn_dim),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, hidden_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


def build_models(dropout: float, *, mode=None, dynamic: bool = False):
    eager = LinearBlock(HIDDEN_DIM, FFN_DIM, dropout)
    compiled = LinearBlock(HIDDEN_DIM, FFN_DIM, dropout)
    compiled.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled, mode=mode, dynamic=dynamic)
    return eager, compiled


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _set_seed(base: int = BASE_SEED):
    torch.manual_seed(base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base)


def _sync(x: torch.Tensor):
    if x.is_cuda:
        torch.cuda.synchronize()


def _timed_run(model, x, backward: bool = False):
    _sync(x)
    t0 = time.time()
    y = model(x)
    if backward:
        (y.square().mean()).backward()
    _sync(x)
    return (time.time() - t0) * 1e3, y


def _cuda_rng_u64_seed_off():
    """Return (seed, offset) extracted from torch.cuda.get_rng_state()."""
    st = torch.cuda.get_rng_state()
    seed = struct.unpack("<Q", st[0:8].cpu().numpy().tobytes())[0]
    off = struct.unpack("<Q", st[8:24].cpu().numpy().tobytes())[0]
    return seed, off


# ───────────────────────────────────────────────────────────────
# Test class (Inductor idioms)
# ───────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (IS_LINUX and HAS_CUDA_AND_TRITON),
    reason="Inductor CUDA dropout alignment tests require Linux and CUDA",
)
@config.patch(align_random_eager=True)
class TestDropoutAlignRandomEager(InductorTestCase):
    @requires_gpu()
    def test_linear_block_compile_parity_forward(self):
        device = torch.device(GPU_TYPE)

        for training in (False, True):
            eager, compiled = build_models(DROPOUT_P)
            eager.to(device)
            compiled.to(device)

            if training:
                eager.train()
                compiled.train()
            else:
                eager.eval()
                compiled.eval()

            x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

            # same seed before both runs (align dropout masks)
            _set_seed(BASE_SEED)
            with torch.no_grad():
                y_eager = eager(x)

            _set_seed(BASE_SEED)
            with torch.no_grad():
                y_comp = compiled(x)

            torch.testing.assert_close(y_eager, y_comp, rtol=1e-3, atol=1e-4)

    @requires_gpu()
    def test_linear_block_compile_parity_backward(self):
        device = torch.device(GPU_TYPE)

        eager, compiled = build_models(DROPOUT_P)
        eager.to(device)
        compiled.to(device)
        eager.train()
        compiled.train()

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        # eager fwd+bwd
        _set_seed(BASE_SEED)
        y_eager = eager(x)
        (y_eager.square().mean()).backward()

        # compiled fwd+bwd (Re-seed)
        for p in compiled.parameters():
            p.grad = None
        _set_seed(BASE_SEED)
        y_comp = compiled(x)
        (y_comp.square().mean()).backward()

        # outputs
        torch.testing.assert_close(
            y_eager.detach(), y_comp.detach(), rtol=1e-3, atol=1e-4
        )
        # grads
        for p_ref, p_new in zip(eager.parameters(), compiled.parameters()):
            assert p_ref.grad is not None and p_new.grad is not None
            torch.testing.assert_close(p_ref.grad, p_new.grad, rtol=1e-3, atol=1e-5)

    @requires_gpu()
    def test_dropout_mask_parity_and_rng_offset_cuda(self):
        device = torch.device(GPU_TYPE)
        H, W = BATCH * SEQ_LEN, FFN_DIM

        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue

            x = torch.ones((H, W), device=device, dtype=dtype)

            # Eager
            _set_seed(BASE_SEED)
            seed0_e, off0_e = _cuda_rng_u64_seed_off()
            drop_e = torch.nn.Dropout(DROPOUT_P).to(device=device, dtype=dtype).train()
            mask_e = drop_e(x) != 0
            seed1_e, off1_e = _cuda_rng_u64_seed_off()
            delta_e = off1_e - off0_e

            # Compiled
            _set_seed(BASE_SEED)
            seed0_c, off0_c = _cuda_rng_u64_seed_off()
            drop_c = torch.nn.Dropout(DROPOUT_P).to(device=device, dtype=dtype)
            drop_c = torch.compile(drop_c)
            drop_c.train()
            mask_c = drop_c(x) != 0
            seed1_c, off1_c = _cuda_rng_u64_seed_off()
            delta_c = off1_c - off0_c

            assert torch.equal(mask_e, mask_c), (
                "Dropout masks differ between eager and compiled"
            )
            assert seed0_e == seed0_c == BASE_SEED
            assert delta_e == delta_c, (
                f"RNG offset delta mismatch: eager={delta_e}, compiled={delta_c}"
            )

    # ───────────────────────────────────────────────────────────
    # multiple dropouts + multiple iterations
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_multi_dropout_multi_iterations_parity(self):
        device = torch.device(GPU_TYPE)

        eager = MultiDropoutBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled = MultiDropoutBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled.load_state_dict(eager.state_dict())
        compiled = torch.compile(compiled)

        eager.train()
        compiled.train()

        num_iters = 10
        for i in range(num_iters):
            seed = BASE_SEED + i
            x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

            _set_seed(seed)
            y_eager = eager(x)

            _set_seed(seed)
            y_comp = compiled(x)

            torch.testing.assert_close(y_eager, y_comp, rtol=1e-3, atol=1e-4)

    # ───────────────────────────────────────────────────────────
    # dynamic shapes test (a)
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_dropout_parity_dynamic_shapes(self):
        device = torch.device(GPU_TYPE)

        eager = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled.load_state_dict(eager.state_dict())
        compiled = torch.compile(compiled, dynamic=True)

        eager.train()
        compiled.train()

        shapes = [
            (BATCH, 512, HIDDEN_DIM),
            (BATCH, 128, HIDDEN_DIM),
        ]

        for shape in shapes:
            x = torch.randn(*shape, device=device)

            _set_seed(BASE_SEED)
            y_eager = eager(x)

            _set_seed(BASE_SEED)
            y_comp = compiled(x)

            torch.testing.assert_close(y_eager, y_comp, rtol=1e-3, atol=1e-4)

    # ───────────────────────────────────────────────────────────
    # cudagraphs test via mode='reduce-overhead' (b)
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_dropout_parity_cudagraphs_reduce_overhead(self):
        device = torch.device(GPU_TYPE)

        eager = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled.load_state_dict(eager.state_dict())
        compiled = torch.compile(compiled, mode="reduce-overhead")

        eager.train()
        compiled.train()

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        _set_seed(BASE_SEED)
        y_eager = eager(x)

        _set_seed(BASE_SEED)
        y_comp = compiled(x)

        torch.testing.assert_close(y_eager, y_comp, rtol=1e-3, atol=1e-4)

    # ───────────────────────────────────────────────────────────
    # Codegen sanity: run_and_get_code + FileCheck
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_inductor_generated_code_contains_dropout(self):
        device = torch.device(GPU_TYPE)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        model = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        model.train()
        compiled = torch.compile(model)

        def fn(inp):
            return compiled(inp)

        _, codes = run_and_get_code(fn, x)
        assert codes, "Expected inductor to generate at least one kernel"

        # Minimal sanity check that generated code mentions dropout.
        FileCheck().check("dropout").run(codes[0])

    # ───────────────────────────────────────────────────────────
    # Optional: perf smoke (GPU only)
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_perf_smoke_cuda(self):
        device = torch.device(GPU_TYPE)
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        eager, compiled = build_models(DROPOUT_P)
        eager.to(device)
        compiled.to(device)
        eager.eval()
        compiled.eval()

        # warm up
        _timed_run(eager, x, backward=False)
        _timed_run(compiled, x, backward=False)

        t_eager, _ = _timed_run(eager, x, backward=False)
        t_comp, _ = _timed_run(compiled, x, backward=False)

        assert t_comp > 0 and t_eager > 0

    # ───────────────────────────────────────────────────────────
    # Helper for primitive random parity (rand / randn / randint)
    # ───────────────────────────────────────────────────────────
    def _run_primitive_random_parity(self, kind, device, shape):
        if kind == "rand":

            def eager():
                return torch.rand(shape, device=device)

            compiled = torch.compile(eager)

        elif kind == "randn":

            def eager():
                return torch.randn(shape, device=device)

            compiled = torch.compile(eager)

        elif kind == "randint":

            def eager():
                return torch.randint(0, 2**31 - 1, shape, device=device)

            compiled = torch.compile(eager)

        else:
            raise AssertionError(f"unknown primitive random kind: {kind}")

        _set_seed(BASE_SEED)
        out_eager = eager()

        _set_seed(BASE_SEED)
        out_comp = compiled()

        torch.testing.assert_close(out_eager, out_comp, rtol=0.0, atol=0.0)

    # ───────────────────────────────────────────────────────────
    # Primitive random fns: rand / randn / randint -> mark as XFAIL
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    @pytest.mark.xfail(
        reason="primitive torch.rand parity is tracked as future work",
        strict=False,
    )
    def test_primitive_rand_parity(self):
        device = torch.device(GPU_TYPE)
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("rand", device, shape)

    @requires_gpu()
    @pytest.mark.xfail(
        reason="primitive torch.randn parity is tracked as future work",
        strict=False,
    )
    def test_primitive_randn_parity(self):
        device = torch.device(GPU_TYPE)
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("randn", device, shape)

    @requires_gpu()
    @pytest.mark.xfail(
        reason="primitive torch.randint parity is tracked as future work",
        strict=False,
    )
    def test_primitive_randint_parity(self):
        device = torch.device(GPU_TYPE)
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("randint", device, shape)

    # ───────────────────────────────────────────────────────────
    # nn.Dropout as primitive RNG consumer (should PASS)
    # ───────────────────────────────────────────────────────────
    @requires_gpu()
    def test_primitive_nn_dropout_parity(self):
        device = torch.device(GPU_TYPE)
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)

        x = torch.ones(shape, device=device)

        drop_eager = torch.nn.Dropout(DROPOUT_P).to(device).train()
        drop_compiled = torch.nn.Dropout(DROPOUT_P).to(device).train()
        drop_compiled.load_state_dict(drop_eager.state_dict())
        drop_compiled = torch.compile(drop_compiled)

        _set_seed(BASE_SEED)
        out_eager = drop_eager(x)

        _set_seed(BASE_SEED)
        out_comp = drop_compiled(x)

        torch.testing.assert_close(out_eager, out_comp, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
