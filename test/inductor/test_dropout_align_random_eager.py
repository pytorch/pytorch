# Owner(s): ["module: inductor"]

import struct
import time
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, IS_LINUX
from torch.testing._internal.inductor_utils import HAS_TRITON


# ───────────────────────────────────────────────────────────────
# Global config
# ───────────────────────────────────────────────────────────────
BASE_SEED = 1234
DROPOUT_P = 0.5
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
def _set_seed(device, base: int = BASE_SEED):
    torch.manual_seed(base)
    dev_mod = torch.get_device_module(device)
    if dev_mod.is_available():
        dev_mod.manual_seed(base)


def _sync(x: torch.Tensor):
    if x.device.type != "cpu":
        torch.get_device_module(x.device).synchronize()


def _timed_run(model, x, backward: bool = False):
    _sync(x)
    t0 = time.time()
    y = model(x)
    if backward:
        (y.square().mean()).backward()
    _sync(x)
    return (time.time() - t0) * 1e3, y


def _rng_seed_and_offset(device):
    """Return (seed, offset) extracted from the current device RNG state.

    The CUDA RNG state is a 16-byte uint8 tensor laid out as
    seed[0:8] + offset[8:16]. The offset is read from bytes [8:16].
    """
    dev_mod = torch.get_device_module(device)
    seed = dev_mod.initial_seed()
    st = dev_mod.get_rng_state()
    off = struct.unpack("<Q", st[8:16].cpu().numpy().tobytes())[0]
    return seed, off


def set_seed(device, seed):
    torch.manual_seed(seed)
    dev_mod = torch.get_device_module(device)
    if dev_mod.is_available():
        dev_mod.manual_seed(seed)


def dropout_parity(device, shape, p=0.3, dtype=torch.float32, seed=1234):
    """Returns (masks_equal, eager_out, compiled_out)."""
    torch._dynamo.reset()
    x = torch.ones(shape, device=device, dtype=dtype)
    drop_e = torch.nn.Dropout(p).to(device).train()
    drop_c = torch.compile(torch.nn.Dropout(p).to(device).train())

    set_seed(device, seed)
    out_e = drop_e(x)
    set_seed(device, seed)
    out_c = drop_c(x)

    masks_eq = torch.equal(out_e != 0, out_c != 0)
    return masks_eq, out_e, out_c


# ───────────────────────────────────────────────────────────────
# Test class (Inductor idioms)
# ───────────────────────────────────────────────────────────────
@unittest.skipIf(
    not (IS_LINUX and HAS_TRITON),
    "Inductor dropout alignment tests require Linux and Triton",
)
class TestDropoutAlignRandomEager(InductorTestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        self._config_ctx = config.patch(align_random_eager=True)
        self._config_ctx.__enter__()

    def tearDown(self):
        self._config_ctx.__exit__(None, None, None)
        super().tearDown()

    def assertSmallMismatchFraction(self, a, b, atol=1e-5, max_fraction=1e-3):
        """Assert that only a small fraction of elements differ significantly.

        The Philox uint32->float32 conversion can produce values on opposite
        sides of the dropout threshold for ~1 in 10^6 elements (architecture-
        and seed-dependent).  One wrong mask bit is then amplified by the
        downstream linear layer, so we check the *fraction* of large
        mismatches rather than requiring every element to be close.
        """
        diff = (a - b).abs()
        bad = (diff > atol).sum().item()
        total = diff.numel()
        fraction = bad / total
        self.assertLessEqual(
            fraction,
            max_fraction,
            f"Mismatch fraction {fraction:.6f} ({bad}/{total}) exceeds {max_fraction}",
        )

    def test_linear_block_compile_parity_forward(self, device):
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
            _set_seed(device, BASE_SEED)
            with torch.no_grad():
                y_eager = eager(x)

            _set_seed(device, BASE_SEED)
            with torch.no_grad():
                y_comp = compiled(x)

            self.assertSmallMismatchFraction(y_eager, y_comp)

    def test_linear_block_compile_parity_backward(self, device):
        eager, compiled = build_models(DROPOUT_P)
        eager.to(device)
        compiled.to(device)
        eager.train()
        compiled.train()

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        # eager fwd+bwd
        _set_seed(device, BASE_SEED)
        y_eager = eager(x)
        (y_eager.square().mean()).backward()

        # compiled fwd+bwd (Re-seed)
        for p in compiled.parameters():
            p.grad = None
        _set_seed(device, BASE_SEED)
        y_comp = compiled(x)
        (y_comp.square().mean()).backward()

        # outputs
        self.assertSmallMismatchFraction(y_eager.detach(), y_comp.detach())
        # grads
        for p_ref, p_new in zip(eager.parameters(), compiled.parameters()):
            self.assertIsNotNone(p_ref.grad)
            self.assertIsNotNone(p_new.grad)
            self.assertSmallMismatchFraction(p_ref.grad, p_new.grad)

    def test_dropout_mask_parity_and_rng_offset(self, device):
        H, W = BATCH * SEQ_LEN, FFN_DIM

        dev_mod = torch.get_device_module(device)
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes:
            if dtype is torch.bfloat16 and not dev_mod.is_bf16_supported():
                continue

            x = torch.ones((H, W), device=device, dtype=dtype)

            # Eager
            _set_seed(device, BASE_SEED)
            seed0_e, off0_e = _rng_seed_and_offset(device)
            drop_e = torch.nn.Dropout(DROPOUT_P).to(device=device, dtype=dtype).train()
            mask_e = drop_e(x) != 0
            seed1_e, off1_e = _rng_seed_and_offset(device)
            delta_e = off1_e - off0_e

            # Compiled
            _set_seed(device, BASE_SEED)
            seed0_c, off0_c = _rng_seed_and_offset(device)
            drop_c = torch.nn.Dropout(DROPOUT_P).to(device=device, dtype=dtype)
            drop_c = torch.compile(drop_c)
            drop_c.train()
            mask_c = drop_c(x) != 0
            seed1_c, off1_c = _rng_seed_and_offset(device)
            delta_c = off1_c - off0_c

            mismatch_ratio = (mask_e != mask_c).float().mean().item()
            self.assertLessEqual(
                mismatch_ratio,
                1e-4,
                msg=lambda msg: (
                    f"{msg}\nDropout mask mismatch ratio too high: {mismatch_ratio:.8f}"
                ),
            )
            self.assertEqual(seed0_e, BASE_SEED)
            self.assertEqual(seed0_c, BASE_SEED)
            self.assertEqual(
                delta_e,
                delta_c,
                msg=lambda msg: (
                    f"{msg}\nRNG offset delta mismatch: eager={delta_e}, compiled={delta_c}"
                ),
            )

    # ───────────────────────────────────────────────────────────
    # multiple dropouts + multiple iterations
    # ───────────────────────────────────────────────────────────
    def test_multi_dropout_multi_iterations_parity(self, device):
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

            _set_seed(device, seed)
            y_eager = eager(x)

            _set_seed(device, seed)
            y_comp = compiled(x)

            mismatch_ratio = ((y_eager != 0) != (y_comp != 0)).float().mean().item()
            self.assertLessEqual(mismatch_ratio, 1e-5)

    # ───────────────────────────────────────────────────────────
    # dynamic shapes test (a)
    # ───────────────────────────────────────────────────────────
    def test_dropout_parity_dynamic_shapes(self, device):
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

            _set_seed(device, BASE_SEED)
            y_eager = eager(x)

            _set_seed(device, BASE_SEED)
            y_comp = compiled(x)

            self.assertSmallMismatchFraction(y_eager, y_comp)

    # ───────────────────────────────────────────────────────────
    # cudagraphs test via mode='reduce-overhead' (b)
    # ───────────────────────────────────────────────────────────
    def test_dropout_parity_cudagraphs_reduce_overhead(self, device):
        eager = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        compiled.load_state_dict(eager.state_dict())
        compiled = torch.compile(compiled, mode="reduce-overhead")

        eager.train()
        compiled.train()

        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        _set_seed(device, BASE_SEED)
        y_eager = eager(x)

        _set_seed(device, BASE_SEED)
        y_comp = compiled(x)

        self.assertSmallMismatchFraction(y_eager, y_comp)

    # ───────────────────────────────────────────────────────────
    # Codegen sanity: run_and_get_code + FileCheck
    # ───────────────────────────────────────────────────────────
    def test_inductor_generated_code_contains_dropout(self, device):
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

        model = LinearBlock(HIDDEN_DIM, FFN_DIM, DROPOUT_P).to(device)
        model.train()
        compiled = torch.compile(model)

        def fn(inp):
            return compiled(inp)

        _, codes = run_and_get_code(fn, x)
        self.assertTrue(codes, msg="Expected inductor to generate at least one kernel")

        # Minimal sanity check that generated code mentions dropout.
        FileCheck().check("dropout").run(codes[0])

    # ───────────────────────────────────────────────────────────
    # Optional: perf smoke (GPU only)
    # ───────────────────────────────────────────────────────────
    def test_perf_smoke(self, device):
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

        self.assertGreater(t_comp, 0)
        self.assertGreater(t_eager, 0)

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

        _set_seed(device, BASE_SEED)
        out_eager = eager()

        _set_seed(device, BASE_SEED)
        out_comp = compiled()

        torch.testing.assert_close(out_eager, out_comp, rtol=0.0, atol=0.0)

    # ───────────────────────────────────────────────────────────
    # Primitive random fns: rand / randn / randint -> mark as XFAIL
    # ───────────────────────────────────────────────────────────
    @unittest.expectedFailure
    def test_primitive_rand_parity(self, device):
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("rand", device, shape)

    @unittest.expectedFailure
    def test_primitive_randn_parity(self, device):
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("randn", device, shape)

    @unittest.expectedFailure
    def test_primitive_randint_parity(self, device):
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)
        self._run_primitive_random_parity("randint", device, shape)

    # ───────────────────────────────────────────────────────────
    # nn.Dropout as primitive RNG consumer (should PASS)
    # ───────────────────────────────────────────────────────────
    def test_primitive_nn_dropout_parity(self, device):
        shape = (BATCH, SEQ_LEN, HIDDEN_DIM)

        x = torch.ones(shape, device=device)

        drop_eager = torch.nn.Dropout(DROPOUT_P).to(device).train()
        drop_compiled = torch.nn.Dropout(DROPOUT_P).to(device).train()
        drop_compiled.load_state_dict(drop_eager.state_dict())
        drop_compiled = torch.compile(drop_compiled)

        _set_seed(device, BASE_SEED)
        out_eager = drop_eager(x)

        _set_seed(device, BASE_SEED)
        out_comp = drop_compiled(x)

        torch.testing.assert_close(out_eager, out_comp, rtol=0.0, atol=0.0)

    # ───────────────────────────────────────────────────────────
    # Large seed (>32-bit) packing truncation
    # Seed and base are packed into int64 as (seed << 32) | base.
    # Seeds > 2^32 overflow.
    # ───────────────────────────────────────────────────────────
    def test_large_seed(self, device):
        for seed in [2**33 + 1, 2**40 + 12345]:
            with self.subTest(seed=seed):
                masks_eq, _, _ = dropout_parity(device, (1024,), seed=seed)
                self.assertTrue(
                    masks_eq, lambda msg: f"{msg}\nseed={seed}: mask mismatch"
                )


instantiate_device_type_tests(
    TestDropoutAlignRandomEager,
    globals(),
    only_for="cuda",
)


if __name__ == "__main__":
    from torch.utils._triton import has_triton

    if has_triton():
        run_tests(needs="filelock")
