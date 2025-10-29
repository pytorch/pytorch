# Owner(s): ["module: inductor"]

import struct
import time
import pytest
import torch

# ───────────────────────────────────────────────────────────────
# Global config
# ───────────────────────────────────────────────────────────────
BASE_SEED = 42
DROPOUT_P = 0.2
FFN_DIM = 4096
HIDDEN_DIM = 1024
BATCH = 3
SEQ_LEN = 2048

# Torch-Inductor knobs
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.freezing = True
torch._inductor.config.align_random_eager = True
# torch._inductor.config.fallback_random = True
# torch._inductor.config.max_autotune_pointwise = True

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

def build_models(dropout: float):
    eager = LinearBlock(HIDDEN_DIM, FFN_DIM, dropout)
    compiled = LinearBlock(HIDDEN_DIM, FFN_DIM, dropout)
    compiled.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled)
    return eager, compiled

# ───────────────────────────────────────────────────────────────
# Pytest fixtures
# ───────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _set_seed():
    torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(BASE_SEED)
    yield

@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    dev = torch.device(request.param)
    if dev.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return dev

@pytest.fixture
def input_tensor(device):
    return torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=device)

# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
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
    off  = struct.unpack("<Q", st[8:24].cpu().numpy().tobytes())[0]
    return seed, off

# ───────────────────────────────────────────────────────────────
# Tests: correctness (forward/backward)
# ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("training", [False, True])
def test_linear_block_compile_parity_forward(device, input_tensor, training):
    # CPU not support Align Random Eager for now：skip compile+dropout parity
    if device.type == "cpu":
        pytest.skip("Align Random Eager not supported on CPU yet; skip compile+dropout parity on CPU")

    eager, compiled = build_models(DROPOUT_P)
    eager.to(device); compiled.to(device)

    if training:
        eager.train(); compiled.train()
    else:
        eager.eval(); compiled.eval()

    # same seed before both runs (align dropout masks)
    torch.manual_seed(BASE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(BASE_SEED)

    with torch.no_grad():
        y_eager = eager(input_tensor)
        # reset seed for compiled run
        torch.manual_seed(BASE_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed(BASE_SEED)
        y_comp = compiled(input_tensor)

    torch.testing.assert_close(y_eager, y_comp, rtol=1e-3, atol=1e-4)

def test_linear_block_compile_parity_backward(device, input_tensor):
    # CPU not support Align Random Eager for now：skip compile+dropout parity
    if device.type == "cpu":
        pytest.skip("Align Random Eager not supported on CPU yet; skip compile+dropout parity on CPU")

    eager, compiled = build_models(DROPOUT_P)
    eager.to(device); compiled.to(device)
    eager.train(); compiled.train()

    # eager fwd+bwd
    torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
    y_eager = eager(input_tensor)
    (y_eager.square().mean()).backward()

    # compiled fwd+bwd (Re-seed)
    for p in compiled.parameters():
        p.grad = None
    torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
    y_comp = compiled(input_tensor)
    (y_comp.square().mean()).backward()

    # outputs
    torch.testing.assert_close(y_eager.detach(), y_comp.detach(), rtol=1e-3, atol=1e-4)
    # grads
    for p_ref, p_new in zip(eager.parameters(), compiled.parameters()):
        assert p_ref.grad is not None and p_new.grad is not None
        torch.testing.assert_close(p_ref.grad, p_new.grad, rtol=1e-3, atol=1e-5)

# ───────────────────────────────────────────────────────────────
# Tests: dropout mask parity & CUDA RNG offset alignment
# ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16] if torch.cuda.is_available() else [],
)
def test_dropout_mask_parity_and_rng_offset_cuda(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        pytest.skip("bf16 requires CUDA")

    device = torch.device("cuda")
    H, W = BATCH * SEQ_LEN, FFN_DIM
    x = torch.ones((H, W), device=device, dtype=dtype)

    # Eager
    torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
    seed0_e, off0_e = _cuda_rng_u64_seed_off()
    drop_e = torch.nn.Dropout(DROPOUT_P).to(device=device, dtype=dtype).train()
    mask_e = drop_e(x) != 0
    seed1_e, off1_e = _cuda_rng_u64_seed_off()
    delta_e = off1_e - off0_e

    # Compiled
    torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
    seed0_c, off0_c = _cuda_rng_u64_seed_off()
    drop_c = torch.compile(torch.nn.Dropout(DROPOUT_P).to(device=device, dtype=dtype))
    drop_c.train()
    mask_c = drop_c(x) != 0
    seed1_c, off1_c = _cuda_rng_u64_seed_off()
    delta_c = off1_c - off0_c

    assert torch.equal(mask_e, mask_c), "Dropout masks differ between eager and compiled"
    assert seed0_e == seed0_c == BASE_SEED
    assert delta_e == delta_c, f"RNG offset delta mismatch: eager={delta_e}, compiled={delta_c}"

# ───────────────────────────────────────────────────────────────
# Optional: perf smoke (GPU only)
# ───────────────────────────────────────────────────────────────
def test_perf_smoke_cuda(input_tensor):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")
    x = input_tensor.to(device)

    eager, compiled = build_models(DROPOUT_P)
    eager.to(device); compiled.to(device)
    eager.eval(); compiled.eval()

    # warm up
    _timed_run(eager, x, backward=False)
    _timed_run(compiled, x, backward=False)

    t_eager, _ = _timed_run(eager, x, backward=False)
    t_comp, _ = _timed_run(compiled, x, backward=False)

    assert t_comp > 0 and t_eager > 0

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
