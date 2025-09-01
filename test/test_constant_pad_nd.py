import pytest
import torch
import torch.nn.functional as F

# --- EAGER BEHAVIOR TESTS (most important) ---

def test_constant_pad_nd_mixed_pos_neg_allows_zero_cpu():
    x = torch.ones(5, 3)
    y = torch.ops.aten.constant_pad_nd.default(x, [-1, -2, 1, 1], 0)
    assert tuple(y.shape) == (7, 0)
    assert y.numel() == 0

def test_constant_pad_nd_negative_size_still_errors_cpu():
    x = torch.ones(5, 3)
    # This would make width 3 - 2 - 2 = -1 → must error
    with pytest.raises(RuntimeError):
        torch.ops.aten.constant_pad_nd.default(x, [-2, -2], 0)

def test_fpad_matches_constant_pad_nd_cpu():
    x = torch.ones(5, 3)
    z = F.pad(x, (-1, -2, 1, 1), mode="constant", value=0)
    assert tuple(z.shape) == (7, 0)
    assert z.numel() == 0

# Optional: run on MPS/CUDA if available (kept simple to avoid internal test infra)
@pytest.mark.parametrize("device", [d for d in ["cpu", "cuda", "mps"] if torch.device(d).type in {"cpu","cuda","mps"} and (d=="cpu" or torch.cuda.is_available() if d=="cuda" else torch.backends.mps.is_available())])
def test_constant_pad_nd_devices(device):
    x = torch.ones(5, 3, device=device)
    y = torch.ops.aten.constant_pad_nd.default(x, [-1, -2, 1, 1], 0)
    assert tuple(y.shape) == (7, 0)
    assert y.numel() == 0