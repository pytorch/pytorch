#    "--profile" "Export chrome-trace JSON"
#    "--perf"    "Print performance comparison"
#    "--dropout" "Dropout prob (default 0.2)"

import torch, argparse, time, json, os, warnings
import torch, matplotlib.pyplot as plt
import numpy as np

ffn_dim        = 4096
hidden_dim     = 1024
batch, seq_len = 3, 2048
BASE_SEED      = 42
DROPOUT_P      = 0.2

# ───────────────────────────────────────────────────────────────────────────
#  Torch-Inductor compile options
# ───────────────────────────────────────────────────────────────────────────
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.coordinate_descent_tuning  = True
torch._inductor.config.freezing                   = True
torch._inductor.config.align_random_eager         = True
#torch._inductor.config.fallback_random            = True
#torch.use_deterministic_algorithms(True)
#torch._inductor.config.max_autotune_pointwise     = True  # enable if needed
# ───────────────────────────────────────────────────────────────────────────

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
    eager = LinearBlock(hidden_dim, ffn_dim, dropout)
    compiled = LinearBlock(hidden_dim, ffn_dim, dropout)
    compiled.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled)
    return eager, compiled

# ───────────────────────────────────────────────────────────────────────────

def _sync(x):
    if x.is_cuda: torch.cuda.synchronize()

def timed_run(model, x, backward: bool = False):
    """Return elapsed milliseconds of fwd (or fwd+bwd)"""
    _sync(x)
    t0 = time.time()
    y = model(x)
    if backward:
        (y.square().mean()).backward()
    _sync(x)
    return (time.time() - t0) * 1e3

def assert_close_report(a, b, rtol=1e-3, atol=1e-4):
    #torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
    #print("✅ torch.testing.assert_close passed.")
    diff = (a - b).abs()
    tol  = rtol * a.abs() + atol
    num_bad = (diff > tol).sum().item()
    total   = a.numel()
    bad_pct = num_bad / total * 100
    avg_pct = (diff.mean() / a.abs().mean().clamp_min(1e-8)).item() * 100

    if num_bad == 0:
        print("✅ torch.testing.assert_close passed.")
    else:
        print(f"❌ outputs differ! avg diff = {avg_pct:.4f}% | "
              f"bad elements = {num_bad}/{total} ({bad_pct:.2f}%)")
        print(a[:32])
        print(b[:32])

def check_gradients(model_ref, model_new, rtol=1e-5, atol=1e-6):
    max_avg_pct = 0.0
    diff_elems = total_elems = 0

    for p_ref, p_new in zip(model_ref.parameters(), model_new.parameters()):
        if p_ref.grad is None or p_new.grad is None:
            continue

        g_ref, g_new = p_ref.grad, p_new.grad
        # ---- layer-wise avg diff (% of mean) ---------------------
        diff_layer = (g_ref - g_new).abs()
        denom      = g_ref.abs().mean().clamp_min(1e-8)
        max_avg_pct = max(max_avg_pct, (diff_layer.mean() / denom).item() * 100)
        # ---- element-wise diff count (only if large gap later) ---
        if max_avg_pct > 0.1:          # defer counting until needed
            tol = rtol * g_ref.abs() + atol
            diff_elems += (diff_layer > tol).sum().item()
            total_elems += g_ref.numel()

    if max_avg_pct < 0.1:
        print(f"✅ gradients close (max avg diff = {max_avg_pct:.4f}%)")
    else:
        pct_elem = diff_elems / total_elems * 100 if total_elems else 0
        print(f"❌ gradients differ! (max avg diff = {max_avg_pct:.4f}%)")
        print(f"   ↳ differing elements: {diff_elems}/{total_elems} "
              f"({pct_elem:.2f}%) > rtol={rtol}, atol={atol}")

# ───────────────────────────────────────────────────────────────────────────
#  Performance & Profiling
# ───────────────────────────────────────────────────────────────────────────
def do_perf(eager, compiled, x):
    # ──────────────── Inference ───────────────────────
    eager.eval(); compiled.eval()

    _ = eager(x)
    t_eager = timed_run(eager, x, backward=False)        # steady-state baseline

    t_comp_first = timed_run(compiled, x, backward=False)
    t_comp = timed_run(compiled, x, backward=False)

    compile_lat_inf = t_comp_first - t_comp
    speed_up_inf = (t_eager - t_comp) / t_eager * 100

    print(f"Inference | eager={t_eager:.2f} ms  "
          f"compiled={t_comp:.2f} ms  "
          f"speed-up={speed_up_inf:.2f}%  "
          f"compile latency={compile_lat_inf:.2f} ms")

    # ──────────────── Training (fwd+bwd) ──────────────
    eager.train(); compiled.train()
    for m in (eager, compiled):
        for p in m.parameters(): p.grad = None

    _ = timed_run(eager, x, backward=True)               # warm-up eager
    for p in eager.parameters(): p.grad = None
    t_eager_tr = timed_run(eager, x, backward=True)      # steady-state baseline

    for p in compiled.parameters(): p.grad = None
    t_comp_first_tr = timed_run(compiled, x, backward=True)
    for p in compiled.parameters(): p.grad = None
    t_comp_tr = timed_run(compiled, x, backward=True)

    compile_lat_tr = t_comp_first_tr - t_comp_tr
    speed_up_tr = (t_eager_tr - t_comp_tr) / t_eager_tr * 100

    print(f"Training  | eager={t_eager_tr:.2f} ms  "
          f"compiled={t_comp_tr:.2f} ms  "
          f"speed-up={speed_up_tr:.2f}%  "
          f"compile latency={compile_lat_tr:.2f} ms")
def do_profile(eager, compiled, x, out_file="linear_block_profile.json"):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities,
                                record_shapes=True) as prof:
        # ───── eager: inference + training ─────
        eager.eval()
        _ = eager(x)
        
        torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
        eager.train()
        (eager(x).square().mean()).backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ───── compiled: inference + training ──
        compiled.eval()
        _ = compiled(x)        
        
        torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
        compiled.train()
        (compiled(x).square().mean()).backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    prof.export_chrome_trace(out_file)
    print(f"Chrome trace JSON saved to {out_file}")
# ───────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LinearBlock compile parity test")
    parser.add_argument("--profile", action="store_true", help="Export chrome-trace JSON")
    parser.add_argument("--perf",    action="store_true", help="Print performance comparison")
    parser.add_argument("--dropout", type=float, default=DROPOUT_P, help="Dropout prob (default 0.2)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(batch, seq_len, hidden_dim, device=device)

    eager, compiled = build_models(args.dropout)
    eager.to(device); compiled.to(device)

    # correctness (forward) & gradients
    eager.eval(); compiled.eval()
    with torch.no_grad():
        assert_close_report(eager(x), compiled(x))
    
    torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
    eager.train(); 
    eager_x = eager(x)
    (eager(x).square().mean()).backward()

    torch.manual_seed(BASE_SEED); torch.cuda.manual_seed(BASE_SEED)
    compiled.train()

    st0 = torch.cuda.get_rng_state()

    compiled_x = compiled(x)

    st1 = torch.cuda.get_rng_state()

    import struct, math
    off0 = struct.unpack("<Q", st0[8:24].cpu().numpy().tobytes())[0]
    off1 = struct.unpack("<Q", st1[8:24].cpu().numpy().tobytes())[0]
    print(off1 - off0)
    print(f"off1={off1}; off0={off0}")
    seed0 = struct.unpack("<Q", st0[0:8].cpu().numpy().tobytes())[0]
    seed1 = struct.unpack("<Q", st1[0:8].cpu().numpy().tobytes())[0]
    print(f"seed0={seed0}; seed1={seed1}")

    (compiled(x).square().mean()).backward()
    
    assert_close_report(eager_x, compiled_x)
    check_gradients(eager, compiled)

    # perf / profile
    if args.perf:
        do_perf(eager, compiled, x)
    if args.profile:
        do_profile(eager, compiled, x)
def visualize_map():
    # --- argparse: dtype switch ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="Tensor dtype (default: float32)")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # --- Runtime Config ---
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print(device, "| dtype:", dtype)

    torch.manual_seed(BASE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(BASE_SEED)

    p = DROPOUT_P
    H, W = batch * seq_len, ffn_dim
    x = torch.ones((H, W), device=device, dtype=dtype)

    # ----- eager -----
    torch.manual_seed(BASE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(BASE_SEED)
    dropout_eager = torch.nn.Dropout(p).to(device=device, dtype=dtype)
    dropout_eager.train()
    mask_eager = dropout_eager(x) != 0
    print(mask_eager)

    # ----- compiled -----
    torch.manual_seed(BASE_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(BASE_SEED)
    dropout_compiled = torch.compile(torch.nn.Dropout(p).to(device=device, dtype=dtype))
    dropout_compiled.train()
    mask_comp = dropout_compiled(x) != 0
    # print(mask_comp)

    # ----- diff (only where different -> red overlay) -----
    diff = (mask_eager ^ mask_comp).to(torch.uint8).cpu().numpy().astype(bool)

    # ----- build grayscale bases -----
    left_gray = (mask_eager.to(torch.uint8).cpu().numpy()) * 255  # 0/255
    right_gray = (mask_comp.to(torch.uint8).cpu().numpy()) * 255

    # ----- draw side-by-side with red alpha overlay (alpha=0.4) -----
    # fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=150)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=150)

    for ax, gray, title in zip(axes, [left_gray, right_gray], ["Eager", "torch.compile"]):
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        # red RGBA overlay where diff==True
        overlay = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.float32)
        overlay[..., 0] = 1.0  # R
        overlay[..., 1] = 0.0  # G
        overlay[..., 2] = 0.0  # B
        overlay[..., 3] = diff * 0.7  # A (only on differing pixels)
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout(pad=0.4)
    plt.savefig("masks_compare.png", bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"  [mask] saved side-by-side image: masks_compare.png  (H={H}, W={W}, diff_pixels={diff.sum()})")

if __name__ == "__main__":
    torch.manual_seed(BASE_SEED)
    warnings.filterwarnings("ignore")
    print("Functional test: (Dropout in a linear block)")
    main()
    print("Visualization of dropout masks:")
    visualize_map()
