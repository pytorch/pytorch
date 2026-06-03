"""
Official PyTorch blocked_autorange benchmark for NLL loss on MPS.
Compares Metal vs MPSGraph baselines (requires restoring old dylib for baseline).
Run after deploying new dylib.
"""
import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer
import itertools

DEVICE = "mps"
_ = torch.randn(1024, device=DEVICE).sum()
torch.mps.synchronize()

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
REDUCTIONS = ["none", "mean", "sum"]
# N=batch_size, C=num_classes
CONFIGS = [(128, 1000), (1024, 100), (65536, 10), (1024, 10000)]


def warmup(fn, *args, **kwargs):
    for _ in range(10):
        try:
            fn(*args, **kwargs)
        except Exception:
            pass
    torch.mps.synchronize()


def hdr(title):
    print(f"\n{'─'*78}")
    print(f"  {title}")
    print(f"{'─'*78}")
    print(f"  {'(N,C)':<18} {'dtype':<10} {'red':<8} {'mean µs':>10} {'median µs':>10}")
    print(f"  {'─'*18} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")


def row(cfg, dtype, red, m):
    print(f"  {str(cfg):<18} {str(dtype).split('.')[-1]:<10} {red:<8}"
          f" {m.mean*1e6:>10.2f} {m.median*1e6:>10.2f}")


hdr("NLLLoss (blocked_autorange)")
for cfg, dtype, red in itertools.product(CONFIGS, DTYPES, REDUCTIONS):
    N, C = cfg
    lp  = F.log_softmax(torch.randn(N, C, dtype=dtype, device=DEVICE), dim=1)
    tgt = torch.randint(0, C, (N,), device=DEVICE)
    try:
        warmup(F.nll_loss, lp, tgt, reduction=red)
        m = Timer(
            stmt="F.nll_loss(lp, tgt, reduction=red)",
            globals={"F": F, "lp": lp, "tgt": tgt, "red": red},
        ).blocked_autorange(min_run_time=2.0)
        row(cfg, dtype, red, m)
    except Exception as e:
        print(f"  SKIP {cfg} {dtype} {red}: {e}")

print()
