import importlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import torch


def noop(score, b, h, q_idx, kv_idx):
    return score


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping repro")
        return
    # Import installed flex_attention and wrap it with AMP harmonization logic
    from torch.nn.attention.flex_attention import flex_attention as _orig_flex_attention

    def flex_attention(query, key, value, *args, **kwargs):
        try:
            device_type = query.device.type
            _fp8_types = (torch.float8_e4m3fn, torch.float8_e5m2)
            if (
                hasattr(torch, "is_autocast_enabled")
                and torch.is_autocast_enabled(device_type=device_type)  # type: ignore[call-arg]
                and query.dtype not in _fp8_types
                and key.dtype not in _fp8_types
                and value.dtype not in _fp8_types
            ):
                target_dtype = torch.get_autocast_dtype(device_type)  # type: ignore[arg-type]
                if (
                    query.dtype != target_dtype
                    or key.dtype != target_dtype
                    or value.dtype != target_dtype
                ):
                    query = query.to(target_dtype)
                    key = key.to(target_dtype)
                    value = value.to(target_dtype)
        except Exception:
            pass
        return _orig_flex_attention(query, key, value, *args, **kwargs)
    q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float32)
    v = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float32)

    print("Running SDPA under autocast (should work):")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print("OK")

    print("Running FlexAttention under autocast (should work after fix):")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = flex_attention(q, k, v, score_mod=noop)
        print("OK, dtype:", out.dtype, "device:", out.device)

    print("Running FlexAttention outside autocast with mixed dtypes (should fail):")
    try:
        _ = flex_attention(q, k, v, score_mod=noop)
        print("Unexpected success")
    except ValueError as e:
        print("Expected failure:", e)


if __name__ == "__main__":
    main()


