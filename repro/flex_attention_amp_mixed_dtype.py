import torch
from torch.nn.attention.flex_attention import flex_attention


def noop(score, b, h, q_idx, kv_idx):
    return score


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping repro")
        return
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


