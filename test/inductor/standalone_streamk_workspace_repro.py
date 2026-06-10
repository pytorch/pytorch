import os


os.environ.setdefault("TORCHINDUCTOR_ENABLE_STREAMK", "1")
os.environ.setdefault("TORCHINDUCTOR_STREAMK_AUTOTUNE", "1")
os.environ.setdefault("TORCHINDUCTOR_STREAMK_ONLY", "1")
os.environ.setdefault("TORCHINDUCTOR_STREAMK_DEBUG", "1")

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this StreamK repro")

    m, n, k = 512, 304, 12288
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).t()

    compiled_mm = torch.compile(torch.mm)
    actual = compiled_mm(a, b)
    expected = torch.mm(a, b)
    torch.testing.assert_close(actual, expected, atol=1.0, rtol=1e-2)


if __name__ == "__main__":
    main()
