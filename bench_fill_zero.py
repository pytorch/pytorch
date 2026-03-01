"""Benchmark fill_ and zero_ operations on MPS across dtypes, sizes, and layouts."""
import torch
import time

device = torch.device("mps")
WARMUP = 20
ITERS = 200


def bench(fn, label):
    for _ in range(WARMUP):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.mps.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / ITERS
    print(f"  {label:<52s}  {ms:.3f} ms")


def make_tensor(shape, dtype):
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, device=device, dtype=dtype)
    elif dtype == torch.bool:
        return torch.zeros(shape, device=device, dtype=dtype)
    else:
        return torch.randint(0, 100, shape, device=device, dtype=dtype)


print(f"{'case':<54s}  time")
print("-" * 68)

# --- dtype sweep: fill_ ---
print("fill_ dtype sweep  [shape=(4096, 4096)]")
for dtype in [torch.float32, torch.float16, torch.bfloat16,
              torch.int32, torch.int64, torch.int8, torch.uint8, torch.bool]:
    t = make_tensor((4096, 4096), dtype)
    fill_val = True if dtype == torch.bool else 42
    name = str(dtype).replace("torch.", "")
    bench(lambda t=t, v=fill_val: t.fill_(v), f"fill_  {name}")

# --- dtype sweep: zero_ ---
print()
print("zero_ dtype sweep  [shape=(4096, 4096)]")
for dtype in [torch.float32, torch.float16, torch.bfloat16,
              torch.int32, torch.int64, torch.int8, torch.uint8, torch.bool]:
    t = make_tensor((4096, 4096), dtype)
    name = str(dtype).replace("torch.", "")
    bench(lambda t=t: t.zero_(), f"zero_  {name}")

# --- size sweep: fill_ vs zero_ ---
print()
print("size sweep  [dtype=float32]")
for numel in [256, 4096, 65536, 1 << 20, 1 << 24, 1 << 27]:
    t = make_tensor((numel,), torch.float32)
    label_base = f"numel={numel // 1024}K" if numel >= 1024 else f"numel={numel}"
    bench(lambda t=t: t.fill_(1.0), f"fill_  {label_base}")
    bench(lambda t=t: t.zero_(), f"zero_  {label_base}")

# --- 2-D size sweep ---
print()
print("2-D size sweep  [dtype=float32]")
for shape in [(64, 64), (256, 256), (512, 512), (1024, 1024),
              (2048, 2048), (4096, 4096), (8192, 8192)]:
    rows, cols = shape
    t = make_tensor(shape, torch.float32)
    label = f"({rows}x{cols})"
    bench(lambda t=t: t.fill_(1.0), f"fill_  {label}")
    bench(lambda t=t: t.zero_(), f"zero_  {label}")

# --- layout: contiguous vs strided (transposed) ---
print()
print("layout  [shape=(4096, 4096), dtype=float32]")
t_dense = make_tensor((4096, 4096), torch.float32)
t_strided = t_dense.t()  # non-contiguous transposed view
bench(lambda t=t_dense: t.fill_(1.0), "fill_  contiguous")
bench(lambda t=t_dense: t.zero_(), "zero_  contiguous")
bench(lambda t=t_strided: t.fill_(1.0), "fill_  strided (transposed view)")
bench(lambda t=t_strided: t.zero_(), "zero_  strided (transposed view)")

# --- rank sweep ---
print()
print("rank sweep  [dtype=float32, ~16M total elements]")
for shape in [(16777216,), (4096, 4096), (256, 256, 256), (64, 64, 64, 64)]:
    t = make_tensor(shape, torch.float32)
    label = f"shape={shape}"
    bench(lambda t=t: t.fill_(1.0), f"fill_  {label}")
    bench(lambda t=t: t.zero_(), f"zero_  {label}")

# --- non-contiguous slices ---
print()
print("sliced (every-other-row) view  [shape base=(8192, 4096), dtype=float32]")
base = make_tensor((8192, 4096), torch.float32)
t_slice = base[::2]  # stride-2 view along dim 0
bench(lambda t=t_slice: t.fill_(1.0), "fill_  stride-2 slice")
bench(lambda t=t_slice: t.zero_(), "zero_  stride-2 slice")
