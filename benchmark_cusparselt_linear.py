import torch
import torch.utils.benchmark as benchmark

def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

print("Running benchmark ...")

device = "cuda"
dtype = torch.float16
batch_size = 2
m, n, k = 16, 16, 16 

print("Creating tensors")
a = torch.randn(batch_size, m, k, device=device, dtype=dtype)
b = torch.randn(k, n, device=device, dtype=dtype)
bias = torch.zeros(n, device=device, dtype=dtype)
res = torch.empty(batch_size, m, n, device=device, dtype=dtype)

print(a)
print(b)
print(res)

c1 = torch.matmul(a, b)

print("Creating class")
cusparse_linear = torch.classes.cusparselt.CusparseLtLinear(a)
print("init")
cusparse_linear.init(b, res, bias)
print("prune")
cusparse_linear.prune()
print("compress")
cusparse_linear.compress()
print("matmul search")
cusparse_linear.search_matmul_algo()
print("masked mm")
cusparse_linear.masked_mm()
# sparse_t = benchmark_torch_function_in_microseconds(cusparse_linear.masked_mm)
# dense_t = benchmark_torch_function_in_microseconds(torch.mm, a, b)

torch.testing.assert_close(res, c1, rtol=1e-3, atol=1e-3)
print(f"sparse_t: {sparse_t:.0f}us dense_t: {dense_t:.0f}us speedup: {dense_t/sparse_t:.2f}x")
