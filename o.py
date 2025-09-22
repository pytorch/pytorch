import torch
from triton.testing import do_bench

torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'ieee'

@torch._dynamo.dont_skip_tracing
def foo(x, y):
    return torch.addmm(x, y, y)

foo_c = torch.compile(foo)
foo_c_max_autotune = torch.compile(foo, options={"max_autotune": True})
# foo_c_exhaustive_autotune = torch.compile(foo, options={"max_autotune": True, "max_autotune_gemm_search_space": "EXHAUSTIVE"})
foo_c_exhaustive_autotune = torch.compile(foo, options={"max_autotune": True})
# foo_c = torch.compile(foo)


# Create common tensors for benchmarking
dtype = torch.bfloat16  # Default dtype set to bfloat16
x_tensor = torch.randn(1024, 1024, device="cuda", dtype=dtype)
y_tensor = torch.randn(1024, 1024, device="cuda", dtype=dtype)

# Initial runs to compile the functions
# print("warm up compile")
# foo_c(x_tensor, y_tensor)
# print("warm up max_autotune")
# foo_c_max_autotune(x_tensor, y_tensor)
print("warm up exhaustive_autotune")
import time
start_time = time.time()
foo_c_exhaustive_autotune(x_tensor, y_tensor)
end_time = time.time()
print(f"Execution time: {(end_time - start_time) * 1000:.4f} ms")

# Benchmark each compiled function
# print("compile ms:", do_bench(lambda: foo_c(x_tensor, y_tensor)))
# print("max autotune compile ms:", do_bench(lambda: foo_c_max_autotune(x_tensor, y_tensor)))
print("exhaustive autotune compile ms:", do_bench(lambda: foo_c_exhaustive_autotune(x_tensor, y_tensor)))
