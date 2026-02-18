import torch

def model_func(grad, input, dim, output):
    out = torch.ops.aten.cumprod_backward(grad, input=input, dim=dim, output=output)
    return out

op_config = {
    'grad': torch.randn([1, 100], dtype=torch.float64, device='cuda') * 0.1,
    'input': torch.randn([1, 100], dtype=torch.float64, device='cuda') * 0.1,
    'dim': 1,
    'output': torch.randn([1, 100], dtype=torch.float64, device='cuda') * 0.1,
}

print("Testing cumprod_backward numerical consistency...")
print(f"PyTorch version: {torch.__version__}")

compiled_eager = torch.compile(model_func, backend="eager")
out1 = compiled_eager(**op_config)
print("Eager backend completed")

compiled_inductor = torch.compile(model_func, backend="inductor")
out_inductor = compiled_inductor(**op_config)
print("Inductor backend completed")

print(f"Eager output shape: {out1.shape}")
print(f"Inductor output shape: {out_inductor.shape}")
print(f"Max absolute difference: {torch.abs(out1 - out_inductor).max().item()}")
print(f"Relative error (L2): {torch.norm(out1 - out_inductor) / torch.norm(out1)}")

try:
    torch.testing.assert_close(out1, out_inductor)
    print("✅ Test PASSED: Results are numerically close")
except AssertionError as e:
    print("❌ Test FAILED: Numerical divergence detected")
    print(f"Error details: {e}")