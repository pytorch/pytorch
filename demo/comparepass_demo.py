import torch


def callback(unfused_outputs, fused_outputs, graph):
    # note: if _jit_nvfuser_set_comparison_callback(False, callback),
    #       then unfused_outputs will be empty
    found_mismatch = False
    for i in range(len(fused_outputs)):
        unfused = unfused_outputs[i]
        fused = fused_outputs[i]
        diff = torch.max(torch.abs(unfused - fused)).item()
        if diff > 1e-9:
            print(f"Found mismatch in NVFuser fusion output #{i}:")
            print(f"Difference: {diff}")
            print(f"Fusion group:\n{graph}")
            found_mismatch = True
    if not found_mismatch:
        print(f"No issues found in this fusion group: {graph}")


torch._C._jit_nvfuser_set_comparison_callback(True, callback)


def fn(x, y):
    a = torch.exp(x)
    b = torch.exp(y)
    return a + b + x


x = torch.rand((10, 10), dtype=torch.half).cuda() + 5
y = torch.rand((10, 10), dtype=torch.half).cuda() + 5

with torch.jit.fuser("fuser2"):
    fn_s = torch.jit.script(fn)
    fn_s(x, y)
    fn_s(x, y)
    torch._C._jit_nvfuser_clear_comparison_callback()
    print("There should be no more callback output after this:")
    fn_s(x, y)
    fn_s(x, y)
