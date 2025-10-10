import torch

def fuzzed_program_2():
    _uniq_wide = torch.unique(torch.arange(1)).float()
    return torch.matmul(_uniq_wide, torch.full((1, 18), 0.5))


result_original = fuzzed_program_2()
print("✅ eager success 2")
print(result_original)
compiled_program_2 = torch.compile(fuzzed_program_2, fullgraph=True, dynamic=True)
result_compiled = compiled_program_2()
print("✅ compile success 2")
print(result_compiled)
assert torch.equal(result_original, result_compiled)
print("✅ results match!")