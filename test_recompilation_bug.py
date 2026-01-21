import torch
def test_topk_out_recompilations(self):
    """Test that topk with out= doesn't cause excessive recompilations"""
    def topk_func(input, k, out):
        torch.topk(input, k, out=out)
    
    opt_model = torch.compile(topk_func)
    
    values = torch.empty(3)
    indices = torch.empty(3, dtype=torch.long)
    
    # Test different input sizes
    x1 = torch.arange(1., 6.)
    opt_model(x1, 3, out=(values, indices))
    
    x2 = torch.arange(1., 8.)  
    opt_model(x2, 3, out=(values, indices))
    
    x3 = torch.arange(1., 10.)
    opt_model(x3, 3, out=(values, indices))
    
    # Should not cause excessive recompilations

def get_num_torch_recompiles():
    guard_failures = torch._dynamo.utils.guard_failures
    num_recompiles = [len(guard_failures[code]) for code in guard_failures]
    return 0 if len(num_recompiles) == 0 else max(num_recompiles)

def topk_func(input, k, out):
    torch.topk(input, k, out=out)

torch._dynamo.reset()
opt_model = torch.compile(topk_func)

values = torch.empty(3)
indices = torch.empty(3, dtype=torch.long)

x = torch.arange(1., 6.)
opt_model(x, 3, out=(values, indices))
print(f"Iter 1: No of recompiles: {get_num_torch_recompiles()}")

x = torch.arange(1., 8.)
opt_model(x, 3, out=(values, indices))
print(f"Iter 2: No of recompiles: {get_num_torch_recompiles()}")

x = torch.arange(1., 10.)
opt_model(x, 3, out=(values, indices))
print(f"Iter 3: No of recompiles: {get_num_torch_recompiles()}")
