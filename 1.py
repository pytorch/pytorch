import torch 

a = [torch.ones(1, 1, dtype=torch.int32) for _ in range(3)]
b = [torch.ones(1, 1, dtype=torch.int32, device='cuda') for _ in range(3)]
print(a)
print(b)

res = torch._foreach_add(a, [1, 2, 3])
print("\nresA = ", res)

torch._foreach_add_(a, [1, 2, 3])
#print("\nresA = ", a)

#res = torch._foreach_add(b, [1, 2, 3])
#print("\n\n\n\nresB = ", res)

torch._foreach_add_(b, [1, 2, 3])
print("\nresB = ", b)