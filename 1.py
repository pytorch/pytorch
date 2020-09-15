import torch 

a = [torch.ones(1, 1, dtype=torch.int16) for _ in range(3)]
b = [torch.ones(1, 1, dtype=torch.int16, device='cuda') for _ in range(3)]
#print(a)
#print(b)


#res = torch._foreach_add(b, 2)

#res = torch._foreach_sub(b, 2)
#res = torch._foreach_mul(b, 2)
#res = torch._foreach_div(b, 2)
#print(res)
#
#res = torch._foreach_add(b, [2, 2, 2])
#res = torch._foreach_sub(b, [2, 2, 2])

print('\n\n\n\n\nhello')
res = torch._foreach_mul(b, [2, 2, 2])
print('\n\n\n\n\nhello div')
res = torch._foreach_div(b, [2, 2, 2])
print(res)
