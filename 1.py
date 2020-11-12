import torch
d1 = torch.int32
d2 = torch.int64

scalar = [1, 2]
a = [torch.tensor([True], dtype=torch.bool, device="cuda") for _ in range(2)]
exp = [t.add(s) for t, s in zip(a, scalar)]
print(exp)
print(exp[0].dtype)

res = torch._foreach_add(a, scalar)
print(res)
print(res[0].dtype)
'''


#a.div_(1)
#print(a)
torch._foreach_div([a], 2)
print(a)

device = 'cpu'
t1 = [torch.zeros(2, 2, device=device, dtype=d1) for n in range(10)]
t2 = [torch.ones(2, 2, device=device, dtype=d2) for n in range(10)]



res = [torch.div(a, b) for a,b in zip(t1, t2)]
print(res)
print(res[0].dtype)

print("\n")
fe_res = torch._foreach_div(t1, t2)
print(fe_res)
print(fe_res[0].dtype)
'''