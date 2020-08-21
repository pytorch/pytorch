import torch
import time

a = [torch.ones(2, 2, device='cuda') for _ in range(3)]
print(a)
res = torch._foreach_add(a, 0)
print(res)
print(a == res)


res = torch._foreach_sqrt(a)
print(res)



'''
a = [torch.ones(1000, 1000, device='cuda', requires_grad=True) for _ in range(499)]
b = [torch.ones(1000, 1000, device='cuda', requires_grad=True) for _ in range(499)]

s = time.time()
torch.cuda.synchronize()
torch._foreach_add(a, b)
e = time.time()
torch.cuda.synchronize()
print(e-s)


s = time.time()
torch.cuda.synchronize()
for a1, b1 in zip(a, b):
    torch.add(a1, b1)
e = time.time()
torch.cuda.synchronize()
print(e-s)
''' 
'''
out = [torch.zeros(100, 200).cuda() for _ in range(12800)]
a = [torch.randn(100, 200).cuda() for _ in range(12800)]
b = [torch.randn(100, 200).cuda() for _ in range(12800)]
ll = [a, b, out]

s = time.time()
torch.cuda.synchronize()
for ai, bi, outi in zip(a, b, out):
    torch.add(ai, bi, out=outi)
torch.cuda.synchronize()
e = time.time()
print(e - s)

s = time.time()
torch.cuda.synchronize()
out2 = torch._foreach_add(a, b)
torch.cuda.synchronize()
e = time.time()
print(e - s)

************************************************************************
out = [torch.zeros(100, 200).cuda() for _ in range(128)]
a = [torch.randn(100, 200).cuda() for _ in range(128)]
b = [torch.randn(100, 200).cuda() for _ in range(128)]
ll = [a, b, out]

t = 0.0
count = 0
while (t < 5.0):
    t0 = time.monotonic()
    torch.cuda.synchronize()
    for ai, bi, outi in zip(a, b, out):
        torch.add(ai, bi, out=outi)
    torch.cuda.synchronize()
    t += time.monotonic() - t0
    count += 1
print("baseline: ", count)

t = 0.0
count = 0
while (t < 5.0):
    t0 = time.monotonic()
    torch.cuda.synchronize()
    out2 = torch._foreach_add(a, b)
    torch.cuda.synchronize()
    t += time.monotonic() - t0
    count += 1
print("count: ", count)
'''