import torch
import os 
import ptvsd

#port=3001
#ptvsd.enable_attach(address=('127.0.0.1', port))
#ptvsd.wait_for_attach()

print("press something.")
print(os.getpid())
input()

a = torch.rand(1, 2, device='cuda')
b = torch.rand(1, 1, device='cuda')

print("\ncalling .add")
res = a.add(b)
print("\n\nPrinting res")
print(res)
print("\n\n\n\n")

print("calling .fe_add")
fe_res = torch._foreach_add([a], [b])
print("Printing fe_res")
print(fe_res)

for r1,r2 in zip(res, fe_res): 
    print(r1 == r2)