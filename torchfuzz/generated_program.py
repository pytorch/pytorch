import torch
import sys

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7):
    t0 = arg0 # size=(1, 1, 1), stride=(1, 1, 1), dtype=float32, device=cpu
    t1 = arg1 # size=(1, 1, 1), stride=(1, 1, 1), dtype=float32, device=cpu
    t2 = t0 + t1 + t0 # size=(1, 1, 1), stride=(1, 1, 1), dtype=float32, device=cpu
    t3 = arg2 # size=(1, 6, 1), stride=(6, 1, 1), dtype=float32, device=cpu
    t4 = arg3 # size=(1, 6, 1), stride=(6, 1, 1), dtype=float32, device=cpu
    t5 = t3 + t4 + t3 + t3 # size=(1, 6, 1), stride=(6, 1, 1), dtype=float32, device=cpu
    t6 = torch.cat([t2, t5], dim=1) # size=(1, 7, 1), stride=(7, 1, 1), dtype=float32, device=cpu
    t7 = arg4 # size=(2, 1, 1), stride=(1, 1, 1), dtype=float32, device=cpu
    t8 = arg5 # size=(2, 2, 1), stride=(2, 1, 1), dtype=float32, device=cpu
    t9 = arg6 # size=(2, 2, 1), stride=(2, 1, 1), dtype=float32, device=cpu
    t10 = torch.cat([t7, t8, t9, t9], dim=1) # size=(2, 7, 1), stride=(7, 1, 1), dtype=float32, device=cpu
    t11 = arg7 # size=(2, 7, 1), stride=(7, 1, 1), dtype=float32, device=cpu
    t12 = t10 + t11 + t10 # size=(2, 7, 1), stride=(7, 1, 1), dtype=float32, device=cpu
    t13 = t12 + t10 + t10 # size=(2, 7, 1), stride=(7, 1, 1), dtype=float32, device=cpu
    t14 = torch.cat([t6, t13], dim=0) # size=(3, 7, 1), stride=(7, 1, 1), dtype=float32, device=cpu
    output = t14  # output tensor
    return output

arg0 = torch.empty([1, 1, 1], dtype=torch.float32, device='cpu') # size=(1, 1, 1), stride=(1, 1, 1), dtype=float32, device=cpu
arg1 = torch.empty([1, 1, 1], dtype=torch.float32, device='cpu') # size=(1, 1, 1), stride=(1, 1, 1), dtype=float32, device=cpu
arg2 = torch.empty([1, 6, 1], dtype=torch.float32, device='cpu') # size=(1, 6, 1), stride=(6, 1, 1), dtype=float32, device=cpu
arg3 = torch.empty([1, 6, 1], dtype=torch.float32, device='cpu') # size=(1, 6, 1), stride=(6, 1, 1), dtype=float32, device=cpu
arg4 = torch.empty([2, 1, 1], dtype=torch.float32, device='cpu') # size=(2, 1, 1), stride=(1, 1, 1), dtype=float32, device=cpu
arg5 = torch.empty([2, 2, 1], dtype=torch.float32, device='cpu') # size=(2, 2, 1), stride=(2, 1, 1), dtype=float32, device=cpu
arg6 = torch.empty([2, 2, 1], dtype=torch.float32, device='cpu') # size=(2, 2, 1), stride=(2, 1, 1), dtype=float32, device=cpu
arg7 = torch.empty([2, 7, 1], dtype=torch.float32, device='cpu') # size=(2, 7, 1), stride=(7, 1, 1), dtype=float32, device=cpu
if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    print('Success!')