import torch
import torch._C._te as te
import time


u8 = te.Dtype.Byte
i32 = te.Dtype.Int
f32 = te.Dtype.Float
f64 = te.Dtype.Double
a = te.VarHandle("a", i32)
_4 = te.ExprHandle.int(4)
scale = te.ExprHandle.double(0.1)
zp = te.ExprHandle.double(130.0)
zpf = te.ExprHandle.float(130.0)

B = torch._C._te.BufHandle('B', [_4], f32)
dim_args = [te.DimArg(_4, 'i')]
def compute(i):
    return te.Cast.make(f32,
               (te.Cast.make(u8,
                  te.Cast.make(f64, B.load([i])) / scale + zp)
                - zpf)
              * scale)
A = te.Compute('A', dim_args, compute)
print(A.stmt())

for cname in ['ir_eval', 'llvm']:
    cg = te.construct_codegen(cname, A.stmt(), [torch._C._te.BufferArg(x) for x in [A, B]])
    tA = torch.rand(4) * 5
    tB = torch.arange(4)*60.0
    cg.call([tA, tB])
    print(cname, tA, tB)

