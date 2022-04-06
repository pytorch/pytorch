import torch._lazy
import torch._lazy.metrics
import torch._lazy.ts_backend


torch._lazy.ts_backend.init()
torch._C._jit_set_bailout_depth(0)
#torch._C._jit_set_profiling_executor(False)

# # test 1
device = 'lazy'
# a = torch.rand(1, 1).to(device=device)
# b = a.expand(3, 1)
# b.add_(5)
# print(b.cpu())
# print(a.cpu())

# # test 2
a = torch.rand(1).to(device=device)
b = a.expand(3, 1)
c = a.expand(2, 1)
b.add_(4)
c.mul_(3)
print(a.cpu())

# test 3
# a = torch.ones(1).to(device="cpu")
# b = a.expand(3, 1)
# b[1, :] += 4
# print(a.cpu())

# slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)


 Optimized SimpleExecutor Graph:
 graph(%p0 : Tensor,
       %p1 : int,
       %p2 : Tensor,
       %p3 : Tensor):
   %24 : int[] = prim::Constant[value=[2, 1]]()
   %19 : int[] = prim::Constant[value=[1]]()
   %15 : int = prim::Constant[value=1]()
   %13 : int = prim::Constant[value=0]()
   %8 : NoneType = prim::Constant()
   %6 : bool = prim::Constant[value=0]()
   %5 : int[] = prim::Constant[value=[3, 1]]()
   %7 : Tensor = aten::expand(%p3, %5, %6)
   %9 : Tensor = aten::clone(%7, %8)
   %10 : Tensor = aten::add(%9, %p2, %p1)
   %12 : Tensor = aten::clone(%p3, %8)
   %17 : Tensor = aten::slice(%10, %13, %13, %15, %15)
   %20 : Tensor = aten::reshape(%17, %19)
   %22 : Tensor = aten::copy_(%12, %20, %6)

   %26 : Tensor = aten::expand(%12, %24, %6)
   %28 : Tensor = aten::clone(%26, %8)
   %29 : Tensor = aten::mul(%28, %p0)
   %31 : Tensor = aten::clone(%12, %8)
   %36 : Tensor = aten::slice(%29, %13, %13, %15, %15)
   %39 : Tensor = aten::reshape(%36, %19)
   %41 : Tensor = aten::copy_(%31, %39, %6)
   return (%31)
