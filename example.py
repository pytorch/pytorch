
import torch
from torch.export import Dim, export
class MyModel(torch.nn.Module):
    def forward(self, x, mask):
        masked_select = x.masked_select(mask)
        view = masked_select.view(-1, 1548)
        contig = view.contiguous()
        return contig + 1
example_inputs = (
    torch.randn((768, 1548), dtype=torch.bfloat16),
    torch.randint(low=0, high=1, size=(768,1), dtype=torch.bool),
)
spec = {
    "x": [Dim.STATIC, Dim.STATIC],
    "mask": [Dim.STATIC, Dim.STATIC],
}

traced = export(MyModel(), example_inputs, strict=True)


# import torch

# class Foo(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         return x

#     @staticmethod
#     def backward(ctx, gradOut):
#         breakpoint()
#         return gradOut*10

# # @torch.compile(fullgraph=True, dynamic=True)
# def func(x):
#     a = Foo.apply(x)   
#     return a

# func(torch.rand(6, requires_grad=True))
# import torch
# @torch.compile(dynamic=True)
# def func(rr, x, y):
#     f = x.size()[0]//2
#     m = x.view(f, 2)
#     summ= m.sum()
#     d = x*x.size()[0]
#     d2 = x*y.size()[0]
#     torch._check(x.size()[0]==y.size()[0])

#     return y*d, rr, d2, summ

# func(10, torch.rand(10, requires_grad=True),torch.rand(10,10, requires_grad=True))

# import torch
# @torch.compile()
# def func(x, y):
#     return x.reshape(-1, y.size()[0], y.size()[1])

# x = torch.rand(100,200, 10, 20)
# torch._dynamo.mark_dynamic(x, 0)
# torch._dynamo.mark_dynamic(x, 1)
# y = torch.rand(10, 20)

# # y = torch.rand(20, 40)
# torch._dynamo.maybe_mark_dynamic(y, 0)
# torch._dynamo.maybe_mark_dynamic(y, 1)
# # mark first two dims dynamic

# import fbvscode;
# fbvscode.set_trace()

# func(x,y)
# func(x,torch.rand(40, 20))
