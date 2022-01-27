import torch
import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as metrics


oldScriptMethod = torch._C.ScriptFunction.__call__


graph_str = """
graph(%a : Tensor, %b : Tensor):
    %2 : int[] = aten::size(%b)
    %3 : Tensor = aten::reshape(%a, %2) 
    return (%3)
"""

graph = torch._C.parse_ir(graph_str)
reshape_bwd = torch._C._create_function_from_graph("reshape_bwd", graph)

def lazyScriptFunction(*args, **kwargs):
    
    # this is JITed function
    self = args[0]

    class AutogradView(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, *view_args): 
            ctx.save_for_backward(input)
            # Note, we reaching to `self`
            # depending on which jit function we are running
            # `self` will be bound to different JIT functions
            nonlocal self
            print ("running AutogradView.forward")
            return lazy_tensor_core._LAZYC._jit_function_call(*((self,) + (input,) + view_args), **{})

        @staticmethod
        def backward(ctx, grad):
            global reshape_bwd
            nonlocal self
            # Note, we need to store a reference to input tensor
            # rather than a ref to a size node since
            # ctx can only save tensors
            (input,) = ctx.saved_tensors
            # _jit_function_call won't work for double backwards
            # since we aren't running through AutogradView
            lazy_tensor_core._LAZYC._print_text("running reshape_bwd")
            out = lazy_tensor_core._LAZYC._jit_function_call(reshape_bwd, grad, input)

            # we need to return gradient for every argument we passed to forward
            # by construction we will be using `AutogradView` to only compute
            # gradient for the first argument. Second argument can still be a tensor
            # but it will be only used to get its size, so no gradient needs to be computed
            nones = [None] * (len(list(self.graph.inputs())) - 1)
            return out, *nones

    if (len(args) > 1 and 'lazy' in str(args[1].device)):
        
        if len(kwargs) > 0:
            raise RuntimeError("Keyword arguments aren't supported!")
        print("inside if")

        return AutogradView.apply(*args[1:])

    return oldScriptMethod(*args, **kwargs)

torch._C.ScriptFunction.__call__ = lazyScriptFunction


@torch.jit.script
def foo(a, b, e: int):
    return a.view(b.size(1), e) 

d = torch.rand(3, 3, device="cuda", requires_grad=True)

a = d.detach().clone().to(device="lazy").requires_grad_(True) #torch.rand(2, 2, device="lazy", requires_grad=True)
c = torch.rand(9, 1, device="lazy", requires_grad=True)
b = foo(a, c, 9)#, 1, 4)
b.sum().backward()
print(a.grad.to(device="cpu").sum())
print("done!")

print(d)
foo(d, c, 9).sum().backward()
print(d.grad.to(device="cpu").sum())

assert(torch.allclose(d.grad.to(device="cpu").sum(), a.grad.to(device="cpu").sum()))
