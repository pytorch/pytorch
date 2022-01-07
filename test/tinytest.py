import torch

"""

There are some common misuse patterns in TorchScript that we should issue
clear error messages for instead of generating generic error that
doesn't capture root cause of error.

Here are a few examples:

Attempting to construct a nn.Module inside TorchScript. This currently
errors out because TorchScript would attempt to compile __init__()
method of module, which usually contains a call to super(), which
isn't supported. Instead, TS should really recognize that a call to
constructor of nn.Module is the real problem.
Calling into known torch.* components that are not scriptable.
For example, torch.distributions currently is not scriptable.
If TS sees a call into torch.distributions methods, it should
warn users about it and prompt them to use jit.trace instead.
Registering new buffers. This isn't supported because it is
effectively modifying type of nn.Module. We should also give a
better error message here.


"""

"""
Attempting to construct a nn.Module inside TorchScript. This currently
errors out because TorchScript would attempt to compile __init__()
method of module, which usually contains a call to super(), which
isn't supported. Instead, TS should really recognize that a call to the
constructor of nn.Module is the real problem.
"""

class M(torch.nn.Module):
    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.add(x, x)

    def __init__(self):
        super().__init__()
        self.nested = self.Nested()

    def forward(self, x):
        return self.nested(x)

scripted = torch.jit.script(M())






#@torch.jit.script
#def foo(a, b):
#    c = a + b
#    e = 2 * a
#    torch.add(c, b, out=e)
#    return e

##print(foo.graph)
#input = torch.rand(2, 3)
#input2 = torch.rand(2, 3)
#foo(input, input2)

#            @torch.jit.script
#            def f(i):
#                # type: (int) -> Tensor
#                l = []
#                for n in [2, 1]:
#                    l.append(torch.zeros(n, i))

#                return l[0]
# torch/csrc/jit/frontend/ir_emitter.cpp:647
# torch/csrc/jit/frontend/schema_matching.cpp

#test_cat_lifts
#test_trace_dict_mix_script
#test_mutable_list_reverse_empty
#test_list_delayed_typing
# test_loop_liveness
# test_torch_tensor_bad_input
