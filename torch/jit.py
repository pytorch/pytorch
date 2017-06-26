import torch.autograd.function as F
import torch._C
from torch.autograd import Variable
import types

# Example how to use:
#
# import torch.jit
# model = model.RNNModel(args.model, ...)
# model = torch.jit.wrap_model(model)

# LIMITATIONS:
# - This assumes that the model will run exactly identically the
#   next time you call forward; if the model looks at some global
#   variables, or has data dependent computation, we will silently
#   give the wrong result.  Adding sanity checking is a TODO.
def wrap_model(model):
    real_forward = model.forward
    def forward(self, *args):
        def flatten(x):
            return tuple(F._iter_variables(x))
        if not hasattr(self, "saved_trace"):
            torch._C._tracer_enter(tuple(self.parameters()) + flatten(args))
            out = real_forward(*args)
            self.saved_trace = torch._C._tracer_exit(flatten(out))
            print(self.saved_trace)
            # TODO: This assignment LEAKS.  Want to ONLY save
            # the shape
            self.saved_outs = out
            return out
        else:
            flat_out = Variable._execution_engine.run_forward(self.saved_trace, tuple(self.parameters()) + flatten(args))
            return F._unflatten(flat_out, self.saved_outs)
    model.forward = types.MethodType(forward, model)
    return model
