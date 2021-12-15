# An introduction to some of the (prototype) compilation features in Functorch

The primary compilation API we provide is something called AOTAutograd.

This is currently a prototype feature.

For example, here are some examples of how to use it.
```
from functorch.compile import aot_function, aot_module, draw_graph
import torch.fx as fx
import torch

# This simply prints out the FX graph of the forwards and the backwards
def print_graph(name):
    def f(fx_g: fx.GraphModule, inps):
        print(name)
        print(fx_g.code)
        return fx_g
    return f

def f(x):
    return x.cos().cos()

nf = aot_function(f, fw_compiler=print_graph("forward"), bw_compiler=print_graph("backward"))
nf(torch.randn(3, requires_grad=True))

# You can do whatever you want before and after, and you can still backprop through the function.
inp = torch.randn(3, requires_grad=True)
inp = inp.cos()
out = nf(inp)
out = out.sin().sum().backward()

def f(x):
    return x.cos().cos()

# This draws out the forwards and the backwards graphs as svg files
def graph_drawer(name):
    def f(fx_g: fx.GraphModule, inps):
        draw_graph(fx_g, name)
        return fx_g
    return f

aot_function(f, fw_compiler=graph_drawer("forward"), bw_compiler=graph_drawer("backward"))(torch.randn(3, requires_grad=True))

# We also have a convenience API for applying AOTAutograd to modules
from torchvision.models import resnet18
aot_module(resnet18(), print_graph("forward"), print_graph("backward"))(torch.randn(1,3,200,200))
# output elided since it's very long

# In practice, you might want to speed it up by sending it to Torchscript. You might also lower it to Torchscript before passing it to another compiler

def f(x):
    return x.cos().cos()

def ts_compiler(fx_g: fx.GraphModule, inps):
    f = torch.jit.script(fx_g)
    print(f.graph)
    f = torch.jit.freeze(f.eval()) # Note: This eval() works fine *even* though we're using this for training
    return f

aot_function(f, ts_compiler, ts_compiler)(torch.randn(3, requires_grad=True))
```
