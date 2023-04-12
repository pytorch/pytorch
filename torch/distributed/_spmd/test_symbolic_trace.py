import os
from typing import cast, List, Optional, Callable, Dict
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx 
from torch._functorch.compilers import aot_function
import torch
import torch._inductor.config
from torch._functorch.compilers import aot_function
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func


torch._dynamo.reset()

# simple model definition
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net1 = nn.Linear(50, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 5)

    def forward(self, x):
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))
    
    def reset_parameters(self, *args, **kwargs):
        self.net1.reset_parameters()
        self.net2.reset_parameters()

# simple train step definition, just an example
def train_step(model, loss_fn, optim, train_batch):
    optim.zero_grad()  
    inputs, labels = train_batch
    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()
    # optim.step()
    return loss
    

graphs = []
def custom_compiler(fx_g, inputs):
    # compiled_g = compile_fx(fx_g, inputs)
    # print("??!!!!")
    # fx_g.print_readable()
    fx_g.graph.print_tabular()
    graphs.append(fx_g)
    return fx_g.forward
    
    # return make_boxed_func(fx_g.forward)

# my_backend = aot_autograd(fw_compiler=custom_compiler)
my_backend = custom_compiler

# compiled_fn = torch.compile(train_step, dynamic=False, backend=my_backend)
model = SimpleMLP()
LR = 0.25
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
def loss_fn(out, labels):
    return (out - labels).sum()
x = torch.randn(100, 50)
y = torch.randn(100, 5)

compiled_fn = torch.compile(train_step, backend=my_backend)
compiled_fn(model, loss_fn, optimizer, (x, y))
# print(graphs)
# print(f">>> forward graph: ")
# print(graphs[0].graph)
# print(f">>> backward graph: ")
# print(graphs[1].graph)
# print(f">>> optim graph: ")
# print(graphs[2].graph)


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward  # return a python callable

torch._dynamo.reset()

@torch.compile(backend=my_compiler)
def fn(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b


fn(torch.randn(10), torch.randn(10))
