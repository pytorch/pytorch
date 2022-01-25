"""
In this exercise, we will look into dynamic shapes: the rank and dimensions of
input data. The performance of many operator kernels highly depends on input
shapes. Compilers usually use different optimization strategies or parameters
for different shapes.
"""

import torch

#########################################################
########### Fusion Kernels for Dynamic Shapes ###########
#########################################################

# Let's define a function that applies two elementwise operations on the input
# tensor: sin and cosine


def fn(a: torch.Tensor):
    b = torch.sin(a)
    return torch.cos(b)


# Let's first transform the function to TorchScript IR using `jit.script` and
# observe the scripted graph
scripted_fn = torch.jit.script(fn)
print(scripted_fn.graph)

# As observed, there are no shape annotations for tensors and operators. Due to
# the lack of shape info, fusion optimizations are not applied to the static
# graph.

# Let's set up the shape of input tensor to be (32, 32), run the scripted graph
# and observe the optimized graph. Here we run the scripted graph twice to
# warm up the profiler in the first run. Based on the profiled shape info,
# TorchScript will create a fusion subgraph in the second run.
scripted_fn(torch.randn(32, 32))
scripted_fn(torch.randn(32, 32))
print(torch.jit.last_executed_optimized_graph())

# As observed, `TensorExprGroup_0` is created to fuse sin and consine operators
# for input tensor with shape (1, 64).

# Let's set up the shape of input tensor to be (64, 64), run the scripted graph
# again to observe the compiler behaviors for a different input shape. Same as
# above, we run the scripted graph twice to warm up the profiler.
scripted_fn(torch.randn(64, 64))
scripted_fn(torch.randn(64, 64))
print(torch.jit.last_executed_optimized_graph())

# TorchScript created a second kernel `TensorExprGroup_1` to fuse
# sin and consine operators for input tensor with shape (64, 64). It cannot be
# observed from the last executed graph yet; we are fixing it now (see a Github
# issue regarding this: https://github.com/pytorch/pytorch/issues/52940).

#########################################################
########### Shape Inference #############################
#########################################################

# Shape inference is important for Compilers to propagate shape info throughout
# the graph. Specifically, compilers compute the shapes of output tensors for
# each operation and pass them down in CFG. Below we will use `sum` as an
# example to show how the shapes of output tensors are computed.

# First, we define a function that sums up the 2nd-dimension values of the input tensor.


def fn_sum(a: torch.Tensor):
    return torch.sum(a, 0)


# We create a TorchScript graph for our sum function.
scripted_fn = torch.jit.script(fn_sum)
print(scripted_fn.graph)

# Next, we run the scripted graph with an input tensor with shape (2, 64) and
# observe the shape info annotated in the graph.
scripted_fn(torch.randn(2, 64))
scripted_fn(torch.randn(2, 64))
print(torch.jit.last_executed_optimized_graph())

# The shape of the output tensor is annotated as (64). Note that this is
# oberseved by enabling JIT_LOG for `tensorexpr_fuser`. There are ongoing
# changes added for the NNC dynamic shape feature so you may not see the
# annotations in the printed graph. Use command line
# `PYTORCH_JIT_LOG_LEVEL=">>tensorexpr_fuser" python exe4.py` to print logs and
# see the shape annotations in the logs as shown below:
"""
[DUMP tensorexpr_fuser.cpp:493] graph(%a.1 : Tensor):
[DUMP tensorexpr_fuser.cpp:493]   %3 : bool = prim::Constant[value=0]()
[DUMP tensorexpr_fuser.cpp:493]   %2 : NoneType = prim::Constant()
[DUMP tensorexpr_fuser.cpp:493]   %1 : int[] = prim::Constant[value=[0]]()
[DUMP tensorexpr_fuser.cpp:493]   %4 : Float(64, strides=[1], requires_grad=0, device=cpu) = aten::sum(%a.1, %1, %3, %2) # exe4.py:62:11
[DUMP tensorexpr_fuser.cpp:493]   return (%4)
"""

# If interested, try 'convolution' and see if the output shape of convolution
# computed by TorchScript matches yours!
