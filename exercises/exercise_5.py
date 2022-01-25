"""
In this exercise, we will look into graph level optimizations more closely
using TorchScript and NNC. First, we show how to use TorchScript passes to
perform common subexpression elimination (CSE). Then we discuss operator
fusion, how it is achieved and what the fused kernel looks like.
"""

import torch

#########################################################
########### Common Subexpression Elimination ############
#########################################################

# Let's define a function that computes the addition, and multiplication of the
# sin and cosine of the input tensor, and returns the sum of all elements.

def fn(a: torch.Tensor):
    b = torch.sin(a) + torch.cos(a)
    c = torch.sin(a) * torch.cos(a)
    return torch.sum(b) + torch.sum(c)

# Inside this function, both `b` and `c` computes the sin and cosine of the
# input tensor, which causes duplicate computation. Let's transform the
# function to TorchScript IR to observe this issue.

scripted_fn = torch.jit.script(fn)
graph = scripted_fn.graph
print(graph)

# As observed, in the graph the computation for sin and cosine of the input
# tensor is replicated. We apply JIT pass "cse" to eliminate the second
# computation.

torch._C._jit_pass_cse(graph)
print(graph)

# As observed, `c` reuses the sin and cosine results after the cse optimization.

# Besides cse, there are a bunch of other JIT passes available on TorchScript.
# Try them out if interested:
# https://github.com/pytorch/pytorch/tree/master/torch/csrc/jit/passes

#########################################################
########### Operator Fusion #############################
#########################################################

# Now let's run the scripted graph twice: it will automatically enable fusion.

scripted_fn(torch.randn(2, 64))
scripted_fn(torch.randn(2, 64))
print(torch.jit.last_executed_optimized_graph())

# As observed, a `TensorExprGroup` is generated which computes the addition and
# multiplication of the sin and cosine of the input tensor. Let's print the NNC
# IR to observe what the fused kernel looks like.

# We first create NNC kernel for the `TensorExprGroup` subgraph.
graph = torch.jit.last_executed_optimized_graph()
node =  graph.findNode("prim::TensorExprGroup", True)
fusion_graph = node.g('Subgraph')
kernel = torch._C._te.TensorExprKernel(fusion_graph)

# Once the kernel is created, we can then print out the corresponding NNC IR.
stmt = kernel.get_codegen_stmt()
print("\nNNC IR:\n", stmt)

# As observed, the fused kernel computes the addition and multiplication of the
# sin and cosine of the input tensor in one nested loop! This saves redundant
# memory accesses to the input tensor which could be quite expensive when the
# its size is large. Moreover, inside the kernel, it vectorized input tensor
# accesses which is also important for fast kernels.
