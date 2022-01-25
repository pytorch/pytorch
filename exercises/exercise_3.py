"""
In this exercise, we will learn about dynamism in models, and how it can affects
tracing.  Dynamism in the deep learning models is manifested in typically two
forms

1) Control flow - Presence of if conditions, for loop, while loop.

2) Dynamic shapes - Dynamic input tensors, operators like unique whose output
shape depends on the input data.

In this exercise, we will look into control flow. Dynamic shapes are covered in
later exercises.
"""

import torch

#########################################################
########### Control Flow and Tracing ####################
#########################################################

# Let's create a small function with control flow. We return either sin or
# cosine of tensor a depending on the value of cond.
def fn(a: torch.Tensor, cond: torch.Tensor):
    if cond:
        return torch.sin(a)
    else:
        return torch.cos(a)


# Naturally, it is worth thinking what would trace look like. While tracing, we
# just collect the operations that are observed for a set of input values.

# Lets set up input cond tensor to be True and observe the traced graph
traced_fn1 = torch.jit.trace(fn, (torch.randn(5), torch.tensor(True)))
print(traced_fn1.graph)

# Lets set up input cond tensor to be False and observe the traced graph
traced_fn2 = torch.jit.trace(fn, (torch.randn(5), torch.tensor(False)))
print(traced_fn2.graph)

# As observed, the traces look different. This means that we can't blindly use
# one trace in the presence of control flow. For example, if we use traced_fn1
# and call it with cond set to False, it will still compute sin, which is
# incorrect. Let's check the correctness one by one.

# First, lets check the correctness for True condition - traced_fn1
inp_a = torch.randn(5)
inp_b = torch.tensor(True)
assert torch.allclose(traced_fn1(inp_a, inp_b), fn(inp_a, inp_b))

# Second, lets check the correctness for False condition - traced_fn2
inp_a = torch.randn(5)
inp_b = torch.tensor(False)
assert torch.allclose(traced_fn2(inp_a, inp_b), fn(inp_a, inp_b))


# Now, lets compare the traced_fn1 (True branch) fn and compare it with fn
# evaluated on False.
inp_a = torch.randn(5)
inp_b = torch.tensor(False)
try:
    assert torch.allclose(traced_fn1(inp_a, inp_b), fn(inp_a, inp_b))
except:
    print("Comparison failed which is expected")


# One way to solve this problem is to use torch.jit.script instead of
# torch.jit.trace. Script mode captures control flow here and works for this
# example atleast. But overall, it is still a subset of Python control flow and
# does not support every type of control flow operation.
scripted_fn = torch.jit.script(fn)
print(scripted_fn.graph)
