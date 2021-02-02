import torch
from torch.fx import symbolic_trace, replace_pattern


'''
How to Use the FX Subgraph Rewriter

For easy subgraph rewriting, FX exposes the utility function:

    replace_pattern(gm : GraphModule,
                    pattern : Callable,
                    replacement : Callable)
                    -> None

`replace_pattern` matches all possible non-overlapping sets of operators
and their data dependencies (`pattern`) in the Graph of a GraphModule
(`gm`), then replaces each of these matched subgraphs with another
subgraph (`replacement).

The docstring for `replace_pattern` (located in `subgraph_rewriter.py`)
gives an in-depth explanation as to how `pattern` and `replacement`
should be specified, what happens during pattern matching, and other
important technical details. This tutorial, therefore, is only meant to
give an overview as to the FX Subgraph Rewriter's basic functionality.
Let's go rewrite a Graph!
'''

# Sample module
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, w2):
        val1 = torch.neg(w1)
        m1 = torch.cat([val1, w2]).sum()
        val2 = torch.neg(w1)
        m2 = torch.cat([val2, w2]).sum()
        return x + torch.max(m1) + torch.max(m2)

# Symbolically trace an instance of `M`
traced = symbolic_trace(M())

# Define the pattern. The FX Subgraph Rewriter will match all
# non-overlapping instances of the pattern in the larger graph.
# Note that Pattern-matching is done based on data dependencies,
# not Node names. Even though we're operating on Nodes named `a1` and
# `a2` instead of `w1` and `w2`, the pattern is still a valid match
# for the two instances of `torch.cat([w1, w2]).sum()` above. Only
# operations that contribute to the single output value of the pattern
# are considered
def pattern(a1, a2):
    val1 = torch.neg(a1)
    return torch.cat([val1, a2]).sum()

# Define the replacement (same rules as the pattern)
def replacement(w1, w2):
    return torch.stack([w1, w2])

# Replace `pattern` with `replacement` in `traced`
replace_pattern(traced, pattern, replacement)

# After calling `replace_pattern`, the generated code is:
'''
def forward(self, x, w1, w2):
    stack_1 = torch.stack([w1, w2])
    sum_1 = stack_1.sum()
    stack_2 = torch.stack([w1, w2])
    sum_2 = stack_2.sum()
    max_1 = torch.max(sum_1)
    add_1 = x + max_1
    max_2 = torch.max(sum_2)
    add_2 = add_1 + max_2
    return add_2
'''
