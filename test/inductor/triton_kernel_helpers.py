import triton
from triton.language import maximum as tl_maximum


@triton.constexpr_function
def transitive_constexpr_helper(x):
    return x + 1


@triton.jit
def maximum(a, b):
    return tl_maximum(a, b) + 100


@triton.jit
def transitive_jit_helper(x):
    return x + 1


@triton.jit
def transitive_triton_alias_kernel(a, b):
    return maximum(a, b)
