# NOTE: like ops.py, this file cannot import the torch module unconditionally

# This file is for PyOps builtins that have special mappings to internal
#   PyTorch C++ operations. They must be implemented as valid Python
#   operations with the same semantics so they can be called from Python, too,
#   but their bodies do not need to be transpilable since they have custom
#   C++ transpliations.

# A Python version of the TORCH_CHECK macro
# See https://github.com/pytorch/pytorch/blob/master/c10/util/Exception.h#L325
def TORCH_CHECK(cond: bool, s: str):
    assert cond, s
