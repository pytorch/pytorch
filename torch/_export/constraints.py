from typing import Optional

import torch


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
def constrain_as_value(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Add min/max constraint on the intermediate symbol at tracing time. If called in eager mode,
    it will still check if the input value is within the specified range.
    """
    torch.sym_constrain_range(symbol, min=min, max=max)


# TODO: we want to hide this min/max stuff under some abstraction similar to
# DynamicDim
def constrain_as_size(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    This indicates that a given int is size-like, and can be used in any context where a size is expected.
    You will typically use this when reading out integers from Tensors, e.g., max.item() or lengths.tolist()
    which then need to be used as tensor constructors. Providing these assertions to PyTorch can help resolve
      GuardOnDataDependentSymNode errors upon export, since we cannot guard on unbacked SymInts.

    This function has unusual semantics which distinguish it from constrain_as_value.
    Specifically, at compile-time, we will unsoundly assume that the resulting int is always >= 2.
    As a result, max value you pass in should always be greater than 2.
    This makes it easier to use the unbacked int in size contexts, as we will often attempt to guard on a size being zero/one
    (e.g., when computing the contiguity of a tensor, or testing if broadcasting can occur),
    which will not work on unbacked SymInts. Assuming that the int is >= 2 allows us to
    report False to these tests. Although this is technically unsound,
    in practice we observe that if your program works for all sizes >= 2,
    it probably works for zero and one too. The reason specifically assume size is >= 2 is because
    lot of PyTorch code is specialized for 0 and 1 which could result in not general graphs.
    At runtime, we only assert that the user provided min/max values are respected.

    To demonstrate in a scenario, suppose you do
    ```
    # Case 1
    # This will assume symbol is between [2, inf) at compile time, but [0, inf) at runtime
    constrain_as_size(symbol, min=0)

    # Case 2
    # This will assume symbol is between [2, N] at compile time, but [0, N] at runtime
    constrain_as_size(symbol, min=0, max=N)

    # Case 3
    # This is not valid case as max is <= 2
    constrain_as_size(symbol, min=0, max=1)

    # Case 4
    # This will assume symbol is between [2, inf) at compile time, AND [2, inf) at runtime
    constrain_as_size(symbol, min=2)

    # Case 5
    # This will assume symbol is between [2, inf) at compile time, but [1, inf) at runtime
    constrain_as_size(symbol, min=1)
    ```
    """
    torch.sym_constrain_range_for_size(symbol, min=min, max=max)
