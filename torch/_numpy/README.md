# NumPy <> PyTorch Compat Layer

This folder contains an implementation of (most of) the NumPy public API using PyTorch tensors.
Note that this folder does not depend on NumPy in any way. This is a standalone implementation.

This implementation is used by Dynamo to through NumPy code and lower it into PyTorch code.

To see design decisions that went into this implementation, please see the [rfc](https://github.com/pytorch/rfcs/pull/54).

## Structure of the code

This folder exports a drop-in replacement for the NumPy namespace and its modules `linalg`, `fft` and `random` via its `__init__.py`.

The implementation is split into files that work with PyTorch objects (PyTorch `Tensor`s, dtypes, etc) and files that
use these PyTorch-only files and convert them into functions/objects that can process all the types that the NumPy functions
accept. In particular, they accept `torch._numpy.dtype`s or `torch._numpy.ndarray`s.

The PyTorch-only files are the `*_impl.py` files, while the wrapper files are those that do not have an `*_impl.py`. This creates a
hierarchy, wherein, for example, `_dtypes.py` will import `_dtypes_impl.py`, but not the other way around. In particular, `*_impl.py`
will only depend on other `*_impl.py` files.

As discussed in the [rfc](https://github.com/pytorch/rfcs/pull/54), we use types as tags in our PyTorch implementations. We then use
a decorator called `normalizer` that will inspect these types and preprocess the inputs before sending them to the function. This
preprocessing is the one in charge of mapping array-like objects into `Tensor`s, dtype-like objects into PyTorch dtypes, implement
the `out=` behaviour and so on.

In the files `_funcs.py` and `_ufuncs.py` we use register the `normalizer` decorator to all the `*_impl.py` functions.

In the file `_ndarray.py` we define the `ndarray` class, which is just a thin wrapper around a PyTorch tensor. We use the free functions
and a bit of metaprogramming to implement many of the methods.

## Adding a new function

You just need to add a function in the relevant `*_impl.py` file. You will need to tag the inputs with the relevant Types. After that, you
can assume that the inputs are all PyTorch objects. Your function should return PyTorch tensors. The `normalizer` will make sure that you
always get PyTorch objects. If in doubt, you can see the implementation of the normalization attached to each type annotation in the file
`_normalizations.py`.

## Debugging

It may be useful to figure out whether a given bug is caused by dynamo or the compatibility layer. You may use the compat layer in eager mode
simply by changing `import numpy as np` by `import torch._numpy as np` in your program, without having to call `torch.compile` at all.
Note that `torch._numpy` will be quite slow when used  in eager mode, and it is in no way a replacement or an alternative to the regular PyTorch API.
This should only be used as a debugging tool.
