---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3.9.12 64-bit ('env')
  language: python
  name: python3
---

# Sparse semantics

+++

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/maskedtensor/blob/main/docs/source/notebooks/sparse.ipynb)

+++

## Introduction

+++

[Sparsity in PyTorch](https://pytorch.org/docs/stable/sparse.html) is a quickly growing area that has found a lot of support and demand due to its efficiency in both memory and compute. This tutorial is meant to be used in conjunction with the the PyTorch link above, as the sparse tensors are ultimately the building blocks for MaskedTensors (just as regular `torch.Tensor`s are as well).

Sparse storage formats have been proven to be powerful in a variety of ways. As a primer, the first use case most practitioners think about is when the majority of elements are equal to zero (a high degree of sparsity), but even in cases of lower sparsity, certain formats (e.g. BSR) can take advantage of substructures within a matrix. There are a number of different [sparse storage formats](https://en.wikipedia.org/wiki/Sparse_matrix) that can be leveraged with various tradeoffs and degrees of adoption.

"Specified" and "unspecified" elements (e.g. elements that are stored vs. not) have a long history in PyTorch without formal semantics and certainly without consistency; indeed, MaskedTensor was partially born out of a build up of issues (e.g. the [nan_grad tutorial](https://pytorch.org/maskedtensor/main/notebooks/nan_grad.html)) that vanilla tensors could not address. A major goal of the MaskedTensor project is to become the primary source of truth for specified/unspecified semantics where they are a first class citizen instead of an afterthought.

<div class="alert alert-info">

Note: Currently, only the COO and CSR sparse storage formats are supported in MaskedTensor (BSR and CSC will be developed in the future). If you have another format that you would like supported, please file an issue!

</div>

+++

## Principles

+++

1. `input` and `mask` must have the same storage format, whether that's `torch.strided`, `torch.sparse_coo`, or `torch.sparse_csr`.

2. `input` and `mask` must have the same size, indicated by `t.size()`

+++

## Sparse COO Tensors

```{code-cell} ipython3
import torch
from maskedtensor import masked_tensor
```

In accordance with Principle #1, a sparse MaskedTensor is created by passing in two sparse tensors, which can be initialized with any of the constructors, e.g. `torch.sparse_coo_tensor`.

As a recap of [sparse COO tensors](https://pytorch.org/docs/stable/sparse.html#sparse-coo-tensors), the COO format stands for "Coordinate format", where the specified elements are stored as tuples of their indices and the corresponding values. That is, the following are provided:

- `indices`: array of size `(ndim, nse)` and dtype `torch.int64`
- `values`: array of size `(nse,)` with any integer or floating point number dtype

where `ndim` is the dimensionality of the tensor and `nse` is the number of specified elements

+++

For both sparse COO and CSR tensors, you can construct them by doing either:

1. `masked_tensor(sparse_tensor_data, sparse_tensor_mask)`
2. `dense_masked_tensor.to_sparse_coo()`

The is second is easier to illustrate so we have shown that below, but for more on the first and the nuances behind the approach, please read the Appendix at the bottom.

```{code-cell} ipython3
# To start, create a MaskedTensor
values = torch.tensor(
     [[0, 0, 3],
      [4, 0, 5]]
)
mask = torch.tensor(
     [[False, False, True],
      [False, False, True]]
)
mt = masked_tensor(values, mask)

sparse_coo_mt = mt.to_sparse_coo()

print("masked tensor:\n", mt)
print("sparse coo masked tensor:\n", sparse_coo_mt)
print("sparse data:\n", sparse_coo_mt.data())
```

## Sparse CSR Tensors

+++

Similarly, MaskedTensor also supports the [CSR (Compressed Sparse Row)](https://pytorch.org/docs/stable/sparse.html#sparse-csr-tensor) sparse tensor format. Instead of storing the tuples of the indices like sparse COO tensors, sparse CSR tensors aim to decrease the memory requirements by storing compressed row indices. In particular, a CSR sparse tensor consists of three 1-D tensors:

- `crow_indices`: array of compressed row indices with size `(size[0] + 1,)`. This array indicates which row a given entry in `values` lives in. The last element is the number of specified elements, while `crow_indices[i+1] - crow_indices[i]` indicates the number of specified elements in row `i`.
- `col_indices`: array of size `(nnz,)`. Indicates the column indices for each value.
- `values`: array of size `(nnz,)`. Contains the values of the CSR tensor.

Of note, both sparse COO and CSR tensors are in a [beta](https://pytorch.org/docs/stable/index.html) state.

By way of example (and again, you can find more examples in the Appendix):

```{code-cell} ipython3
mt_sparse_csr = mt.to_sparse_csr()

print("values:\n", mt_sparse_csr.data())
print("mask:\n", mt_sparse_csr.mask())
print("mt:\n", mt_sparse_csr)
```

## Supported Operations

+++

### Unary

+++

[All unary operations are supported](https://pytorch.org/maskedtensor/main/unary.html), e.g.:

```{code-cell} ipython3
mt.sin()
```

### Binary

+++

[Binary operations are also supported](https://pytorch.org/maskedtensor/main/binary.html), but the input masks from the two masked tensors must match.

```{code-cell} ipython3
i = [[0, 1, 1],
     [2, 0, 2]]
v1 = [3, 4, 5]
v2 = [20, 30, 40]
m = torch.tensor([True, False, True])

s1 = torch.sparse_coo_tensor(i, v1, (2, 3))
s2 = torch.sparse_coo_tensor(i, v2, (2, 3))
mask = torch.sparse_coo_tensor(i, m, (2, 3))

mt1 = masked_tensor(s1, mask)
mt2 = masked_tensor(s2, mask)
```

```{code-cell} ipython3
print("mt1:\n", mt1)
print("mt2:\n", mt2)
print("torch.div(mt2, mt1):\n", torch.div(mt2, mt1))
print("torch.mul(mt1, mt2):\n", torch.mul(mt1, mt2))
```

### Reductions

+++

At the moment, when the underlying data is sparse, only [reductions](https://pytorch.org/maskedtensor/main/reductions.html) across all dimensions are supported and not a particular dimension (e.g. `mt.sum()` is supported but not `mt.sum(dim=1)`). This is next in line to work on.

```{code-cell} ipython3
print("mt:\n", mt)
print("mt.sum():\n", mt.sum())
print("mt.amin():\n", mt.amin())
```

## MaskedTensor methods and sparse

+++

`to_dense()`

```{code-cell} ipython3
mt.to_dense()
```

`to_sparse_coo()`

```{code-cell} ipython3
v = [[3, 0, 0],
     [0, 4, 5]]
m = [[True, False, False],
     [False, True, True]]
mt = masked_tensor(torch.tensor(v), torch.tensor(m))

mt_sparse = mt.to_sparse_coo()
```

`to_sparse_csr()`

```{code-cell} ipython3
v = [[3, 0, 0],
     [0, 4, 5]]
m = [[True, False, False],
     [False, True, True]]
mt = masked_tensor(torch.tensor(v), torch.tensor(m))

mt_sparse_csr = mt.to_sparse_csr()
```

`is_sparse` / `is_sparse_coo` / `is_sparse_csr`

```{code-cell} ipython3
print("mt.is_sparse: ", mt.is_sparse())
print("mt_sparse.is_sparse: ", mt_sparse.is_sparse())

print("mt.is_sparse_coo: ", mt.is_sparse_coo())
print("mt_sparse.is_sparse_coo: ", mt_sparse.is_sparse_coo())

print("mt.is_sparse_csr: ", mt.is_sparse_csr())
print("mt_sparse_csr.is_sparse_csr: ", mt_sparse_csr.is_sparse_csr())
```

## Appendix

+++

### Sparse COO construction

+++

Recall in our original example, we created a MaskedTensor and then converted it to a sparse COO MaskedTensor with `mt.to_sparse_coo()`

Alternatively, we can also construct a sparse COO MaskedTensor by passing in two sparse COO tensors!

```{code-cell} ipython3
values = torch.tensor([[0, 0, 3], [4, 0, 5]]).to_sparse()
mask = torch.tensor([[False, False, True], [False, False, True]]).to_sparse()

mt = masked_tensor(values, mask)

print("values:\n", values)
print("mask:\n", mask)
print("mt:\n", mt)
```

Instead of doing `dense_tensor.to_sparse()`, we can also create the sparse COO tensors directly, which brings us to a word of warning: when using a function like `.to_sparse_coo()`, if the user does not specify the indices like in the above example, then 0 values will be default "unspecified"

```{code-cell} ipython3
i = [[0, 1, 1],
     [2, 0, 2]]
v =  [3, 4, 5]
m = torch.tensor([True, False, True])

values = torch.sparse_coo_tensor(i, v, (2, 3))
mask = torch.sparse_coo_tensor(i, m, (2, 3))
mt2 = masked_tensor(values, mask)

print("values:\n", values)
print("mask:\n", mask)
print("mt2:\n", mt2)
```

Note that `mt` and `mt2` will have the same value in the vast majority of operations, but this brings us to a note on the implementation under the hood:

`input` and `mask` - only for sparse formats - can have a different number of elements (`tensor.nnz()`) **at creation**, but the indices of `mask` must then be a subset of the indices from `input`. In this case, `input` will assume the shape of mask using the function `input.sparse_mask(mask)`; in other words, any of the elements in `input` that are not `True` in `mask` will be thrown away

Therefore, under the hood, the data looks slightly different; `mt2` has the 4 value masked out and `mt` is completely without it. In other words, their underlying data still has different shapes, so `mt + mt2` is invalid.

```{code-cell} ipython3
print("mt.masked_data:\n", mt.data())
print("mt2.masked_data:\n", mt2.data())
```

### Sparse CSR

+++

We can also construct a sparse CSR MaskedTensor using sparse CSR tensors, and like the example above, they have a similar treatment under the hood.

```{code-cell} ipython3
crow_indices = torch.tensor([0, 2, 4])
col_indices = torch.tensor([0, 1, 0, 1])
values = torch.tensor([1, 2, 3, 4])
mask_values = torch.tensor([True, False, False, True])

csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.double)
mask = torch.sparse_csr_tensor(crow_indices, col_indices, mask_values, dtype=torch.bool)

mt = masked_tensor(csr, mask)

print("csr tensor:\n", csr.to_dense())
print("mask csr tensor:\n", mask.to_dense())
print("masked tensor:\n", mt)
```
