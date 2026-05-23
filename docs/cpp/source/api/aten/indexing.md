---
myst:
  html_meta:
    description: Tensor indexing in PyTorch C++ — Slice, None, Ellipsis, boolean masks, and advanced indexing with index() and index_put_().
    keywords: PyTorch, C++, tensor indexing, Slice, index, index_put_, boolean mask, fancy indexing
---

# Tensor Indexing

The PyTorch C++ API provides tensor indexing similar to Python. Use
`torch::indexing` namespace for index types:

```cpp
using namespace torch::indexing;
```

The main difference from Python is that instead of using the `[]` operator,
the C++ API uses the `index` and `index_put_` methods:

- `torch::Tensor::index` — read elements
- `torch::Tensor::index_put_` — write elements

## Index Types

The `TensorIndex` class accepts six types of indices via implicit constructors:

```{list-table}
:widths: 25 35 40
:header-rows: 1

* - Type
  - C++
  - Python equivalent
* - None (unsqueeze)
  - `None`
  - `None`
* - Ellipsis
  - `Ellipsis` or `"..."`
  - `...`
* - Integer
  - `0`, `1`, `-1`
  - `0`, `1`, `-1`
* - Boolean
  - `true`, `false`
  - `True`, `False`
* - Slice
  - `Slice(start, stop, step)`
  - `start:stop:step`
* - Tensor
  - `torch::tensor({0, 2})`
  - `torch.tensor([0, 2])`
```

## Getter Operations

```{list-table}
:widths: 40 60
:header-rows: 1

* - Python
  - C++
* - `tensor[0]`
  - `tensor.index({0})`
* - `tensor[-1]`
  - `tensor.index({-1})`
* - `tensor[1, 2]`
  - `tensor.index({1, 2})`
* - `tensor[1, :, 3]`
  - `tensor.index({1, Slice(), 3})`
* - `tensor[None]`
  - `tensor.index({None})`
* - `tensor[:, None]`
  - `tensor.index({Slice(), None})`
* - `tensor[...]`
  - `tensor.index({Ellipsis})` or `tensor.index({"..."})`
* - `tensor[..., 0]`
  - `tensor.index({Ellipsis, 0})`
* - `tensor[1::2]`
  - `tensor.index({Slice(1, None, 2)})`
* - `tensor[True]`
  - `tensor.index({true})`
* - `tensor[torch.tensor([1, 2])]`
  - `tensor.index({torch::tensor({1, 2})})`
* - `tensor[bool_mask]`
  - `tensor.index({bool_mask})`
* - `tensor[:, torch.tensor([[0,1],[4,3]])]`
  - `tensor.index({Slice(), torch::tensor({{0,1},{4,3}})})`
* - `tensor[cond > 0]`
  - `tensor.index({cond > 0})`
```

## Setter Operations

```{list-table}
:widths: 40 60
:header-rows: 1

* - Python
  - C++
* - `tensor[0] = 1`
  - `tensor.index_put_({0}, 1)`
* - `tensor[1, 2] = 1`
  - `tensor.index_put_({1, 2}, 1)`
* - `tensor[1] = torch.arange(5)`
  - `tensor.index_put_({1}, torch::arange(5))`
* - `tensor[1::2] = 1`
  - `tensor.index_put_({Slice(1, None, 2)}, 1)`
* - `tensor[0, 1::2] = torch.tensor([3., 4.])`
  - `tensor.index_put_({0, Slice(1, None, 2)}, torch::tensor({3., 4.}))`
* - `tensor[...] = 0`
  - `tensor.index_put_({Ellipsis}, 0)`
* - `tensor[None] = value`
  - `tensor.index_put_({None}, value)`
* - `tensor[bool_mask] = 0`
  - `tensor.index_put_({bool_mask}, 0)`
* - `tensor[torch.tensor([0, 2])] = value`
  - `tensor.index_put_({torch::tensor({0, 2})}, value)`
* - `tensor[1:2, torch.tensor([1,2])] = 0`
  - `tensor.index_put_({Slice(1, 2), torch::tensor({1, 2})}, 0)`
```

The `index_put_` method also accepts an optional `accumulate` parameter.
When `true`, values are added to existing values instead of replacing them:

```cpp
tensor.index_put_({mask}, values, /*accumulate=*/true);
```

## Slice Syntax

The `Slice` constructor signature is:

```cpp
Slice(
    std::optional<c10::SymInt> start = std::nullopt,
    std::optional<c10::SymInt> stop  = std::nullopt,
    std::optional<c10::SymInt> step  = std::nullopt);
```

Pass `None` for open-ended bounds:

```{list-table}
:widths: 30 70
:header-rows: 1

* - Python
  - C++
* - `:` or `::`
  - `Slice()` or `Slice(None, None)`
* - `1:`
  - `Slice(1, None)`
* - `:3`
  - `Slice(None, 3)`
* - `1:3`
  - `Slice(1, 3)`
* - `1:3:2`
  - `Slice(1, 3, 2)`
* - `::2`
  - `Slice(None, None, 2)`
```

## Full Example

```cpp
#include <torch/torch.h>

using namespace torch::indexing;

auto tensor = torch::arange(2 * 3 * 4).reshape({2, 3, 4});

// Basic indexing
auto row = tensor.index({0});             // tensor[0]
auto elem = tensor.index({1, 2, 3});      // tensor[1, 2, 3]

// Slicing
auto sliced = tensor.index({Slice(), Slice(0, 2)});  // tensor[:, 0:2]

// None (unsqueeze) and Ellipsis
auto unsqueezed = tensor.index({None});              // tensor[None]
auto last_dim = tensor.index({Ellipsis, -1});         // tensor[..., -1]

// Boolean mask indexing
auto mask = tensor > 10;
auto selected = tensor.index({mask});     // tensor[tensor > 10]

// Integer tensor (fancy) indexing
auto idx = torch::tensor({0, 2});
auto gathered = tensor.index({Slice(), idx});  // tensor[:, [0, 2]]

// Setting values
tensor.index_put_({0, Slice(), 0}, 99);        // tensor[0, :, 0] = 99
tensor.index_put_({mask}, 0);                  // tensor[tensor > 10] = 0
```
