## Summary

The goal of this project is to create an alternative Python API for PyTorch which is as similar as possible to NumPy, while retaining distinguishing features like simple GPU support and gradient computation, as illustrated below.

**Basic Example**:
```python
>>> import torch.numpy as np
>>> a = np.arange(10)
>>> a.sum()
tensor(45)
```

## Goals

* The project should, when complete, implement (at least) a large subset of the NumPy ndarray API.
* Implementing NumPy functions should not involve duplicating code if substantially similar PyTorch functions exist.
* Users should be able to seamlessly convert between a `torch.Tensor` and a `torch.numpy.ndarray` and use both at will.
* The new functionality must cause zero performance overhead for users of the traditional PyTorch API, and negligible (ideally zero) overhead for using the `torch.numpy` API.
* Any functions not existing in NumPy but necessary for interaction with PyTorch features should be clearly marked by underscores, signifying immediately to readers that they are PyTorch-specific extensions (e.g. `ndarray._cuda`, `ndarray._backward`, etc).
* PyTorch developers should be able to extend the API to add new NumPy functions in an easy and intuitive way.

## Data Model

The [torch.numpy.ndarray](__init__.py) class is implemented similarly to `torch.Tensor`: it is a python object wrapping a python extension API object ([torch._np_compat._ndarray_base](../csrc/np.cpp)) which ultimately wraps a `Variable`. This approach was chosen for two major reasons:

* Flexibility: implementing a new type (rather than inheriting from `torch.Tensor`) allows us to implement exactly the set of methods we want on the `torch.numpy.ndarray` type.
* Ease of implementation: A similar implementation to `torch.Tensor` enables existing Python binding codegen to be leveraged with minimal changes.

Tensors and ndarrays can be freely converted one to another:

```python
import torch, torch.numpy as np
>>> a = torch.randn(10)._np_compat()
>>> b = torch.randn(10)._np_compat()
>>> c = np.hypot(a, b)
>>> type(c)
<class 'torch.numpy.ndarray'>
>>> t = c._torch()
>>> type(t)
<class 'torch.Tensor'>
```

## Binding Generation

Code generation logic has been extended to allow NumPy API bindings to be defined in [native_functions.yaml](../../aten/src/ATen/native/native_functions.yaml) similarly to any other native function bindings. For example:

```yaml
- func: sum(np.ndarray a) -> np.ndarray
  variants: function, method
  np_compat: True
```

causes a new signature to be added to the argument parser in the generated binding function `THPVariable_sum`, the same function that already handles `torch.Tensor.sum`.

In order to distinguish between the two cases, we make the generated binding function accept a template parameter `<bool compat>`, controlling whether it should parse arguments according to the NumPy compatibility API. Then in the list of methods for the python extension objects backing `torch.Tensor` and `torch.numpy.ndarray`, we would add `THPVariable_sum<false>` and `THPVariable_sum<true>`, respectively.

Other than the bindings, this declaration of `sum` does not cause any code to be generated. The actual functionality will be implemented by the existing `at::native::sum` after appropriately translating arguments, as described below.

## Argument parsing and translation

The [argument parsing](../csrc/utils/python_arg_parser.cpp) logic is extended to support the new compatibility mode. Parsers are now initialized with two separate lists of signatures: one for the traditional API and one for the new one. When invoked in the old mode, the new-API signatures are ignored, and everything works the same as always.

When invoked in the new compatibility mode, the argument parsing works in two steps. First, the arguments are parsed against the compatiblity signatures. If a match is found, the argument names are translated into PyTorch equivalents (e.g., `a` is replaced by `input`, `keepdims` by `keepdim`, and so on), and argument types are converted if necessary (e.g., any `ndarray` is unwrapped and re-wrapped as a `torch.Tensor`). This new set of arguments is then matched against the PyTorch API set of signatures, and dispatched as appropriate.

A set of common argument name translations (for now: `a`, `keepdims`, and `axis`) is provided by default. It is also possible to add custom translations for a particular binding. The following example causes `shape` to be replaced by `size`.

```yaml
- func: ones(int[1] shape, np.dtype dtype=float) -> np.ndarray
  variants: function
  np_compat: True
  additional_translations:
    shape: size
```

## Adding new functions

Obviously, if a function is supported by NumPy and not by PyTorch, we need to actually implement it, not just rely on argument translation magic.

The most straightfoward way to do this is to create a PyTorch binding, mark it as hidden, and then define a NumPy compatibility binding depending on it. For example:

```yaml
- func: hypot(Tensor input, Tensor b) -> Tensor
  variants: function
  hidden: True
  dispatch:
    CPU: hypot
    CUDA: hypot

- func: hypot(np.ndarray a, np.ndarray b) -> np.ndarray
  variants: function
  np_compat: True
```

The required underlying function `at::native::hypot(Tensor const& input, Tensor const& b)` can then be implemented as usual, and `torch.numpy.hypot(a, b)` will return the equivalent of `sqrt(a*a + b*b)`,  as expected.

## CUDA support

A `torch.numpy.ndarray` can be created on the GPU in either of two ways: either by creating it as usual in PyTorch and converting it to an ndarray using `torch.Tensor._np_compat` (which just involves wrapping and unwrapping some objects, not copying any data), or by calling `_cuda` on an existing `torch.numpy.ndarray`. The ndarray can then be used as usual:

```python
>>> import torch.numpy as np
>>> cpu = np.arange(10)
>>> cpu.sum()
tensor(45)
>>> gpu = cpu._cuda()
>>> gpu.sum()
tensor(45, device='cuda:0')
```

## Differentiation support

**Not yet implemented**.

Since a `torch.numpy.ndarray` wraps a variable, in principle tracking gradients should be straightforward.

However, currently, much of the logic for `backward` is implemented in python, on the `torch.Tensor` type and in the `torch.autograd` package, and so is not available to `torch.numpy.ndarray`. In order to make this work, we can refactor the relevant code in order to share it between both types.

Keeping with the convention of prefixing API extensions that don't exist in NumPy with underscores, this functionality would be accessed via functions like `ndarry._backward`, `ndarray._grad`, `ndarray._requires_grad`, and so on.


## NumPy concepts not existing in PyTorch

**None of these are implemented yet in my proof of concept**

### dtypes

NumPy supports a [very rich](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html) notion of dtype allowing complex structures, whereas PyTorch tensors are made of scalars: float, double, and so on.

Unless we decide that it's worth making a fundamental refactor of PyTorch in order to support this, it is out of scope.

### Type promotion

Some work has already been done on designing and implementing NumPy-like type promotion in PyTorch: pytorch/pytorch#5795 and pytorch/pytorch#9515. Now that we are implementing this NumPy compatibility layer, the importance of that project increases.

Implementing this type promotion feature would involve:

1. Finalizing the design elaborated by Sam and Tugrul, which appears mostly complete
2. Implementing it in code
3. Adding an option to use exactly the same rules as NumPy, in cases where the design requires slightly different behavior in PyTorch. This option would be used for the NumPy compatibility API.
4. Provide options in the NumPy API to control this behavior (in particular, whether to use different type promotion rules on CUDA, when differences in performance can depend on data type width to an extreme extent)

### Multiple packages

NumPy functionality is spread throughout many different packages, to a much greater extent than PyTorch. For example, while PyTorch has `torch.randn`, NumPy has `numpy.random.randn`.

We can specify these with an option in the YAML defining the binding:

```yaml
- func: randn(int[] size)
  np_compat: True
  package: 'random'
```

Implementing this is straightforward: we will define a set of different packages (`torch.numpy.random`, `torch.numpy.matlib`, `torch.numpy.linalg`, and so on), and when generating the bindings, we will add each one to the list of functions for the appropriate module.

### Common NumPy parameters

NumPy ufuncs have a few parameters with no real PyTorch equivalent.

* `where`: This is used for masking. Masking support in PyTorch has been discussed for a while. If we decide to implement it, we can then re-use the same implementation in the NumPy API.
* `order`: NumPy allows passing an "order" parameter determining the layout of function outputs. We can implement this by transforming calls like `a.foo(order='K')` to a call to the corresponding `foo_out`, passing a tensor with the correct strides.
