(jit_unsupported)=

# TorchScript Unsupported PyTorch Constructs

## Torch and Tensor Unsupported Attributes

TorchScript supports most methods defined on `torch` and `torch.Tensor`, but we do not have full coverage.
Here are specific known ops and categories of ops which have diverging behavior between
Python and TorchScript. If you encounter something else that is not supported please
file a GitHub issue. Deprecated ops are not listed below.

```{eval-rst}
.. automodule:: torch.jit.unsupported_tensor_ops
```

### Functions Not Correctly Bound on Torch

The following functions will fail if used in TorchScript, either because they
are not bound on `torch` or because Python expects a different schema than
TorchScript.

- {func}`torch.tensordot`
- {func}`torch.nn.init.calculate_gain`
- {func}`torch.nn.init.eye_`
- {func}`torch.nn.init.dirac_`
- {func}`torch.nn.init.kaiming_normal_`
- {func}`torch.nn.init.orthogonal_`
- {func}`torch.nn.init.sparse`

### Ops With Divergent Schemas Between Torch & Python

The following categories of ops have divergent schemas:

Functions which construct tensors from non-tensor inputs do not support the `requires_grad`
argument, except for `torch.tensor`. This covers the following ops:

- {func}`torch.norm`
- {func}`torch.bartlett_window`
- {func}`torch.blackman_window`
- {func}`torch.empty`
- {func}`torch.empty_like`
- {func}`torch.empty_strided`
- {func}`torch.eye`
- {func}`torch.full`
- {func}`torch.full_like`
- {func}`torch.hamming_window`
- {func}`torch.hann_window`
- {func}`torch.linspace`
- {func}`torch.logspace`
- {func}`torch.normal`
- {func}`torch.ones`
- {func}`torch.rand`
- {func}`torch.rand_like`
- {func}`torch.randint_like`
- {func}`torch.randn`
- {func}`torch.randn_like`
- {func}`torch.randperm`
- {func}`torch.tril_indices`
- {func}`torch.triu_indices`
- {func}`torch.vander`
- {func}`torch.zeros`
- {func}`torch.zeros_like`

The following functions require `dtype`, `layout`, `device` as parameters in TorchScript,
but these parameters are optional in Python.

- {func}`torch.randint`
- {func}`torch.sparse_coo_tensor`
- {func}`torch.Tensor.to`

## PyTorch Unsupported Modules and Classes

TorchScript cannot currently compile a number of other commonly used PyTorch
constructs. Below are listed the modules that TorchScript does not support, and
an incomplete list of PyTorch classes that are not supported. For unsupported modules
we suggest using {meth}`torch.jit.trace`.

- {class}`torch.nn.RNN`
- {class}`torch.nn.AdaptiveLogSoftmaxWithLoss`
- {class}`torch.autograd.Function`
- {class}`torch.autograd.enable_grad`
