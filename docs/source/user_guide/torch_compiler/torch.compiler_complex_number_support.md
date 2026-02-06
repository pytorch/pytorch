(torch.compiler_complex_number_support)=

# Complex Number Support in `torch.compile`
PyTorch, as of version 2.11, has experimental opt-in support for compilation of complex-valued
tensors. The following code shows an example of how to use the complex number support.

```python
import torch
import torch._functorch.config

# Enable compilation of complex-valued tensors
torch._functorch.config.enable_complex_wrapper = True

@torch.compile
def some_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = a + b
    d = a * b
    e = torch.sin(c)
    f = torch.cos(d)
    return torch.atan(f / e)

a = torch.randn((5, 1), dtype=torch.complex64)
b = torch.randn((5, 1), dtype=torch.complex64)

out = some_function(a, b)
```

This is implemented via the `torch._subclasses.complex_tensor.ComplexTensor` subclass, which
decomposes complex-valued operations into real-valued ones. Not all operations are supported for
compilation. If there are some operations you'd like supported, check the list of known issues in
[this list](https://github.com/pytorch/pytorch/issues?q=sort%3Aupdated-desc%20state%3Aopen%20label%3A%22module%3A%20complex%22%20label%3Adynamo-functorch).
If there's no existing issue open for your proposed operation, open an issue.
