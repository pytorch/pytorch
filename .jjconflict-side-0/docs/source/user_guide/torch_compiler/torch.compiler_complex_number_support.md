(torch.compiler_complex_number_support)=

# Complex Number Support in `torch.compile`
PyTorch, as of version 2.14, has experimental opt-in support for compilation of complex-valued
tensors. The following code shows an example of how to use the complex number support.

```py
import torch
import torch._functorch.config

def some_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = a + b
    d = a * b
    e = torch.sin(c)
    f = torch.cos(d)
    return torch.atan(f / e)

a = torch.randn((5, 1), dtype=torch.complex64)
b = torch.randn((5, 1), dtype=torch.complex64)

# Enable compilation of complex-valued tensors
with torch._functorch.config.patch(enable_complex_wrapper=True)
    out = torch.compile(some_function)(a, b)
```

This is implemented via the `torch._subclasses.complex_tensor.ComplexTensor` subclass, which
decomposes complex-valued operations into real-valued ones. This is done by storing two
separate, contiguous tensors for the real and imaginary components instead of one tensor with
interleaved real and imaginary parts.

The benefit is that this layout can easily use the existing optimized hardware kernels that
aren't available for tensors holding complex numbers, notably for matrix multiplication.
The downside is that when entering/exiting a `torch.compile` block, there is a one-time cost
for converting the tensor into the two-component format or back.

Not all operations are supported for
compilation. If there are some operations you'd like supported, check the list of known issues in
[this list](https://github.com/pytorch/pytorch/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22module%3A%20complex%22%20label%3A%22module%3A%20functorch%22).
If there's no existing issue open for your proposed operation, open an issue.

## Limitations
The largest limitation coming from this approach is that it's impossible to maintain aliasing
semantics for certain operations; notably those which require the interleaved layout.

The two most common examples of this are the operations ``torch.view_as_real`` and
``torch.view_as_complex``. Another common case is modifying a complex input to `torch.compile`.

However, use of these operations may not show up in the compiled graph due to fusion; and therefore
many functions with these operations present may actually compile successfully despite
using the operations in user code.
