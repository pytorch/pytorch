(glossary)=
# PyTorch Glossary

This glossary provides definitions for terms commonly used in PyTorch documentation.

```{glossary}

ATen
   Short for "A Tensor Library". The foundational tensor and mathematical
   operation library on which all else is built.

Operation
   A unit of work. For example, the work of matrix multiplication is an operation
   called `aten::matmul`.

Native Operation
   An operation that comes natively with PyTorch ATen, for example `aten::matmul`.

Custom Operation
   An Operation that is defined by users and is usually a {term}`Compound Operation`.
   For example, this [tutorial](https://pytorch.org/docs/stable/notes/extending.html)
   details how to create Custom Operations.

Kernel
   Implementation of a PyTorch operation, specifying what should be done when an
   operation executes.

Compound Operation
   A Compound Operation is composed of other operations. Its kernel is usually
   device-agnostic. There are two variants of this: Composite Implicit Autograd, where no
   autograd formula is required as it is derived implicitly from the operations composing this
   one. And Composite Explicit Autograd, where there is an explicit autograd formula.

Composite Operation
   Same as {term}`Compound Operation`.

Non-Leaf Operation
   Same as {term}`Compound Operation`.

Leaf Operation
   An operation that's considered a basic operation, as opposed to a {term}`Compound
   Operation`. Leaf Operation always has dispatch functions defined, usually has a
   derivative function defined as well.

Device Kernel
   Device-specific kernel of a {term}`Leaf Operation`.

Compound Kernel
   Opposed to {term}`Device Kernels<Device Kernel>`, Compound kernels are usually
   device-agnostic and belong to {term}`Compound Operations<Compound Operation>`.

JIT
   Just-In-Time Compilation.

Tracing
   Using `torch.jit.trace` on a function to get an executable that can be optimized
   using just-in-time compilation.

```
