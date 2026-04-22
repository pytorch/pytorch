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
   For example, this [tutorial](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html
   details how to create Custom Operations.

Kernel
   Implementation of a PyTorch operation, specifying what should be done when an
   operation executes.

Composite Operation
   A Composite Operation is composed of other operations. Its kernel is usually
   device-agnostic. There are two variants of this: Composite Implicit Autograd, where no
   autograd formula is required as it is derived implicitly from the operations composing this
   one. And Composite Explicit Autograd, where there is an explicit autograd formula

Non-Leaf Operation
   Same as {term}`Composite Operation`.

Device Kernel
   Device-specific kernel of a {term}`Leaf Operation`.

JIT
   Just-In-Time Compilation.

Tracing
   In PyTorch, tracing is a way to convert a PyTorch model (or function) into a static computation graph by running it once (or a few times) with example inputs and recording (“tracing”) the tensor operations that actually execute.

```
