# PyTorch Glossary

<!-- toc -->

- [Operation and Kernel](#operation-and-kernel)
  - [ATen](#aten)
  - [Operation](#operation)
  - [Native Operation](#native-operation)
  - [Custom Operation](#custom-operation)
  - [Kernel](#kernel)
  - [Compound Operation](#compound-operation)
  - [Composite Operation](#composite-operation)
  - [Non-Leaf Operation](#non-leaf-operation)
  - [Leaf Operation](#leaf-operation)
  - [Device Kernel](#device-kernel)
  - [Compound Kernel](#compound-kernel)
- [JIT Compilation](#jit-compilation)
  - [JIT](#jit)
  - [TorchScript](#torchscript)
  - [Tracing](#tracing)
  - [Scripting](#scripting)

<!-- tocstop -->

# Operation and Kernel

## ATen
Short for "A Tensor Library". The foundational tensor and mathematical
operation library on which all else is built.

## Operation
A unit of work. For example, the work of matrix multiplication is an operation
called aten::matmul.

## Native Operation
An operation that comes natively with PyTorch ATen, for example aten::matmul.

## Custom Operation
An Operation that is defined by users and is usually a Compound Operation.
For example, this
[tutorial](https://pytorch.org/docs/stable/notes/extending.html) details how
to create Custom Operations.

## Kernel
Implementation of a PyTorch operation, specifying what should be done when an
operation executes.

## Compound Operation
A Compound Operation is composed of other operations. Its kernel is usually
device-agnostic. Normally it doesn't have its own derivative functions defined.
Instead, AutoGrad automatically computes its derivative based on operations it
uses.

## Composite Operation
Same as Compound Operation.

## Non-Leaf Operation
Same as Compound Operation.

## Leaf Operation
An operation that's considered a basic operation, as opposed to a Compound
Operation. Leaf Operation always has dispatch functions defined, usually has a
derivative function defined as well.

## Device Kernel
Device-specific kernel of a leaf operation.

## Compound Kernel
Opposed to Device Kernels, Compound kernels are usually device-agnostic and belong to Compound Operations.

# JIT Compilation

## JIT
Just-In-Time Compilation.

## TorchScript
An interface to the TorchScript JIT compiler and interpreter.

## Tracing
Using `torch.jit.trace` on a function to get an executable that can be optimized
using just-in-time compilation.

## Scripting
Using `torch.jit.script` on a function to inspect source code and compile it as
TorchScript code.
