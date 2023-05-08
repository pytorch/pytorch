# PyTorch Glossary

Several terms in this glossary are used frequently throughout PyTorch documentation and warrant basic definitions.

<!-- toc -->

- [PyTorch Glossary](#pytorch-glossary)
  - [Model, Tensor, Operation and Kernel](#model-tensor-operation-and-kernel)
    - [Model](#model)
    - [Tensor](#tensor)
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

## Model, Tensor, Operation and Kernel

### Model

An algorithm for making predictions from data. (E.g. AI Model)

### Tensor

(1:formal) A multilinear map, in the form of a specialized storage data structure, similar to arrays and matrices.
(2:informal) A multi-dimensional array/matrix containing elements (data) of a single data type (int, float, ...).
(3:PyTorch implementation) A low-level representation of the specialized storage data structure, containing pointers to storage where the data and metadata are located.

- In PyTorch tensors are used to encode the data inputs & outputs of a model and the model’s parameters.

### ATen

Short for "A Tensor Library." The foundational (low-level) tensor and mathematical
operation library on which PyTorch is built.

### Operation

A unit of work typically performed on or for one or more tensors. For example, the work of matrix multiplication is an operation called `aten::matmul`. In this example the function `at::matmul`, from the c++ library, may be used to perform a matrix multiplication operation on two tensors.

### Native Operation

Any operation included with the PyTorch Tensor Library (ATen), for example `aten::matmul` is an operation native to PyTorch.

### Custom Operation

A non-native operation [Extending Pytorch](https://pytorch.org/docs/stable/notes/extending.html)

### Kernel

Implementation of a PyTorch operation, specifying what should be done when an
operation executes.

### Compound Operation

A Compound Operation is composed of other operations. Its kernel is usually
device-agnostic. Normally it doesn't have its own derivative functions defined.
Instead, AutoGrad automatically computes its derivative based on operations it
uses.

### Composite Operation

Same as Compound Operation.

### Non-Leaf Operation

Same as Compound Operation.

### Leaf Operation

An operation that's considered a basic operation, as opposed to a Compound
Operation. Leaf Operation always has dispatch functions defined, usually has a
derivative function defined as well.

### Device Kernel

Device-specific kernel of a leaf operation.

### Compound Kernel

Opposed to Device Kernels, Compound kernels are usually device-agnostic and belong to Compound Operations.

## JIT Compilation

### JIT

Just-In-Time Compilation.

### TorchScript

An interface to the TorchScript JIT compiler and interpreter.

### Tracing

Using `torch.jit.trace` on a function to get an executable that can be optimized
using just-in-time compilation.

### Scripting

Using `torch.jit.script` on a function to inspect source code and compile it as
TorchScript code.
