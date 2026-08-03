(pytorch_main_components)=
# PyTorch Main Components

PyTorch is a flexible and powerful library for deep learning that provides a comprehensive set of tools for building, training, and deploying machine learning models.

## PyTorch Components for Basic Deep Learning

Some of the basic PyTorch components include:

* **Tensors** - N-dimensional arrays that serve as PyTorch's fundamental
data structure. They support automatic differentiation, hardware acceleration, and provide a comprehensive API for mathematical operations.

* **Autograd** - PyTorch's automatic differentiation engine
that tracks operations performed on tensors and builds a computational
graph dynamically to be able to compute gradients.

* **Neural Network API** - A modular framework for building neural networks with pre-defined layers,
activation functions, and loss functions. The {mod}`nn.Module` base class provides a clean interface
for creating custom network architectures with parameter management.

* **DataLoaders** - Tools for efficient data handling that provide
features like batching, shuffling, and parallel data loading. They abstract away the complexities
of data preprocessing and iteration, allowing for optimized training loops.


## PyTorch Compiler

The PyTorch compiler is a suite of tools that optimize model execution and
reduce resource requirements. You can learn more about the PyTorch compiler [here](https://docs.pytorch.org/docs/stable/torch.compiler_get_started.html).
