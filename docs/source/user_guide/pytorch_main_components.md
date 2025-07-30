(pytorch_main_components)=
# PyTorch Main Components

PyTorch is a flexible and powerful library for deep learning that provides a comprehensive set of tools for building, training, and deploying machine learning models.

## PyTorch Components for Basic Deep Learning

Some of the basic PyTorch components include:

* **Tensors** ({class}`torch.tensor`)- N-dimensional arrays that serve as PyTorch's fundamental
data structure. They support automatic differentiation, GPU acceleration, and provide a comprehensive
API for mathematical operations. Tensors can seamlessly move between CPU and GPU for
optimized computation.

* **Autograd** - ({mod}`torch.autograd`)PyTorch's automatic differentiation engine
that tracks operations performed on tensors and builds a computational
graph dynamically. It enables efficient gradient computation for backpropagation
during model training with minimal overhead.

* **Neural Network API** ({mod}`nn.Module`)- A modular framework for building neural networks with pre-defined layers,
activation functions, and loss functions. The {mod}`nn.Module` base class provides a clean interface
for creating custom network architectures with parameter management.

* **DataLoaders** ([torch.utils.data](data.html))- Tools for efficient data handling that provide
features like batching, shuffling, and parallel data loading. They abstract away the complexities
of data preprocessing and iteration, allowing for optimized training loops.


## PyTorch Components for Production-Grade Performance and Deployment

PyTorch extends beyond basic deep learning capabilities with advanced features designed for
production-grade performance and deployment. These components optimize model execution,
reduce resource requirements, and enable scaling across multiple compute devices.
These components include:

### PyTorch Compiler

The PyTorch compiler is a suite of tools that optimize model execution and
reduce resource requirements. It includes:
* **torch.compile** ({func}`torch.compile`)- Just-in-Time (JIT) compilation for accelerated execution.
torch.compile transforms PyTorch code into optimized computational graphs at runtime.
It can provide significant speedups with minimal code changes by analyzing execution
patterns and applying hardware-specific optimizations.
* **torch.export** ({func}`torch.export`)- Exporting models for deployment in
resource-constrained environments. torch.export generates standalone artifacts
that can run without the PyTorch runtime. It supports various deployment targets,
including mobile, embedded, and cloud.
* **Inductor** ([TorchInductor](../torch.compiler_inductor_profiling.html))- The default backend
for {func}`torch.compile` that converts PyTorch operations
into efficient machine code. Uses TorchIR as an intermediate representation to apply
optimizations like operator fusion, memory planning, and loop transformations.
* **AOTInductor** ({ref}`torch.compiler_aot_inductor`)- Compiles models Ahead-Of-Time (AOT) for deployment environments
where JIT compilation isn't feasible. **AOTInductor** generates standalone artifacts
that can run without the PyTorch runtime.

### Deployment and Optimization

PyTorch provides tools for optimizing model performance and deployment in various environments. These include:

* **Quantization** ([torchao](https://docs.pytorch.org/ao/stable/index.html))- Precision-reduction
techniques for model efficiency. Qunatization features of PyTorch reduce model precision
from 32-bit to 8-bit or lower formats. They support post-training quantization, quantization-aware training, and dynamic
quantization to balance accuracy and efficiency.
* **Edge Deployment** ([Executorch](../index.html))- ExecuTorch is a PyTorch-compatible library that supports
resource-constrained environments.
* **Distributed Training** ([torch.distributed](../distributed.html)) -  Includes data parallelism (`DistributedDataParallel`),
model parallelism, and pipeline parallelism options. Supports communication backends
like NCCL and Gloo for efficient multi-node training. **`Fully Sharded Data Parallel`(FSDP2)** provides
memory-efficient training for large models by sharding model parameters, gradients,
and optimizer states across devices while maintaining training efficiency.
* **Profiling and Monitoring** - Tools like {class}`torch.profiler` help identify bottlenecks,
visualize execution traces, and monitor resource utilization during training and inference.
