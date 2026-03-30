% PyTorch documentation master file, created by
%  sphinx-quickstart on Fri Dec 23 13:31:47 2016.
%  You can adapt this file completely to your liking, but it should at least
%  contain the root `toctree` directive.

% :github_url: https://github.com/pytorch/pytorch

# PyTorch Documentation

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

Features described in this documentation are classified by release status:

- **Stable (API-Stable):** Maintained long-term with backwards compatibility.
- **Unstable (API-Unstable):** Under active development; APIs may change.

**Popular:** {bdg-link-secondary}`torch.nn <search.html?q=torch.nn>`
{bdg-link-secondary}`torch.compile <search.html?q=torch.compile>`
{bdg-link-secondary}`DataLoader <search.html?q=DataLoader>`
{bdg-link-secondary}`autograd <search.html?q=autograd>`
{bdg-link-secondary}`FSDP <search.html?q=FSDP>`
{bdg-link-secondary}`torch.export <search.html?q=torch.export>`

:::{dropdown} Browse all APIs
:icon: list-unordered

**Core**

- [torch](https://docs.pytorch.org/docs/stable/torch.html)
- [torch.nn](https://docs.pytorch.org/docs/stable/nn.html)
- [torch.nn.functional](https://docs.pytorch.org/docs/stable/nn.functional.html)
- [torch.nn.init](https://docs.pytorch.org/docs/stable/nn.init.html)
- [torch.nn.attention](https://docs.pytorch.org/docs/stable/nn.attention.html)
- [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html)
- [Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
- [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [torch.autograd](https://docs.pytorch.org/docs/stable/autograd.html)
- [torch.optim](https://docs.pytorch.org/docs/stable/optim.html)
- [torch.utils.data](https://docs.pytorch.org/docs/stable/data.html)
- [torch.library](https://docs.pytorch.org/docs/stable/library.html)

**Math & Signal**

- [torch.fft](https://docs.pytorch.org/docs/stable/fft.html)
- [torch.linalg](https://docs.pytorch.org/docs/stable/linalg.html)
- [torch.special](https://docs.pytorch.org/docs/stable/special.html)
- [torch.signal](https://docs.pytorch.org/docs/stable/signal.html)
- [Complex Numbers](https://docs.pytorch.org/docs/stable/complex_numbers.html)

**Hardware Acceleration**

- [torch.accelerator](https://docs.pytorch.org/docs/stable/accelerator.html)
- [torch.cpu](https://docs.pytorch.org/docs/stable/cpu.html)
- [torch.cuda](https://docs.pytorch.org/docs/stable/cuda.html)
- [torch.cuda.memory](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [torch.mps](https://docs.pytorch.org/docs/stable/mps.html)
- [torch.xpu](https://docs.pytorch.org/docs/stable/xpu.html)
- [torch.mtia](https://docs.pytorch.org/docs/stable/mtia.html)
- [torch.mtia.memory](https://docs.pytorch.org/docs/stable/mtia.memory.html)
- [torch.mtia.mtia_graph](https://docs.pytorch.org/docs/stable/mtia.mtia_graph.html)
- [Meta device](https://docs.pytorch.org/docs/stable/meta.html)
- [torch.amp](https://docs.pytorch.org/docs/stable/amp.html)
- [torch.backends](https://docs.pytorch.org/docs/stable/backends.html)

**Compiler & Export**

- [torch.compiler](https://docs.pytorch.org/docs/stable/torch.compiler_api.html)
- [torch.export](https://docs.pytorch.org/docs/stable/export.html)
- [torch.fx](https://docs.pytorch.org/docs/stable/fx.html)
- [torch.fx.experimental](https://docs.pytorch.org/docs/stable/fx.experimental.html)
- [torch.func](https://docs.pytorch.org/docs/stable/func.html)
- [torch.onnx](https://docs.pytorch.org/docs/stable/onnx.html)
- [torch.nativert](https://docs.pytorch.org/docs/stable/nativert.html)

**Distributed**

- [torch.distributed](https://docs.pytorch.org/docs/stable/distributed.html)
- [torch.distributed.tensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [torch.distributed.algorithms.join](https://docs.pytorch.org/docs/stable/distributed.algorithms.join.html)
- [torch.distributed.elastic](https://docs.pytorch.org/docs/stable/distributed.elastic.html)
- [torch.distributed.fsdp](https://docs.pytorch.org/docs/stable/fsdp.html)
- [torch.distributed.fsdp.fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [torch.distributed.tensor.parallel](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [torch.distributed.optim](https://docs.pytorch.org/docs/stable/distributed.optim.html)
- [torch.distributed.pipelining](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
- [torch.distributed._symmetric_memory](https://docs.pytorch.org/docs/stable/symmetric_memory.html)
- [torch.distributed.checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [DDP Communication Hooks](https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html)
- [Distributed RPC Framework](https://docs.pytorch.org/docs/stable/rpc.html)

**Quantization & Performance**

- [Quantization](https://docs.pytorch.org/docs/stable/quantization.html)
- [torch.profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [torch.sparse](https://docs.pytorch.org/docs/stable/sparse.html)
- [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html)

**Utilities**

- [torch.distributions](https://docs.pytorch.org/docs/stable/distributions.html)
- [torch.futures](https://docs.pytorch.org/docs/stable/futures.html)
- [torch.hub](https://docs.pytorch.org/docs/stable/hub.html)
- [torch.monitor](https://docs.pytorch.org/docs/stable/monitor.html)
- [torch.overrides](https://docs.pytorch.org/docs/stable/torch.overrides.html)
- [torch.package](https://docs.pytorch.org/docs/stable/package.html)
- [torch.random](https://docs.pytorch.org/docs/stable/random.html)
- [torch.masked](https://docs.pytorch.org/docs/stable/masked.html)
- [torch.nested](https://docs.pytorch.org/docs/stable/nested.html)
- [torch.Size](https://docs.pytorch.org/docs/stable/size.html)
- [torch.Storage](https://docs.pytorch.org/docs/stable/storage.html)
- [torch.testing](https://docs.pytorch.org/docs/stable/testing.html)
- [torch.utils](https://docs.pytorch.org/docs/stable/utils.html)
- [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [torch.utils.cpp_extension](https://docs.pytorch.org/docs/stable/cpp_extension.html)
- [torch.utils.data](https://docs.pytorch.org/docs/stable/data.html)
- [torch.utils.deterministic](https://docs.pytorch.org/docs/stable/deterministic.html)
- [torch.utils.jit](https://docs.pytorch.org/docs/stable/jit_utils.html)
- [torch.utils.dlpack](https://docs.pytorch.org/docs/stable/dlpack.html)
- [torch.utils.mobile_optimizer](https://docs.pytorch.org/docs/stable/mobile_optimizer.html)
- [torch.utils.model_zoo](https://docs.pytorch.org/docs/stable/model_zoo.html)
- [torch.utils.tensorboard](https://docs.pytorch.org/docs/stable/tensorboard.html)
- [torch.utils.module_tracker](https://docs.pytorch.org/docs/stable/module_tracker.html)
- [Multiprocessing](https://docs.pytorch.org/docs/stable/multiprocessing.html)

**Other**

- [C++](https://docs.pytorch.org/cppdocs/)
- [Type Info](https://docs.pytorch.org/docs/stable/type_info.html)
- [Named Tensors](https://docs.pytorch.org/docs/stable/named_tensor.html)
- [Named Tensors operator coverage](https://docs.pytorch.org/docs/stable/name_inference.html)
- [torch.\_\_config\_\_](https://docs.pytorch.org/docs/stable/config_mod.html)
- [torch.\_\_future\_\_](https://docs.pytorch.org/docs/stable/future_mod.html)
- [torch.\_logging](https://docs.pytorch.org/docs/stable/logging.html)
- [Torch Environment Variables](https://docs.pytorch.org/docs/stable/torch_environment_variables.html)
:::

---

## Get Started

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`download;1.5em` Install PyTorch
:link: https://pytorch.org/get-started/locally/

Select your platform and package manager to install PyTorch locally.
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em` Tutorials
:link: https://docs.pytorch.org/tutorials/

Hands-on tutorials from beginner basics to advanced topics.
:::

:::{grid-item-card} {octicon}`book;1.5em` User Guide
:link: user_guide/index.html

Step-by-step guides for common PyTorch workflows.
:::

::::

---

## Core API

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`flame;1.5em` torch
:link: torch.html

Core tensor library — creation, indexing, math, serialization.
:::

:::{grid-item-card} {octicon}`cpu;1.5em` torch.nn
:link: nn.html

Neural network layers, loss functions, and containers.
:::

:::{grid-item-card} {octicon}`graph;1.5em` torch.nn.functional
:link: nn.functional.html

Functional interface for neural network operations.
:::

:::{grid-item-card} {octicon}`iterations;1.5em` torch.optim
:link: optim.html

Optimization algorithms — SGD, Adam, AdamW, and more.
:::

:::{grid-item-card} {octicon}`workflow;1.5em` torch.autograd
:link: autograd.html

Automatic differentiation engine powering neural network training.
:::

:::{grid-item-card} {octicon}`database;1.5em` torch.utils.data
:link: data.html

Dataset and DataLoader utilities for efficient data pipelines.
:::

:::{grid-item-card} {octicon}`package;1.5em` Tensors
:link: tensors.html

Tensor class reference — dtypes, views, and operations.
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em` Tensor Attributes
:link: tensor_attributes.html

dtype, device, layout, and memory format details.
:::

:::{grid-item-card} {octicon}`pulse;1.5em` torch.fft
:link: fft.html

Discrete Fourier Transform operations.
:::

:::{grid-item-card} {octicon}`triangle-right;1.5em` torch.linalg
:link: linalg.html

Linear algebra operations — decompositions, solves, norms.
:::

:::{grid-item-card} {octicon}`sparkle-fill;1.5em` torch.special
:link: special.html

Special mathematical functions.
:::

:::{grid-item-card} {octicon}`unmute;1.5em` torch.signal
:link: signal.html

Signal processing — window functions and filtering.
:::

::::

---

## Hardware Acceleration

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`zap;1.5em` torch.cuda
:link: cuda.html

CUDA tensor types, streams, and GPU operations.
:::

:::{grid-item-card} {octicon}`cpu;1.5em` CPU
:link: cpu.html

CPU-specific operations and optimizations.
:::

:::{grid-item-card} {octicon}`device-desktop;1.5em` torch.mps
:link: mps.html

Apple Metal Performance Shaders backend.
:::

:::{grid-item-card} {octicon}`server;1.5em` torch.xpu
:link: xpu.html

Intel XPU device support.
:::

:::{grid-item-card} {octicon}`rocket;1.5em` torch.accelerator
:link: accelerator.html

Unified accelerator abstraction layer.
:::

:::{grid-item-card} {octicon}`meter;1.5em` torch.amp
:link: amp.html

Automatic mixed precision training.
:::

::::

---

## Compiler & Export

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`gear;1.5em` torch.compile
:link: torch.compiler_api.html

Compiler-based model optimization with TorchDynamo and TorchInductor.
:::

:::{grid-item-card} {octicon}`file-symlink-file;1.5em` torch.export
:link: export.html

Export models to a portable, standardized representation.
:::

:::{grid-item-card} {octicon}`code;1.5em` torch.fx
:link: fx.html

Python-to-Python code transformation framework.
:::

:::{grid-item-card} {octicon}`share-android;1.5em` ONNX
:link: onnx.html

Export models to the ONNX interchange format.
:::

:::{grid-item-card} {octicon}`plug;1.5em` NativeRT
:link: nativert.html

Native runtime for executing exported programs.
:::

:::{grid-item-card} {octicon}`beaker;1.5em` torch.func
:link: func.html

JAX-like composable function transforms (vmap, grad, etc.).
:::

::::

---

## Distributed Training

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`globe;1.5em` torch.distributed
:link: distributed.html

Distributed communication primitives and process groups.
:::

:::{grid-item-card} {octicon}`stack;1.5em` FSDP
:link: distributed.fsdp.fully_shard.html

Fully Sharded Data Parallel for memory-efficient training.
:::

:::{grid-item-card} {octicon}`versions;1.5em` DTensor
:link: distributed.tensor.html

Distributed tensor abstraction for sharding strategies.
:::

:::{grid-item-card} {octicon}`git-branch;1.5em` Tensor Parallel
:link: distributed.tensor.parallel.html

Tensor parallelism for large model training.
:::

:::{grid-item-card} {octicon}`arrow-switch;1.5em` Pipeline Parallel
:link: distributed.pipelining.html

Pipeline parallelism for splitting model stages across devices.
:::

:::{grid-item-card} {octicon}`download;1.5em` Distributed Checkpoint
:link: distributed.checkpoint.html

Save and load distributed model state.
:::

:::{grid-item-card} {octicon}`broadcast;1.5em` RPC
:link: rpc.html

Remote Procedure Call framework for distributed operations.
:::

:::{grid-item-card} {octicon}`link-external;1.5em` Elastic Training
:link: distributed.elastic.html

Fault-tolerant, elastic distributed training with torchelastic.
:::

:::{grid-item-card} {octicon}`sync;1.5em` Distributed Optim
:link: distributed.optim.html

Distributed optimizer implementations.
:::

::::

---

## Quantization & Performance

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`meter;1.5em` Quantization
:link: quantization.html

Post-training quantization and quantization-aware training.
:::

:::{grid-item-card} {octicon}`stopwatch;1.5em` Profiler
:link: profiler.html

Performance profiling for CPU and GPU workloads.
:::

:::{grid-item-card} {octicon}`tools;1.5em` Benchmark Utils
:link: benchmark_utils.html

Utilities for precise microbenchmarking.
:::

:::{grid-item-card} {octicon}`graph;1.5em` Sparse Tensors
:link: sparse.html

Sparse tensor support and operations.
:::

::::

---

## Utilities & Ecosystem

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`archive;1.5em` torch.hub
:link: hub.html

Load pre-trained models from GitHub repositories.
:::

:::{grid-item-card} {octicon}`multi-select;1.5em` Multiprocessing
:link: multiprocessing.html

Shared-memory multiprocessing utilities.
:::

:::{grid-item-card} {octicon}`codescan-checkmark;1.5em` torch.testing
:link: testing.html

Testing utilities and assertions.
:::

:::{grid-item-card} {octicon}`checklist;1.5em` torch.utils.checkpoint
:link: checkpoint.html

Gradient checkpointing for memory-efficient training.
:::

:::{grid-item-card} {octicon}`file-binary;1.5em` C++ Extension
:link: cpp_extension.html

Build custom C++/CUDA extensions for PyTorch.
:::

:::{grid-item-card} {octicon}`log;1.5em` TensorBoard
:link: tensorboard.html

TensorBoard integration for visualization.
:::

:::{grid-item-card} {octicon}`comment-discussion;1.5em` torch.distributions
:link: distributions.html

Probability distributions and sampling.
:::

:::{grid-item-card} {octicon}`shield-check;1.5em` torch.library
:link: library.html

Register custom operators with PyTorch dispatcher.
:::

:::{grid-item-card} {octicon}`package-dependents;1.5em` torch.package
:link: package.html

Serialize and package PyTorch models with dependencies.
:::

::::

---

## Developer Resources

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`note;1.5em` Developer Notes
:link: notes.html

Design docs, best practices, and technical details.
:::

:::{grid-item-card} {octicon}`people;1.5em` Community
:link: community/index.html

Contribution guide, governance, and design philosophy.
:::

:::{grid-item-card} {octicon}`code-square;1.5em` C++ API
:link: https://docs.pytorch.org/cppdocs/

Full C++ frontend and library reference.
:::

:::{grid-item-card} {octicon}`info;1.5em` Environment Variables
:link: torch_environment_variables.html

Comprehensive list of PyTorch environment variables.
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em` Full API Reference
:link: pytorch-api.html

Complete list of all PyTorch Python and C++ APIs.
:::

::::

---

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`

```{toctree}
:hidden:
:maxdepth: 2

Install PyTorch <https://pytorch.org/get-started/locally/>
user_guide/index
pytorch-api
notes
community/index
```
