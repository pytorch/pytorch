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

```{raw} html
<div class="landing-search" role="search">
  <form id="landing-search-form" action="search.html" method="get">
    <input
      id="landing-search-input"
      type="search"
      name="q"
      placeholder="Search APIs — e.g. torch.nn.Linear, autograd, DataLoader …"
      autocomplete="off"
      aria-label="Search PyTorch documentation"
    />
    <button type="submit" aria-label="Search">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
           stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
    </button>
  </form>
  <div class="landing-search-tags">
    <span class="landing-search-label">Popular:</span>
    <a href="search.html?q=torch.nn">torch.nn</a>
    <a href="search.html?q=torch.compile">torch.compile</a>
    <a href="search.html?q=DataLoader">DataLoader</a>
    <a href="search.html?q=autograd">autograd</a>
    <a href="search.html?q=FSDP">FSDP</a>
    <a href="search.html?q=torch.export">torch.export</a>
  </div>
</div>
```

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
:link: user_guide/torch_compiler/export.html

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
