# Pytorch-LLGA Bridge

For more details about oneDNN Graph, please check out the [RFC](https://github.com/pytorch/pytorch/issues/49444)

## Background

[**oneDNN Graph API**](https://spec.oneapi.com/onednn-graph/latest/introduction.html) (codename: LLGA) _extends_ oneDNN with a high-level graph API, as a complement to the **oneDNN Primitive API**.

oneDNN Graph API accepts a deep learning computation graph based on its [Opset](https://spec.oneapi.com/onednn-graph/latest/ops/index.html) as input and performs graph partitioning, where nodes that are candidates for fusion are grouped together. oneDNN Graph compiles and executes a group of operators in a partition as a fused operation. [The programming model](https://spec.oneapi.com/onednn-graph/latest/programming_model.html) is designed for easy integration with deep learning frameworks.

The PyTorch integration is mainly composed of two parts:

1. Graph Rewrite

    - **Passing graph**: Translate TorchScript IR into oneDNN Graph IR represented by the `graph` object. All PyTorch `Node`s are mapped to oneDNN Graph `op`s and passed with `graph::add_op(op)`.

    - **Partitioning**: oneDNN Graph picks out regions of interest and returns `std::vector<partition>`. Each `partition` corresponds to an unordered list of ops to form a fusion op.

    - **Rewriting**: Cluster JIT nodes into fusion ops based on the partitions returned from the partitioning stage.

2. Runtime

    Each oneDNN Graph partition is [compiled](https://spec.oneapi.com/onednn-graph/latest/programming_model.html#partition) and [executed](https://spec.oneapi.com/onednn-graph/latest/programming_model.html#compiled-partition) in the implementation of the fusion op.

## Quick Start

You can get started with a simple Conv-Relu example provided in the test script. Enabling log outputs to familiarize with the whole pipeline:

```bash
DNNL_VERBOSE=1 PYTORCH_JIT_LOG_LEVEL=">>graph_helper:>>graph_fuser:>>kernel:>>interface" python -u test/test_jit_llga_fuser.py -k test_conv2d_eltwise
```

All available patterns are covered by unit tests

```bash
pytest test/test_jit_llga_fuser.py
```

Also, we've put all popular torchvision models in this script (torchvision & scipy required)

```bash
python test/test_jit_llga_fuser.py -k resnet50
```

## How to Use

oneDNN Graph fuser is designed to be worked with both traced and scripted modules. You will have to add this line in the script to enable the oneDNN Graph optimization pass:

```python
torch._C._jit_set_llga_enabled(True)
```

### Caveats for CNN

If you are working on a CNN model, make sure you have disabled the profiling executor. Since profiling executor will add some Bailout nodes that reduce the fusion opportunities.

```python
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
```

Additionally, if you are optimizing the forward pass (inference only), you need to disable autograd by putting the model execution within a `torch.no_grad()` context. Otherwise, operators like `Conv`, `Batchnorm` will not be grouped into the `DifferentiableGraph`, and thus they will not go through any custom optimization passes.

```python
with torch.no_grad():
    y = model(x)
```

## Codebase structure

Most of the source code are placed in

```bash
torch/csrc/jit/codegen/onednn/*
```

Tensor related code are located at

```bash
aten/src/ATen/native/mkldnn/LlgaTensorImpl.h
aten/src/ATen/native/mkldnn/LlgaTensorImpl.cpp
```

CMake in which bridge code are included

```bash
caffe2/CMakeLists.txt
```
