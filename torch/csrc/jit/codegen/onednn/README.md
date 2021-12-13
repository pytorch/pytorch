# Pytorch - oneDNN Graph API Bridge
This integration will add the infrastructure of a new PyTorch JIT graph fuser based on oneDNN Graph API. The oneDNN Graph API provides flexible API for aggressive fusion. The current preview2 version supports fusion for FP32 inference.

## Tests

```bash
pytest test/test_jit_llga_fuser.py
```

## Quick Start

A simple cascaded Conv-Relu example is provided in test. Please consider enabling log outputs to familiarize yourself with the whole pipeline:

**Mutation Removal -> DecomposeOps -> Prepare Binary -> Defer Size Check -> Graph Fuser -> Layout Propagation -> Type Guard -> Kernel Execution**

```bash
DNNL_VERBOSE=1 PYTORCH_JIT_LOG_LEVEL=">>graph_helper:>>graph_fuser:>>kernel:>>interface" python -u test/test_jit_llga_fuser.py -k test_conv2d_eltwise
```

## Codebase structure

Most of the source code is placed in

```bash
torch/csrc/jit/codegen/onednn/*
```

Tensor related code is located at

```bash
torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h
torch/csrc/jit/codegen/onednn/LlgaTensorImpl.cpp
```

CMake where bridge code is included:

```bash
caffe2/CMakeLists.txt
```

CMake where oneDNN Graph submodule are included:

```bash
third_party/mkl-dnn
cmake/public/mkldnn.cmake
cmake/Modules/FindMKLDNN.cmake
cmake/Dependencies.cmake
```

## How to use

Use `export DNNL_GRAPH_CONSTANT_CACHE=1` to enable the weight cache (This is a temporary API).


```python
# enable oneDNN graph fusion globally
torch.jit.enable_onednn_fusion(True)

# define the model
def MyModel(torch.nn.Module):
    ...

# construct the model
model = MyModel(…)
with torch.no_grad():
    model.eval()
    model = torch.jit.trace(model, torch.rand(args.batch_size, 3, 224, 224))

# run the model
with torch.no_grad():
    # oneDNN graph fusion will be trigerred during runtime
    output = model(images)
```
