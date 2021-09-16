# Pytorch-LLGA Bridge

**Note:** Due to rapid changes in the llga submodule, please make sure that the submodule is up-to-date each time you pulls the code:

```bash
git submodule sync && git submodule update --init --recursive
```

## Build

### 1. Prepare Environment

```bash
# gcc >= 5
conda create -n pyllga python numpy ninja pyyaml pytest mkl mkl-include setuptools cmake cffi typing
conda activate pyllga
```

### 2. Build from source

```bash
git clone https://gitlab.devtools.intel.com/pinzhen1/pytorch_llga.git --recursive
cd pytorch_llga
python setup.py develop
# Or if you are going to enable debug builds:
# DEBUG=1 python setup.py develop
```

### 3. Build torchvision

```bash
git clone -b v0.8.0-rc2 https://github.com/pytorch/vision.git
cd vision
python setup.py develop
```

### 4. Tests

```bash
pytest test/test_jit_llga_fuser.py
```

## Quick Start

A simple cascaded Conv-Relu example is provided in test. Enabling log outputs to familiarize with the whole pipeline:

**Mutation Removal -> DecomposeOps -> Prepare Binary -> Defer Size Check -> Graph Fuser -> Layout Propagation -> Kernel Execution**

```bash
DNNL_VERBOSE=1 PYTORCH_JIT_LOG_LEVEL=">>graph_helper:>>graph_fuser:>>kernel:>>interface" python -u test/test_jit_llga_fuser.py -k test_conv2d_eltwise
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

CMake where bridge code are included in

```bash
caffe2/CMakeLists.txt
```

CMake where LLGA submodule are included in

```bash
third_party/llga
aten/src/ATen/CMakeLists.txt
cmake/Dependencies.cmake
cmake/Modules/FindLLGA.cmake
```

## How to use

Use `export DNNL_GRAPH_CONSTANT_CACHE=1` to enable the weight cache (This is a temporary API).


```python
# enable oneDNN graph fusion globally 
torch._C._jit_set_llga_enabled(True)

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