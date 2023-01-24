# Pytorch - oneDNN Graph API Bridge
This is a PyTorch JIT graph fuser based on [oneDNN Graph API](https://spec.oneapi.io/onednn-graph/latest/programming_model.html), which provides a flexible API for aggressive fusion. Float & BFloat16 inference is supported. However, BFloat16 only performs well on Intel Xeon Cooper Lake platform & beyond, as they have native BFloat16 support. Also, currently, PyTorch has divergent AMP support in JIT & eager modes, so one should disable JIT AMP support & leverage eager mode AMP support to use BFloat16. Please refer to the BFloat16 example below.

Currently, speedup is achieved only for static shapes, although we'd soon add dynamic-shape support. When oneDNN Graph is enabled, weights are cached, as they're constant during inference.

## Graph Optimization
We have registered optimization passes in the custom pre-passes set of PyTorch:

1. Alias and mutation reduction

    The operators of oneDNN graph are pure functional while PyTorch has operators in in-place forms or create views for buffer sharing.
    Due to the semantic gaps between the backend operators and the PyTorch operators, we have a pass to reduce mutation with best effort at the beginning.

2. Graph passing

    With a PyTorch TorchScript graph, the integration maps PyTorch operators on the graph to the corresponding oneDNN Graph operators to form a backend graph.

3. Partitioning

    The backend selects regions to be fused in the graph and returns a list of partitions. Each partition corresponds to a set of fused operators.

4. Graph rewriting

    The original PyTorch JIT graph will be re-written based on the partitions returned from the backend. The operators in one partition will be grouped together to form a JIT operator, referred to as a oneDNN Graph fusion group.

5. Layout propagation

    This pass is to eliminate unnecessary layout conversions at partition boundaries. We set different formats to the output of a partition so that the backend could perform layout conversion internally. When `ANY` is set, the layout at boundaries will be fully decided by the backend. Otherwise, the backend should follow the layout set by PyTorch. Currently, we set `ANY` layout for a tensor that's an output of a oneDNN Graph partition, and an input to another.

## Graph Executor
During runtime execution of a (re-written) PyTorch JIT graph, oneDNN graph partitions will be dispatched to the oneDNN graph JIT variadic Operator.
Inside the oneDNN graph JIT Op, input PyTorch tensors of each partition will be mapped to oneDNN graph tensors. The partition will then be [compiled](https://spec.oneapi.io/onednn-graph/latest/programming_model.html#partition) and [executed](https://spec.oneapi.io/onednn-graph/latest/programming_model.html#compiled-partition). The output oneDNN graph tensor will be mapped back to PyTorch tensors to be fed to the next operator on the PyTorch JIT graph.


## Tests

```bash
pytest test/test_jit_llga_fuser.py
```

## Quick Start

A simple cascaded Conv-Relu example is provided in test. Please consider enabling log outputs to familiarize yourself with the whole pipeline:

**Mutation Removal -> Prepare Binary -> Defer Size Check -> Graph Fuser -> Layout Propagation -> Type Guard -> Kernel Execution**

oneDNN Graph was formerly known as LLGA (Low Level Graph API),
and thus LLGA in the codebase corresponds to oneDNN Graph.

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

CMake files where bridge code is included:

```bash
caffe2/CMakeLists.txt
```

CMake files where oneDNN Graph submodule are included:

```bash
third_party/ideep/mkl-dnn
cmake/public/mkldnn.cmake
cmake/Modules/FindMKLDNN.cmake
cmake/Dependencies.cmake
```

To map another op to oneDNN Graph, you should add an entry for it in in createOperator in torch/csrc/jit/codegen/onednn/graph_helper.cpp.
If it has an inplace variant, you should add it in the lambda being passed to RemoveTensorMutation in
torch/csrc/jit/codegen/onednn/interface.cpp. You might also want to add it to canFuseNode in torch/csrc/jit/codegen/onednn/register_interface.cpp.

## Example with Float


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
    # oneDNN graph fusion will be triggered during runtime
    output = model(images)
```

## Example with BFloat16

```python
# Assuming we have a model of the name 'model'

example_input = torch.rand(1, 3, 224, 224)

# enable oneDNN Graph
torch.jit.enable_onednn_fusion(True)
# Disable AMP for JIT
torch._C._jit_set_autocast_mode(False)
with torch.no_grad(), torch.cpu.amp.autocast():
    model = torch.jit.trace(model, (example_input))
    model = torch.jit.freeze(model)
     # 2 warm-ups (2 for tracing/scripting with an example, 3 without an example)
    model(example_input)
    model(example_input)

    # speedup would be observed in subsequent runs.
    model(example_input)
```
