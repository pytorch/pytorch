# NestedTensors
So you decided to look at the source code.. let's give you a quick overview of the codebase.

### NestedTensor Data Structure

NestedTensors are a generalization of torch Tensors which eases working with data of different shapes and lengths. NestedTenosrs are mainly used to represent a list of N tensor_components. The N tensor_components are flattened and concatenated into a single tensor. The NestedTensor data structure contains the following information to recover the original tensor_components:

- size_tensor: 2d tensor of n_tensor_components x n_dims
- stride_tensor: 2d tensor of n_tensor_components x n_dims
- offsets: 1d tensor of offsets corresponding to the start position of each tensor component

NestedTensors inherit from c10::TensorImpl and their definition can be found here: [NestedTensorImpl.h](../../NestedTensorImpl.h).

When constructing a NestedTensor in C++ you will likely not be using the NestedTensorImpl constructor directly but use the `wrap_buffer` function defined in [NestedTensorUtils.h](NestedTensorUtils.h). This is a thin wrapper around the NestedTensorImpl constructor that insures that the input tensor is contiguous. This is because when constructing a NestedTensor from a dense tensor we create a shallow copy of the input's Storage object and the if the input tensor did not satisfy tensor.numel() == tensor.storage.numel() this could lead to undefined behavior.

###  Code Structure

The NestedTensor code is split into two parts: the C++ code and the Python code. The C++ code is located in [aten/src/ATen/native/nested](.) and the Python code is located in [torch/nested/__init__.py](/torch/nested/__init__.py). The C++ code is split into the following files:

- `NestedTensorImpl.h | NestedTensorImpl.cpp`: The NestedTensor data structure and its methods.
- `NestedTensorUtils.h | NestedTensorUtils.cpp`: Utility functions for working with NestedTensors. (This is where you will find  `map_nested_tensor` which is discussed below in the section on implementing new functions.)
- `NestedTensorUnaryOps.cpp`: Unary operations on NestedTensors (functions that can be efficiently implemented via map_nt)
- `NestedTensorBinaryOps.h | NestedTensorBinaryOps.cpp`: Binary operations on NestedTensors (functions that can be efficiently implemented via NestedTensor_elementwise_Tensor which can be found in the cpp file)
- `NestedTensorFactories.h | NestedTensorFactories.cpp`: Functions for creating NestedTensors (e.g. empty_like)
- `NestedTensorMath.h | NestedTensorMath.cpp`: Math functions on NestedTensors (e.g. softmax, embedding)
- `NestedTensorMatmul.cpp`: Matmul functions on NestedTensors (e.g. matmul, linear, bmm)
- `NestedTensorTransformerFunctions.h | NestedTensorTransformerFunctions.cpp`: Functions for enabling the BetterTransformer workstream
-


###  Implementing new functions

###  Best Practices