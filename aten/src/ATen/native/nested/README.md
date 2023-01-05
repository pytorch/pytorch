# NestedTensors
So you decided to look at the source code.. let's give you a quick overview of the codebase.

## NestedTensor Data Structure

NestedTensors are a generalization of torch Tensors which eases working with data of different shapes and lengths. NestedTensors are mainly used to represent a list of N tensor_components. The N tensor_components are flattened and concatenated into a single tensor. The NestedTensor data structure contains the following information to recover the original tensor_components:

- size_tensor: 2d tensor of n_tensor_components x n_dims
- stride_tensor: 2d tensor of n_tensor_components x n_dims
- offsets: 1d tensor of offsets corresponding to the start position of each tensor component

NestedTensors inherit from c10::TensorImpl and their definition can be found here: [NestedTensorImpl.h](../../NestedTensorImpl.h).

When constructing a NestedTensor in C++ you will likely not be using the NestedTensorImpl constructor directly but use the `wrap_buffer` function defined in [NestedTensorUtils.h](NestedTensorUtils.h). This is a thin wrapper around the NestedTensorImpl constructor that insures that the input tensor is contiguous. This is because when constructing a NestedTensor from a dense tensor we create a shallow copy of the input's Storage object and the if the input tensor did not satisfy tensor.numel() == tensor.storage.numel() this could lead to undefined behavior.

##  Code Structure

The NestedTensor code is split into two parts: the C++ code and the Python code. The C++ code is located in [aten/src/ATen/native/nested](.) and the Python code is located in [torch/nested/__init__.py](/torch/nested/__init__.py). The C++ code is split into the following files:

- `NestedTensorImpl.h | NestedTensorImpl.cpp`: The NestedTensor data structure and its methods.
- `NestedTensorUtils.h | NestedTensorUtils.cpp`: Utility functions for working with NestedTensors. (This is where you will find  `map_nested_tensor` which is discussed below in the section on implementing new functions.)
- `NestedTensorUnaryOps.cpp`: Unary operations on NestedTensors (functions that can be efficiently implemented via map_nt)
- `NestedTensorBinaryOps.h | NestedTensorBinaryOps.cpp`: Binary operations on NestedTensors (functions that can be efficiently implemented via NestedTensor_elementwise_Tensor which can be found in the cpp file)
- `NestedTensorFactories.h | NestedTensorFactories.cpp`: Functions for creating NestedTensors (e.g. empty_like)
- `NestedTensorMath.h | NestedTensorMath.cpp`: Math functions on NestedTensors (e.g. softmax, embedding)
- `NestedTensorMatmul.cpp`: Matmul functions on NestedTensors (e.g. matmul, linear, bmm)
- `NestedTensorTransformerFunctions.h | NestedTensorTransformerFunctions.cpp`: Functions for enabling the BetterTransformer work stream
- `cuda/`: CUDA implementations of the NestedTensor functions

##  Implementing new functions
There are two main classes of functions that can be implemented on NestedTensors: functions that can be efficiently implemented by viewing the NestedTensor as dense tensor where the raggedness has been folded into the dense dimensions and those that can't. Examples of the former are all unary ops (e.g. abs, log, relu) and binary ops (e.g. add, mul, div). If the function acts element wise on one NestedTensor or multiple NestedTensors than an efficient implementation is relative straight forward.

The definition of map_nt is:

```cpp
template <typename Func>
Tensor map_nt(const Tensor& nt, Func f) {
  auto* nt_impl = get_nested_tensor_impl(nt);
  const auto& sizes = nt_impl->get_nested_size_tensor();
  return at::detail::make_tensor<NestedTensorImpl>(f(nt_impl->get_buffer()), sizes);
}
```
1. Get the NestedTensorImpl from the input NestedTensor.
2. Get the sizes of the NestedTensor.
3. Call get_buffer() which returns a flat, dense tensor whose storage shares that of the input NestedTensor.
4. Call the function f on the dense tensor.
5. Construct a new NestedTensor from the output of f and the sizes of the input NestedTensor.

There are also important functions that, under certain conditions of regularity, can be implemented effectively by accessing the underlying buffer and viewing it in a special manner. For a good example of this see the implementation of `linear` in [NestedTensorTransformerFunctions.cpp](NestedTensorTransformerFunctions.cpp).

The second class of functions can't be efficiently implemented by viewing the NestedTensor as a dense tensor. An example of this is `softmax_nested`. The implementation of this function can be found in [NestedTensorMath.cpp](NestedTensorMath.cpp). This can occur when there are strict problem boundaries that occur at the tensor_component level. For example, when computing the softmax of a NestedTensor we need to compute the softmax of each tensor_component individually. This is because the softmax function is not defined on tensors of different shapes. In this case we can't view the NestedTensor as a dense tensor and apply the softmax function to the entire tensor at once. Instead we need to apply the softmax function to each tensor_component individually.

The problem with this though is that iterating over a potentially large number of tensor_components and launching individual cuda kernels for each one is very inefficient. Ideally we would launch one kernel that operates on all the tensor_components in parallel.

If performance is not your main concern and you would like to enable coverage the function `map_nested_tensor` can be found in [NestedTensorUtils.h](NestedTensorUtils.h). This function iterates over tensor components to applying a function to each tensor_component individually. Be warned though that this is likely very inefficient, especially for cuda implementations of the function.

## Triton

##  Best Practices
