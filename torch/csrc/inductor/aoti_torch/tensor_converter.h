#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <ATen/Tensor.h>

namespace torch {
namespace aot_inductor {

// Functions declared here are not meant to be called from the AOTInductor
// generated model.so

// No ownership transfer, just pointer type conversion
TORCH_API at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle);

// No ownership transfer, just pointer type conversion
TORCH_API AtenTensorHandle tensor_pointer_to_tensor_handle(at::Tensor* tensor);

TORCH_API AtenTensorHandle new_tensor_handle(at::Tensor&& tensor);

// unsafe_alloc_new_handles_from_tensors is used for allocating new aten
// tensor objects and return them as a vector of AtenTensorHandle (raw
// pointers), and those pointers will be stolen by model.so.
TORCH_API std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    std::vector<at::Tensor>& tensors);

// alloc_tensors_by_stealing_from_handles is used for creating a vector of aten
// tensors by stealing from an array of handles. Only the handles are stolen,
// and the array itself is borrowed.
//
// WARNING: Can NOT be called in model.so unless in the non-ABI-compatible mode
TORCH_API std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    AtenTensorHandle* handles,
    size_t length);

} // namespace aot_inductor
} // namespace torch
