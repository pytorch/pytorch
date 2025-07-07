#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

namespace torch::aot_inductor {

// Functions declared here are not meant to be called from the AOTInductor
// generated model.so

// unsafe_alloc_new_handles_from_tensors is used for allocating new aten
// tensor objects and return them as a vector of AtenTensorHandle (raw
// pointers), and those pointers will be stolen by model.so.
TORCH_API std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    const std::vector<at::Tensor>& tensors);

// alloc_tensors_by_stealing_from_handles is used for creating a vector of aten
// tensors by stealing from an array of handles. Only the handles are stolen,
// and the array itself is borrowed.
//
// WARNING: Can NOT be called in model.so
TORCH_API std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    AtenTensorHandle* handles,
    size_t length);

} // namespace torch::aot_inductor
