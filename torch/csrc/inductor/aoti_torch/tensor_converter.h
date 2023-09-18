#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <ATen/Tensor.h>

namespace torch {
namespace aot_inductor {

// Functions declared here are not meant to be called from the AOTInductor
// generated model.so

TORCH_API at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle);

TORCH_API AtenTensorHandle tensor_pointer_to_tensor_handle(at::Tensor* tensor);

// borrow_tensors_to_handles is used for turning a vector of aten tensors into
// a vector of AtenTensorHandles, and then pass that into model.so.
//
// Let's start with borrowing ownership instead of stealing, correctness first
// and then optimize for performance
TORCH_API std::vector<AtenTensorHandle> borrow_tensors_to_handles(
    std::vector<at::Tensor>& tensors);

// borrow_handles_to_tensors is used for turning a vector of AtenTensorHandles
// into a vector of aten tensors.
//
// Let's start with borrowing ownership instead of stealing, correctness first
// and then optimize for performance
TORCH_API std::vector<at::Tensor> borrow_handles_to_tensors(
    std::vector<AtenTensorHandle>& handles);

} // namespace aot_inductor
} // namespace torch
