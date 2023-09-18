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

// create_handles_from_tensors is used for turning a vector of aten tensors into
// a vector of AtenTensorHandles, and then pass that into model.so. Right now we
// create new references and return their raw pointers as a vector, and the
// returned raw pointers will be wrapped by RAIIAtenTensorHandle in model.so.
TORCH_API std::vector<AtenTensorHandle> create_handles_from_tensors(
    std::vector<at::Tensor>& tensors);

// create_tensors_from_handles is used for turning a vector of AtenTensorHandles
// into a vector of aten tensors. We don't free the passed in AtenTensorHandles
// as they are owned by RAIIAtenTensorHandle in model.so.
TORCH_API std::vector<at::Tensor> create_tensors_from_handles(
    std::vector<AtenTensorHandle>& handles);

} // namespace aot_inductor
} // namespace torch
