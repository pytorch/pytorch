#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <ATen/Tensor.h>

namespace torch {
namespace aot_inductor {

TORCH_API at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle);

TORCH_API AtenTensorHandle tensor_pointer_to_tensor_handle(at::Tensor* tensor);

TORCH_API std::vector<AtenTensorHandle> borrow_tensors_to_handles(
    std::vector<at::Tensor>& tensors);

TORCH_API std::vector<AtenTensorHandle> steal_tensors_to_handles(
    std::vector<at::Tensor>& tensors);

TORCH_API std::vector<at::Tensor> borrow_handles_to_tensors(
    std::vector<AtenTensorHandle>& handles);

TORCH_API std::vector<at::Tensor> steal_handles_to_tensors(
    std::vector<AtenTensorHandle>& handles);

} // namespace aot_inductor
} // namespace torch
