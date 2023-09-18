
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

namespace torch {
namespace aot_inductor {

at::Tensor* tensor_handle_to_tensor_pointer(AtenTensorHandle handle) {
  return reinterpret_cast<at::Tensor*>(handle);
}

AtenTensorHandle tensor_pointer_to_tensor_handle(at::Tensor* tensor) {
  return reinterpret_cast<AtenTensorHandle>(tensor);
}

std::vector<AtenTensorHandle> create_handles_from_tensors(
    std::vector<at::Tensor>& tensors) {
  std::vector<AtenTensorHandle> result;
  result.reserve(tensors.size());
  for (auto tensor : tensors) {
    auto allocated = new at::Tensor(std::move(tensor));
    result.push_back(tensor_pointer_to_tensor_handle(allocated));
  }
  return result;
}

std::vector<at::Tensor> create_tensors_from_handles(
    std::vector<AtenTensorHandle>& handles) {
  std::vector<at::Tensor> result;
  result.reserve(handles.size());
  for (auto handle : handles) {
    auto tensor = *tensor_handle_to_tensor_pointer(handle);
    result.emplace_back(std::move(tensor));
  }
  return result;
}

} // namespace aot_inductor
} // namespace torch
