
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

std::vector<AtenTensorHandle> borrow_tensors_to_handles(
    std::vector<at::Tensor>& tensors) {
  std::vector<AtenTensorHandle> result;
  result.reserve(tensors.size());
  for (auto tensor : tensors) {
    auto allocated = new at::Tensor(std::move(tensor));
    result.push_back(tensor_pointer_to_tensor_handle(allocated));
  }
  return result;
}

std::vector<AtenTensorHandle> steal_tensors_to_handles(
    std::vector<at::Tensor>& tensors) {
  auto result = borrow_tensors_to_handles(tensors);
  tensors.clear();
  return result;
}

std::vector<at::Tensor> borrow_handles_to_tensors(
    std::vector<AtenTensorHandle>& handles) {
  std::vector<at::Tensor> result;
  result.reserve(handles.size());
  for (auto handle : handles) {
    auto tensor = *tensor_handle_to_tensor_pointer(handle);
    result.emplace_back(std::move(tensor));
  }
  return result;
}

std::vector<at::Tensor> steal_handles_to_tensors(
    std::vector<AtenTensorHandle>& handles) {
  auto result = borrow_handles_to_tensors(handles);
  for (auto handle : handles) {
    aoti_torch_delete_tensor_object(handle);
  }
  handles.clear();
  return result;
}

} // namespace aot_inductor
} // namespace torch
