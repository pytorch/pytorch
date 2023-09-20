
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

std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    std::vector<AtenTensorHandle>& handles) {
  std::vector<at::Tensor> result;
  result.reserve(handles.size());
  for (auto handle : handles) {
    result.emplace_back(std::move(*tensor_handle_to_tensor_pointer(handle)));
  }
  handles.clear();
  return result;
}

RAIITensorToHandleAllocator::RAIITensorToHandleAllocator(
    std::vector<at::Tensor>& tensors) {
  size_t size = tensors.size();
  handles_ = new AtenTensorHandle[size];
  for (size_t i = 0; i < size; i++) {
    at::Tensor* allocated = new at::Tensor(tensors[i]);
    handles_[i] = tensor_pointer_to_tensor_handle(allocated);
  }
}

} // namespace aot_inductor
} // namespace torch
