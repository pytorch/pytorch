
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

AtenTensorHandle new_tensor_handle(at::Tensor&& tensor) {
  at::Tensor* new_tensor = new at::Tensor(std::move(tensor));
  return tensor_pointer_to_tensor_handle(new_tensor);
}

std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    std::vector<at::Tensor>& tensors) {
  std::vector<AtenTensorHandle> result;
  result.reserve(tensors.size());
  for (auto tensor : tensors) {
    auto allocated = new at::Tensor(std::move(tensor));
    result.push_back(tensor_pointer_to_tensor_handle(allocated));
  }
  return result;
}

std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    AtenTensorHandle* handles,
    size_t length) {
  std::vector<at::Tensor> result;
  result.reserve(length);
  for (size_t i = 0; i < length; i++) {
    result.emplace_back(
        std::move(*tensor_handle_to_tensor_pointer(handles[i])));
    aoti_torch_delete_tensor_object(handles[i]);
    handles[i] = nullptr;
  }
  return result;
}

} // namespace aot_inductor
} // namespace torch
