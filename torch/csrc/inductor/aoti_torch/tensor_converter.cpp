#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

namespace torch {
namespace aot_inductor {

std::vector<AtenTensorHandle> unsafe_alloc_new_handles_from_tensors(
    const std::vector<at::Tensor>& tensors) {
  std::vector<AtenTensorHandle> result;
  result.reserve(tensors.size());
  for (auto tensor : tensors) {
    auto allocated = new at::Tensor(std::move(tensor));
    result.push_back(tensor_pointer_to_tensor_handle(allocated));
  }
  return result;
}

std::vector<at::Tensor> alloc_tensors_by_stealing_from_handles(
    std::vector<AtenTensorHandle>& handles) {
  std::vector<at::Tensor> result;
  result.reserve(handles.size());
  for (auto handle : handles) {
    result.emplace_back(std::move(*tensor_handle_to_tensor_pointer(handle)));
    aoti_torch_delete_tensor_object(handle);
  }
  handles.clear();
  return result;
}

} // namespace aot_inductor
} // namespace torch
