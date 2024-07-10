#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

namespace torch {
namespace aot_inductor {

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
  // Find duplicates by recording the last known index for each handle.
  std::unordered_map<AtenTensorHandle, size_t> lastKnownIdx;
  for (size_t i = 0; i < length; i++) {
    lastKnownIdx[handles[i]] = i;
  }

  std::vector<at::Tensor> result;
  result.reserve(length);
  for (size_t i = 0; i < length; i++) {
    if (handles[i] == nullptr) {
      result.emplace_back();
      continue;
    }

    at::Tensor tensor = *tensor_handle_to_tensor_pointer(handles[i]);
    if (lastKnownIdx[handles[i]] != i) {
      result.emplace_back(tensor);
    } else {
      result.emplace_back(std::move(tensor));
      aoti_torch_delete_tensor_object(handles[i]);
    }
    handles[i] = nullptr;
  }

  return result;
}

} // namespace aot_inductor
} // namespace torch
