#include <ATen/core/TensorBody.h>

#include <torch/csrc/autograd/cpp_hook.h>

namespace torch {
namespace utils {
namespace hooks {

unsigned RemovableHandle::next_id = 0;

RemovableHandle::RemovableHandle(std::shared_ptr<torch::autograd::hooks_dict> hooks_dict)
  : hooks_dict_ref_(hooks_dict),
    id_(RemovableHandle::next_id) {
  RemovableHandle::next_id++;
}

void RemovableHandle::remove() {
  if (auto hooks_dict = hooks_dict_ref.lock() && hooks_dict->contains(id_)) {
    hooks_dict->erase(id_);
  }
}

} // namespace hooks
} // namespace utils
} // namespace torch
