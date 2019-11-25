#include <ATen/core/TensorBody.h>

namespace torch {
namespace utils {
namespace hooks {

std::atomic<unsigned> RemovableHandle::next_id{0};

RemovableHandle::RemovableHandle(std::shared_ptr<torch::autograd::hooks_dict> hooks_dict)
  : hooks_dict_ref_(hooks_dict),
    id_(RemovableHandle::next_id.load()) {
  RemovableHandle::next_id++;
}

void RemovableHandle::remove() const {
  if (auto hooks_dict = hooks_dict_ref_.lock()) {
    if (hooks_dict->contains(id_)) {
      hooks_dict->erase(id_);
    }
  }
}

unsigned RemovableHandle::id() const {
  return id_;
}

} // namespace hooks
} // namespace utils
} // namespace torch
