#include <ATen/core/TensorBody.h>

namespace torch {
namespace utils {
namespace hooks {

unsigned RemovableHandle::next_id = 0;

RemovableHandle::RemovableHandle(std::shared_ptr<torch::autograd::hooks_dict> hooks_dict)
  : hooks_dict_ref_(hooks_dict),
    id_(RemovableHandle::next_id) {
  RemovableHandle::next_id++;  // yf225 TODO: we need to protect this with atomic variable, otherwise there can be race condition!
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
