#include <torch/csrc/autograd/forward_grad.h>

namespace torch { namespace autograd {

namespace {
    static uint64_t next_forward_idx_ = 0;
    static std::vector<std::shared_ptr<ForwardADLevel>> all_forward_levels_;
    static std::mutex all_forward_levels_mutex_;

    static at::Tensor singleton_undefined_tensor;

}

uint64_t ForwardADLevel::get_next_idx() {
    std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
    TORCH_CHECK(next_forward_idx_ == 0, "Nested forward mode AD is not supported at the moment");
    auto new_index = next_forward_idx_++;
    TORCH_INTERNAL_ASSERT(new_index == all_forward_levels_.size());
    all_forward_levels_.push_back(std::make_shared<ForwardADLevel>(new_index));
    return new_index;
}

void ForwardADLevel::release_idx(uint64_t idx) {
    std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
    TORCH_CHECK(idx == all_forward_levels_.size() - 1, "Exiting a forward AD level that is not the "
                "last that was created is not support. Ensure they are released in the reverse "
                "order they were created.");
    TORCH_CHECK(idx >= 0, "No forward AD level was created so you cannot exit it.");
    next_forward_idx_--;
    all_forward_levels_.pop_back();

}
std::shared_ptr<ForwardADLevel> ForwardADLevel::get_by_idx(uint64_t idx) {
    std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
    TORCH_CHECK(idx < all_forward_levels_.size(), "Trying to access a forward AD level with an invalid index. "
                "This index was either not created or is already deleted.");
    return all_forward_levels_[idx];
}

ForwardADLevel::~ForwardADLevel() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = grads_.begin();
    while (it != grads_.end()) {
        (*it)->reset(idx_, /* update_level */ false);
        it = grads_.erase(it);
    }
}

const at::Tensor& ForwardGrad::value(uint64_t level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& it = content_.find(level);
    return it == content_.end() ? singleton_undefined_tensor : (*it).second;
}

}} // namespace torch::autograd
