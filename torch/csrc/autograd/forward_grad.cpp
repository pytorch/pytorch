#include <torch/csrc/autograd/forward_grad.h>

namespace torch { namespace autograd {

namespace {
    // See discussion in forward_grad.h for why these are global variables and not
    // thread local

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex all_forward_levels_mutex_;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static uint64_t next_forward_idx_ = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static std::vector<std::shared_ptr<ForwardADLevel>> all_forward_levels_;

    const static at::Tensor singleton_undefined_tensor;

    // Temporary flag to disable forward mode
    // TODO(alband) remove these when perf issues are solved
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static bool is_forward_grad_enabled = false;
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

std::shared_ptr<ForwardADLevel> ForwardADLevel::try_get_by_idx(uint64_t idx) {
    std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
    if (idx < all_forward_levels_.size()) {
        return all_forward_levels_[idx];
    } else {
        return nullptr;
    }
}

ForwardADLevel::~ForwardADLevel() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = grads_.begin();
    while (it != grads_.end()) {
        // Warning this will lock *it mutex
        // This is ok as this function is the *only* one to call back into another class's method.
        (*it)->reset(idx_, /* update_level */ false);
        it = grads_.erase(it);
    }
}

const at::Tensor& ForwardGrad::value(uint64_t level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& it = content_.find(level);
    return it == content_.end() ? singleton_undefined_tensor : (*it).second;
}

const at::Tensor& ForwardGrad::undef_grad() {
    return singleton_undefined_tensor;
}

}} // namespace torch::autograd
