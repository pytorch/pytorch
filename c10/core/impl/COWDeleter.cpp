#include <c10/core/impl/COWDeleter.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  static_cast<cow::COWDeleterContext*>(ctx)->decrement_refcount();
}

cow::COWDeleterContext::COWDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data,
    c10::Device original_device,
    c10::Allocator* original_allocator)
    : data_(std::move(data)),
      original_device_(original_device),
      original_allocator_(original_allocator) {
  // We never wrap a COWDeleterContext.
  TORCH_INTERNAL_ASSERT(data_.get_deleter() != cow::cow_deleter);
}

auto cow::COWDeleterContext::increment_refcount() -> void {
  auto refcount = ++refcount_;
  TORCH_INTERNAL_ASSERT(refcount > 1);
}

auto cow::COWDeleterContext::decrement_refcount()
    -> std::variant<NotLastReference, LastReference> {
  auto refcount = --refcount_;
  TORCH_INTERNAL_ASSERT(refcount >= 0, refcount);
  if (refcount == 0) {
    std::unique_lock lock(mutex_);
    auto result = std::move(data_);
    lock.unlock();
    delete this;
    return {std::move(result)};
  }

  return std::shared_lock(mutex_);
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

c10::Device cow::COWDeleterContext::original_device() {
  return original_device_;
}

c10::Allocator* cow::COWDeleterContext::original_allocator() {
  return original_allocator_;
}

} // namespace c10::impl
