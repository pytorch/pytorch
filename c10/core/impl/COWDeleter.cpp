#include <c10/core/impl/COWDeleter.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  std::unique_ptr<cow::COWReference> ref(static_cast<cow::COWReference*>(ctx));
  ref->context->decrement_refcount();
}

cow::COWDeleterContext::COWDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data)
    : data_(std::move(data)) {
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

auto cow::COWDeleterContext::is_unique() const -> bool {
  return refcount_ == 1;
}

auto cow::COWDeleterContext::record_copy_stream(c10::Stream stream) -> void {
  std::lock_guard lock(copy_stream_mutex_);
  if (!copy_stream_.has_value()) {
    copy_stream_ = stream;
  } else if (*copy_stream_ != stream) {
    copy_streams_differ_ = true;
  }
}

auto cow::COWDeleterContext::copies_ordered_before(c10::Stream stream) -> bool {
  std::lock_guard lock(copy_stream_mutex_);
  return !copy_streams_differ_ &&
      (!copy_stream_.has_value() || *copy_stream_ == stream);
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

} // namespace c10::impl
