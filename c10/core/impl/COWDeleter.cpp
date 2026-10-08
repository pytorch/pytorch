#include <c10/core/impl/COWDeleter.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  static_cast<cow::COWDeleterContext*>(ctx)->decrement_refcount();
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

auto cow::COWDeleterContext::set_stream(c10::Stream stream, bool captured)
    -> void {
  stream_ = stream;
  captured_ = captured;
  std::lock_guard lock(copy_event_mutex_);
  copy_event_.reset();
}

auto cow::COWDeleterContext::record_copy_event() -> void {
  TORCH_INTERNAL_ASSERT(stream_.has_value());
  std::lock_guard lock(copy_event_mutex_);
  if (!copy_event_.has_value()) {
    copy_event_.emplace(stream_->device_type());
  }
  // Supersedes the previous event: copies are all enqueued on stream_.
  copy_event_->record(*stream_);
}

auto cow::COWDeleterContext::wait_for_copies(c10::Stream stream) -> void {
  std::lock_guard lock(copy_event_mutex_);
  if (copy_event_.has_value() && stream != stream_) {
    copy_event_->block(stream);
  }
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

} // namespace c10::impl
