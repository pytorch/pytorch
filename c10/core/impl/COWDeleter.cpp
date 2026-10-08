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

auto cow::COWDeleterContext::init_stream(c10::Stream stream, bool captured)
    -> void {
  std::lock_guard lock(stream_mutex_);
  stream_ = stream;
  captured_ = captured;
}

auto cow::COWDeleterContext::share(c10::Stream stream, bool captured) -> bool {
  std::lock_guard lock(stream_mutex_);
  // Under the lock, the refcount can't be incremented concurrently, and it
  // can't drop to zero while the caller holds its reference.
  if (refcount_ == 1) {
    // Moving to another stream: that stream must wait for pending copies,
    // and recording the event on it again (after the wait) keeps them
    // ordered before later steals. During capture, we can't wait for eager
    // work, and no copies happen (see materialize_cow), so the event is left
    // for steals after the capture.
    if (copy_event_.has_value() && !captured && copy_event_stream_ != stream) {
      copy_event_->block(stream);
      copy_event_->record(stream);
      copy_event_stream_ = stream;
    }
    stream_ = stream;
    captured_ = captured;
  } else if (stream_ != stream || captured_ != captured) {
    return false;
  }
  increment_refcount();
  return true;
}

auto cow::COWDeleterContext::record_copy_event() -> void {
  std::lock_guard lock(stream_mutex_);
  TORCH_INTERNAL_ASSERT(stream_.has_value());
  if (!copy_event_.has_value()) {
    copy_event_.emplace(stream_->device_type());
  }
  // Supersedes the previous event: copies are all enqueued on stream_, which
  // waited for the previous copies if it changed (see share()).
  copy_event_->record(*stream_);
  copy_event_stream_ = stream_;
}

auto cow::COWDeleterContext::wait_for_copies(c10::Stream stream) -> void {
  std::lock_guard lock(stream_mutex_);
  if (copy_event_.has_value() && copy_event_stream_ != stream) {
    copy_event_->block(stream);
  }
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

} // namespace c10::impl
