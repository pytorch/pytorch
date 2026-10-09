#include <c10/core/impl/COWDeleter.h>

#include <c10/core/Event.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl {

namespace {

// Makes `waiter` wait (on the device) for all work enqueued on `stream` so
// far, if it is another stream. Skipped if `stream` is being captured into a
// graph: eager work can't wait for captured work (and waiting on an event
// recorded during capture would make `waiter` join the capture).
void wait_for_stream(c10::Stream waiter, c10::Stream stream) {
  if (waiter == stream || stream.is_capturing()) {
    return;
  }
  c10::Event event(stream.device_type());
  event.record(stream);
  event.block(waiter);
}

} // namespace

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
    // Moving to another stream: it waits for the work enqueued on the
    // previous one, so that a later fence on it (see fence()) covers that
    // work too. Not during capture: a capture can't wait for eager work,
    // and captured work only runs when the graph is replayed, which the user
    // already orders after the eager work it depends on.
    TORCH_INTERNAL_ASSERT(stream_.has_value());
    if (!captured) {
      wait_for_stream(stream, *stream_);
    }
    stream_ = stream;
    captured_ = captured;
  } else if (stream_ != stream || captured_ != captured) {
    return false;
  }
  increment_refcount();
  return true;
}

auto cow::COWDeleterContext::fence(c10::Stream stream) -> void {
  std::lock_guard lock(stream_mutex_);
  TORCH_INTERNAL_ASSERT(stream_.has_value());
  wait_for_stream(stream, *stream_);
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

} // namespace c10::impl
