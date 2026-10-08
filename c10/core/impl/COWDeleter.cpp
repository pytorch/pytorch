#include <c10/core/impl/COWDeleter.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  auto* ref = static_cast<cow::COWReference*>(ctx);
  ref->context->decrement_refcount();
  delete ref;
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
    LastReference result{std::move(data_), std::move(copy_events_)};
    lock.unlock();
    delete this;
    return {std::move(result)};
  }

  return std::shared_lock(mutex_);
}

auto cow::COWDeleterContext::is_unique() const -> bool {
  return refcount_ == 1;
}

auto cow::COWDeleterContext::record_copy_event(c10::Stream stream) -> void {
  std::lock_guard lock(copy_events_mutex_);
  for (auto& [copy_stream, event] : copy_events_) {
    if (copy_stream == stream) {
      // Supersedes the previous event recorded on the same stream.
      event.record(stream);
      return;
    }
  }
  c10::Event event(stream.device_type());
  event.record(stream);
  copy_events_.emplace_back(stream, std::move(event));
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

} // namespace c10::impl
