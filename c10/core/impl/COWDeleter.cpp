#include <c10/core/impl/COWDeleter.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  static_cast<cow::COWDeleterContext*>(ctx)->decrement_refcount();
}

cow::COWDeleterContext::COWDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data,
    std::optional<Stream> stream)
    : data_(std::move(data)), stream_(stream) {
  // We never wrap a COWDeleterContext.
  TORCH_INTERNAL_ASSERT(data_.get_deleter() != cow::cow_deleter);
}

auto cow::COWDeleterContext::increment_refcount() -> void {
  auto refcount = ++refcount_;
  TORCH_INTERNAL_ASSERT(refcount > 1);
  ++references_;
}

void cow::COWDeleterContext::release(std::optional<ReleaseEvent> event) {
  std::lock_guard lock(release_mutex_);
  ++releases_;
  if (event) {
    release_events_.push_back(std::move(*event));
  }
}

void cow::COWDeleterContext::add_release_event(ReleaseEvent event) {
  std::lock_guard lock(release_mutex_);
  release_events_.push_back(std::move(event));
}

bool cow::COWDeleterContext::can_steal(bool released) {
  std::lock_guard lock(release_mutex_);
  return refcount_ == 1 && releases_ - released >= references_ - 1;
}

void cow::COWDeleterContext::wait_for_releases(const Stream& stream) {
  std::vector<ReleaseEvent> events;
  {
    std::lock_guard lock(release_mutex_);
    const auto* impl = getDeviceGuardImpl(stream.device_type());
    for (const auto& event : release_events_) {
      impl->block(event.event, stream);
    }
    events.swap(release_events_);
  }
  // `events` is destroyed outside of the lock, since dropping a Python event
  // acquires the GIL.
}

auto cow::COWDeleterContext::decrement_refcount()
    -> std::variant<NotLastReference, LastReference> {
  auto refcount = --refcount_;
  TORCH_INTERNAL_ASSERT(refcount >= 0, refcount);
  if (refcount == 0) {
    std::unique_lock lock(mutex_);
    auto result = std::move(data_);
    lock.unlock();
    if (stream_) {
      // Unless the caller steals the data (and has already waited for the
      // release events), it is freed, and the allocator orders its reuse
      // after the allocation stream.
      try {
        wait_for_releases(*stream_);
      } catch (const std::exception& e) {
        TORCH_WARN(
            "Leaking copy-on-write data, since waiting for its release "
            "events failed: ",
            e.what());
        (void)result.release();
      }
    }
    delete this;
    return {std::move(result)};
  }

  return std::shared_lock(mutex_);
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

} // namespace c10::impl
