#pragma once

#include <c10/core/Stream.h>
#include <c10/macros/Export.h>
#include <c10/util/UniqueVoidPtr.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <variant>
#include <vector>

namespace c10::impl::cow {

// An event recorded after a reference to stream-ordered copy-on-write data is
// done with it. See "Releasing lazy clones" in README-cow.md.
struct C10_API ReleaseEvent {
  // A backend event handle, as used by DeviceGuardImplInterface::block().
  void* event;
  // Keeps `event` alive.
  std::shared_ptr<void> owner;
};

// A COWDeleterContext object is used as the `ctx` argument for DataPtr
// to implement a Copy-on-write (COW) DataPtr.
class C10_API COWDeleterContext {
 public:
  // Creates an instance, holding the pair of data and original
  // deleter.
  //
  // Note that the deleter will only be called in our destructor if
  // the last reference to this goes away without getting
  // materialized.
  //
  // `stream` is the allocation stream of stream-ordered (e.g. CUDA) data, on
  // which all references to it are created. See README-cow.md.
  explicit COWDeleterContext(
      std::unique_ptr<void, DeleterFnPtr> data,
      std::optional<Stream> stream = std::nullopt);

  const std::optional<Stream>& stream() const {
    return stream_;
  }

  // Increments the current refcount.
  void increment_refcount();

  // Counts one of the references to stream-ordered data as released: it is
  // done with the data once `event` completes (or already, if nullopt).
  void release(std::optional<ReleaseEvent> event);

  // Adds an event to wait for to a reference which was already released.
  void add_release_event(ReleaseEvent event);

  // Whether the only remaining reference may take the stream-ordered data
  // after waiting for the release events: every other reference ever created
  // has been released. `released` is whether the remaining one was.
  bool can_steal(bool released);

  // Makes `stream` wait for the release events, which are then dropped.
  void wait_for_releases(const Stream& stream);

  // See README.md in this directory to understand the locking
  // strategy.

  // Represents a reference to the context.
  //
  // This is returned by decrement_refcount to allow the caller to
  // copy the data under the shared lock.
  using NotLastReference = std::shared_lock<std::shared_mutex>;

  // Represents the last reference to the context.
  //
  // This will be returned by decrement_refcount when it is the last
  // reference remaining and after any pending copies have completed.
  using LastReference = std::unique_ptr<void, DeleterFnPtr>;

  // Decrements the refcount, returning a handle indicating what to
  // do with it.
  std::variant<NotLastReference, LastReference> decrement_refcount();

 private:
  // The destructor is hidden, this should only ever be used within
  // UniqueVoidPtr using cow::delete_context as the deleter.
  ~COWDeleterContext();

  std::shared_mutex mutex_;
  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 1;
  const std::optional<Stream> stream_;

  // The number of references ever created, and how many were released.
  std::atomic<std::int64_t> references_ = 1;
  std::mutex release_mutex_;
  std::int64_t releases_ = 0;
  std::vector<ReleaseEvent> release_events_;
};

// `cow_deleter` is used as the `ctx_deleter` for DataPtr to implement a COW
// DataPtr.
//
// Warning: This should only be called on a pointer to a COWDeleterContext that
// was allocated on the heap with `new`, because when the refcount reaches 0,
// the context is deleted with `delete`.
C10_API void cow_deleter(void* ctx);

} // namespace c10::impl::cow
