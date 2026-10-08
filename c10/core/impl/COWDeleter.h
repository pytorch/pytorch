#pragma once

#include <c10/core/Event.h>
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

namespace c10::impl::cow {

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
  explicit COWDeleterContext(std::unique_ptr<void, DeleterFnPtr> data);

  // Increments the current refcount.
  void increment_refcount();

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

  // Returns true if there is only a single reference to this context. Only
  // meaningful when called through that reference, since no new references
  // can be created concurrently in that case.
  bool is_unique() const;

  // On devices whose streams are tracked (see COW.cpp), all references to
  // the data share the stream they were made on, and whether that stream was
  // being captured into a graph then. They are set when the first lazy clone
  // is made, and may only be changed through the only remaining reference
  // (after waiting for copies, see wait_for_copies()).
  void set_stream(c10::Stream stream, bool captured);
  std::optional<c10::Stream> stream() const {
    return stream_;
  }
  bool captured() const {
    return captured_;
  }

  // Records an event after a copy of the data was enqueued on stream().
  // Must be called before the reference that made the copy is decremented.
  void record_copy_event();

  // Makes `stream` wait (on the device) for the copies recorded with
  // record_copy_event(), if it isn't stream(). Must be called through the
  // only remaining reference.
  void wait_for_copies(c10::Stream stream);

 private:
  // The destructor is hidden, this should only ever be used within
  // UniqueVoidPtr using cow::delete_context as the deleter.
  ~COWDeleterContext();

  std::shared_mutex mutex_;
  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 1;

  std::optional<c10::Stream> stream_;
  bool captured_ = false;
  std::mutex copy_event_mutex_;
  std::optional<c10::Event> copy_event_;
};

// `cow_deleter` is used as the `ctx_deleter` for DataPtr to implement a COW
// DataPtr.
//
// Warning: This should only be called on a pointer to a COWDeleterContext that
// was allocated on the heap with `new`, because when the refcount reaches 0,
// the context is deleted with `delete`.
C10_API void cow_deleter(void* ctx);

} // namespace c10::impl::cow
