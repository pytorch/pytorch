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

namespace c10::impl::cow {

// A COWDeleterContext object holds the data shared by all the COW
// DataPtrs that are lazy copies of one another. Each such DataPtr has a
// COWReference (below) as its `ctx`, which points to the shared context.
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

  // Records that a reference is about to be materialized by copying the data
  // on `stream`. Must be called before the corresponding decrement_refcount().
  void record_copy_stream(c10::Stream stream);

  // Returns true if every copy recorded with record_copy_stream() was on
  // `stream`, i.e., if work enqueued on `stream` is ordered after all of them.
  bool copies_ordered_before(c10::Stream stream);

 private:
  // The destructor is hidden, this should only ever be used within
  // UniqueVoidPtr using cow::delete_context as the deleter.
  ~COWDeleterContext();

  std::shared_mutex mutex_;
  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 1;

  std::mutex copy_stream_mutex_;
  std::optional<c10::Stream> copy_stream_;
  bool copy_streams_differ_ = false;
};

// The `ctx` of a COW DataPtr. There is one of these per DataPtr, while the
// COWDeleterContext is shared by all lazy copies of the same data.
struct COWReference {
  COWDeleterContext* context;
  // For a lazy clone, the stream that was current when the clone was made.
  // The clone has the semantics of a clone() enqueued on this stream, so it
  // may only be materialized by copying on this stream. Unset for the storage
  // that was lazily cloned from, which may only be materialized by copying on
  // the stream it was allocated on.
  std::optional<c10::Stream> clone_stream;
};

// `cow_deleter` is used as the `ctx_deleter` for DataPtr to implement a COW
// DataPtr.
//
// Warning: This should only be called on a pointer to a COWReference that was
// allocated on the heap with `new`, pointing to a COWDeleterContext that was
// allocated on the heap with `new`, because both are deleted with `delete`
// (the latter when its refcount reaches 0).
C10_API void cow_deleter(void* ctx);

} // namespace c10::impl::cow
