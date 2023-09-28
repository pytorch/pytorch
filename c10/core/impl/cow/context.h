#pragma once

#include <c10/macros/Export.h>
#include <c10/util/UniqueVoidPtr.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <variant>

namespace c10::impl::cow {

/// The c10::DataPtr context for copy-on-write storage.
class C10_API Context {
 public:
  /// Creates an instance, holding the pair of data and original
  /// deleter.
  ///
  /// Note that the deleter will only be called in our destructor if
  /// the last reference to this goes away without getting
  /// materialized.
  explicit Context(std::unique_ptr<void, DeleterFnPtr> data);

  /// Increments the current refcount.
  auto increment_refcount() -> void;

  // See README.md in this directory to understand the locking
  // strategy.

  /// Represents a reference to the context.
  ///
  /// This is returned by decrement_refcount to allow the caller to
  /// copy the data under the shared lock.
  using NotLastReference = std::shared_lock<std::shared_mutex>;

  /// Represents the last reference to the context.
  ///
  /// This will be returned by decrement_refcount when it is the last
  /// reference remaining and after any pending copies have completed.
  using LastReference = std::unique_ptr<void, DeleterFnPtr>;

  /// Decrements the refcount, returning a handle indicating what to
  /// do with it.
  auto decrement_refcount() -> std::variant<NotLastReference, LastReference>;

 private:
  // The destructor is hidden, this should only ever be used within
  // UniqueVoidPtr using cow::delete_context as the deleter.
  ~Context();

  std::shared_mutex mutex_;
  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 1;
};

} // namespace c10::impl::cow
