#pragma once

#include <c10/core/StorageImpl.h>
#include <c10/util/UniqueVoidPtr.h>

#include <atomic>
#include <cstdint>
#include <memory>

namespace c10::impl::cow {

// Encapsulates a c10::DataPtr context that is copy on write.
class Context {
 public:
  // Creates an instance, wrapping an existing context.
  explicit Context(std::unique_ptr<void, DeleterFnPtr> data);

  // The destructor is hidden, this should only ever be used within
  // UniqueVoidPtr using this as the deleter.
  static auto delete_instance(void* ctx) -> void;

  // Gets the current refcount.
  auto refcount() const -> std::int64_t;

  // Increments the current refcount.
  auto increment_refcount() -> void;

  // Decrements the current refcount.
  auto decrement_refcount() -> void;

 private:
  ~Context();

  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 0;
};

} // namespace c10::impl::cow
