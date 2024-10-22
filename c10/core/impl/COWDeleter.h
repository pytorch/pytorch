#pragma once

#include <c10/macros/Export.h>
#include <c10/util/UniqueVoidPtr.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
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

  // Wrap the current underlying data a wrapper function. The wrapper
  // function takes a unique void pointer and a size, and returns a
  // unique void pointer.
  void WrapDataPtr(
    std::function<
      std::unique_ptr<void, DeleterFnPtr>(
        std::unique_ptr<void, DeleterFnPtr>, size_t
      )
    > wrapper_func, size_t nbytes
  );

  const void* GetConstDataPtr() const { return data_.get(); }
  DeleterFnPtr GetDataDeleter() const { return data_.get_deleter(); }
 private:
  // The destructor is hidden, this should only ever be used within
  // UniqueVoidPtr using cow::delete_context as the deleter.
  ~COWDeleterContext();

  std::shared_mutex mutex_;
  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 1;
};

// `cow_deleter` is used as the `ctx_deleter` for DataPtr to implement a COW
// DataPtr.
//
// Warning: This should only be called on a pointer to a COWDeleterContext that
// was allocated on the heap with `new`, because when the refcount reaches 0,
// the context is deleted with `delete`.
C10_API void cow_deleter(void* ctx);


class C10_API UnifiedMemoryDataPtrContext {
 public:
  // Provided the original_data context, initialize the mapped data pointer.
  virtual void InitializeMappedData() = 0;
  // Empty destructor
  virtual ~UnifiedMemoryDataPtrContext() {};

  // Transfer the ownership of the original data ctx
  virtual std::unique_ptr<void, DeleterFnPtr> move_original_data_ctx() = 0;
  // Get the original data ctx
  virtual void* get_original_data_ctx() const = 0;
  // Get the mapped data ctx
  virtual void* get_mapped_data_ctx() const = 0;

  // Whether the memory is allocated by the CPU
  virtual bool memory_backed_by_cpu() const = 0;
};

// Deleter for UnifiedMemoryDataPtrContext
C10_API void unified_memory_data_ptr_ctx_deleter(void *ctx);

} // namespace c10::impl::cow
