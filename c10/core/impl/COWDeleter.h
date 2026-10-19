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

namespace c10::impl::cow {

struct COWCompletion {
  explicit COWCompletion(DeviceType device_type) : event(device_type) {}
  Event event;
  std::unique_ptr<COWCompletion> next;
};

// Shared ownership and completion dependencies for one backing allocation.
class C10_API COWDeleterContext {
 public:
  // Creates an instance, holding the pair of data and original
  // deleter.
  //
  // Note that the deleter will only be called in our destructor if
  // the last reference to this goes away without getting
  // materialized.
  explicit COWDeleterContext(
      std::unique_ptr<void, DeleterFnPtr> data,
      std::optional<Stream> stream = std::nullopt,
      uint64_t capture_id = 0);

  bool is_unique() const {
    return refcount_ == 1;
  }

  const std::optional<Stream>& stream() const {
    return stream_;
  }

  uint64_t capture_id() const {
    return capture_id_;
  }

  void wait_for_completion(Stream stream);

  // Increments the current refcount.
  void increment_refcount();

  using LastReference = std::unique_ptr<void, DeleterFnPtr>;

  // Returns ownership on the last release, otherwise null. CUDA callers
  // must establish completion dependencies before taking ownership.
  LastReference decrement_refcount(
      std::unique_ptr<COWCompletion> completion = nullptr);

 private:
  // Destroyed by decrement_refcount() when the last reference is released.
  ~COWDeleterContext();

  std::mutex mutex_;
  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 1;
  // CUDA allocation/reuse stream; nullopt for CPU and other synchronous
  // allocators. Immutable because callers rely on it for allocation lifetime.
  std::optional<Stream> stream_;
  uint64_t capture_id_;
  std::unique_ptr<COWCompletion> completions_;
  bool needs_stream_wait_ = false;
};

// Each DataPtr has its own reference, which follows it into retired storage
// metadata on materialization. Access after record_done starts a new use.
struct C10_API COWReference {
  explicit COWReference(COWDeleterContext* context = nullptr) : ctx(context) {}
  ~COWReference();

  void record_done();
  COWDeleterContext::LastReference release();

  COWDeleterContext* ctx;
  std::atomic<bool> completed = false;
  std::unique_ptr<COWCompletion> completion;
};

// Deletes a heap-allocated COWReference.
C10_API void cow_deleter(void* ctx);

} // namespace c10::impl::cow
