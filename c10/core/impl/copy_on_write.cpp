#include <c10/core/impl/copy_on_write.h>

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/UniqueVoidPtr.h>

#include <atomic>
#include <cassert>
#include <memory>

namespace c10::impl {

namespace {

// Encapsulates a c10::DataPtr context that is copy on write.
class CopyOnWriteContext {
 public:
  // Creates an instance, wrapping an existing context.
  explicit CopyOnWriteContext(std::unique_ptr<void, DeleterFnPtr> data);

  // The destructor is hidden, this should only ever be used within
  // UniqueVoidPtr using this as the deleter.
  static auto delete_instance(void* ctx) -> void;

  // Gets the current refcount.
  auto refcount() const -> std::int64_t;

  // Increments the current refcount.
  auto increment_refcount() -> void;

 private:
  ~CopyOnWriteContext();

  auto decrement_refcount() -> void;

  std::unique_ptr<void, DeleterFnPtr> data_;
  std::atomic<std::int64_t> refcount_ = 0;
};

// A std::unique_ptr deleter that is expected to never execute.
struct AssertFalseDeleter {
  template <typename T>
  auto operator()(T* /* ptr */) const -> void {
    assert(false); // expected to never be called
  }
};

// Wraps a DataPtr with a copy-on-write DataPtr.
auto make_data_ptr(at::DataPtr const& data_ptr, CopyOnWriteContext& ctx)
    -> at::DataPtr {
  ctx.increment_refcount();
  return at::DataPtr(
      data_ptr.get(),
      &ctx,
      CopyOnWriteContext::delete_instance,
      data_ptr.device());
}

} // namespace

auto C10_API make_copy_on_write(Storage const& storage)
    -> c10::intrusive_ptr<StorageImpl> {
  StorageImpl& storage_impl = *storage.unsafeGetStorageImpl();
  std::lock_guard<std::mutex> guard_ctx = storage_impl.guard_ctx();
  at::DataPtr& data_ptr = storage_impl.data_ptr();

  if (data_ptr.get_deleter() == CopyOnWriteContext::delete_instance) {
    // Already a copy-on-write storage.
    CopyOnWriteContext& ctx = *data_ptr.cast_context<CopyOnWriteContext>(
        CopyOnWriteContext::delete_instance);
    return c10::make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        storage_impl.sym_nbytes(),
        make_data_ptr(data_ptr, ctx),
        storage_impl.allocator(),
        storage_impl.resizable());
  }

  if (data_ptr.get() != data_ptr.get_context()) {
    // Non-trivial context that is *not* a CopyOnWriteContext as
    // verified above. We can't handle this case.
    return nullptr;
  }

  // We have a simple data pointer: wrap it.
  std::unique_ptr<void, DeleterFnPtr> original_ctx = data_ptr.move_context();
  assert(original_ctx.get() == data_ptr.get());

  std::unique_ptr<CopyOnWriteContext, AssertFalseDeleter> ctx(
      new CopyOnWriteContext(std::move(original_ctx)));
  storage_impl.set_data_ptr_noswap(make_data_ptr(data_ptr, *ctx));

  return c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      storage_impl.sym_nbytes(),
      make_data_ptr(data_ptr, *ctx.release()),
      storage_impl.allocator(),
      storage_impl.resizable());
}

auto copy_on_write_refcount(Storage const& storage)
    -> std::optional<std::int64_t> {
  StorageImpl const& storage_impl = *storage.unsafeGetStorageImpl();
  at::DataPtr const& data_ptr = storage_impl.data_ptr();
  if (data_ptr.get_deleter() != CopyOnWriteContext::delete_instance) {
    return std::nullopt;
  }
  CopyOnWriteContext& ctx = *data_ptr.cast_context<CopyOnWriteContext>(
      CopyOnWriteContext::delete_instance);
  return ctx.refcount();
}

namespace {

CopyOnWriteContext::CopyOnWriteContext(std::unique_ptr<void, DeleterFnPtr> data)
    : data_(std::move(data)) {
  assert(
      data_.get_deleter() !=
      delete_instance); // we never wrap a CopyOnWriteContext
}

/* static */ auto CopyOnWriteContext::delete_instance(void* ctx) -> void {
  static_cast<CopyOnWriteContext*>(ctx)->decrement_refcount();
}

auto CopyOnWriteContext::refcount() const -> std::int64_t {
  auto refcount = refcount_.load();
  assert(refcount > 0);
  return refcount;
}

auto CopyOnWriteContext::increment_refcount() -> void {
  [[maybe_unused]] auto refcount = ++refcount_;
  assert(refcount >= 1);
}

auto CopyOnWriteContext::decrement_refcount() -> void {
  auto refcount = --refcount_;
  assert(refcount >= 0);
  if (refcount == 0) {
    delete this;
  }
}

CopyOnWriteContext::~CopyOnWriteContext() {
  assert(refcount_ == 0);
}

} // namespace

} // namespace c10::impl
