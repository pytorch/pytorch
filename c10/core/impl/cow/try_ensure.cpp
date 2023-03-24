#include <c10/core/impl/cow/try_ensure.h>

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/cow/context.h>
#include <c10/util/Exception.h>
#include <c10/util/UniqueVoidPtr.h>

#include <memory>

namespace c10::impl {

namespace {

// A std::unique_ptr deleter that is expected to never execute.
struct AssertFalseDeleter {
  template <typename T>
  auto operator()(T* /* ptr */) const -> void {
    TORCH_INTERNAL_ASSERT(false); // expected to never be called
  }
};

// Wraps a DataPtr with a copy-on-write DataPtr.
auto make_data_ptr(at::DataPtr const& data_ptr, cow::Context& ctx)
    -> at::DataPtr {
  ctx.increment_refcount();
  return at::DataPtr(
      data_ptr.get(), &ctx, cow::Context::delete_instance, data_ptr.device());
}

} // namespace

auto C10_API cow::try_ensure(Storage const& storage)
    -> c10::intrusive_ptr<StorageImpl> {
  StorageImpl& storage_impl = *storage.unsafeGetStorageImpl();
  std::lock_guard<std::mutex> guard_ctx = storage_impl.guard_ctx();
  at::DataPtr& data_ptr = storage_impl.data_ptr();

  if (data_ptr.get_deleter() == cow::Context::delete_instance) {
    // Already a copy-on-write storage.
    auto& ctx =
        *data_ptr.cast_context<cow::Context>(cow::Context::delete_instance);
    return make_intrusive<StorageImpl>(
        StorageImpl::use_byte_size_t(),
        storage_impl.sym_nbytes(),
        make_data_ptr(data_ptr, ctx),
        storage_impl.allocator(),
        storage_impl.resizable());
  }

  if (data_ptr.get() != data_ptr.get_context()) {
    // Non-trivial context that is *not* a cow::Context as
    // verified above. We can't handle this case.
    return nullptr;
  }

  // We have a simple data pointer: wrap it.
  std::unique_ptr<void, DeleterFnPtr> original_ctx = data_ptr.move_context();
  TORCH_INTERNAL_ASSERT(original_ctx.get() == data_ptr.get());

  std::unique_ptr<cow::Context, AssertFalseDeleter> ctx(
      new cow::Context(std::move(original_ctx)));
  storage_impl.set_data_ptr_noswap(make_data_ptr(data_ptr, *ctx));

  return c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      storage_impl.sym_nbytes(),
      make_data_ptr(data_ptr, *ctx.release()),
      storage_impl.allocator(),
      storage_impl.resizable());
}

} // namespace c10::impl
