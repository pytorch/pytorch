#include <c10/core/StorageImpl.h>

#include <c10/core/impl/CopyOnWriteContext.h>

namespace c10 {

c10::intrusive_ptr<StorageImpl> StorageImpl::copy_on_write() {
  std::lock_guard<std::mutex> lock(copy_on_write_mutex_.mutex_);

  // Get the CopyOnWriteContext, transforming the current storage into copy
  // on write if necessary.
  auto* cow_ctx = data_ptr_.cast_context<impl::CopyOnWriteContext>(&impl::deleteCopyOnWriteContext);

  if ((data_ptr_.get_context() != data_ptr_.get() && !cow_ctx) || received_cuda_) {
    // This is a nontrivial storage pointer, we cannot easily copy-on-write
    // it.  Return a nullptr back to caller and have them manually perform
    // the copy.  (Copy cannot be done here as copy is a per-device concept
    // and there is no virtual method for it.)
    return c10::intrusive_ptr<StorageImpl>();
  }

  if (!cow_ctx) {
    // We need to turn the original storage into copy on write storage
    auto new_ctx = impl::newCopyOnWriteContext(data_ptr_.get(), data_ptr_.get_deleter());
    cow_ctx = new_ctx.get();
    auto new_void_ctx = static_cast<std::unique_ptr<void, DeleterFnPtr>>(std::move(new_ctx));
    data_ptr_.swap_context(new_void_ctx);
    new_void_ctx.release();  // we preserved the deleter on new_ctx
  }

  // Incref the copy on write context, and share the data pointer on the new
  // storage
  cow_ctx->incref();
  return c10::make_intrusive<StorageImpl>(
    use_byte_size_t{},
    size_bytes_,
    DataPtr(data_ptr_.get(), cow_ctx, &impl::deleteCopyOnWriteContext, data_ptr_.device()),
    allocator_,
    resizable_
  );
}

} // namespace c10
