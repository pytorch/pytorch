#include <c10/core/impl/cow/materialize.h>

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/cow/context.h>

namespace c10::impl {

auto C10_API cow::materialize(Storage const& storage) -> void {
  StorageImpl& storage_impl = *storage.unsafeGetStorageImpl();
  at::DataPtr& data_ptr = storage_impl.data_ptr();
  if (data_ptr.get_deleter() != cow::Context::delete_instance) {
    return;
  }
  auto& ctx =
      *data_ptr.cast_context<cow::Context>(cow::Context::delete_instance);

  auto refcount = ctx.refcount();
  assert(refcount >= 1);
  if (refcount == 1) {
    // This is the only alias remaining, nothing to do here.
    // TODO Change this to downgrade to non copy-on-write storage.
    return;
  }
  assert(refcount > 1);
  // We have multiple aliases, let's materialize the storage for this
  // view family.
  DataPtr copy = storage.allocator()->allocate(storage.nbytes());
  std::memcpy(copy.get(), data_ptr.get(), storage.nbytes());
  storage.set_data_ptr_noswap(std::move(copy));
}

} // namespace c10::impl
