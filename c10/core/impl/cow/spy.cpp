#include <c10/core/impl/cow/spy.h>

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace c10::impl {

/* static */ auto cow::Spy::get_generation(Storage const& storage)
    -> std::uint64_t {
  TORCH_INTERNAL_ASSERT(storage);
  return storage.unsafeGetStorageImpl()
      ->copy_on_write_state_.physical_generation();
}

/* static */ auto cow::Spy::get_shadow_storage(TensorImpl const& tensor)
    -> ShadowStorage const* {
  return tensor.shadow_storage();
}

} // namespace c10::impl
