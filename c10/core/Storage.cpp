#include <c10/core/Storage.h>

#include <c10/util/Exception.h>

namespace c10 {

intrusive_ptr<impl::cow::ShadowStorage> Storage::simulate_copy_on_write(
    impl::cow::ShadowStorage* shadow_storage) const {
  TORCH_INTERNAL_ASSERT(storage_impl_ != nullptr);
  return storage_impl_.get()->simulate_copy_on_write(shadow_storage);
}

void Storage::maybe_bump_copy_on_write_generation(
    impl::cow::ShadowStorage* shadow_storage) {
  TORCH_INTERNAL_ASSERT(storage_impl_ != nullptr);
  storage_impl_->maybe_bump_copy_on_write_generation(shadow_storage);
}

} // namespace c10
