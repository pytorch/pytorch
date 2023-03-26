#include <c10/core/StorageImpl.h>

namespace c10 {

intrusive_ptr<impl::cow::ShadowStorage> StorageImpl::simulate_copy_on_write(
    impl::cow::ShadowStorage* shadow_storage) {
  return copy_on_write_state_.simulate_lazy_copy(shadow_storage);
}

void StorageImpl::maybe_bump_copy_on_write_generation(
    impl::cow::ShadowStorage* shadow_storage) {
  copy_on_write_state_.maybe_bump(shadow_storage);
}

} // namespace c10
