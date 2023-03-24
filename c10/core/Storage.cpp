#include <c10/core/Storage.h>

#include <c10/util/Exception.h>

namespace c10 {

intrusive_ptr<impl::cow::Simulator> Storage::simulate_copy_on_write(
    impl::cow::Simulator* simulator) const {
  TORCH_INTERNAL_ASSERT(storage_impl_ != nullptr);
  return storage_impl_.get()->simulate_copy_on_write(simulator);
}

void Storage::maybe_bump_copy_on_write_generation(
    impl::cow::Simulator* simulator) {
  TORCH_INTERNAL_ASSERT(storage_impl_ != nullptr);
  storage_impl_->maybe_bump_copy_on_write_generation(simulator);
}

} // namespace c10
