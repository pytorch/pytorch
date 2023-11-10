#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/cow/COW.h>

namespace at::native {

Tensor _lazy_clone(Tensor const& self) {
  c10::StorageImpl* self_storage = self.storage().unsafeGetStorageImpl();
  c10::intrusive_ptr<c10::StorageImpl> storage =
    c10::impl::cow::lazy_clone_storage(*self_storage);
  TORCH_CHECK(storage != nullptr);
  auto tensor = c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(std::move(storage)),
      self.key_set(),
      self.dtype());
  tensor->set_sizes_and_strides(self.sym_sizes(),
                                self.sym_strides(),
                                self.sym_storage_offset());
  return Tensor(std::move(tensor));
}

} // namespace at::native
