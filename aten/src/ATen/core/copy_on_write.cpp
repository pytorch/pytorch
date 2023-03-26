#include <ATen/core/copy_on_write.h>

#include <ATen/core/TensorBase.h>
#include <c10/util/Exception.h>

namespace at {

auto simulate_materialize_copy_on_write(TensorBase const& tensor) -> void {
  if (!tensor.has_storage()) {
    return;
  }

  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  TORCH_INTERNAL_ASSERT(tensor_impl != nullptr);
  tensor_impl->maybe_bump_copy_on_write_generation();
}

} // namespace at
