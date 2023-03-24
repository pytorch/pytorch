#include <ATen/core/copy_on_write.h>

#include <ATen/core/TensorBase.h>
#include <c10/core/impl/cow/materialize.h>

namespace at {

auto materialize_copy_on_write(TensorBase const& tensor) -> void {
  if (!tensor.has_storage()) {
    return;
  }

  c10::impl::cow::materialize(tensor.storage());
}

} // namespace at
