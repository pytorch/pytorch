#include <ATen/core/copy_on_write.h>

#include <ATen/core/TensorBase.h>
#include <c10/core/impl/copy_on_write.h>
#include <c10/util/Exception.h>

#include <cassert>
#include <optional>

namespace at {

auto materialize_copy_on_write(TensorBase const& tensor) -> void {
  std::optional<std::int64_t> refcount = c10::impl::copy_on_write_refcount(tensor.storage());
  if (!refcount.has_value()) {
    // This is not a copy-on-write tensor, nothing to do here.
    return;
  }
  assert(*refcount >= 1);
  if (*refcount > 1) {
    TORCH_WARN_ONCE(
        "You have written through a view created by calling reshape(). In the "
        "future, reshape() will never create a view but will instead return a "
        "lazily copied tensor. If you wish to preserve the aliasing "
        "properties, you should rewrite your reshape() as a view().");
  }
}

} // namespace at
