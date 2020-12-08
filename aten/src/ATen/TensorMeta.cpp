#include <ATen/TensorMeta.h>
#include <ATen/ATen.h>

namespace at {

Tensor meta_tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty_meta(meta.sizes, meta.options);
}

Tensor tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty(meta.sizes, meta.options);
}

// Analogous to self.new_empty(sizes)
TensorMeta new_meta(const Tensor& self, IntArrayRef sizes) {
  return TensorMeta(sizes, self.options());
}

} // namespace at
