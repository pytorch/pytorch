#include <ATen/TensorMeta.h>
#include <ATen/ATen.h>  // bad

namespace at {

Tensor meta_tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty_meta(meta.sizes, meta.options);
}

Tensor tensor_from_meta(const TensorMeta& meta) {
  // TODO: eliminate indirection
  return at::empty(meta.sizes, meta.options);
}

TensorMeta new_meta(const Tensor& self, IntArrayRef sizes) {
  return TensorMeta(sizes, self.options());
}

}
