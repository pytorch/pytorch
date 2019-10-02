#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Nonzero.h>

namespace at { namespace native {

DEFINE_DISPATCH(nonzero_stub);

Tensor nonzero(const Tensor& self) {
  auto indices = at::empty({}, self.options().dtype(kLong));
  native::nonzero_out(indices, self);
  return indices;
}

Tensor& nonzero_out(Tensor& indices, const Tensor& self) {
  TORCH_CHECK(indices.scalar_type() == ScalarType::Long,
      "nonzero expects Long tensor out, got: ", indices.scalar_type());
  nonzero_stub(self.type().device_type(), indices, self);
  return indices;
}

}} // namespace at::native