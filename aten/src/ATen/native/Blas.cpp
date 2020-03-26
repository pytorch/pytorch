#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Blas.h>
#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

DEFINE_DISPATCH(addmv_stub);

Tensor &addmv_out(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  { // scope of NoNamesGuard

  at::NoNamesGuard guard;
  result.resize_({mat.size(0)});

  Tensor self_ = self;
  if (self.dim() == 0 || self.size(0) == 1) {
    self_ = self.expand({mat.size(0)});
  }

  TORCH_CHECK((mat.dim() == 2 && vec.dim() == 1 && self_.dim() == 1),
    "vector + matrix @ vector expected, got ", self_.dim(), ", ", mat.dim(), ", ", vec.dim());
  TORCH_CHECK((mat.size(1) == vec.size(0) && mat.size(0) == self_.size(0)),
    "size mismatch, get ", self_.size(0), ", ", mat.size(0), "x", mat.size(1), ",", vec.size(0));

  if (!result.is_same(self_)) {
    at::native::copy_(result, self_);
  }

  if (result.numel() != 0) {
    auto device_type = self_.device().type();
    addmv_stub(self_.device().type(), result, self_, mat, vec, beta, alpha);
  }

  } // scope of NoNamesGuard
  at::namedinference::propagate_names_for_addmv(result, mat, vec, self);
  return result;
}

Tensor addmv(const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  Tensor result = at::empty({mat.size(0)}, mat.options());
  return native::addmv_out(result, self, mat, vec, beta, alpha);
}

Tensor &addmv_(Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  return native::addmv_out(self, self, mat, vec, beta, alpha);
}

Tensor &mv_out(Tensor& result, const Tensor &self, const Tensor &vec) {
  return native::addmv_out(result, result, self, vec, 0, 1);
}

Tensor mv(const Tensor &self, const Tensor &vec) {
  Tensor result = at::empty({self.size(0)}, self.options());
  return native::mv_out(result, self, vec);
}

}}  // namespace at::native
