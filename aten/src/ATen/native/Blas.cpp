#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

// Related code are removed and pasted here:

// [[
//   name: _th_addmv
//   cname: addmv
//   cpu_bfloat16: True
//   cuda_bfloat16: True
//   variants: function
//   return: argument 0
//   scalar_check: false
//   arguments:
//     - arg: THTensor* result
//       output: True
//     - arg: THTensor* self
//       broadcast: mat,vec dims:mat.dim0
//     - THTensor* mat
//     - THTensor* vec
//     - real beta
//     - real alpha
// ]]
// [[
//   name: _th_addmv_
//   cname: addmv
//   cpu_bfloat16: True
//   cuda_bfloat16: True
//   variants: function
//   return: self
//   arguments:
//     - THTensor* self
//     - THTensor* self
//     - THTensor* mat
//     - THTensor* vec
//     - real beta
//     - real alpha
// ]]
// [[
//   name: _th_mv
//   cpu_bfloat16: True
//   cuda_bfloat16: True
//   cname: addmv
//   variants: function
//   return: argument 0
//   scalar_check: false
//   arguments:
//     - arg: THTensor* result
//       output: True
//       resize: [ [self, 0] ]
//       cpu_zero: True
//     - argument 0
//     - THTensor* self
//     - THTensor* vec
//     - CONSTANT AS_REAL(0)
//     - CONSTANT AS_REAL(1)
// ]]

namespace at { namespace native {

DEFINE_DISPATCH(addmv_stub);

Tensor addmv_out(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {

#ifdef BUILD_NAMEDTENSOR
    at::NoNamesGuard guard;
#endif

  TORCH_CHECK((mat.dim() == 2 && vec.dim() == 1 && self.dim() == 1),
    "vector + matrix @ vector expected, got ", self.dim(), ", ", mat.dim(), ", ", vec.dim());
  TORCH_CHECK((mat.size(1) == vec.size(0) && mat.size(0) == self.size(0)),
    "size mismatch, get ", self.size(0), ", ", mat.size(0), "x", mat.size(1), ",", vec.size(0));

  addmv_stub(result, self, mat, vec, beta, alpha);

#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names_for_addmv(r_, mat, vec, t);
#endif
  return result;
}

Tensor addmv(const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  Tensor result;
  return native::addmv_out(result, self, mat, vec, beta, alpha);
}

Tensor addmv_(Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
  return native::addmv_out(self, self, mat, vec, beta, alpha);
}

Tensor mv_out(Tensor& result, const Tensor &self, const Tensor &vec) {
  return native::addmv_out(result, result, self, vec, 0, 1);
}

Tensor mv(const Tensor &self, const Tensor &vec) {
  Tensor result;
  return native::mv_out(result, self, vec, 0, 1);
}

}}  // namespace at::native