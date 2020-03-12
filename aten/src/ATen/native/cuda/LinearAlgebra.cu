#include <ATen/ATen.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

namespace at { namespace native {

Tensor baddbmm_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::cuda::_th_baddbmm(self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm_out_cuda(Tensor &result, const Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::cuda::_th_baddbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor& baddbmm__cuda(Tensor& self, const Tensor& batch1, const Tensor& batch2, Scalar beta, Scalar alpha) {
  return legacy::cuda::_th_baddbmm_out(self, self, batch1, batch2, beta, alpha);
}

Tensor bmm_cuda(const Tensor& self, const Tensor& mat2) {
  return legacy::cuda::_th_bmm(self, mat2);
}

Tensor& bmm_out_cuda(Tensor &result, const Tensor& batch1, const Tensor& batch2) {
  return legacy::cuda::_th_bmm_out(result, batch1, batch2);
}

Tensor& addmm_out_impl(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  return legacy::cuda::_th_addmm_out(result, self, mat1, mat2, beta, alpha);
}

Tensor& mm_out_cuda(Tensor& result, const Tensor& self, const Tensor& mat2) {
  result.resize_({ self.size(0), mat2.size(1) });
  return addmm_out_impl(result, result, self, mat2, 0, 1);
}

Tensor mm_cuda(const Tensor& self, const Tensor& mat2) {
  Tensor result = at::empty({ self.size(0), mat2.size(1) }, self.options());
  return addmm_out_impl(result, result, self, mat2, 0, 1);
}

} }
