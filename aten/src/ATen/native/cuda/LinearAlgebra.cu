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

} }
