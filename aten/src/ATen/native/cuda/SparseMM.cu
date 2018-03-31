#include "ATen/ATen.h"
#include "ATen/Error.h"
#include "ATen/NativeFunctions.h"

namespace at { namespace native {
Tensor& _sspaddmm_out_cuda(Tensor& result, const Tensor& self,
    const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  AT_ERROR("NYI: CUDA sspaddmm is not implemented");
}
}} // namespace at::native
