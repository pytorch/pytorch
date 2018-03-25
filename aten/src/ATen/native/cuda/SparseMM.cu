#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

Tensor& _sspaddmm_out_cuda(Tensor& result, const Tensor& self,
    const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  runtime_error("NYI: CUDA sspaddmm is not implemented");
  return result;
}

}}
