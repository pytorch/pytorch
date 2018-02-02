#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"

#include <THC/THC.h>
#include <THCUNN/THCUNN.h>
#undef THNN_
#undef THCIndexTensor_
#include <THCS/THCS.h>
#undef THCIndexTensor_

#include "ATen/CUDAFloatTensor.h"
#include "ATen/CUDADoubleTensor.h"
#include "ATen/SparseCUDAFloatTensor.h"
#include "ATen/SparseCUDADoubleTensor.h"

namespace at {
namespace native {

// Calling into TH for sspaddmm because ATen code generation currently
// doesn't support Sparse x Dense operations
template <class scalar>
void TH_cuda_sspaddmm(Tensor & result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
  runtime_error("sspaddmm NYI for passed in types\n");
}

template <>
void TH_cuda_sspaddmm<float>(Tensor& result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
  auto result_ = checked_cast_tensor<SparseCUDAFloatTensor>(result.pImpl,"result",0, false);
  auto beta_ = beta.toFloat();
  auto self_ = checked_cast_tensor<SparseCUDAFloatTensor>(self.pImpl,"self",2, false);
  auto alpha_ = alpha.toFloat();
  auto mat1_ = checked_cast_tensor<SparseCUDAFloatTensor>(mat1.pImpl,"mat1",4, false);
  auto mat2_ = checked_cast_tensor<CUDAFloatTensor>(mat2.pImpl,"mat2",5, false);
  THCSFloatTensor_sspaddmm(result.type().get_context().thc_state, result_->tensor,
      beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
  result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
}

template <>
void TH_cuda_sspaddmm<double>(Tensor& result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
  auto result_ = checked_cast_tensor<SparseCUDADoubleTensor>(result.pImpl,"result",0, false);
  auto beta_ = beta.toDouble();
  auto self_ = checked_cast_tensor<SparseCUDADoubleTensor>(self.pImpl,"self",2, false);
  auto alpha_ = alpha.toDouble();
  auto mat1_ = checked_cast_tensor<SparseCUDADoubleTensor>(mat1.pImpl,"mat1",4, false);
  auto mat2_ = checked_cast_tensor<CUDADoubleTensor>(mat2.pImpl,"mat2",5, false);
  THCSDoubleTensor_sspaddmm(result.type().get_context().thc_state, result_->tensor,
      beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
  result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
}

template <typename scalar>
struct SspaddmmCUDAOp {
  static void apply(Tensor& result, Scalar beta, const Tensor& self,
      Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
    TH_cuda_sspaddmm<scalar>(result, beta, self, alpha, mat1, mat2);
  }
};

Tensor& _sspaddmm_out_cuda(Tensor& result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
  dispatch_floating_types<void, SspaddmmCUDAOp>(self.type(), "sspaddmm_cuda",
      result, beta, self, alpha, mat1, mat2);
  return result;
}

}}
