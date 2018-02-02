#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"

#include <TH/TH.h>
#include <THNN/THNN.h>
#undef THNN_
#include <THS/THS.h>

#include "ATen/CPUFloatTensor.h"
#include "ATen/CPUDoubleTensor.h"
#include "ATen/SparseCPUFloatTensor.h"
#include "ATen/SparseCPUDoubleTensor.h"

namespace at {
namespace native {

// Calling into TH for sspaddmm because ATen code generation currently
// doesn't support Sparse x Dense operations on Sparse tensors
template <class scalar>
void TH_sspaddmm(Tensor & result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
  runtime_error("sspaddmm NYI for passed in types\n");
}

template <>
void TH_sspaddmm<float>(Tensor& result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
  auto result_ = checked_cast_tensor<SparseCPUFloatTensor>(result.pImpl,"result",0, false);
  auto beta_ = beta.toFloat();
  auto self_ = checked_cast_tensor<SparseCPUFloatTensor>(self.pImpl,"self",2, false);
  auto alpha_ = alpha.toFloat();
  auto mat1_ = checked_cast_tensor<SparseCPUFloatTensor>(mat1.pImpl,"mat1",4, false);
  auto mat2_ = checked_cast_tensor<CPUFloatTensor>(mat2.pImpl,"mat2",5, false);
  THSFloatTensor_sspaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
  result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
}

template <>
void TH_sspaddmm<double>(Tensor& result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
  auto result_ = checked_cast_tensor<SparseCPUDoubleTensor>(result.pImpl,"result",0, false);
  auto beta_ = beta.toDouble();
  auto self_ = checked_cast_tensor<SparseCPUDoubleTensor>(self.pImpl,"self",2, false);
  auto alpha_ = alpha.toDouble();
  auto mat1_ = checked_cast_tensor<SparseCPUDoubleTensor>(mat1.pImpl,"mat1",4, false);
  auto mat2_ = checked_cast_tensor<CPUDoubleTensor>(mat2.pImpl,"mat2",5, false);
  THSDoubleTensor_sspaddmm(result_->tensor, beta_, self_->tensor, alpha_, mat1_->tensor, mat2_->tensor);
  result_->maybeScalar(self_->isScalar() && mat1_->isScalar() && mat2_->isScalar());
}

template <typename scalar>
struct SspaddmmOp {
  static void apply(Tensor& result, Scalar beta, const Tensor& self,
      Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
    TH_sspaddmm<scalar>(result, beta, self, alpha, mat1, mat2);
  }
};

// sparse, real, sparse, real, sparse, dense -> sparse
Tensor& _sspaddmm_out_cpu(Tensor& result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
  dispatch_floating_types<void, SspaddmmOp>(self.type(), "sspaddmm",
      result, beta, self, alpha, mat1, mat2);
  return result;
}

// sparse, real, sparse, real, sparse, dense -> sparse
Tensor& sspaddmm_out(Tensor& result, const Tensor& self, const Tensor& mat1,
    const Tensor& mat2, Scalar beta, Scalar alpha) {
  return mat2.type()._sspaddmm_out(result, beta, self, alpha, mat1, mat2);
}

// real, sparse, real, sparse, dense -> sparse
Tensor sspaddmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    Scalar beta, Scalar alpha) {
  // There's no easy way to make an empty sparse tensor right now
  auto result = self.clone();
  // Dispatch on the DENSE type, because native_functions.yaml
  // doesn't support specifying sparse backend dispatch
  mat2.type().sspaddmm_out(result, self, mat1, mat2, beta, alpha);
  return result;
}

}}
