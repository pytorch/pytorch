#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"

#include <TH/TH.h>
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
template <class scalar_t>
void sspaddmm_TH_dispatch(Tensor & result, Scalar beta, const Tensor& self,
    Scalar alpha, const Tensor& mat1, const Tensor& mat2) {
 AT_ERROR("sspaddmm NYI for types %s %s %s",
      self.type().toString(), mat1.type().toString(), mat2.type().toString());
}

template <>
void sspaddmm_TH_dispatch<float>(Tensor& result, Scalar beta, const Tensor& self,
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
void sspaddmm_TH_dispatch<double>(Tensor& result, Scalar beta, const Tensor& self,
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

// sparse, sparse, sparse, dense, real, real -> sparse
Tensor& _sspaddmm_out_cpu(
    Tensor& result,
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "sspaddmm", [&]{
    sspaddmm_TH_dispatch<scalar_t>(result, beta, self, alpha, mat1, mat2);
  });
  return result;
}

// sparse, sparse, sparse, dense, real, real -> sparse
Tensor& _sspaddmm_out_only_sparse(Tensor& result, const Tensor& self,
    const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
 AT_ERROR("tensor.sspaddmm(...) can only be called on sparse tensors");
  return result;
}

// sparse, dense -> sparse
Tensor smm(const Tensor& self, const Tensor& mat2) {
  auto result = self.type().tensor();
  self.type().sspaddmm_out(result, result, self, mat2, 0.0, 1.0);
  return result;
}

// sparse, sparse, dense, real, real -> sparse
Tensor sspaddmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    Scalar beta, Scalar alpha) {
  auto result = self.type().tensor();
  self.type().sspaddmm_out(result, self, mat1, mat2, beta, alpha);
  return result;
}

}} // namespace at::native
