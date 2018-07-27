#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/LinearAlgebraUtils.h"
#include "Getri.h"

#include "TH.h" // for USE_LAPACK

#include <vector>

#ifdef USE_LAPACK
extern "C" void dgetri_(
    int *n, double *a, int *lda,
    int *ipiv, double *work, int *lwork,
    int *info);
extern "C" void sgetri_(
    int *n, float *a, int *lda,
    int *ipiv, double *work, int *lwork,
    int *info);
#endif

namespace at {
namespace native {

template<class scalar_t>
void lapackGetri(
    int n, scalar_t* a, int lda,
    int *ipiv, double *work, int lwork,
    int *info) {
  AT_ERROR("getri only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGetri<float>(
    int n, float *a, int lda,
    int *ipiv, double *work, int lwork,
    int* info) {
  sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template<> void lapackGetri<double>(
    int n, double *a, int lda,
    int *ipiv, double *work, int lwork,
    int* info) {
  dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}
#endif

template <typename scalar_t>
static void applyGetri(Tensor& self, std::vector<int64_t> infos) {
#ifndef USE_LAPACK
  AT_ERROR("getri: LAPACK library not found in compilation");
#endif
  auto self_data = self.data<scalar_t>();
  auto self_matrix_stride = matrixStride(self);

  auto batch_size = batchCount(self);
  auto n = self.size(-2);

  auto ipiv = at::empty({n}, self.type().toScalarType(kInt));
  auto work = at::empty({1}, self.type().toScalarType(kDouble));
  int lwork = -1;  // compute the optimal work size

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    lapackGetri<scalar_t>(n, self_working_ptr, n, ipiv.data<int>(),
                          work.data<double>(), lwork, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

Tensor _inverse_helper_cpu(const Tensor& self) {
  std::vector<int64_t> infos(batchCount(self), 0);
  auto self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "inverse", [&]{
    applyGetri<scalar_t>(self_working_copy, infos);
  });
  checkErrors(infos);
  return self_working_copy;
}

Tensor inverse(const Tensor &self) {
  if (self.dim() == 2) {
    return at::_getri_single(self);
  }
  return self.type()._inverse_helper(self);
}

Tensor& inverse_out(Tensor &result, const Tensor &self) {
  return at::_getri_single_out(result, self);
}

} // namespace native
} // namespace at
