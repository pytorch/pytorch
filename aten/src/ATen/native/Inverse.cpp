#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/LinearAlgebraUtils.h"
#include "Inverse.h"

#include "TH.h" // for USE_LAPACK

#include <vector>

#ifdef USE_LAPACK
extern "C" void dgetri_(
    int *n, double *a, int *lda,
    int *ipiv, double *work, int *lwork,
    int *info);
extern "C" void sgetri_(
    int *n, float *a, int *lda,
    int *ipiv, float *work, int *lwork,
    int *info);
extern "C" void dgetrf_(
    int *m, int *n, double *a,
    int *lda, int *ipiv,
    int *info);
extern "C" void sgetrf_(
    int *m, int *n, float *a,
    int *lda, int *ipiv,
    int *info);
#endif

namespace at {
namespace native {

template<class scalar_t>
void lapackGetri(
    int n, scalar_t *a, int lda,
    int *ipiv, scalar_t *work, int lwork,
    int *info) {
  AT_ERROR("getri only takes float or double Tensors");
}

template<class scalar_t>
void lapackGetrf(
    int m, int n, scalar_t* a,
    int lda, int *ipiv, int *info) {
  AT_ERROR("getrf only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGetri<double>(
    int n, double *a, int lda,
    int *ipiv, double *work, int lwork,
    int *info) {
  dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template<> void lapackGetri<float>(
    int n, float *a, int lda,
    int *ipiv, float *work, int lwork,
    int *info) {
  sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template<> void lapackGetrf<double>(
    int m, int n, double *a,
    int lda, int *ipiv, int *info) {
  dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackGetrf<float>(
    int m, int n, float *a,
    int lda, int *ipiv, int *info) {
  sgetrf_(&m, &n, a, &lda, ipiv, info);
}
#endif

template <typename scalar_t>
static void applyInverse(
  Tensor& self, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("inverse: LAPACK library not found in compilation");
#endif
  auto self_data = self.data<scalar_t>();
  auto self_matrix_stride = matrixStride(self);

  auto batch_size = batchCount(self);
  auto n = self.size(-2);

  auto ipiv = at::empty({n}, self.type().toScalarType(kInt));
  int lwork;
  scalar_t wkopt;
  Tensor work;

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    lapackGetrf<scalar_t>(n, n, self_working_ptr, n, ipiv.data<int>(),
                          &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }

    // Run twice, first to get the optimum work size
    lwork = -1;
    lapackGetri<scalar_t>(n, self_working_ptr, n, ipiv.data<int>(),
                          &wkopt, lwork, &info);

    lwork = static_cast<int>(wkopt);
    work = at::empty({lwork}, self.type());

    // now to compute the actual inverse
    lapackGetri<scalar_t>(n, self_working_ptr, n, ipiv.data<int>(),
                          work.data<scalar_t>(), lwork, &info);
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
    applyInverse<scalar_t>(self_working_copy, infos);
  });
  
  checkErrors(infos, "inverse");
  return self_working_copy;
}

Tensor inverse(const Tensor &self) {
  if (self.size(-1) == 0) {
    return at::empty_like(self);
  }
  if (self.dim() == 2) {
    return at::_getri_single(self);
  }
  checkInputs(self);
  return self.type()._inverse_helper(self);
}

Tensor& inverse_out(Tensor &result, const Tensor &self) {
  if (self.size(-1) == 0) {
    return result.resize_as_(self);
  }
  result.copy_(native::inverse(self));
  return result;
}

} // namespace native
} // namespace at
