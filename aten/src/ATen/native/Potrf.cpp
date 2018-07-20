#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/LinearAlgebraUtils.h"

#include "TH.h" // for USE_LAPACK

#include <vector>

#ifdef USE_LAPACK
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);
#endif

namespace at { namespace native {

template<class scalar_t>
void lapackPotrf(char uplo, int n, scalar_t * a , int lda, int* info) {
  AT_ERROR("potrf only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackPotrf<float>(char uplo , int n , float * a , int lda, int* info) {
  spotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackPotrf<double>(char uplo , int n , double * a , int lda, int* info) {
  dpotrf_(&uplo, &n, a, &lda, info);
}
#endif

template <typename scalar_t>
static void applyPotrf(Tensor& A, bool upper) {
#ifndef USE_LAPACK
  AT_ERROR("potrf: LAPACK library not found in compilation");
#endif
  auto A_data = A.data<scalar_t>();
  auto A_stride = matrixStride(A);
  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  char uplo = upper ? 'U' : 'L';

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    scalar_t* A_working_ptr = &A_data[i * A_stride];
    lapackPotrf<scalar_t>(uplo, n, A_working_ptr, n, &info);
  }
}

Tensor _potrf_helper_cpu(const Tensor& A, bool upper) {
  auto A_working_copy = cloneBatchedColumnMajor(A);
  AT_DISPATCH_FLOATING_TYPES(A.type(), "potrf", [&]{
    applyPotrf<scalar_t>(A_working_copy, upper);
  });
  return A_working_copy;
}

// Supports arbitrary batch dimensions for self and A
Tensor potrf(const Tensor& A, bool upper) {
 if (A.dim() <= 2 && A.dim() <= 2) {
   int64_t n = A.size(0);
   IntList size({1, n, n});
   auto new_tensor = A.view(size);
   return new_tensor.type()._potrf_helper(new_tensor, upper)
     .view_as(A);
 }
 // broadcast the batch dimensions of self and A.
 return A.type()._potrf_helper(A, upper);
}

Tensor& potrf_out(
    Tensor& output, const Tensor& self, const bool upper) {
  if (self.dim() > 2) {
    AT_ERROR("torch.potrf() with the `out` keyword does not support batching. "
                  "A.dim() (%lld) must be 2.",
             (long long)self.dim());
  }
  return at::_potrf_single_out(output, self, upper);
}

}}  // namespace at::native
