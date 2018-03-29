#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/LinearAlgebraUtils.h"
#include "ATen/native/Gesv.h"

#include "TH.h"  // for USE_LAPACK

#include <vector>

#ifdef USE_LAPACK
extern "C" void dgesv_(
    int* n, int* nrhs, double* a, int* lda,
    int *ipiv, double* b, int* ldb, int* info);
extern "C" void sgesv_(
    int* n, int* nrhs, float* a, int* lda,
    int* ipiv, float* b, int* ldb, int* info);
#endif

namespace at { namespace native {

template<class real>
void lapackGesv(
    int n, int nrhs, real* a, int lda, int* ipiv,
    real* b, int ldb, int* info) {
  runtime_error("gesv only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGesv<float>(
    int n, int nrhs, float* a, int lda, int* ipiv,
    float* b, int ldb, int* info) {
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGesv<double>(
    int n, int nrhs, double* a, int lda, int* ipiv,
    double* b, int ldb, int* info) {
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}
#endif

template <typename real>
static void applyGesv(Tensor& b, Tensor& A, std::vector<int64_t> infos) {
#ifndef USE_LAPACK
  runtime_error("gesv: LAPACK library not found in compilation");
#endif
  real* A_data = (real*)A.data_ptr();
  real* b_data = (real*)b.data_ptr();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto nrhs = b.size(-1);

  auto ipiv = b.type().toScalarType(kLong).tensor(n);

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    real* A_working_ptr = &A_data[i * A_mat_stride];
    real* b_working_ptr = &b_data[i * b_mat_stride];
    lapackGesv<real>(n, nrhs, A_working_ptr, n, (int*)ipiv.data_ptr(),
        b_working_ptr, n, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

std::tuple<Tensor,Tensor> _gesv_helper_cpu(const Tensor& self, const Tensor& A) {
  std::vector<int64_t> infos(batchCount(A), 0);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  auto b_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    applyGesv<scalar_t>(b_working_copy, A_working_copy, infos);
  });
  checkErrors(infos);
  return std::tuple<Tensor,Tensor>(b_working_copy, A_working_copy);
}

// Supports arbitrary batch dimensions for self and A
std::tuple<Tensor,Tensor> gesv(const Tensor& self, const Tensor& A) {
  if (self.dim() <= 2 && A.dim() <= 2) {
    return at::_gesv(self, A);
  }

  checkInputs(self, A);

  // broadcast the batch dimensions of self and A.
  IntList self_batch_sizes(self.sizes().data(), self.ndimension() - 2);
  IntList A_batch_sizes(A.sizes().data(), A.ndimension() - 2);
  std::vector<int64_t> expand_batch_portion =
      infer_size(self_batch_sizes, A_batch_sizes);

  std::vector<int64_t> self_expand_size({expand_batch_portion});
  self_expand_size.insert(self_expand_size.end(),
      { self.size(-2), self.size(-1) });

  std::vector<int64_t> A_expand_size({expand_batch_portion});
  A_expand_size.insert(A_expand_size.end(),
      { A.size(-2), A.size(-1) });

  Tensor self_broadcasted  = self.expand(self_expand_size);
  Tensor A_broadcasted = A.expand(A_expand_size);
  return self.type()._gesv_helper(self_broadcasted, A_broadcasted);
}

std::tuple<Tensor&,Tensor&> gesv_out(
    Tensor& solution, Tensor& lu, const Tensor& self, const Tensor& A) {
  if (self.dim() > 2 || A.dim() > 2) {
    runtime_error("torch.gesv() with the `out` keyword does not support batching. "
                  "b.dim() (%lld) and A.dim() (%lld) must both be 2.",
                  (long long)self.dim(), (long long)A.dim());
  }
  return at::_gesv_out(solution, lu, self, A);
}

}}  // namespace at::native
