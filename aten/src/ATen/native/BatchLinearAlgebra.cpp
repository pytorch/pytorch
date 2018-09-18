#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/native/LinearAlgebraUtils.h"

#include "TH.h"  // for USE_LAPACK

#include <vector>

// First the required LAPACK implementations are registered here.
// A comment above the registered LAPACK routine suggest which batched
// linear algebra function uses that routine
#ifdef USE_LAPACK

// gesv
extern "C" void dgesv_(int* n, int* nrhs, double* a, int* lda, int *ipiv, double* b, int* ldb, int* info);
extern "C" void sgesv_(int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);

// inverse
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
extern "C" void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
extern "C" void sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);

// potrf
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);

#endif

namespace at {
namespace native {

// Define the per-batch functions to be used in the main implementation of the batched
// linear algebra operations
template<class scalar_t>
void lapackGesv(int n, int nrhs, scalar_t* a, int lda, int* ipiv, scalar_t* b, int ldb, int* info) {
  AT_ERROR("gesv only takes float or double Tensors");
}

template<class scalar_t>
void lapackGetrf(int m, int n, scalar_t* a, int lda, int *ipiv, int *info) {
  AT_ERROR("getrf only takes float or double Tensors");
}

template<class scalar_t>
void lapackGetri(int n, scalar_t *a, int lda, int *ipiv, scalar_t *work, int lwork, int *info) {
  AT_ERROR("getri only takes float or double Tensors");
}

template<class scalar_t>
void lapackPotrf(char *uplo, int *n, scalar_t *a, int *lda, int *info) {
  AT_ERROR("potrf only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGesv<double>(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb, int* info) {
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGesv<float>(int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb, int* info) {
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGetri<double>(int n, double *a, int lda, int *ipiv, double *work, int lwork, int *info) {
  dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template<> void lapackGetri<float>(int n, float *a, int lda, int *ipiv, float *work, int lwork, int *info) {
  sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template<> void lapackGetrf<double>(int m, int n, double *a, int lda, int *ipiv, int *info) {
  dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackGetrf<float>(int m, int n, float *a, int lda, int *ipiv, int *info) {
  sgetrf_(&m, &n, a, &lda, ipiv, info);
}

template<> void lapackPotrf<double>(char *uplo, int *n, double *a, int *lda, int *info) {
  dpotrf_(uplo, n, a, lda, info);
}

template<> void lapackPotrf<float>(char *uplo, int *n, float *a, int *lda, int *info) {
  spotrf_(uplo, n, a, lda, info);
}

#endif

// Below of the definitions of the functions operating on a batch that are going to be dispatched
// in the main helper functions for the linear algebra operations

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ gesv ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_gesv(Tensor& b, Tensor& A, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("gesv: LAPACK library not found in compilation");
#endif
  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto b_mat_stride = matrixStride(b);

  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto nrhs = b.size(-1);

  auto ipiv = at::empty({n}, b.type().toScalarType(kInt));

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
    lapackGesv<scalar_t>(n, nrhs, A_working_ptr, n, ipiv.data<int>(), b_working_ptr, n, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

// These utilities are specified in LinearAlgebraUtils.h
LINALG_HELPER_2_ARGS(gesv, self, A, cpu)

// Supports arbitrary batch dimensions for self and A
std::tuple<Tensor,Tensor> gesv(const Tensor& self, const Tensor& A) {
  if (self.dim() <= 2 && A.dim() <= 2) {
    // TODO: #7102: It's not necessary to have gesv (single) bindings for both
    // TH and ATen. We should remove the TH gesv bindings, especially
    // since the lapackGesv function is already in ATen.
    return at::_gesv_single(self, A);
  }

  gesvCheckInputs(self, A);

  // broadcast the batch dimensions of self and A.
  IntList self_batch_sizes(self.sizes().data(), self.ndimension() - 2);
  IntList A_batch_sizes(A.sizes().data(), A.ndimension() - 2);
  std::vector<int64_t> expand_batch_portion = infer_size(self_batch_sizes, A_batch_sizes);

  std::vector<int64_t> self_expand_size({expand_batch_portion});
  self_expand_size.insert(self_expand_size.end(), { self.size(-2), self.size(-1) });

  std::vector<int64_t> A_expand_size({expand_batch_portion});
  A_expand_size.insert(A_expand_size.end(), { A.size(-2), A.size(-1) });

  Tensor self_broadcasted  = self.expand(self_expand_size);
  Tensor A_broadcasted = A.expand(A_expand_size);
  return at::_gesv_helper(self_broadcasted, A_broadcasted);
}

std::tuple<Tensor&,Tensor&> gesv_out(Tensor& solution, Tensor& lu, const Tensor& self, const Tensor& A) {
  AT_CHECK(self.dim() == 2 && A.dim() == 2, 
           "torch.gesv() with the `out` keyword does not support batching. "
           "b.dim() (", self.dim(), ") and A.dim() (", A.dim(), ") must both be 2.");
  return at::_gesv_single_out(solution, lu, self, A);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_inverse(Tensor& self, std::vector<int64_t>& infos) {
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
    lapackGetrf<scalar_t>(n, n, self_working_ptr, n, ipiv.data<int>(), &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }

    // Run twice, first to get the optimum work size
    lwork = -1;
    lapackGetri<scalar_t>(n, self_working_ptr, n, ipiv.data<int>(), &wkopt, lwork, &info);

    lwork = static_cast<int>(wkopt);
    work = at::empty({lwork}, self.type());

    // now to compute the actual inverse
    lapackGetri<scalar_t>(n, self_working_ptr, n, ipiv.data<int>(), work.data<scalar_t>(), lwork, &info);
    infos[i] = info;
    if (info != 0) {
      return;
    }
  }
}

LINALG_HELPER_1_ARGS(inverse, self, cpu)

Tensor inverse(const Tensor &self) {
  if (self.size(-1) == 0) {
    return at::empty_like(self);
  }
  if (self.dim() == 2) {
    return at::_getri_single(self);
  }
  inverseCheckInputs(self);
  return at::_inverse_helper(self);
}

Tensor& inverse_out(Tensor &result, const Tensor &self) {
  if (self.size(-1) == 0) {
    return result.resize_as_(self);
  }
  result.copy_(native::inverse(self));
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ potrf ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_potrf(Tensor& A, bool upper) {
#ifndef USE_LAPACK
  AT_ERROR("not compiled with LAPACK");
#endif

  char uplo = upper ? 'U' : 'L';

  auto A_data = A.data<scalar_t>();
  auto A_mat_stride = matrixStride(A);

  auto batch_size = batchCount(A);
  int n = A.size(-2);
  AT_CHECK(A.size(-1) == n, "last two dimensions must be of equal size");
  //THArgCheck(THTensor_nDimensionLegacyAll(a) == 2, 1, "A should be 2 dimensional");
  int lda = n;

  for (int64_t i = 0; i < batch_size; i++) {
    int info;
    lapackPotrf(&uplo, &n, A_data+i*A_mat_stride, &lda, &info);
    AT_CHECK(info == 0, "The leading minor of order ", info, " is not positive definite");
  }
}

Tensor potrf_cpu(const Tensor &self, bool upper) {
  if (self.dim() == 0) {
    return self.sqrt();
  } else if (self.size(-1) == 0) {
    return at::empty_like(self);
  }
  AT_CHECK(self.dim() >= 2, "tensor must be at least two-dimensional");
  Tensor result = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(result.type(), "potrf", [&] {
      apply_potrf<scalar_t>(result, upper);
    });
  if (upper) {
    result.triu_();
  } else {
    result.tril_();
  }
  return result;
}

Tensor& potrf_out(Tensor& result, const Tensor &self, bool upper) {
  // should check if out is of the right format and copy before...
  result.resize_as_(self).copy_(at::potrf(self));
  return result;
}

template <typename scalar_t, bool inplace, bool upper>
void apply_triu_tril(Tensor& result, const Tensor& self, int64_t k) {
  auto n = self.size(-2);
  auto m = self.size(-1);
  auto self_batched_ = self.view({-1, n, m});
  auto self_batched = self_batched_.accessor<scalar_t, 3>();
  auto result_batched_ = result.view({-1, n, m});
  auto result_batched = result_batched_.accessor<scalar_t, 3>();
  auto batch_size = self_batched.size(0);
  AT_CHECK(result_batched.size(0) == batch_size, "matrix sizes don't match");
  constexpr int64_t zero = 0;
  for (int64_t b = 0; b < batch_size; b++) {
    auto self1 = self_batched[b];
    auto result1 = result_batched[b];
    for (int64_t i = 0; i < n; i++) {
      auto result_row = result1[i];
      if (upper) { // triu
	  int64_t sz = std::min(m, i+k);
	  for (int64_t j = 0; j < sz; j++) {
	  result_row[j] = 0;
	}
	if (! inplace) {
	  auto self_row = self1[i];
	  for (int64_t j = std::max(zero, i+k); j < m; j++) {
	    result_row[j] = self_row[j];
	  }
	}
      } else {     // tril
	for (int64_t j = std::max(zero, i+k+1); j < m; j++) {
	  result_row[j] = 0;
	}
	if (! inplace) {
	  auto self_row = self1[i];
	  int64_t sz = std::min(m, i+k+1);
	  for (int64_t j = 0; j < sz; j++) {
	    result_row[j] = self_row[j];
	  }
	}
      }
    }
  }
}

Tensor tril(const Tensor &self, int64_t k) {
  auto result = at::empty_like(self);
  at::tril_out(result, self, k);
  return result;
}

Tensor& tril_cpu_(Tensor &self, int64_t k) {
  AT_DISPATCH_ALL_TYPES(self.type(), "tril", [&] {
      apply_triu_tril<scalar_t, true, false>(self, self, k);
    });
  return self;
}

Tensor& tril_cpu_out(Tensor &result, const Tensor& self, int64_t k) {
  result.resize_as_(self);
  AT_DISPATCH_ALL_TYPES(self.type(), "tril", [&] {
      apply_triu_tril<scalar_t, false, false>(result, self, k);
    });
  return result;
}

Tensor triu(const Tensor &self, int64_t k) {
  auto result = at::empty_like(self);
  at::triu_out(result, self, k);
  return result;
}

Tensor& triu_cpu_(Tensor &self, int64_t k) {
  AT_DISPATCH_ALL_TYPES(self.type(), "triu", [&] {
      apply_triu_tril<scalar_t, true, true>(self, self, k);
    });
  return self;
}

Tensor& triu_cpu_out(Tensor &result, const Tensor& self, int64_t k) {
  result.resize_as_(self);
  AT_DISPATCH_ALL_TYPES(self.type(), "triu", [&] {
      apply_triu_tril<scalar_t, false, true>(result, self, k);
    });
  return result;
}

}}  // namespace at::native
