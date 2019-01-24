#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctions.h>

#include <ATen/native/LinearAlgebraUtils.h>

#include <TH/TH.h>  // for USE_LAPACK

#include <vector>

// First the required LAPACK implementations are registered here.
// A comment above the registered LAPACK routine suggest which batched
// linear algebra function uses that routine
#ifdef USE_LAPACK

// gesv
extern "C" void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);
extern "C" void sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info);

// inverse
extern "C" void dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv, int *info);
extern "C" void sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info);
extern "C" void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int *lwork, int *info);
extern "C" void sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info);

// potrs
extern "C" void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, int *info);
extern "C" void spotrs_(char *uplo, int *n, int *nrhs, float *a, int *lda, float *b, int *ldb, int *info);

// potrf
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void spotrf_(char *uplo, int *n, float *a, int *lda, int *info);
#endif

namespace at {
namespace native {

// Define the per-batch functions to be used in the main implementation of the batched
// linear algebra operations
template<class scalar_t>
void lapackGesv(int n, int nrhs, scalar_t *a, int lda, int *ipiv, scalar_t *b, int ldb, int *info) {
  AT_ERROR("gesv only takes float or double Tensors");
}

template<class scalar_t>
void lapackGetrf(int m, int n, scalar_t *a, int lda, int *ipiv, int *info) {
  AT_ERROR("getrf only takes float or double Tensors");
}

template<class scalar_t>
void lapackGetri(int n, scalar_t *a, int lda, int *ipiv, scalar_t *work, int lwork, int *info) {
  AT_ERROR("getri only takes float or double Tensors");
}

template<class scalar_t>
void lapackCholeskySolve(char uplo, int n, int nrhs, scalar_t *a, int lda, scalar_t *b, int ldb, int *info) {
  AT_ERROR("cholesky_solve only takes float or double Tensors");
}

template<class scalar_t>
void lapackCholesky(char uplo, int n, scalar_t *a, int lda, int *info) {
  AT_ERROR("cholesky only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGesv<double>(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, int *info) {
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGesv<float>(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, int *info) {
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

template<> void lapackCholeskySolve<double>(char uplo, int n, int nrhs, double *a, int lda, double *b, int ldb, int *info) {
  dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholeskySolve<float>(char uplo, int n, int nrhs, float *a, int lda, float *b, int ldb, int *info) {
  spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
}

template<> void lapackCholesky<double>(char uplo, int n, double *a, int lda, int *info) {
  dpotrf_(&uplo, &n, a, &lda, info);
}

template<> void lapackCholesky<float>(char uplo, int n, float *a, int lda, int *info) {
  spotrf_(&uplo, &n, a, &lda, info);
}
#endif

// Below of the definitions of the functions operating on a batch that are going to be dispatched
// in the main helper functions for the linear algebra operations

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ gesv ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_gesv(Tensor& b, Tensor& A, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("gesv: LAPACK library not found in compilation");
#else
  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto n = A.size(-2);
  auto nrhs = b.size(-1);

  auto ipiv = at::empty({n}, b.type().toScalarType(kInt));

  int info;
  if (b.dim() == 2) {
    lapackGesv<scalar_t>(n, nrhs, A_data, n, ipiv.data<int>(), b_data, n, &info);
    infos[0] = info;
  } else {
    auto A_mat_stride = matrixStride(A);
    auto b_mat_stride = matrixStride(b);
    auto batch_size = batchCount(A);

    for (int64_t i = 0; i < batch_size; i++) {
      scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
      scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
      lapackGesv<scalar_t>(n, nrhs, A_working_ptr, n, ipiv.data<int>(), b_working_ptr, n, &info);
      infos[i] = info;
      if (info != 0) {
        return;
      }
    }
  }
#endif
}

std::tuple<Tensor, Tensor> _gesv_helper_cpu(const Tensor& self, const Tensor& A) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  std::vector<int64_t> infos(batchCount(self), 0);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "gesv", [&]{
    apply_gesv<scalar_t>(self_working_copy, A_working_copy, infos);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos, "gesv");
  } else {
    singleCheckErrors(infos[0], "gesv");
  }
  return std::tuple<Tensor, Tensor>(self_working_copy, A_working_copy);
}

// Supports arbitrary batch dimensions for self and A
std::tuple<Tensor,Tensor> gesv(const Tensor& self, const Tensor& A) {
  AT_CHECK(self.dim() >= 2,
           "B should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  AT_CHECK(A.dim() >= 2,
           "A should have at least 2 dimensions, but has ", A.dim(), " dimensions instead");
  Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) = _linear_solve_broadcast_args(self, A);
  return at::_gesv_helper(self_broadcasted, A_broadcasted);
}

std::tuple<Tensor&,Tensor&> gesv_out(Tensor& solution, Tensor& lu, const Tensor& self, const Tensor& A) {
  AT_CHECK(self.dim() == 2 && A.dim() == 2, 
           "torch.gesv() with the `out` keyword does not support batching. "
           "b.dim() (", self.dim(), ") and A.dim() (", A.dim(), ") must both be 2.");
  Tensor solution_tmp, lu_tmp;
  std::tie(solution_tmp, lu_tmp) = at::_gesv_helper(self, A);
  solution.resize_as_(solution_tmp).copy_(solution_tmp);
  lu.resize_as_(lu_tmp).copy_(lu_tmp);
  return std::tuple<Tensor&, Tensor&>(solution, lu);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_inverse(Tensor& self, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("inverse: LAPACK library not found in compilation");
#else
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
#endif
}

Tensor _inverse_helper_cpu(const Tensor& self) {
  std::vector<int64_t> infos(batchCount(self), 0);
  auto self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "inverse", [&]{
    apply_inverse<scalar_t>(self_working_copy, infos);
  });
  batchCheckErrors(infos, "inverse");
  return self_working_copy;
}

Tensor inverse(const Tensor &self) {
  if (self.size(-1) == 0) {
    return at::empty_like(self);
  }
  if (self.dim() == 2) {
    return at::legacy::th::_th_getri_single(self);
  }
  squareCheckInputs(self);
  return at::_inverse_helper(self);
}

Tensor& inverse_out(Tensor &result, const Tensor &self) {
  if (self.size(-1) == 0) {
    return result.resize_as_(self);
  }
  result.copy_(native::inverse(self));
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky_solve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_cholesky_solve(Tensor& b, Tensor& A, bool upper, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("cholesky_solve: LAPACK library not found in compilation");
#else
  char uplo = upper ? 'U' : 'L';

  auto A_data = A.data<scalar_t>();
  auto b_data = b.data<scalar_t>();
  auto n = A.size(-2);
  auto nrhs = b.size(-1);

  int info;
  if (b.dim() == 2) {
    lapackCholeskySolve<scalar_t>(uplo, n, nrhs, A_data, n, b_data, n, &info);
    infos[0] = info;
  } else {
    auto A_mat_stride = matrixStride(A);
    auto b_mat_stride = matrixStride(b);
    auto batch_size = batchCount(A);
    for (int64_t i = 0; i < batch_size; i++) {
      scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
      scalar_t* b_working_ptr = &b_data[i * b_mat_stride];
      lapackCholeskySolve<scalar_t>(uplo, n, nrhs, A_working_ptr, n, b_working_ptr, n, &info);
      infos[i] = info;
      if (info != 0) {
        return;
      }
    }
  }
#endif
}

Tensor _cholesky_solve_helper_cpu(const Tensor& self, const Tensor& A, bool upper) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  auto A_working_copy = cloneBatchedColumnMajor(A);
  std::vector<int64_t> infos(batchCount(self), 0);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "cholesky_solve", [&]{
    apply_cholesky_solve<scalar_t>(self_working_copy, A_working_copy, upper, infos);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos, "cholesky_solve");
  } else {
    singleCheckErrors(infos[0], "cholesky_solve");
  }
  return self_working_copy;
}

// Supports arbitrary batch dimensions for self and A
Tensor cholesky_solve(const Tensor& self, const Tensor& A, bool upper) {
  AT_CHECK(self.dim() >= 2,
           "b should have at least 2 dimensions, but has ", self.dim(), " dimensions instead");
  AT_CHECK(A.dim() >= 2,
           "u should have at least 2 dimensions, but has ", A.dim(), " dimensions instead");
  Tensor self_broadcasted, A_broadcasted;
  std::tie(self_broadcasted, A_broadcasted) = _linear_solve_broadcast_args(self, A);
  return at::_cholesky_solve_helper(self_broadcasted, A_broadcasted, upper);
}

Tensor& cholesky_solve_out(Tensor& result, const Tensor& self, const Tensor& A, bool upper) {
  AT_CHECK(self.dim() == 2 && A.dim() == 2,
           "torch.cholesky_solve() with the `out` keyword does not support batching. "
           "b.dim() (", self.dim(), ") and A.dim() (", A.dim(), ") must both be 2.");
  result = at::_cholesky_solve_helper(self, A, upper);
  return result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cholesky ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<typename scalar_t>
static void apply_cholesky(Tensor& self, bool upper, std::vector<int64_t>& infos) {
#ifndef USE_LAPACK
  AT_ERROR("cholesky: LAPACK library not found in compilation");
#else
  char uplo = upper ? 'U' : 'L';

  auto self_data = self.data<scalar_t>();
  auto n = self.size(-2);

  int info;
  if (self.dim() == 2) {
    lapackCholesky<scalar_t>(uplo, n, self_data, n, &info);
    infos[0] = info;
  } else {
    auto self_matrix_stride = matrixStride(self);
    auto batch_size = batchCount(self);
    for (int64_t i = 0; i < batch_size; i++) {
      scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
      lapackCholesky<scalar_t>(uplo, n, self_working_ptr, n, &info);
      infos[i] = info;
      if (info != 0) {
        return;
      }
    }
  }
#endif
}

Tensor _cholesky_helper_cpu(const Tensor& self, bool upper) {
  std::vector<int64_t> infos(batchCount(self), 0);
  auto self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "cholesky", [&]{
    apply_cholesky<scalar_t>(self_working_copy, upper, infos);
  });
  if (self.dim() > 2) {
    batchCheckErrors(infos, "cholesky");
  } else {
    singleCheckErrors(infos[0], "cholesky");
  }
  return self_working_copy;
}

Tensor cholesky(const Tensor &self, bool upper) {
  if (self.size(-1) == 0) {
    return at::empty_like(self);
  }
  squareCheckInputs(self);

  auto raw_cholesky_output = at::_cholesky_helper(self, upper);
  if (upper) {
    return raw_cholesky_output.triu_();
  } else {
    return raw_cholesky_output.tril_();
  }
}

Tensor& cholesky_out(Tensor &result, const Tensor &self, bool upper) {
  if (self.size(-1) == 0) {
    return result.resize_as_(self);
  }
  result.copy_(native::cholesky(self, upper));
  return result;
}

template <typename scalar_t, bool inplace, bool upper>
static void apply_triu_tril_single(
    scalar_t* result, scalar_t* self,
    int64_t k, int64_t n, int64_t m,
    int64_t res_row_stride, int64_t res_col_stride,
    int64_t self_row_stride, int64_t self_col_stride) {

  constexpr int64_t zero = 0;
  int64_t i;

  if (upper) {
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {
      for (int64_t j = 0; j < std::min(m, i + k); j++) {
        result[i * res_row_stride + j * res_col_stride] = 0;
      }
      if (!inplace) {  // copy the rest of the self if not inplace
        for (int64_t j = std::max(zero, i + k); j < m; j++) {
          result[i * res_row_stride + j * res_col_stride] = self[i * self_row_stride + j * self_col_stride];
        }
      }
    }
  } else {
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {
      for (int64_t j = std::max(zero, i + k + 1); j < m; j++) {
        result[i * res_row_stride + j * res_col_stride] = 0;
      }
      if (!inplace) {  // copy the rest of the self if not inplace
        for (int64_t j = zero; j < std::min(m, i + k + 1); j++) {
          result[i * res_row_stride + j * res_col_stride] = self[i * self_row_stride + j * self_col_stride];
        }
      }
    }
  }
}

template <typename scalar_t, bool inplace, bool upper>
void apply_triu_tril(Tensor& result, const Tensor& self, int64_t k) {
  auto n = self.size(-2);
  auto m = self.size(-1);
  auto self_data = self.data<scalar_t>();
  auto self_stride = self.dim() > 2 ? self.stride(-3) : 1;
  auto batchsize = batchCount(self);
  auto self_row_stride = self.stride(-2);
  auto self_column_stride = self.stride(-1);

  auto result_data = result.data<scalar_t>();
  int64_t result_stride, result_row_stride, result_column_stride;
  if (result_data != self_data) {
    result_stride = result.dim() > 2 ? result.stride(-3) : 1;
    result_row_stride = result.stride(-2);
    result_column_stride = result.stride(-1);
  } else {
    result_stride = self_stride;
    result_row_stride = self_row_stride;
    result_column_stride = self_column_stride;
  }

  int64_t b;
  #pragma omp parallel for private(b)
  for (b = 0; b < batchsize; b++) {
    scalar_t* self_batch = &self_data[b * self_stride];
    scalar_t* result_batch = &result_data[b * result_stride];
    apply_triu_tril_single<scalar_t, inplace, upper>(
        result_batch, self_batch, k, n, m,
        result_row_stride, result_column_stride, self_row_stride, self_column_stride);
  }
}

Tensor tril(const Tensor& self, int64_t k) {
  Tensor result = at::empty({0}, self.options());
  at::tril_out(result, self, k);
  return result;
}

Tensor& tril_cpu_(Tensor &self, int64_t k) {
  if (self.numel() == 0) {
    return self;
  }
  if (!checkTrilTriuBatchContiguous(self)) self = self.contiguous();
  AT_DISPATCH_ALL_TYPES(self.type(), "tril", [&]{
    apply_triu_tril<scalar_t, true, false>(self, self, k);
  });
  return self;
}

Tensor& tril_cpu_out(Tensor &result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  Tensor self_c = checkTrilTriuBatchContiguous(self) ? self : self.contiguous();
  AT_DISPATCH_ALL_TYPES(self.type(), "tril", [&]{
    apply_triu_tril<scalar_t, false, false>(result, self_c, k);
  });
  return result;
}

Tensor triu(const Tensor& self, int64_t k) {
  Tensor result = at::empty({0}, self.options());
  at::triu_out(result, self, k);
  return result;
}

Tensor& triu_cpu_(Tensor &self, int64_t k) {
  if (self.numel() == 0) {
    return self;
  }
  if (!checkTrilTriuBatchContiguous(self)) self = self.contiguous();
  AT_DISPATCH_ALL_TYPES(self.type(), "triu", [&]{
    apply_triu_tril<scalar_t, true, true>(self, self, k);
  });
  return self;
}

Tensor& triu_cpu_out(Tensor &result, const Tensor& self, int64_t k) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return result;
  }
  Tensor self_c = checkTrilTriuBatchContiguous(self) ? self : self.contiguous();
  AT_DISPATCH_ALL_TYPES(self.type(), "triu", [&]{
    apply_triu_tril<scalar_t, false, true>(result, self_c, k);
  });
  return result;
}

}}  // namespace at::native
