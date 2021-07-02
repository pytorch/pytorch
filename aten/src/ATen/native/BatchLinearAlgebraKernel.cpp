#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cpu/zmath.h>

#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

/*
  Computes the Cholesky decomposition of matrices stored in `input`.
  This is an in-place routine and the content of 'input' is overwritten with the result.

  Args:
  * `input` - [in] Input tensor for the Cholesky decomposition
              [out] Cholesky decomposition result
  * `info` -  [out] Tensor filled with LAPACK error codes,
                    positive values indicate that the matrix is not positive definite.
  * `upper` - controls whether the upper (true) or lower (false) triangular portion of `input` is used

  For further details, please see the LAPACK documentation for POTRF.
*/
template <typename scalar_t>
void apply_cholesky(const Tensor& input, const Tensor& info, bool upper) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.cholesky on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  char uplo = upper ? 'U' : 'L';
  auto input_data = input.data_ptr<scalar_t>();
  auto info_data = info.data_ptr<int>();
  auto input_matrix_stride = matrixStride(input);
  auto batch_size = batchCount(input);
  auto n = input.size(-2);
  auto lda = std::max<int64_t>(1, n);

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    int* info_working_ptr = &info_data[i];
    lapackCholesky<scalar_t>(uplo, n, input_working_ptr, lda, info_working_ptr);
  }
#endif
}

// This is a type dispatching helper function for 'apply_cholesky'
void cholesky_kernel(const Tensor& input, const Tensor& infos, bool upper) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "cholesky_cpu", [&]{
    apply_cholesky<scalar_t>(input, infos, upper);
  });
}

/*
Copies the lower (or upper) triangle of the square matrix to the other half and conjugates it.
This operation is performed in-place.
*/
template <typename scalar_t>
void apply_reflect_conj_tri_single(scalar_t* self, int64_t n, int64_t stride, bool upper) {
  std::function<void(int64_t, int64_t)> loop = [](int64_t, int64_t){};
  if (upper) {
    loop = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        for (int64_t j = i + 1; j < n; j++) {
          self[i * stride + j] = conj_impl(self[j * stride + i]);
        }
      }
    };
  } else {
    loop = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        for (int64_t j = 0; j < i; j++) {
          self[i * stride + j] = conj_impl(self[j * stride + i]);
        }
      }
    };
  }
  // For small matrices OpenMP overhead is too large
  if (n < 256) {
    loop(0, n);
  } else {
    at::parallel_for(0, n, 0, loop);
  }
}

/*
Computes the inverse of a symmetric (Hermitian) positive-definite matrix n-by-n matrix 'input' using the Cholesky factorization
This is an in-place routine, content of 'input' is overwritten.
'infos' is an int Tensor containing error codes for each matrix in the batched input.
For more information see LAPACK's documentation for POTRI routine.
*/
template <typename scalar_t>
void apply_cholesky_inverse(Tensor& input, Tensor& infos, bool upper) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "cholesky_inverse: LAPACK library not found in compilation");
#else
  char uplo = upper ? 'U' : 'L';

  auto input_data = input.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();
  auto input_matrix_stride = matrixStride(input);
  auto batch_size = batchCount(input);
  auto n = input.size(-2);
  auto lda = std::max<int64_t>(1, n);

  for (int64_t i = 0; i < batch_size; i++) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    int* info_working_ptr = &infos_data[i];
    lapackCholeskyInverse<scalar_t>(uplo, n, input_working_ptr, lda, info_working_ptr);
    // LAPACK writes to only upper/lower part of the matrix leaving the other side unchanged
    apply_reflect_conj_tri_single<scalar_t>(input_working_ptr, n, lda, upper);
  }
#endif
}

// This is a type dispatching helper function for 'apply_cholesky_inverse'
Tensor& cholesky_inverse_kernel_impl(Tensor& result, Tensor& infos, bool upper) {
  // This function calculates the inverse matrix in-place
  // result should be in column major order and contain matrices to invert
  // the content of result is overwritten by 'apply_cholesky_inverse'
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "cholesky_inverse_out_cpu", [&]{
    apply_cholesky_inverse<scalar_t>(result, infos, upper);
  });
  return result;
}

template <typename scalar_t>
void apply_eig(const Tensor& self, bool eigenvectors, Tensor& vals_, Tensor& vecs_, int64_t* info_ptr) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "Calling torch.eig on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  char jobvr = eigenvectors ? 'V' : 'N';
  int64_t n = self.size(-1);
  auto self_data = self.data_ptr<scalar_t>();

  auto vals_data = vals_.data_ptr<scalar_t>();
  scalar_t* wr = vals_data;

  scalar_t* vecs_data = eigenvectors ? vecs_.data_ptr<scalar_t>() : nullptr;
  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  int ldvr = eigenvectors ? n : 1;

  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (self.is_complex()) {
    ScalarType real_dtype = toValueType(typeMetaToScalarType(self.dtype()));
    rwork = at::empty({n*2}, self.options().dtype(real_dtype));
    rwork_data = rwork.data_ptr<value_t>();
  }

  if (n > 0) {
    // call lapackEig once to get the optimal size for work data
    scalar_t wkopt;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int info;
    lapackEig<scalar_t, value_t>('N', jobvr, n, self_data, n, wr,
      nullptr, 1, vecs_data, ldvr, &wkopt, -1, rwork_data, &info);
    int lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));

    // call again to do the actual work
    Tensor work = at::empty({lwork}, self.dtype());
    lapackEig<scalar_t, value_t>('N', jobvr, n, self_data, n, wr,
      nullptr, 1, vecs_data, ldvr, work.data_ptr<scalar_t>(), lwork, rwork_data, &info);
    *info_ptr = info;
  }
#endif
}

std::tuple<Tensor, Tensor> eig_kernel_impl(const Tensor& self, bool& eigenvectors) {
  int64_t n = self.size(-1);
  // lapackEig function expects the input to be column major, or stride {1, n},
  // so we must set the stride manually since the default stride for tensors is
  // row major, {n, 1}
  Tensor self_ = at::empty_strided(
      {n, n},
      {1, n},
      at::TensorOptions(self.dtype()));
  self_.copy_(self);

  auto options = self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // the API is slightly different for the complex vs real case: if the input
  // is complex, eigenvals will be a vector of complex. If the input is real,
  // eigenvals will be a (n, 2) matrix containing the real and imaginary parts
  // in each column
  Tensor vals_;
  if (self.is_complex()) {
      vals_ = at::empty({n}, options);
  } else {
      vals_ = at::empty_strided({n, 2}, {1, n}, options);
  }
  Tensor vecs_ = eigenvectors
                 ? at::empty_strided({n, n}, {1, n}, options)
                 : Tensor();

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t info;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "eig_cpu", [&]{
    apply_eig<scalar_t>(self_, eigenvectors, vals_, vecs_, &info);
  });
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  singleCheckErrors(info, "eig_cpu");
  return std::tuple<Tensor, Tensor>(vals_, vecs_);
}

/*
  Computes the eigenvalues and eigenvectors of n-by-n matrix 'input'.
  This is an in-place routine, content of 'input', 'values', 'vectors' is overwritten.
  'infos' is an int Tensor containing error codes for each matrix in the batched input.
  For more information see LAPACK's documentation for GEEV routine.
*/
template <typename scalar_t>
void apply_linalg_eig(Tensor& values, Tensor& vectors, Tensor& input, Tensor& infos, bool compute_eigenvectors) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "Calling torch.linalg.eig on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  char jobvr = compute_eigenvectors ? 'V' : 'N';
  char jobvl = 'N';  // only right eigenvectors are computed
  auto n = input.size(-1);
  auto lda = std::max<int64_t>(1, n);
  auto batch_size = batchCount(input);
  auto input_matrix_stride = matrixStride(input);
  auto values_stride = values.size(-1);
  auto input_data = input.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<scalar_t>();
  auto infos_data = infos.data_ptr<int>();
  auto rvectors_data = compute_eigenvectors ? vectors.data_ptr<scalar_t>() : nullptr;
  scalar_t* lvectors_data = nullptr;  // only right eigenvectors are computed
  int64_t ldvr = compute_eigenvectors ? lda : 1;
  int64_t ldvl = 1;

  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (input.is_complex()) {
    ScalarType real_dtype = toValueType(input.scalar_type());
    rwork = at::empty({lda * 2}, input.options().dtype(real_dtype));
    rwork_data = rwork.data_ptr<value_t>();
  }

  // call lapackEig once to get the optimal size for work data
  scalar_t work_query;
  lapackEig<scalar_t, value_t>(jobvl, jobvr, n, input_data, lda, values_data,
    lvectors_data, ldvl, rvectors_data, ldvr, &work_query, -1, rwork_data, &infos_data[0]);

  int lwork = std::max<int>(1, static_cast<int>(real_impl<scalar_t, value_t>(work_query)));
  Tensor work = at::empty({lwork}, input.dtype());
  auto work_data = work.data_ptr<scalar_t>();

  for (auto i = decltype(batch_size){0}; i < batch_size; i++) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* values_working_ptr = &values_data[i * values_stride];
    scalar_t* rvectors_working_ptr = compute_eigenvectors ? &rvectors_data[i * input_matrix_stride] : nullptr;
    int* info_working_ptr = &infos_data[i];
    lapackEig<scalar_t, value_t>(jobvl, jobvr, n, input_working_ptr, lda, values_working_ptr,
      lvectors_data, ldvl, rvectors_working_ptr, ldvr, work_data, lwork, rwork_data, info_working_ptr);
  }
#endif
}

// This is a type dispatching helper function for 'apply_linalg_eig'
void linalg_eig_kernel(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, const Tensor& input, bool compute_eigenvectors) {
  // This function calculates the non-symmetric eigendecomposition in-place
  // tensors should be in batched column major memory format
  // the content of eigenvalues, eigenvectors and infos is overwritten by 'apply_linalg_eig'

  // apply_linalg_eig modifies in-place provided input matrix, therefore we need a copy
  Tensor input_working_copy = at::empty(input.transpose(-2, -1).sizes(), input.options());
  input_working_copy.transpose_(-2, -1);  // make input_working_copy to have Fortran contiguous memory layout
  input_working_copy.copy_(input);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_eig_out_cpu", [&]{
    apply_linalg_eig<scalar_t>(eigenvalues, eigenvectors, input_working_copy, infos, compute_eigenvectors);
  });
}

/*
  Computes eigenvalues and eigenvectors of the input that is stored initially in 'vectors'.
  The computation is done in-place: 'vectors' stores the input and will be overwritten,
  'values' should be an allocated empty array.
  'infos' is used to store information for possible checks for error.
  'upper' controls the portion of input matrix to consider in computations
  'compute_eigenvectors' controls whether eigenvectors should be computed.
  This function doesn't do any error checks and it's assumed that every argument is valid.
*/
template <typename scalar_t>
void apply_lapack_eigh(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.eigh or eigvalsh on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  char uplo = upper ? 'U' : 'L';
  char jobz = compute_eigenvectors ? 'V' : 'N';

  auto n = vectors.size(-1);
  auto lda = std::max<int64_t>(1, n);
  auto batch_size = batchCount(vectors);

  auto vectors_stride = matrixStride(vectors);
  auto values_stride = values.size(-1);

  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<int>();

  // Using 'int' instead of int32_t or int64_t is consistent with the current LAPACK interface
  // It really should be changed in the future to something like lapack_int that depends on the specific LAPACK library that is linked
  // or switch to supporting only 64-bit indexing by default.
  int lwork = -1;
  int lrwork = -1;
  int liwork = -1;
  scalar_t lwork_query;
  value_t rwork_query;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int iwork_query;

  // call lapackSyevd once to get the optimal size for work data
  lapackSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_data, lda, values_data,
    &lwork_query, lwork, &rwork_query, lrwork, &iwork_query, liwork, infos_data);

  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(lwork_query));
  Tensor work = at::empty({lwork}, vectors.options());
  auto work_data = work.data_ptr<scalar_t>();

  liwork = std::max<int>(1, iwork_query);
  Tensor iwork = at::empty({liwork}, vectors.options().dtype(at::kInt));
  auto iwork_data = iwork.data_ptr<int>();

  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (vectors.is_complex()) {
    lrwork = std::max<int>(1, rwork_query);
    rwork = at::empty({lrwork}, values.options());
    rwork_data = rwork.data_ptr<value_t>();
  }

  // Now call lapackSyevd for each matrix in the batched input
  for (const auto i : c10::irange(batch_size)) {
    scalar_t* vectors_working_ptr = &vectors_data[i * vectors_stride];
    value_t* values_working_ptr = &values_data[i * values_stride];
    int* info_working_ptr = &infos_data[i];
    lapackSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_working_ptr, lda, values_working_ptr,
      work_data, lwork, rwork_data, lrwork, iwork_data, liwork, info_working_ptr);
    // The current behaviour for Linear Algebra functions to raise an error if something goes wrong
    // or input doesn't satisfy some requirement
    // therefore return early since further computations will be wasted anyway
    if (*info_working_ptr != 0) {
      return;
    }
  }
#endif
}

// This is a type dispatching helper function for 'apply_lapack_eigh'
void linalg_eigh_kernel(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  // This function calculates the symmetric/hermitian eigendecomposition
  // in-place tensors should be in batched column major memory format the
  // content of eigenvalues, eigenvectors and infos is overwritten by
  // 'apply_lapack_eigh'
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      eigenvectors.scalar_type(), "linalg_eigh_cpu", [&] {
        apply_lapack_eigh<scalar_t>(
            eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
      });
}

/*
  The geqrf function computes the QR decomposition of matrices stored in `input`.
  However, rather than producing a Q matrix directly, it produces a sequence of
  elementary reflectors which may later be composed to construct Q - for example
  with the orgqr or ormqr functions.

  Args:
  * `input` - [in] Input tensor for QR decomposition
              [out] QR decomposition result which contains:
              i)  The elements of R, on and above the diagonal.
              ii) Directions of the reflectors implicitly defining Q.
             Tensor with the directions of the elementary reflectors below the diagonal,
              it will be overwritten with the result
  * `tau` - [out] Tensor which will contain the magnitudes of the reflectors
            implicitly defining Q.

  For further details, please see the LAPACK documentation for GEQRF.
*/
template <typename scalar_t>
static void apply_geqrf(const Tensor& input, const Tensor& tau) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.geqrf on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto input_data = input.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto input_matrix_stride = matrixStride(input);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(input);
  auto m = input.size(-2);
  auto n = input.size(-1);
  auto lda = std::max<int>(1, m);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;
  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  int lwork = -1;
  scalar_t wkopt;
  lapackGeqrf<scalar_t>(m, n, input_data, lda, tau_data, &wkopt, lwork, &info);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);

  // if lwork is less than 'n' then a warning is printed:
  // Intel MKL ERROR: Parameter 7 was incorrect on entry to SGEQRF.
  lwork = std::max<int>(std::max<int>(1, n), real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, input.options());

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // now compute the actual QR and tau
    lapackGeqrf<scalar_t>(m, n, input_working_ptr, lda, tau_working_ptr, work.data_ptr<scalar_t>(), lwork, &info);

    // info from lapackGeqrf only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_geqrf'
void geqrf_kernel(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_cpu", [&]{
    apply_geqrf<scalar_t>(input, tau);
  });
}

/*
  The orgqr function allows reconstruction of an orthogonal (or unitary) matrix Q,
  from a sequence of elementary reflectors, such as produced by the geqrf function.

  Args:
  * `self` - Tensor with the directions of the elementary reflectors below the diagonal,
              it will be overwritten with the result
  * `tau` - Tensor containing the magnitudes of the elementary reflectors

  For further details, please see the LAPACK documentation for ORGQR and UNGQR.
*/
template <typename scalar_t>
inline void apply_orgqr(Tensor& self, const Tensor& tau) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "Calling torch.orgqr on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  // Some LAPACK implementations might not work well with empty matrices:
  // workspace query might return lwork as 0, which is not allowed (requirement is lwork >= 1)
  // We don't need to do any calculations in this case, so let's return early
  if (self.numel() == 0) {
    return;
  }

  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto self_matrix_stride = matrixStride(self);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(self);
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = tau.size(-1);
  auto lda = std::max<int64_t>(1, m);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int info;

  // LAPACK's requirement
  TORCH_INTERNAL_ASSERT(m >= n);
  TORCH_INTERNAL_ASSERT(n >= k);

  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  int lwork = -1;
  scalar_t wkopt;
  lapackOrgqr<scalar_t>(m, n, k, self_data, lda, tau_data, &wkopt, lwork, &info);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, self.options());

  for (int64_t i = 0; i < batch_size; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // now compute the actual Q
    lapackOrgqr<scalar_t>(m, n, k, self_working_ptr, lda, tau_working_ptr, work.data_ptr<scalar_t>(), lwork, &info);

    // info from lapackOrgqr only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_orgqr'
Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "orgqr_cpu", [&]{
    apply_orgqr<scalar_t>(result, tau);
  });
  return result;
}

// we use `enum class LapackLstsqDriverType` as keys in an unordered_map.
// Clang5 and Gcc5 do not support std::hash for enum classes, hence
// we provide our own hash function.
struct LapackLstsqDriverTypeHash {
  std::size_t operator()(const LapackLstsqDriverType& driver_type) const {
    return static_cast<std::size_t>(driver_type);
  }
};

/*
  Solves a least squares problem. That is minimizing ||B - A X||.

  Input args:
  * 'input' - Tensor containing batches of m-by-n matrix A.
  * 'other' - Tensor containing batches of max(m, n)-by-nrhs matrix B.
  * 'cond' - relative tolerance for determining rank of A.
  * 'driver' - the name of the LAPACK driver that is used to compute the solution.
  Output args (modified in-place):
  * 'solution' - Tensor to store the solution matrix X.
  * 'residuals' - Tensor to store values of ||B - A X||.
  * 'rank' - Tensor to store the rank of A.
  * 'singular_values' - Tensor to store the singular values of A.
  * 'infos' - Tensor to store error codes of linear algebra math library.

  For further details, please see the LAPACK documentation for GELS/GELSY/GELSS/GELSD routines.
*/
template <typename scalar_t>
void apply_lstsq(const Tensor& A, Tensor& B, Tensor& rank, Tensor& singular_values, Tensor& infos, double rcond, LapackLstsqDriverType driver_type) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.lstsq on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  using driver_t = at::native::LapackLstsqDriverType;

  auto lapack_func = lapackLstsq<driver_t::Gelsd, scalar_t, value_t>;
  static auto driver_type_to_func
    = std::unordered_map<driver_t, decltype(lapack_func), LapackLstsqDriverTypeHash>({
    {driver_t::Gels, lapackLstsq<driver_t::Gels, scalar_t, value_t>},
    {driver_t::Gelsy, lapackLstsq<driver_t::Gelsy, scalar_t, value_t>},
    {driver_t::Gelsd, lapackLstsq<driver_t::Gelsd, scalar_t, value_t>},
    {driver_t::Gelss, lapackLstsq<driver_t::Gelss, scalar_t, value_t>}
  });
  lapack_func = driver_type_to_func[driver_type];

  char trans = 'N';

  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto m = A.size(-2);
  auto n = A.size(-1);
  auto nrhs = B.size(-1);
  auto lda = std::max<int64_t>(1, m);
  auto ldb = std::max<int64_t>(1, std::max(m, n));
  auto infos_data = infos.data_ptr<int>();

  // only 'gels' driver does not compute the rank
  int rank_32;
  int64_t* rank_data;
  int64_t* rank_working_ptr = nullptr;
  if (driver_t::Gels != driver_type) {
    rank_data = rank.data_ptr<int64_t>();
    rank_working_ptr = rank_data;
  }

  // 'gelsd' and 'gelss' are SVD-based algorithms
  // so we can get singular values
  value_t* s_data;
  value_t* s_working_ptr = nullptr;
  int64_t s_stride;
  if (driver_t::Gelsd == driver_type || driver_t::Gelss == driver_type) {
    s_data = singular_values.data_ptr<value_t>();
    s_working_ptr = s_data;
    s_stride = singular_values.size(-1);
  }

  // 'jpvt' workspace array is used only for 'gelsy' which uses QR factorization with column pivoting
  Tensor jpvt;
  int* jpvt_data = nullptr;
  if (driver_t::Gelsy == driver_type) {
    jpvt = at::empty({std::max<int64_t>(1, n)}, A.options().dtype(at::kInt));
    jpvt_data = jpvt.data_ptr<int>();
  }

  // Run once the driver, first to get the optimal workspace size
  int lwork = -1; // default value to decide the opt size for workspace arrays
  scalar_t work_opt;
  value_t rwork_opt;
  int iwork_opt;
  lapack_func(trans, m, n, nrhs,
    A_data, lda,
    B_data, ldb,
    &work_opt, lwork,
    infos_data,
    jpvt_data,
    static_cast<value_t>(rcond),
    &rank_32,
    &rwork_opt,
    s_working_ptr,
    &iwork_opt);

  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(work_opt));
  Tensor work = at::empty({lwork}, A.options());
  scalar_t* work_data = work.data_ptr<scalar_t>();

  // 'rwork' only used for complex inputs and 'gelsy', 'gelsd' and 'gelss' drivers
  Tensor rwork;
  value_t* rwork_data;
  if (A.is_complex() && driver_t::Gels != driver_type) {
    int64_t rwork_len;
    switch (driver_type) {
      case driver_t::Gelsy:
        rwork_len = std::max<int64_t>(1, 2 * n);
        break;
      case driver_t::Gelss:
        rwork_len = std::max<int64_t>(1, 5 * std::min(m, n));
        break;
      // case driver_t::Gelsd:
      default:
        rwork_len = std::max<int64_t>(1, rwork_opt);
    }
    rwork = at::empty({rwork_len}, A.options().dtype(c10::toValueType(A.scalar_type())));
    rwork_data = rwork.data_ptr<value_t>();
  }

  // 'iwork' workspace array is relevant only for 'gelsd'
  Tensor iwork;
  int* iwork_data;
  if (driver_t::Gelsd == driver_type) {
    iwork = at::empty({std::max<int>(1, iwork_opt)}, A.options().dtype(at::kInt));
    iwork_data = iwork.data_ptr<int>();
  }

  at::native::batch_iterator_with_broadcasting<scalar_t>(A, B,
    [&](scalar_t* A_working_ptr, scalar_t* B_working_ptr, int64_t A_linear_batch_idx) {
      rank_working_ptr = rank_working_ptr ? &rank_data[A_linear_batch_idx] : nullptr;
      s_working_ptr = s_working_ptr ? &s_data[A_linear_batch_idx * s_stride] : nullptr;
      int* infos_working_ptr = &infos_data[A_linear_batch_idx];

      lapack_func(trans, m, n, nrhs,
        A_working_ptr, lda,
        B_working_ptr, ldb,
        work_data, lwork,
        infos_working_ptr,
        jpvt_data,
        static_cast<value_t>(rcond),
        &rank_32,
        rwork_data,
        s_working_ptr,
        iwork_data);

      // we want the output `rank` Tensor to be of type int64_t,
      // however LAPACK accepts int. That is why we use an integer
      // variable that then gets promoted and written into `rank`.
      // We use this approach over a tensor cast for better performance.
      if (rank_working_ptr) {
        *rank_working_ptr = static_cast<int64_t>(rank_32);
      }
    }
  );
#endif
}

// This is a type and driver dispatching helper function for 'apply_lstsq'
void lstsq_kernel(const Tensor& a, Tensor& b, Tensor& rank, Tensor& singular_values, Tensor& infos, double rcond, std::string driver_name) {

  static auto driver_string_to_type = std::unordered_map<c10::string_view, LapackLstsqDriverType>({
    {"gels", at::native::LapackLstsqDriverType::Gels},
    {"gelsy", at::native::LapackLstsqDriverType::Gelsy},
    {"gelsd", at::native::LapackLstsqDriverType::Gelsd},
    {"gelss", at::native::LapackLstsqDriverType::Gelss}
  });
  auto driver_type = driver_string_to_type[driver_name];

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "linalg_lstsq_cpu", [&]{
    apply_lstsq<scalar_t>(a, b, rank, singular_values, infos, rcond, driver_type);
  });
}

/*
  The ormqr function multiplies Q with another matrix from a sequence of
  elementary reflectors, such as is produced by the geqrf function.

  Args:
  * `input`     - Tensor with elementary reflectors below the diagonal,
                  encoding the matrix Q.
  * `tau`       - Tensor containing the magnitudes of the elementary
                  reflectors.
  * `other`     - [in] Tensor containing the matrix to be multiplied.
                  [out] result of the matrix multiplication with Q.
  * `left`      - bool, determining whether `other` is left- or right-multiplied with Q.
  * `transpose` - bool, determining whether to transpose (or conjugate transpose) Q before multiplying.

  For further details, please see the LAPACK documentation.
*/
template <typename scalar_t>
void apply_ormqr(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "Calling torch.ormqr on a CPU tensor requires compiling ",
    "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  char side = left ? 'L' : 'R';
  char trans = transpose ? (input.is_complex() ? 'C' : 'T') : 'N';

  auto input_data = input.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto other_data = other.data_ptr<scalar_t>();

  auto input_matrix_stride = matrixStride(input);
  auto other_matrix_stride = matrixStride(other);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(input);
  auto m = other.size(-2);
  auto n = other.size(-1);
  auto k = tau.size(-1);
  auto lda = std::max<int64_t>(1, left ? m : n);
  auto ldc = std::max<int64_t>(1, m);
  int info = 0;

  // LAPACK's requirement
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY((left ? m : n) >= k);

  // Query for the optimal size of the workspace tensor
  int lwork = -1;
  scalar_t wkopt;
  lapackOrmqr<scalar_t>(side, trans, m, n, k, input_data, lda, tau_data, other_data, ldc, &wkopt, lwork, &info);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, input.options());

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* other_working_ptr = &other_data[i * other_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // now compute the actual result
    lapackOrmqr<scalar_t>(
        side, trans, m, n, k,
        input_working_ptr, lda,
        tau_working_ptr,
        other_working_ptr, ldc,
        work.data_ptr<scalar_t>(), lwork, &info);

    // info from lapackOrmqr only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_ormqr'
void ormqr_kernel(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "ormqr_cpu", [&]{
    apply_ormqr<scalar_t>(input, tau, other, left, transpose);
  });
}

/*
Solves the matrix equation op(A) X = B
X and B are n-by-nrhs matrices, A is a unit, or non-unit, upper or lower triangular matrix
and op(A) is one of op(A) = A or op(A) = A^T or op(A) = A^H.
This is an in-place routine, content of 'B' is overwritten.
'upper' controls the portion of input matrix to consider in computations,
'transpose' if true then op(A) = A^T,
'unitriangular' if true then the diagonal elements of A are assumed to be 1
and the actual diagonal values are not used.
'infos' is an int Tensor containing error codes for each matrix in the batched input.
For more information see LAPACK's documentation for TRTRS routine.
*/
template<typename scalar_t>
void apply_triangular_solve(Tensor& A, Tensor& B, Tensor& infos, bool upper, bool transpose, bool conjugate_transpose, bool unitriangular) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.triangular_solve on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  char uplo = upper ? 'U' : 'L';
  char trans = transpose ? 'T' : 'N';
  trans = conjugate_transpose ? 'C' : trans;
  char diag = unitriangular ? 'U' : 'N';

  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto B_mat_stride = matrixStride(B);
  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto nrhs = B.size(-1);
  auto lda = std::max<int64_t>(1, n);
  auto infos_data = infos.data_ptr<int>();

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* B_working_ptr = &B_data[i * B_mat_stride];
    int* info_working_ptr = &infos_data[i];
    lapackTriangularSolve<scalar_t>(uplo, trans, diag, n, nrhs, A_working_ptr, lda, B_working_ptr, lda, info_working_ptr);
    // The current behaviour for linear algebra functions to raise an error if something goes wrong
    // or input doesn't satisfy some requirement
    // therefore return early since further computations will be wasted anyway
    if (*info_working_ptr != 0) {
      return;
    }
  }
#endif
}

void triangular_solve_kernel(Tensor& A, Tensor& B, Tensor& infos, bool upper, bool transpose, bool conjugate_transpose, bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cpu", [&]{
    apply_triangular_solve<scalar_t>(A, B, infos, upper, transpose, conjugate_transpose, unitriangular);
  });
}

/*
  Computes the LU decomposition of a m×n matrix or batch of matrices in 'input' tensor.
  This is an in-place routine, content of 'input', 'pivots', and 'infos' is overwritten.

  Args:
  * `input` - [in] the input matrix for LU decomposition
              [out] the LU decomposition
  * `pivots` - [out] the pivot indices
  * `infos` - [out] error codes, positive values indicate singular matrices
  * `compute_pivots` - should always be true (can be false only for CUDA)

  For further details, please see the LAPACK documentation for GETRF.
*/
template <typename scalar_t>
void apply_lu(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.lu on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  TORCH_CHECK(compute_pivots, "lu without pivoting is not implemented on the CPU");

  auto input_data = input.data_ptr<scalar_t>();
  auto pivots_data = pivots.data_ptr<int>();
  auto infos_data = infos.data_ptr<int>();
  auto input_matrix_stride = matrixStride(input);
  auto pivots_stride = pivots.size(-1);
  auto batch_size = batchCount(input);
  auto m = input.size(-2);
  auto n = input.size(-1);
  auto leading_dimension = std::max<int64_t>(1, m);

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    int* pivots_working_ptr = &pivots_data[i * pivots_stride];
    int* infos_working_ptr = &infos_data[i];
    lapackLu<scalar_t>(m, n, input_working_ptr, leading_dimension, pivots_working_ptr, infos_working_ptr);
  }
#endif
}

// This is a type dispatching helper function for 'apply_lu'
void lu_kernel(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "lu_cpu", [&]{
    apply_lu<scalar_t>(input, pivots, infos, compute_pivots);
  });
}

/*
  Solves the matrix equation A X = B
  X and B are n-by-nrhs matrices, A is represented using the LU factorization.
  This is an in-place routine, content of `b` is overwritten.

  Args:
  * `b` -  [in] the right hand side matrix B
           [out] the solution matrix X
  * `lu` - [in] the LU factorization of matrix A (see at::_lu_with_info)
  * `pivots` - [in] the pivot indices (see at::_lu_with_info)

  For further details, please see the LAPACK documentation for GETRS.
*/
template <typename scalar_t>
void apply_lu_solve(const Tensor& b, const Tensor& lu, const Tensor& pivots) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.lu_solve on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  char trans = 'N';
  auto b_data = b.data_ptr<scalar_t>();
  auto lu_data = lu.data_ptr<scalar_t>();
  auto pivots_data = pivots.data_ptr<int>();
  auto b_stride = matrixStride(b);
  auto lu_stride = matrixStride(lu);
  auto pivots_stride = pivots.size(-1);
  auto batch_size = batchCount(b);

  auto n = lu.size(-2);
  auto nrhs = b.size(-1);
  auto leading_dimension = std::max<int64_t>(1, n);

  int info = 0;
  for (const auto i : c10::irange(batch_size)) {
    scalar_t* b_working_ptr = &b_data[i * b_stride];
    scalar_t* lu_working_ptr = &lu_data[i * lu_stride];
    int* pivots_working_ptr = &pivots_data[i * pivots_stride];

    lapackLuSolve<scalar_t>(trans, n, nrhs, lu_working_ptr, leading_dimension, pivots_working_ptr,
                            b_working_ptr, leading_dimension, &info);

    // info from lapackLuSolve only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_lu_solve'
void lu_solve_kernel(const Tensor& b, const Tensor& lu, const Tensor& pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(b.scalar_type(), "lu_solve_cpu", [&]{
    apply_lu_solve<scalar_t>(b, lu, pivots);
  });
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(cholesky_stub, DEFAULT, &cholesky_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(cholesky_stub, &cholesky_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(cholesky_stub, &cholesky_kernel);
REGISTER_VSX_DISPATCH(cholesky_stub, &cholesky_kernel);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(cholesky_inverse_stub, DEFAULT, &cholesky_inverse_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);
REGISTER_VSX_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(eig_stub, DEFAULT, &eig_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(eig_stub, &eig_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(eig_stub, &eig_kernel_impl);
REGISTER_VSX_DISPATCH(eig_stub, &eig_kernel_impl);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(linalg_eig_stub, DEFAULT, &linalg_eig_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);
REGISTER_VSX_DISPATCH(linalg_eig_stub, &linalg_eig_kernel);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(linalg_eigh_stub, DEFAULT, &linalg_eigh_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);
REGISTER_VSX_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(geqrf_stub, DEFAULT, &geqrf_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(geqrf_stub, &geqrf_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(geqrf_stub, &geqrf_kernel);
REGISTER_VSX_DISPATCH(geqrf_stub, &geqrf_kernel);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(orgqr_stub, DEFAULT, &orgqr_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(orgqr_stub, &orgqr_kernel_impl);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(orgqr_stub, &orgqr_kernel_impl);
REGISTER_VSX_DISPATCH(orgqr_stub, &orgqr_kernel_impl);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(ormqr_stub, DEFAULT, &ormqr_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(ormqr_stub, &ormqr_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(ormqr_stub, &ormqr_kernel);
REGISTER_VSX_DISPATCH(ormqr_stub, &ormqr_kernel);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(lstsq_stub, DEFAULT, &lstsq_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(lstsq_stub, &lstsq_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(lstsq_stub, &lstsq_kernel);
REGISTER_VSX_DISPATCH(lstsq_stub, &lstsq_kernel);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_ARCH_DISPATCH(triangular_solve_stub, DEFAULT, &triangular_solve_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);
REGISTER_VSX_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);

REGISTER_ARCH_DISPATCH(lu_stub, DEFAULT, &lu_kernel);
REGISTER_AVX_DISPATCH(lu_stub, &lu_kernel);
REGISTER_AVX2_DISPATCH(lu_stub, &lu_kernel);
REGISTER_VSX_DISPATCH(lu_stub, &lu_kernel);

REGISTER_ARCH_DISPATCH(lu_solve_stub, DEFAULT, &lu_solve_kernel);
REGISTER_AVX_DISPATCH(lu_solve_stub, &lu_solve_kernel);
REGISTER_AVX2_DISPATCH(lu_solve_stub, &lu_solve_kernel);
REGISTER_VSX_DISPATCH(lu_solve_stub, &lu_solve_kernel);

}} // namespace at::native
