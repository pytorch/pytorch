#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cpu/zmath.h>

#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#endif
namespace at::native {

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
      for (const auto i : c10::irange(start, end)) {
        for (int64_t j = i + 1; j < n; j++) {
          self[i * stride + j] = conj_impl(self[j * stride + i]);
        }
      }
    };
  } else {
    loop = [&](int64_t start, int64_t end) {
      for (const auto i : c10::irange(start, end)) {
        for (const auto j : c10::irange(i)) {
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

  for (const auto i : c10::irange(batch_size)) {
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

/*
 LAPACK query functions return workspace size as floating point value, which means
 that it might not be accurately represented if it's size exceed mantissa of the
 corresponding type. Fix it by adding 1ULP to the value before casting to it
 For more info see https://github.com/pytorch/pytorch/issues/145801#issuecomment-2631781776
*/
template <typename T>
static inline
std::enable_if_t<std::is_floating_point_v<T>, int> lapack_work_to_int(const T val) {
    const auto next_after = std::nextafter(val, std::numeric_limits<T>::infinity());
    return std::max<int>(1, std::ceil(next_after));
}
template <typename T>
static inline
std::enable_if_t<c10::is_complex<T>::value, int> lapack_work_to_int(const T val) {
    return lapack_work_to_int(val.real());
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
    ScalarType real_dtype = toRealValueType(input.scalar_type());
    rwork = at::empty({lda * 2}, input.options().dtype(real_dtype));
    rwork_data = rwork.mutable_data_ptr<value_t>();
  }

  // call lapackEig once to get the optimal size for work data
  scalar_t work_query;
  lapackEig<scalar_t, value_t>(jobvl, jobvr, n, input_data, lda, values_data,
    lvectors_data, ldvl, rvectors_data, ldvr, &work_query, -1, rwork_data, &infos_data[0]);

  int lwork = lapack_work_to_int(work_query);
  Tensor work = at::empty({lwork}, input.dtype());
  auto work_data = work.mutable_data_ptr<scalar_t>();

  for (const auto i : c10::irange(batch_size)) {
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
  Tensor input_working_copy = at::empty(input.mT().sizes(), input.options());
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
  int iwork_query = 0;

  // call lapackSyevd once to get the optimal size for work data
  lapackSyevd<scalar_t, value_t>(jobz, uplo, n, vectors_data, lda, values_data,
    &lwork_query, lwork, &rwork_query, lrwork, &iwork_query, liwork, infos_data);

  lwork = lapack_work_to_int(lwork_query);

  Tensor work = at::empty({lwork}, vectors.options());
  auto work_data = work.mutable_data_ptr<scalar_t>();

  liwork = std::max<int>(1, iwork_query);
  Tensor iwork = at::empty({liwork}, vectors.options().dtype(at::kInt));
  auto iwork_data = iwork.mutable_data_ptr<int>();

  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (vectors.is_complex()) {
    lrwork = lapack_work_to_int(rwork_query);
    rwork = at::empty({lrwork}, values.options());
    rwork_data = rwork.mutable_data_ptr<value_t>();
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
  auto input_data = input.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto input_matrix_stride = matrixStride(input);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(input);
  auto m = input.size(-2);
  auto n = input.size(-1);
  auto lda = std::max<int64_t>(1, m);

  int info = 0;
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
  lwork = std::max<int>(static_cast<int>(n), lapack_work_to_int(wkopt));
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

  auto self_data = self.data_ptr<scalar_t>();
  auto tau_data = tau.const_data_ptr<scalar_t>();
  auto self_matrix_stride = matrixStride(self);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(self);
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto k = tau.size(-1);
  auto lda = std::max<int64_t>(1, m);
  int info = 0;

  // LAPACK's requirement
  TORCH_INTERNAL_ASSERT(m >= n);
  TORCH_INTERNAL_ASSERT(n >= k);

  // Run once, first to get the optimum work size.
  // Since we deal with batches of matrices with the same dimensions, doing this outside
  // the loop saves (batch_size - 1) workspace queries which would provide the same result
  // and (batch_size - 1) calls to allocate and deallocate workspace using at::empty()
  int lwork = -1;
  scalar_t wkopt;
  lapackOrgqr<scalar_t>(m, n, k, self_data, lda, const_cast<scalar_t*>(tau_data), &wkopt, lwork, &info);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  lwork = lapack_work_to_int(wkopt);
  Tensor work = at::empty({lwork}, self.options());

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    const scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // now compute the actual Q
    lapackOrgqr<scalar_t>(m, n, k, self_working_ptr, lda, const_cast<scalar_t*>(tau_working_ptr), work.data_ptr<scalar_t>(), lwork, &info);

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
    = std::unordered_map<driver_t, decltype(lapack_func)>({
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
  int rank_32 = 0;
  int64_t* rank_data = nullptr;
  int64_t* rank_working_ptr = nullptr;
  if (driver_t::Gels != driver_type) {
    rank_data = rank.data_ptr<int64_t>();
    rank_working_ptr = rank_data;
  }

  // 'gelsd' and 'gelss' are SVD-based algorithms
  // so we can get singular values
  value_t* s_data = nullptr;
  value_t* s_working_ptr = nullptr;
  int64_t s_stride = 0;
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
    jpvt_data = jpvt.mutable_data_ptr<int>();
  }

  // Run once the driver, first to get the optimal workspace size
  int lwork = -1; // default value to decide the opt size for workspace arrays
  scalar_t work_opt;
  value_t rwork_opt;
  int iwork_opt = 0;
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

  lwork = lapack_work_to_int(work_opt);
  Tensor work = at::empty({lwork}, A.options());
  scalar_t* work_data = work.mutable_data_ptr<scalar_t>();

  // 'rwork' only used for complex inputs and 'gelsy', 'gelsd' and 'gelss' drivers
  Tensor rwork;
  value_t* rwork_data = nullptr;
  if (A.is_complex() && driver_t::Gels != driver_type) {
    int64_t rwork_len = 0;
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
    rwork = at::empty({rwork_len}, A.options().dtype(c10::toRealValueType(A.scalar_type())));
    rwork_data = rwork.mutable_data_ptr<value_t>();
  }

  // 'iwork' workspace array is relevant only for 'gelsd'
  Tensor iwork;
  int* iwork_data = nullptr;
  if (driver_t::Gelsd == driver_type) {
    iwork = at::empty({std::max<int>(1, iwork_opt)}, A.options().dtype(at::kInt));
    iwork_data = iwork.mutable_data_ptr<int>();
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

  static auto driver_string_to_type = std::unordered_map<std::string_view, LapackLstsqDriverType>({
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

  auto input_data = input.const_data_ptr<scalar_t>();
  auto tau_data = tau.const_data_ptr<scalar_t>();
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
  lapackOrmqr<scalar_t>(side, trans, m, n, k, const_cast<scalar_t*>(input_data), lda, const_cast<scalar_t*>(tau_data), other_data, ldc, &wkopt, lwork, &info);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, input.options());

  for (const auto i : c10::irange(batch_size)) {
    const scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* other_working_ptr = &other_data[i * other_matrix_stride];
    const scalar_t* tau_working_ptr = &tau_data[i * tau_stride];

    // now compute the actual result
    lapackOrmqr<scalar_t>(
        side, trans, m, n, k,
        const_cast<scalar_t*>(input_working_ptr), lda,
        const_cast<scalar_t*>(tau_working_ptr),
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
'transpose' chooses op(A)
'unitriangular' if true then the diagonal elements of A are assumed to be 1
and the actual diagonal values are not used.
*/
template<typename scalar_t>
void apply_triangular_solve(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
#if !AT_BUILD_WITH_BLAS()
  TORCH_CHECK(
      false,
      "Calling torch.triangular_solve on a CPU tensor requires compiling ",
      "PyTorch with BLAS. Please use PyTorch built with BLAS support.");
#else
  char uplo = upper ? 'U' : 'L';
  char diag = unitriangular ? 'U' : 'N';
  char side = left ? 'L' : 'R';
  const char trans = to_blas(transpose);

  auto A_data = A.const_data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto B_mat_stride = matrixStride(B);
  auto batch_size = batchCount(A);
  // This allows to pass rectangular A and B when left = True
  auto m = left ? A.size(-1) : B.size(-2);
  auto n = B.size(-1);
  auto lda = std::max<int64_t>(1, A.size(-2));
  auto ldb = std::max<int64_t>(1, B.size(-2));

  for (const auto i : c10::irange(batch_size)) {
    const scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* B_working_ptr = &B_data[i * B_mat_stride];
    blasTriangularSolve<scalar_t>(side, uplo, trans, diag, m, n, const_cast<scalar_t*>(A_working_ptr), lda, B_working_ptr, ldb);
  }
#endif
}

void triangular_solve_kernel(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cpu", [&]{
    apply_triangular_solve<scalar_t>(A, B, left, upper, transpose, unitriangular);
  });
}

template <typename scalar_t>
void apply_ldl_factor(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.ldl_factor on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(A) > 0);
  auto batch_size = batchCount(A);
  auto n = A.size(-2);
  auto leading_dim = A.stride(-1);
  auto uplo = upper ? 'U' : 'L';

  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  auto a_data = A.data_ptr<scalar_t>();
  auto pivots_data = pivots.data_ptr<int>();
  auto info_data = info.data_ptr<int>();

  auto ldl_func =
      hermitian ? lapackLdlHermitian<scalar_t> : lapackLdlSymmetric<scalar_t>;

  scalar_t wkopt;
  ldl_func(uplo, n, a_data, leading_dim, pivots_data, &wkopt, -1, info_data);
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  int lwork = std::max<int>(1, real_impl<scalar_t, value_t>(wkopt));
  Tensor work = at::empty({lwork}, A.dtype());
  auto work_data = work.mutable_data_ptr<scalar_t>();

  for (const auto i : c10::irange(batch_size)) {
    scalar_t* a_working_ptr = &a_data[i * a_stride];
    auto* pivots_working_ptr = &pivots_data[i * pivots_stride];
    auto* info_working_ptr = &info_data[i];
    ldl_func(
        uplo,
        n,
        a_working_ptr,
        leading_dim,
        pivots_working_ptr,
        work_data,
        lwork,
        info_working_ptr);
  }
#endif
}

void ldl_factor_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_factor_kernel_cpu", [&] {
        apply_ldl_factor<scalar_t>(LD, pivots, info, upper, hermitian);
      });
}

template <typename scalar_t>
void apply_ldl_solve(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& B,
    bool upper,
    bool hermitian) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.ldl_factor on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(A) > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(pivots.unsqueeze(-1)) > 0);
  auto batch_size = batchCount(B);
  auto n = A.size(-2);
  auto nrhs = B.size(-1);
  auto lda = A.stride(-1);
  auto ldb = B.stride(-1);
  auto uplo = upper ? 'U' : 'L';

  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  auto b_stride = B.dim() > 2 ? B.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  auto a_data = A.const_data_ptr<scalar_t>();
  auto b_data = B.data_ptr<scalar_t>();
  auto pivots_ = pivots.to(kInt);
  auto pivots_data = pivots_.const_data_ptr<int>();

  auto ldl_solve_func = hermitian ? lapackLdlSolveHermitian<scalar_t>
                                  : lapackLdlSolveSymmetric<scalar_t>;

  int info = 0;
  for (const auto i : c10::irange(batch_size)) {
    const scalar_t* a_working_ptr = &a_data[i * a_stride];
    scalar_t* b_working_ptr = &b_data[i * b_stride];
    const auto* pivots_working_ptr = &pivots_data[i * pivots_stride];
    ldl_solve_func(
        uplo,
        n,
        nrhs,
        const_cast<scalar_t*>(a_working_ptr),
        lda,
        const_cast<int*>(pivots_working_ptr),
        b_working_ptr,
        ldb,
        &info);
  }
  TORCH_INTERNAL_ASSERT(info == 0);
#endif
}

void ldl_solve_kernel(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& result,
    bool upper,
    bool hermitian) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_solve_kernel_cpu", [&] {
        apply_ldl_solve<scalar_t>(LD, pivots, result, upper, hermitian);
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
void apply_lu_factor(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling torch.linalg.lu_factor on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  TORCH_CHECK(compute_pivots, "linalg.lu_factor: LU without pivoting is not implemented on the CPU");

  auto input_data = input.data_ptr<scalar_t>();
  auto pivots_data = pivots.data_ptr<int>();
  auto infos_data = infos.data_ptr<int>();
  auto input_matrix_stride = matrixStride(input);
  auto pivots_stride = pivots.size(-1);
  auto batch_size = batchCount(input);
  auto m = input.size(-2);
  auto n = input.size(-1);
  auto leading_dimension = std::max<int64_t>(1, m);

  const auto loop = [&](int64_t start, int64_t end) {
    for (const auto i : c10::irange(start, end)) {
      scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
      int* pivots_working_ptr = &pivots_data[i * pivots_stride];
      int* infos_working_ptr = &infos_data[i];
      lapackLu<scalar_t>(
          m,
          n,
          input_working_ptr,
          leading_dimension,
          pivots_working_ptr,
          infos_working_ptr);
    }
  };
  // avoid overflow
  float matrix_rank = float(std::min(m, n));
  // A heuristic tested on a 32 core/socket ICX system
  // https://github.com/pytorch/pytorch/pull/93037#discussion_r1090112948
  int64_t chunk_size_per_thread = int64_t(
      std::min(1.0, 3200.0 / (matrix_rank * matrix_rank * matrix_rank)));
  int64_t grain_size = chunk_size_per_thread * at::get_num_threads();
  at::parallel_for(0, batch_size, grain_size, loop);
#endif
}

// This is a type dispatching helper function for 'apply_lu'
void lu_factor_kernel(const Tensor& input, const Tensor& pivots, const Tensor& infos, bool compute_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "lu_cpu", [&]{
    apply_lu_factor<scalar_t>(input, pivots, infos, compute_pivots);
  });
}

/*
  Solves the matrix equation A X = B
  X and B are n-by-nrhs matrices, A is represented using the LU factorization.
  This is an in-place routine, content of `b` is overwritten.

  Args:
  * `b` -  [in] the right hand side matrix B
           [out] the solution matrix X
  * `lu` - [in] the LU factorization of matrix A (see at::linalg_lu_factor)
  * `pivots` - [in] the pivot indices (see at::linalg_lu_factor)

  For further details, please see the LAPACK documentation for GETRS.
*/
template <typename scalar_t>
void apply_lu_solve(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(
      false,
      "Calling linalg.lu_solve on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  auto b_data = B.data_ptr<scalar_t>();
  auto lu_data = LU.const_data_ptr<scalar_t>();
  const auto trans = to_blas(transpose);
  auto pivots_data = pivots.const_data_ptr<int>();
  auto b_stride = matrixStride(B);
  auto lu_stride = LU.dim() > 2 ? LU.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;
  auto batch_size = batchCount(B);

  auto n = LU.size(-2);
  auto nrhs = B.size(-1);
  auto leading_dimension = std::max<int64_t>(1, n);

  int info = 0;

  // lu and pivots tensors can be broadcast to B
  // here we construct a helper indexing tensor to linearly index into LU and pivots
  IntArrayRef lu_batch_shape(LU.sizes().data(), LU.dim() - 2);
  IntArrayRef b_batch_shape(B.sizes().data(), B.dim() - 2);
  BroadcastLinearIndices lu_index(
      batchCount(LU), lu_batch_shape, b_batch_shape);

  for (const auto i : c10::irange(batch_size)) {
    int64_t lu_index_i = lu_index(i);
    scalar_t* b_working_ptr = &b_data[i * b_stride];
    const scalar_t* lu_working_ptr = &lu_data[lu_index_i * lu_stride];
    const int* pivots_working_ptr = &pivots_data[lu_index_i * pivots_stride];

    lapackLuSolve<scalar_t>(trans, n, nrhs, const_cast<scalar_t*>(lu_working_ptr), leading_dimension, const_cast<int*>(pivots_working_ptr),
                            b_working_ptr, leading_dimension, &info);

    // info from lapackLuSolve only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
  }
#endif
}

// This is a type dispatching helper function for 'apply_lu_solve'
void lu_solve_kernel(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  // Lapack will write into unrelated memory if pivots are not in the right range so we do
  // some simple sanity checks here for the CPU version
  TORCH_CHECK(pivots.gt(0).all().item<bool>(),
              "Pivots given to lu_solve must all be greater or equal to 1. "
              "Did you properly pass the result of lu_factor?");
  TORCH_CHECK(pivots.le(LU.size(-2)).all().item<bool>(),
              "Pivots given to lu_solve must all be smaller or equal to LU.size(-2). "
              "Did you properly pass the result of lu_factor?");

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "linalg.lu_solve_cpu", [&]{
    apply_lu_solve<scalar_t>(LU, pivots, B, trans);
  });
}

template <typename scalar_t>
static void apply_svd(const Tensor& A,
                      const bool full_matrices,
                      const bool compute_uv,
                      const Tensor& U,
                      const Tensor& S,
                      const Tensor& Vh,
                      const Tensor& info) {
#if !AT_BUILD_WITH_LAPACK()
  TORCH_CHECK(false, "svd: LAPACK library not found in compilation");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  const auto A_data = A.data_ptr<scalar_t>();
  const auto U_data = compute_uv ? U.data_ptr<scalar_t>() : nullptr;
  const auto S_data = S.data_ptr<value_t>();
  const auto info_data = info.data_ptr<int>();
  const auto Vh_data = compute_uv ? Vh.data_ptr<scalar_t>() : nullptr;
  const auto A_stride = matrixStride(A);
  const auto S_stride = S.size(-1);
  const auto U_stride = compute_uv ? matrixStride(U) : 1;
  const auto Vh_stride = compute_uv ? matrixStride(Vh) : 1;
  const auto batchsize = batchCount(A);
  const char jobz = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';

  const auto m = A.size(-2);
  const auto n = A.size(-1);
  const auto lda = A.stride(-1);
  const auto ldu= compute_uv ? U.stride(-1) : 1;
  const auto ldvh = compute_uv ? Vh.stride(-1) : 1;

  auto iwork = std::vector<int>(8 * std::min(m, n));
  auto* const iwork_data = iwork.data();

  // rwork is just used for the complex decomposition
  auto rwork = std::vector<value_t>{};
  if (A.is_complex()) {
    rwork.resize(std::max(computeLRWorkDim(jobz, m, n), int64_t{1}));
  }
  auto* const rwork_data = rwork.data();

  // Query svd for the optimal lwork size
  int lwork = -1;
  {
    scalar_t wkopt;
    lapackSvd<scalar_t, value_t>(jobz, m, n, A_data, lda, S_data, U_data, ldu, Vh_data, ldvh, &wkopt, lwork, rwork_data, iwork_data, info_data);
    lwork = lapack_work_to_int(wkopt);
  }
  auto work = std::vector<scalar_t>(lwork);
  auto* const work_data = work.data();

  for (const auto i : c10::irange(batchsize)) {
    auto* const A_working_ptr = &A_data[i * A_stride];
    auto* const S_working_ptr = &S_data[i * S_stride];
    auto* const U_working_ptr = compute_uv ? &U_data[i * U_stride] : nullptr;
    auto* const Vh_working_ptr = compute_uv ? &Vh_data[i * Vh_stride] : nullptr;

    // Compute S, U (optionally) and Vh (optionally)
    lapackSvd<scalar_t, value_t>(jobz, m, n, A_working_ptr, lda,
                        S_working_ptr, U_working_ptr, ldu, Vh_working_ptr, ldvh, work_data, lwork, rwork_data, iwork_data, info_data + i);
  }
#endif
}

void svd_kernel(const Tensor& A,
                const bool full_matrices,
                const bool compute_uv,
                const std::optional<std::string_view>& driver,
                const Tensor& U,
                const Tensor& S,
                const Tensor& Vh,
                const Tensor& infos) {
  TORCH_INTERNAL_ASSERT(!driver.has_value(), "svd_kernel: driver shouldn't have a value here. ");
  // Need to copy A as column major, as its contents will be destroyed in the LAPACK call.
  // FIXME It'd be more efficient, rather than cloning A, to copy it into `U` or `Vh` (depending on m > n
  // or m < n) and call jobz='O'
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "linalg_svd_cpu", [&]{
    apply_svd<scalar_t>(cloneBatchedColumnMajor(A), full_matrices, compute_uv, U, S, Vh, infos);
  });
}

void unpack_pivots_cpu_kernel(TensorIterator& iter, const int64_t dim_size, const int64_t max_pivot) {
  if (iter.numel() == 0 || dim_size == 0) {
    return;
  }
  auto loop = [&](char* const* const  data, const int64_t* const strides, const int64_t nelems) {
    auto* perm_ptr = data[0];
    const auto* pivots_ptr = data[1];

    for ([[maybe_unused]] const auto elem : c10::irange(nelems)) {
      // WARNING: linalg.lu_factor returns int32 pivots,
      // this behavior could change in the future.
      const auto perm_data = reinterpret_cast<int64_t*>(perm_ptr);
      const auto pivots_data = reinterpret_cast<const int32_t*>(pivots_ptr);

      for (const auto i : c10::irange(dim_size)) {
        auto new_idx = pivots_data[i] - 1;
        TORCH_CHECK(new_idx >= 0 && new_idx < max_pivot,
                    "pivots passed to lu_unpack must be between 1 and LU.size(-2) inclusive."
                    "Did you properly pass the result of lu_factor?");
        std::swap(
          perm_data[i],
          perm_data[new_idx]
        );
      }

      perm_ptr += strides[0];
      pivots_ptr += strides[1];
    }
  };

  iter.for_each(loop);
}
} // anonymous namespace

REGISTER_ARCH_DISPATCH(cholesky_stub, DEFAULT, &cholesky_kernel)
REGISTER_AVX512_DISPATCH(cholesky_stub, &cholesky_kernel)
REGISTER_AVX2_DISPATCH(cholesky_stub, &cholesky_kernel)
REGISTER_VSX_DISPATCH(cholesky_stub, &cholesky_kernel)
REGISTER_ZVECTOR_DISPATCH(cholesky_stub, &cholesky_kernel)
REGISTER_SVE_DISPATCH(cholesky_stub, &cholesky_kernel)

REGISTER_ARCH_DISPATCH(cholesky_inverse_stub, DEFAULT, &cholesky_inverse_kernel_impl)
REGISTER_AVX512_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl)
REGISTER_AVX2_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl)
REGISTER_VSX_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl)
REGISTER_ZVECTOR_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl)
REGISTER_SVE_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl)

REGISTER_ARCH_DISPATCH(linalg_eig_stub, DEFAULT, &linalg_eig_kernel)
REGISTER_AVX512_DISPATCH(linalg_eig_stub, &linalg_eig_kernel)
REGISTER_AVX2_DISPATCH(linalg_eig_stub, &linalg_eig_kernel)
REGISTER_VSX_DISPATCH(linalg_eig_stub, &linalg_eig_kernel)
REGISTER_ZVECTOR_DISPATCH(linalg_eig_stub, &linalg_eig_kernel)
REGISTER_SVE_DISPATCH(linalg_eig_stub, &linalg_eig_kernel)

REGISTER_ARCH_DISPATCH(linalg_eigh_stub, DEFAULT, &linalg_eigh_kernel)
REGISTER_AVX512_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)
REGISTER_AVX2_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)
REGISTER_VSX_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)
REGISTER_ZVECTOR_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)
REGISTER_SVE_DISPATCH(linalg_eigh_stub, &linalg_eigh_kernel)

REGISTER_ARCH_DISPATCH(geqrf_stub, DEFAULT, &geqrf_kernel)
REGISTER_AVX512_DISPATCH(geqrf_stub, &geqrf_kernel)
REGISTER_AVX2_DISPATCH(geqrf_stub, &geqrf_kernel)
REGISTER_VSX_DISPATCH(geqrf_stub, &geqrf_kernel)
REGISTER_ZVECTOR_DISPATCH(geqrf_stub, &geqrf_kernel)
REGISTER_SVE_DISPATCH(geqrf_stub, &geqrf_kernel)

REGISTER_ARCH_DISPATCH(orgqr_stub, DEFAULT, &orgqr_kernel_impl)
REGISTER_AVX512_DISPATCH(orgqr_stub, &orgqr_kernel_impl)
REGISTER_AVX2_DISPATCH(orgqr_stub, &orgqr_kernel_impl)
REGISTER_VSX_DISPATCH(orgqr_stub, &orgqr_kernel_impl)
REGISTER_ZVECTOR_DISPATCH(orgqr_stub, &orgqr_kernel_impl)
REGISTER_SVE_DISPATCH(orgqr_stub, &orgqr_kernel_impl)

REGISTER_ARCH_DISPATCH(ormqr_stub, DEFAULT, &ormqr_kernel)
REGISTER_AVX512_DISPATCH(ormqr_stub, &ormqr_kernel)
REGISTER_AVX2_DISPATCH(ormqr_stub, &ormqr_kernel)
REGISTER_VSX_DISPATCH(ormqr_stub, &ormqr_kernel)
REGISTER_ZVECTOR_DISPATCH(ormqr_stub, &ormqr_kernel)
REGISTER_SVE_DISPATCH(ormqr_stub, &ormqr_kernel)

REGISTER_ARCH_DISPATCH(lstsq_stub, DEFAULT, &lstsq_kernel)
REGISTER_AVX512_DISPATCH(lstsq_stub, &lstsq_kernel)
REGISTER_AVX2_DISPATCH(lstsq_stub, &lstsq_kernel)
REGISTER_VSX_DISPATCH(lstsq_stub, &lstsq_kernel)
REGISTER_ZVECTOR_DISPATCH(lstsq_stub, &lstsq_kernel)
REGISTER_SVE_DISPATCH(lstsq_stub, &lstsq_kernel)

REGISTER_ARCH_DISPATCH(triangular_solve_stub, DEFAULT, &triangular_solve_kernel)
REGISTER_AVX512_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)
REGISTER_AVX2_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)
REGISTER_VSX_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)
REGISTER_ZVECTOR_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)
REGISTER_SVE_DISPATCH(triangular_solve_stub, &triangular_solve_kernel)

REGISTER_ARCH_DISPATCH(lu_factor_stub, DEFAULT, &lu_factor_kernel)
REGISTER_AVX512_DISPATCH(lu_factor_stub, &lu_factor_kernel)
REGISTER_AVX2_DISPATCH(lu_factor_stub, &lu_factor_kernel)
REGISTER_VSX_DISPATCH(lu_factor_stub, &lu_factor_kernel)
REGISTER_ZVECTOR_DISPATCH(lu_factor_stub, &lu_factor_kernel)
REGISTER_SVE_DISPATCH(lu_factor_stub, &lu_factor_kernel)

REGISTER_ARCH_DISPATCH(ldl_factor_stub, DEFAULT, &ldl_factor_kernel)
REGISTER_AVX512_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_AVX2_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_VSX_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_ZVECTOR_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)
REGISTER_SVE_DISPATCH(ldl_factor_stub, &ldl_factor_kernel)

REGISTER_ARCH_DISPATCH(ldl_solve_stub, DEFAULT, &ldl_solve_kernel)
REGISTER_AVX512_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)
REGISTER_AVX2_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)
REGISTER_VSX_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)
REGISTER_ZVECTOR_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)
REGISTER_SVE_DISPATCH(ldl_solve_stub, &ldl_solve_kernel)

REGISTER_ARCH_DISPATCH(lu_solve_stub, DEFAULT, &lu_solve_kernel)
REGISTER_AVX512_DISPATCH(lu_solve_stub, &lu_solve_kernel)
REGISTER_AVX2_DISPATCH(lu_solve_stub, &lu_solve_kernel)
REGISTER_VSX_DISPATCH(lu_solve_stub, &lu_solve_kernel)
REGISTER_ZVECTOR_DISPATCH(lu_solve_stub, &lu_solve_kernel)
REGISTER_SVE_DISPATCH(lu_solve_stub, &lu_solve_kernel)

REGISTER_ARCH_DISPATCH(svd_stub, DEFAULT, &svd_kernel)
REGISTER_AVX512_DISPATCH(svd_stub, &svd_kernel)
REGISTER_AVX2_DISPATCH(svd_stub, &svd_kernel)
REGISTER_VSX_DISPATCH(svd_stub, &svd_kernel)
REGISTER_ZVECTOR_DISPATCH(svd_stub, &svd_kernel)
REGISTER_SVE_DISPATCH(svd_stub, &svd_kernel)

REGISTER_ARCH_DISPATCH(unpack_pivots_stub, DEFAULT, &unpack_pivots_cpu_kernel)
REGISTER_AVX512_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel)
REGISTER_AVX2_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel)
REGISTER_VSX_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel)
REGISTER_ZVECTOR_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel)
REGISTER_SVE_DISPATCH(unpack_pivots_stub, &unpack_pivots_cpu_kernel)
} // namespace at::native
