#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cpu/zmath.h>

#include <c10/util/irange.h>

#include <TH/TH.h>  // for USE_LAPACK

namespace at { namespace native {

namespace {

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
#ifndef USE_LAPACK
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
#ifndef USE_LAPACK
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

  int64_t info;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "eig_cpu", [&]{
    apply_eig<scalar_t>(self_, eigenvectors, vals_, vecs_, &info);
  });
  singleCheckErrors(info, "eig_cpu");
  return std::tuple<Tensor, Tensor>(vals_, vecs_);
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
#ifndef USE_LAPACK
  TORCH_CHECK(
      false,
      "Calling torch.linalg.lstsq on a CPU tensor requires compiling ",
      "PyTorch with LAPACK. Please use PyTorch built with LAPACK support.");
#else
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  auto lapack_func = lapackLstsq<LapackLstsqDriverType::Gelsd, scalar_t, value_t>;
  static auto driver_type_to_func
    = std::unordered_map<LapackLstsqDriverType, decltype(lapack_func), LapackLstsqDriverTypeHash>({
    {LapackLstsqDriverType::Gels, lapackLstsq<LapackLstsqDriverType::Gels, scalar_t, value_t>},
    {LapackLstsqDriverType::Gelsy, lapackLstsq<LapackLstsqDriverType::Gelsy, scalar_t, value_t>},
    {LapackLstsqDriverType::Gelsd, lapackLstsq<LapackLstsqDriverType::Gelsd, scalar_t, value_t>},
    {LapackLstsqDriverType::Gelss, lapackLstsq<LapackLstsqDriverType::Gelss, scalar_t, value_t>}
  });
  lapack_func = driver_type_to_func[driver_type];

  char trans = 'N';

  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto batch_size = batchCount(A);
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
  if (LapackLstsqDriverType::Gels != driver_type) {
    rank_data = rank.data_ptr<int64_t>();
    rank_working_ptr = rank_data;
  }

  // 'gelsd' and 'gelss' are SVD-based algorithms
  // so we can get singular values
  value_t* s_data;
  value_t* s_working_ptr = nullptr;
  int64_t s_stride;
  if (LapackLstsqDriverType::Gelsd == driver_type || LapackLstsqDriverType::Gelss == driver_type) {
    s_data = singular_values.data_ptr<value_t>();
    s_working_ptr = s_data;
    s_stride = singular_values.size(-1);
  }

  // 'jpvt' workspace array is used only for 'gelsy' which uses QR factorization with column pivoting
  Tensor jpvt;
  int* jpvt_data = nullptr;
  if (LapackLstsqDriverType::Gelsy == driver_type) {
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
  if (A.is_complex() && LapackLstsqDriverType::Gels != driver_type) {
    int64_t rwork_len;
    switch (driver_type) {
      case LapackLstsqDriverType::Gelsy:
        rwork_len = std::max<int64_t>(1, 2 * n);
        break;
      case LapackLstsqDriverType::Gelss:
        rwork_len = std::max<int64_t>(1, 5 * std::min(m, n));
        break;
      // case LapackLstsqDriverType::Gelsd:
      default:
        rwork_len = std::max<int64_t>(1, rwork_opt);
    }
    rwork = at::empty({rwork_len}, A.options().dtype(c10::toValueType(A.scalar_type())));
    rwork_data = rwork.data_ptr<value_t>();
  }

  // 'iwork' workspace array is relevant only for 'gelsd'
  Tensor iwork;
  int* iwork_data;
  if (LapackLstsqDriverType::Gelsd == driver_type) {
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

  static auto driver_string_to_type = std::unordered_map<std::string, LapackLstsqDriverType>({
    {"gels", LapackLstsqDriverType::Gels},
    {"gelsy", LapackLstsqDriverType::Gelsy},
    {"gelsd", LapackLstsqDriverType::Gelsd},
    {"gelss", LapackLstsqDriverType::Gelss}
  });
  auto driver_type = driver_string_to_type[driver_name];

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "linalg_lstsq_cpu", [&]{
    apply_lstsq<scalar_t>(a, b, rank, singular_values, infos, rcond, driver_type);
  });
}

// This is a type dispatching helper function for 'apply_orgqr'
Tensor& orgqr_kernel_impl(Tensor& result, const Tensor& tau, Tensor& infos, int64_t n_columns) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "orgqr_cpu", [&]{
    apply_orgqr<scalar_t>(result, tau, infos, n_columns);
  });
  return result;
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
#ifndef USE_LAPACK
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

} // anonymous namespace

REGISTER_ARCH_DISPATCH(cholesky_inverse_stub, DEFAULT, &cholesky_inverse_kernel_impl);
REGISTER_AVX_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);
REGISTER_AVX2_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);
REGISTER_VSX_DISPATCH(cholesky_inverse_stub, &cholesky_inverse_kernel_impl);

REGISTER_ARCH_DISPATCH(eig_stub, DEFAULT, &eig_kernel_impl);
REGISTER_AVX_DISPATCH(eig_stub, &eig_kernel_impl);
REGISTER_AVX2_DISPATCH(eig_stub, &eig_kernel_impl);
REGISTER_VSX_DISPATCH(eig_stub, &eig_kernel_impl);

REGISTER_ARCH_DISPATCH(lstsq_stub, DEFAULT, &lstsq_kernel);
REGISTER_AVX_DISPATCH(lstsq_stub, &lstsq_kernel);
REGISTER_AVX2_DISPATCH(lstsq_stub, &lstsq_kernel);
REGISTER_VSX_DISPATCH(lstsq_stub, &lstsq_kernel);

REGISTER_ARCH_DISPATCH(orgqr_stub, DEFAULT, &orgqr_kernel_impl);
REGISTER_AVX_DISPATCH(orgqr_stub, &orgqr_kernel_impl);
REGISTER_AVX2_DISPATCH(orgqr_stub, &orgqr_kernel_impl);
REGISTER_VSX_DISPATCH(orgqr_stub, &orgqr_kernel_impl);

REGISTER_ARCH_DISPATCH(triangular_solve_stub, DEFAULT, &triangular_solve_kernel);
REGISTER_AVX_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);
REGISTER_AVX2_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);
REGISTER_VSX_DISPATCH(triangular_solve_stub, &triangular_solve_kernel);

}} // namespace at::native
