// Note [BatchLinearAlgebraLib split implementation files]
//
// There are two files that implement the interfaces found in
// BatchLinearAlgebraLib.h
// - BatchLinearAlgebraLib.cpp
// - BatchLinearAlgebraLibBlas.cpp (this file)
//
// In order to support the ROCm build target, the use of cublas and
// cusolver APIs needed to be split into separate source files to
// accomodate the hipify step of the ROCm build process.
//
// To create this current file, the original file
// BatchLinearAlgebraLib.cpp was copied to
// BatchLinearAlgebraLibBlas.cpp, then any functions that used cusolver
// APIs were removed. Similarly, in the original file
// BatchLinearAlgebraLib.cpp, any use of cublas APIs was removed.
// The net result is a split of the BatchLinearAlgebraLib
// implementation files. The original file BatchLinearAlgebraLib.cpp
// contains the full, original git history for both files.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/cuda/linalg/CUDASolver.h>
#include <ATen/native/cuda/linalg/BatchLinearAlgebraLib.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

static cublasOperation_t to_cublas(TransposeType trans) {
  switch (trans) {
    case TransposeType::NoTranspose: return CUBLAS_OP_N;
    case TransposeType::Transpose: return CUBLAS_OP_T;
    case TransposeType::ConjTranspose: return CUBLAS_OP_C;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

// Some cuBLAS and cuSOLVER batched routines require input to be a device array of pointers to device individual matrices
// 'input' must be a contiguous tensor
template <typename scalar_t>
static Tensor get_device_pointers(const Tensor& input) {
  auto input_data = input.const_data_ptr<scalar_t>();
  int64_t input_mat_stride = matrixStride(input);

  // cublas/cusolver interface requires 'int'
  int batch_size = cuda_int_cast(batchCount(input), "batch_size");

  // if batch_size==0, then start=0 and end=0
  // if input_mat_stride==0, then step=sizeof(scalar_t)
  return at::arange(
      /*start=*/reinterpret_cast<int64_t>(input_data),
      /*end=*/reinterpret_cast<int64_t>(input_data + batch_size * input_mat_stride),
      /*step=*/static_cast<int64_t>(std::max<int64_t>(input_mat_stride, 1) * sizeof(scalar_t)),
      input.options().dtype(at::kLong));
}

template <typename scalar_t>
void apply_geqrf_batched(const Tensor& input, const Tensor& tau) {
  auto batch_size = cuda_int_cast(batchCount(input), "batch_size");
  auto m = cuda_int_cast(input.size(-2), "m");
  auto n = cuda_int_cast(input.size(-1), "n");
  auto lda = std::max<int>(1, m);

  // cuBLAS batched geqrf requires input to be the device array of pointers to device single matrices
  Tensor input_ptr_array = get_device_pointers<scalar_t>(input);
  Tensor tau_ptr_array = get_device_pointers<scalar_t>(tau.unsqueeze(-1));
  auto input_ptr_array_data = reinterpret_cast<scalar_t**>(input_ptr_array.data_ptr());
  auto tau_ptr_array_data = reinterpret_cast<scalar_t**>(tau_ptr_array.data_ptr());

  int info;
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  at::cuda::blas::geqrfBatched(handle, m, n, input_ptr_array_data, lda, tau_ptr_array_data, &info, batch_size);

  // info only indicates wrong arguments to geqrfBatched call
  // info is a host variable, we can check it without device synchronization
  TORCH_INTERNAL_ASSERT(info == 0);
}

void geqrf_batched_cublas(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_batched_cuda", [&]{
    apply_geqrf_batched<scalar_t>(input, tau);
  });
}

template <typename scalar_t>
static void apply_lu_factor_batched_cublas(const Tensor& A, const Tensor& pivots, const Tensor& infos, bool get_pivots) {
  // This function just works with square matrices
  TORCH_INTERNAL_ASSERT(A.size(-2) == A.size(-1));

  auto batch_size = cuda_int_cast(batchCount(A), "batch_size");;
  auto n = cuda_int_cast(A.size(-2), "n");
  auto lda = cuda_int_cast(std::max<int>(1, n), "lda");

  auto pivots_data = get_pivots ? pivots.data_ptr<int>() : nullptr;
  auto infos_data = infos.data_ptr<int>();
  Tensor a_ptr_array = get_device_pointers<scalar_t>(A);
  auto a_ptr_array_data = reinterpret_cast<scalar_t**>(a_ptr_array.data_ptr());

  at::cuda::blas::getrfBatched(n, a_ptr_array_data, lda, pivots_data, infos_data, batch_size);
}

void lu_factor_batched_cublas(const Tensor& A, const Tensor& pivots, const Tensor& infos, bool get_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "lu_factor_cublas", [&]{
    apply_lu_factor_batched_cublas<scalar_t>(A, pivots, infos, get_pivots);
  });
}

template <typename scalar_t>
static void apply_lu_solve_batched_cublas(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
  TORCH_INTERNAL_ASSERT(batchCount(LU) == batchCount(B), "batch_size of LU and B must be the same");
  TORCH_INTERNAL_ASSERT(batchCount(LU) == batchCount(pivots.unsqueeze(-1)), "batch_size of LU and pivots must be the same");
  const auto trans = to_cublas(transpose);

  auto pivots_data = pivots.data_ptr<int>();
  auto batch_size = cuda_int_cast(batchCount(LU), "batch_size");;
  auto m = cuda_int_cast(LU.size(-2), "m");
  auto nrhs = cuda_int_cast(B.size(-1), "nrhs");
  auto lda = cuda_int_cast(std::max<int>(1, m), "lda");
  int info = 0;

  Tensor lu_ptr_array = get_device_pointers<scalar_t>(LU);
  Tensor b_ptr_array = get_device_pointers<scalar_t>(B);
  auto lu_ptr_array_data = reinterpret_cast<scalar_t**>(lu_ptr_array.data_ptr());
  auto b_ptr_array_data = reinterpret_cast<scalar_t**>(b_ptr_array.data_ptr());

  auto handle = at::cuda::getCurrentCUDABlasHandle();
  at::cuda::blas::getrsBatched(handle, trans, m, nrhs, lu_ptr_array_data,
    lda, pivots_data, b_ptr_array_data, lda, &info, batch_size);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
}

void lu_solve_batched_cublas(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_cublas", [&]{
    apply_lu_solve_batched_cublas<scalar_t>(LU, pivots, B, trans);
  });
}

template <typename scalar_t>
static void apply_triangular_solve(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const auto trans = to_cublas(transpose);
  cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto B_mat_stride = matrixStride(B);
  auto batch_size = batchCount(A);
  // This allows to pass rectangular A and B when left = True
  auto m = cuda_int_cast(left ? A.size(-1) : B.size(-2), "m");
  auto n = cuda_int_cast(B.size(-1), "n");
  auto lda = std::max<int>(1, cuda_int_cast(A.size(-2), "lda"));
  auto ldb = std::max<int>(1, cuda_int_cast(B.size(-2), "ldb"));

  auto alpha = scalar_t{1};

  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* B_working_ptr = &B_data[i * B_mat_stride];
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::trsm(handle, side, uplo, trans, diag, m, n, &alpha, A_working_ptr, lda, B_working_ptr, ldb);
  }
}

void triangular_solve_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cuda", [&]{
    apply_triangular_solve<scalar_t>(A, B, left, upper, transpose, unitriangular);
  });
}

template <typename scalar_t>
static void apply_triangular_solve_batched(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const auto trans = to_cublas(transpose);
  cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  auto batch_size = cuda_int_cast(batchCount(A), "batch_size");
  // This allows to pass rectangular A and B when left = True
  auto m = cuda_int_cast(left ? A.size(-1) : B.size(-2), "m");
  auto n = cuda_int_cast(B.size(-1), "n");
  auto lda = std::max<int>(1, cuda_int_cast(A.size(-2), "lda"));
  auto ldb = std::max<int>(1, cuda_int_cast(B.size(-2), "ldb"));

  auto alpha = scalar_t{1};

  // cuBLAS batched trsm requires input to be the device array of pointers to device single matrices
  Tensor A_ptr_array = get_device_pointers<scalar_t>(A);
  Tensor B_ptr_array = get_device_pointers<scalar_t>(B);
  auto A_ptr_array_data = reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr());
  auto B_ptr_array_data = reinterpret_cast<scalar_t**>(B_ptr_array.data_ptr());

  auto handle = at::cuda::getCurrentCUDABlasHandle();
  at::cuda::blas::trsmBatched(handle, side, uplo, trans, diag, m, n, &alpha, A_ptr_array_data, lda, B_ptr_array_data, ldb, batch_size);
}

void triangular_solve_batched_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cuda", [&]{
    apply_triangular_solve_batched<scalar_t>(A, B, left, upper, transpose, unitriangular);
  });
}

template <typename scalar_t>
inline void apply_gels_batched(const Tensor& A, Tensor& B, Tensor& infos) {
  auto trans = CUBLAS_OP_N;
  auto m = cuda_int_cast(A.size(-2), "m");
  auto n = cuda_int_cast(A.size(-1), "n");

  auto nrhs = cuda_int_cast(B.size(-1), "nrhs");
  // cuBLAS from cuda10 and older doesn't work with nrhs == 0 (cuda11 works)
  // so we need to put this early return
  if (nrhs == 0) {
    return;
  }

  auto batch_size = cuda_int_cast(batchCount(B), "batch_size");
  auto lda = std::max<int>(1, m);
  auto ldb = std::max<int>(1, m);

  // cuBLAS's requirement
  TORCH_CHECK(
    m >= n,
    "torch.linalg.lstsq: only overdetermined systems (input.size(-2) >= input.size(-1)) are allowed on CUDA with cuBLAS backend.");

  // cuBLAS documentation says:
  // Matrices Aarray[i] should not overlap; otherwise, undefined behavior is expected.
  // explicitly broadcast the batch dimensions of A
  IntArrayRef A_batch_sizes(A.sizes().data(), A.dim() - 2);
  IntArrayRef B_batch_sizes(B.sizes().data(), B.dim() - 2);
  std::vector<int64_t> expand_batch_portion = at::infer_size(A_batch_sizes, B_batch_sizes);
  expand_batch_portion.insert(expand_batch_portion.end(), {A.size(-2), A.size(-1)});
  Tensor A_expanded = A.expand({expand_batch_portion});
  Tensor A_broadcasted = cloneBatchedColumnMajor(A_expanded);

  // cuBLAS batched gels requires input to be the device array of pointers to device single matrices
  Tensor A_ptr_array = get_device_pointers<scalar_t>(A_broadcasted);
  Tensor B_ptr_array = get_device_pointers<scalar_t>(B);
  auto A_ptr_array_data = reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr());
  auto B_ptr_array_data = reinterpret_cast<scalar_t**>(B_ptr_array.data_ptr());

  auto infos_data = infos.data_ptr<int>();
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  int info;

  at::cuda::blas::gelsBatched<scalar_t>(
    handle, trans, m, n, nrhs,
    A_ptr_array_data, lda,
    B_ptr_array_data, ldb,
    &info,
    infos_data,
    batch_size);

  // negative info indicates that an argument to gelsBatched call is invalid
  TORCH_INTERNAL_ASSERT(info == 0);
}

// This is a type dispatching helper function for 'apply_gels_batched'
void gels_batched_cublas(const Tensor& a, Tensor& b, Tensor& infos) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "gels_batched_cublas", [&]{
    apply_gels_batched<scalar_t>(a, b, infos);
  });
}

} // namespace at::native
