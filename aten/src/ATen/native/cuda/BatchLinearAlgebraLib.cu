#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDASolver.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/cuda/BatchLinearAlgebraLib.h>

namespace at {
namespace native {

// Some cuBLAS and cuSOLVER batched routines require input to be a device array of pointers to device individual matrices
// 'input' must be a contiguous tensor
template <typename scalar_t>
static Tensor get_device_pointers(const Tensor& input) {
  auto input_data = input.data_ptr<scalar_t>();
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
// AMD ROCm backend is implemented via rewriting all CUDA calls to HIP
// rocBLAS does not implement BLAS-like extensions of cuBLAS, they're in rocSOLVER
// rocSOLVER is currently not used in ATen, therefore we raise an error in this case
#ifndef CUDART_VERSION
  TORCH_CHECK(false, "geqrf: Batched version is supported only with cuBLAS backend.")
#else
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
#endif
}

void geqrf_batched_cublas(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_batched_cuda", [&]{
    apply_geqrf_batched<scalar_t>(input, tau);
  });
}

template <typename scalar_t>
static void apply_triangular_solve(Tensor& A, Tensor& B, bool upper, bool transpose, bool conjugate_transpose, bool unitriangular) {
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  trans = conjugate_transpose ? CUBLAS_OP_C : trans;
  cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;

  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto B_mat_stride = matrixStride(B);
  auto batch_size = batchCount(A);
  auto m = cuda_int_cast(A.size(-2), "m");
  auto n = cuda_int_cast(A.size(-1), "n");
  auto nrhs = cuda_int_cast(B.size(-1), "nrhs");
  auto lda = std::max<int>(1, m);

  auto alpha = scalar_t{1};

  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* B_working_ptr = &B_data[i * B_mat_stride];
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    at::cuda::blas::trsm(handle, side, uplo, trans, diag, n, nrhs, &alpha, A_working_ptr, lda, B_working_ptr, lda);
  }
}

void triangular_solve_cublas(Tensor& A, Tensor& B, Tensor& infos, bool upper, bool transpose, bool conjugate_transpose, bool unitriangular) {
  (void)infos; // unused
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cuda", [&]{
    apply_triangular_solve<scalar_t>(A, B, upper, transpose, conjugate_transpose, unitriangular);
  });
}

template <typename scalar_t>
static void apply_triangular_solve_batched(Tensor& A, Tensor& B, bool upper, bool transpose, bool conjugate_transpose, bool unitriangular) {
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  trans = conjugate_transpose ? CUBLAS_OP_C : trans;
  cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
  cublasSideMode_t side = CUBLAS_SIDE_LEFT;

  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto B_mat_stride = matrixStride(B);
  auto batch_size = cuda_int_cast(batchCount(A), "batch_size");
  auto m = cuda_int_cast(A.size(-2), "m");
  auto n = cuda_int_cast(A.size(-1), "n");
  auto nrhs = cuda_int_cast(B.size(-1), "nrhs");
  auto lda = std::max<int>(1, m);

  auto alpha = scalar_t{1};

  // cuBLAS batched trsm requires input to be the device array of pointers to device single matrices
  Tensor A_ptr_array = get_device_pointers<scalar_t>(A);
  Tensor B_ptr_array = get_device_pointers<scalar_t>(B);
  auto A_ptr_array_data = reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr());
  auto B_ptr_array_data = reinterpret_cast<scalar_t**>(B_ptr_array.data_ptr());

  auto handle = at::cuda::getCurrentCUDABlasHandle();
  at::cuda::blas::trsmBatched(handle, side, uplo, trans, diag, n, nrhs, &alpha, A_ptr_array_data, lda, B_ptr_array_data, lda, batch_size);
}

void triangular_solve_batched_cublas(Tensor& A, Tensor& B, Tensor& infos, bool upper, bool transpose, bool conjugate_transpose, bool unitriangular) {
  (void)infos; // unused
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "triangular_solve_cuda", [&]{
    apply_triangular_solve_batched<scalar_t>(A, B, upper, transpose, conjugate_transpose, unitriangular);
  });
}

template <typename scalar_t>
inline void apply_gels_batched(const Tensor& A, Tensor& B, Tensor& infos) {
// AMD ROCm backend is implemented via rewriting all CUDA calls to HIP
// rocBLAS does not implement BLAS-like extensions of cuBLAS, they're in rocSOLVER
// rocSOLVER is currently not used in ATen, therefore we raise an error in this case
#ifndef CUDART_VERSION
  TORCH_CHECK(false, "torch.linalg.lstsq: Batched version is supported only with cuBLAS backend.")
#else
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
#endif
}

// This is a type dispatching helper function for 'apply_gels_batched'
void gels_batched_cublas(const Tensor& a, Tensor& b, Tensor& infos) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "gels_batched_cublas", [&]{
    apply_gels_batched<scalar_t>(a, b, infos);
  });
}

#ifdef USE_CUSOLVER

inline static Tensor column_major_identity_matrix_like(const Tensor& self) {
  auto size = self.sizes();
  auto size_slice = IntArrayRef(size.data(), size.size()-1);
  return at::ones(size_slice, self.options()).diag_embed().transpose(-2, -1);
}

template <typename scalar_t>
inline static void _apply_single_inverse_helper(scalar_t* self_ptr, scalar_t* self_inv_ptr, int* ipiv_ptr, int* info_getrf_ptr, int* info_getrs_ptr, int n, int lda) {
  // self_inv_ptr should already be an identity matrix

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  at::cuda::solver::getrf<scalar_t>(handle, n, n, self_ptr, lda, ipiv_ptr, info_getrf_ptr);
  at::cuda::solver::getrs<scalar_t>(handle, n, n, self_ptr, lda, ipiv_ptr, self_inv_ptr, lda, info_getrs_ptr);
}

template <typename scalar_t>
static void apply_batched_inverse_lib(Tensor& self, Tensor& self_inv, Tensor& infos_getrf, Tensor& infos_getrs) {
  const int batch_size = cuda_int_cast(batchCount(self), "batchCount");
  const int n = cuda_int_cast(self.size(-2), "self.size(-2)");
  const int lda = std::max<int>(1, n);

  auto self_data = self.data_ptr<scalar_t>();
  auto self_mat_stride = matrixStride(self);
  auto self_inv_data = self_inv.data_ptr<scalar_t>();
  auto self_inv_mat_stride = matrixStride(self_inv);

  auto infos_getrf_data = infos_getrf.data_ptr<int>();
  auto infos_getrs_data = infos_getrs.data_ptr<int>();

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();

  // Heuristic: For small batch size or large matrix size, we use for-loop to iterate over the batches instead of
  //            calling the batched cublas routine.
  if (batch_size <= 8 || /* batch_size > 8 && */ n >= 512) {
    for (int64_t i = 0; i < batch_size; i++) {
      auto dataPtr = allocator.allocate(sizeof(int) * lda);
      int* pivot = reinterpret_cast<int*>(dataPtr.get());

      int* infos_getrf_working_ptr = &infos_getrf_data[i];
      int* infos_getrs_working_ptr = &infos_getrs_data[i];

      _apply_single_inverse_helper<scalar_t>(
        &self_data[i * self_mat_stride], &self_inv_data[i * self_inv_mat_stride], pivot, infos_getrf_working_ptr, infos_getrs_working_ptr, n, lda);
    }
  } else {
    // cublas batched kernels require input be "device array of device pointers"
    Tensor self_array = at::arange(
      reinterpret_cast<int64_t>(self_data),
      reinterpret_cast<int64_t>(&self_data[(batch_size-1) * self_mat_stride]) + 1,
      static_cast<int64_t>(self_mat_stride * sizeof(scalar_t)), self.options().dtype(at::kLong));
    Tensor self_inv_array = at::arange(
      reinterpret_cast<int64_t>(self_inv_data),
      reinterpret_cast<int64_t>(&self_inv_data[(batch_size-1) * self_inv_mat_stride]) + 1,
      static_cast<int64_t>(self_inv_mat_stride * sizeof(scalar_t)), self.options().dtype(at::kLong));

    auto dataPtr = allocator.allocate(sizeof(int)*batch_size*lda);
    int* ipiv_array = reinterpret_cast<int*>(dataPtr.get());

    at::cuda::blas::getrfBatched<scalar_t>(n, reinterpret_cast<scalar_t**>(self_array.data_ptr()), lda,
      ipiv_array, infos_getrf_data, batch_size);

    at::cuda::blas::getriBatched<scalar_t>(n, reinterpret_cast<scalar_t**>(self_array.data_ptr()), lda,
      ipiv_array, reinterpret_cast<scalar_t**>(self_inv_array.data_ptr()), lda, infos_getrs_data, batch_size);
  }
}

template <typename scalar_t>
static void apply_single_inverse_lib(const Tensor& self, Tensor& self_inv, Tensor& infos_getrf, Tensor& infos_getrs) {
  int n = cuda_int_cast(self.size(-2), "self.size(-2)");
  int lda = std::max<int>(1, n);

  Tensor ipiv = at::empty({lda}, self.options().dtype(at::kInt));

  _apply_single_inverse_helper<scalar_t>(
    self.data_ptr<scalar_t>(), self_inv.data_ptr<scalar_t>(), ipiv.data_ptr<int>(), infos_getrf.data_ptr<int>(), infos_getrs.data_ptr<int>(), n, lda);
}

// This is a type dispatching helper function for 'apply_batched_inverse_lib' and 'apply_single_inverse_lib'
Tensor& _linalg_inv_out_helper_cuda_lib(Tensor& result, Tensor& infos_getrf, Tensor& infos_getrs) {
  // assuming result is in column major order and contains the matrices to invert
  Tensor input_working_copy = cloneBatchedColumnMajor(result);

  // for getrf + getrs (cusolver path)
  // result should be filled with identity matrices
  result.zero_();
  result.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);

  const int batch_size = cuda_int_cast(batchCount(result), "batchCount");

  if (result.dim() > 2) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "linalg_inv_out_cuda", [&]{
      apply_batched_inverse_lib<scalar_t>(
        input_working_copy, result, infos_getrf, infos_getrs);
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "linalg_inv_out_cuda", [&]{
      apply_single_inverse_lib<scalar_t>(input_working_copy, result, infos_getrf, infos_getrs);
    });
  }

  return result;
}

// entrance of calculations of `inverse` using cusolver getrf + getrs, cublas getrfBatched + getriBatched
Tensor _inverse_helper_cuda_lib(const Tensor& self) {
  Tensor self_working_copy = cloneBatchedColumnMajor(self);
  Tensor self_inv_working_copy = column_major_identity_matrix_like(self_working_copy);
  const int batch_size = cuda_int_cast(batchCount(self), "batchCount");

  if (self.dim() > 2 && batch_size > 1) {
    Tensor infos_getrf = at::zeros({std::max<int64_t>(1, batchCount(self))}, self.options().dtype(kInt));
    Tensor infos_getrs = at::zeros({std::max<int64_t>(1, batchCount(self))}, self.options().dtype(kInt));
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "inverse_cuda", [&]{
      apply_batched_inverse_lib<scalar_t>(
        self_working_copy, self_inv_working_copy, infos_getrf, infos_getrs);
    });
    batchCheckErrors(infos_getrf, "inverse_cuda");
    batchCheckErrors(infos_getrs, "inverse_cuda");
  } else {
    Tensor infos_getrf = at::zeros({1}, self.options().dtype(kInt));
    Tensor infos_getrs = at::zeros({1}, self.options().dtype(kInt));
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "inverse_cuda", [&]{
      apply_single_inverse_lib<scalar_t>(self_working_copy, self_inv_working_copy, infos_getrf, infos_getrs);
    });
    batchCheckErrors(infos_getrf, "inverse_cuda");
    batchCheckErrors(infos_getrs, "inverse_cuda");
  }

  return self_inv_working_copy;
}

// call cusolver gesvdj function to calculate svd
template<typename scalar_t>
inline static void _apply_svd_lib_gesvdj(const Tensor& self, Tensor& U, Tensor& S, Tensor& VT, Tensor& infos, bool compute_uv, bool some) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto VT_data = VT.data_ptr<scalar_t>();
  auto self_stride = matrixStride(self);
  auto U_stride = matrixStride(U);
  auto S_stride = S.size(-1);
  auto VT_stride = matrixStride(VT);

  int batchsize = cuda_int_cast(batchCount(self), "batch size");
  int m = cuda_int_cast(self.size(-2), "m");
  int n = cuda_int_cast(self.size(-1), "n");
  int lda = std::max<int>(1, m);
  int ldvt = std::max<int>(1, n);

  for(int i = 0; i < batchsize; i++){
    // gesvdj_params controls the numerical accuracy of cusolver gesvdj iterations on GPU
    gesvdjInfo_t gesvdj_params;
    TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
    // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, 1.0e-7));
    // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 15));

    auto handle = at::cuda::getCurrentCUDASolverDnHandle();
    auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    at::cuda::solver::gesvdj<scalar_t>(
      handle, jobz, /*econ=*/ some ? 1 : 0, m, n,
      self_data + i * self_stride,
      lda,
      S_data + i * S_stride,
      U_data + i * U_stride,
      lda,
      VT_data + i * VT_stride,
      ldvt,
      infos.data_ptr<int>() + i,
      gesvdj_params
    );

    TORCH_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
  }
}

// wrapper around _apply_svd_lib_gesvdj that handles dtype dispatch,
// creates a working copy of the input, and creates V^H from the V returned by gesvdj
inline static void apply_svd_lib_gesvdj(const Tensor& self, Tensor& U, Tensor& S, Tensor& VT, Tensor& infos, bool compute_uv, bool some) {
  const int64_t m = self.size(-2);
  const int64_t n = self.size(-1);
  Tensor self_working_copy = cloneBatchedColumnMajor(self);
  VT = VT.transpose(-2, -1);  // gesvdj returns V instead of V^H

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda_gesvdj", [&] {
    _apply_svd_lib_gesvdj<scalar_t>(self_working_copy, U, S, VT, infos, compute_uv, some);
  });
}

// call cusolver gesvdj batched function to calculate svd
template<typename scalar_t>
inline static void _apply_svd_lib_gesvdjBatched(const Tensor& self, Tensor& U, Tensor& S, Tensor& VT, Tensor& infos, bool compute_uv) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto VT_data = VT.data_ptr<scalar_t>();
  auto self_stride = matrixStride(self);
  auto U_stride = matrixStride(U);
  auto S_stride = S.size(-1);
  auto VT_stride = matrixStride(VT);

  int batchsize = cuda_int_cast(batchCount(self), "batch size");
  int m = cuda_int_cast(self.size(-2), "m");
  int n = cuda_int_cast(self.size(-1), "n");
  int lda = std::max<int>(1, m);
  int ldvt = std::max<int>(1, n);

  TORCH_INTERNAL_ASSERT(m <= 32 && n <= 32, "gesvdjBatched requires both matrix dimensions not greater than 32, but got "
                        "m = ", m, " n = ", n);

  // gesvdj_params controls the numerical accuracy of cusolver gesvdj iterations on GPU
  gesvdjInfo_t gesvdj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
  // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, 1.0e-7));
  // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 15));
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(gesvdj_params, 1));

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  at::cuda::solver::gesvdjBatched<scalar_t>(
    handle, jobz, m, n, self_data, lda, S_data, U_data, lda, VT_data, ldvt,
    infos.data_ptr<int>(), gesvdj_params, batchsize
  );

  TORCH_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

// wrapper around _apply_svd_lib_gesvdjBatched that handles dtype dispatch,
// creates a working copy of the input, and creates V^H from the V returned by gesvdj
inline static void apply_svd_lib_gesvdjBatched(const Tensor& self, Tensor& U, Tensor& S, Tensor& VT, Tensor& infos, bool compute_uv) {
  const int64_t m = self.size(-2);
  const int64_t n = self.size(-1);
  Tensor self_working_copy = cloneBatchedColumnMajor(self);
  VT = VT.transpose(-2, -1);  // gesvdj returns V instead of V^H

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda_gesvdjBatched", [&] {
    _apply_svd_lib_gesvdjBatched<scalar_t>(self_working_copy, U, S, VT, infos, compute_uv);
  });
}

// entrance of calculations of `svd` using cusolver gesvdj and gesvdjBatched
std::tuple<Tensor, Tensor, Tensor> _svd_helper_cuda_lib(const Tensor& self, bool some, bool compute_uv) {
  const int64_t batch_size = batchCount(self);
  at::Tensor infos = at::zeros({batch_size}, self.options().dtype(at::kInt));
  const int64_t m = self.size(-2);
  const int64_t n = self.size(-1);
  const int64_t k = std::min(m, n);

  Tensor U_working_copy, S_working_copy, VT_working_copy;
  std::tie(U_working_copy, S_working_copy, VT_working_copy) = \
    _create_U_S_VT(self, some, compute_uv, /* svd_use_cusolver = */ true);
  // U, S, V working copies are already column majored now

  // heuristic for using `gesvdjBatched` over `gesvdj`
  if (m <= 32 && n <= 32 && batch_size > 1 && (!some || m == n)) {
    apply_svd_lib_gesvdjBatched(self, U_working_copy, S_working_copy, VT_working_copy, infos, compute_uv);
  } else {
    apply_svd_lib_gesvdj(self, U_working_copy, S_working_copy, VT_working_copy, infos, compute_uv, some);
  }

  // A device-host sync will be performed.
  batchCheckErrors(infos, "svd_cuda");

  if (!compute_uv) {
    VT_working_copy.zero_();
    U_working_copy.zero_();
  }

  if (some) {
    VT_working_copy = VT_working_copy.narrow(-2, 0, k);
  }

  // so far we have computed VT, but torch.svd returns V instead. Adjust accordingly.
  VT_working_copy.transpose_(-2, -1);
  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}


// Implementation of Cholesky decomposition using looped cusolverDn<T>potrf or cusolverDnXpotrf (64-bit)
template<typename scalar_t>
inline static void apply_cholesky_cusolver_potrf_looped(const Tensor& self_working_copy, bool upper, const Tensor& infos) {
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const int64_t n = self_working_copy.size(-1);
  const int64_t lda = std::max<int64_t>(1, n);
  const int64_t batch_size = batchCount(self_working_copy);
  const int64_t matrix_stride = matrixStride(self_working_copy);

  scalar_t* self_working_copy_ptr = self_working_copy.data_ptr<scalar_t>();
  int* infos_ptr = infos.data_ptr<int>();

#ifdef USE_CUSOLVER_64_BIT
  size_t worksize_device;
  size_t worksize_host;
  cusolverDnParams_t params;
  cudaDataType datatype = at::cuda::solver::get_cusolver_datatype<scalar_t>();
  TORCH_CUSOLVER_CHECK(cusolverDnCreateParams(&params));
  at::cuda::solver::xpotrf_buffersize(handle, params, uplo, n, datatype, nullptr, lda, datatype, &worksize_device, &worksize_host);

  // allocate workspace storage
  auto& device_allocator = *at::cuda::getCUDADeviceAllocator();
  auto workdata_device = device_allocator.allocate(worksize_device * batch_size);
  void* workdata_device_ptr = workdata_device.get();

  auto& host_allocator = *at::getCPUAllocator();
  auto workdata_host = host_allocator.allocate(worksize_host * batch_size);
  void* workdata_host_ptr = workdata_host.get();

  for (int64_t i = 0; i < batch_size; i++) {
    at::cuda::solver::xpotrf(
      handle, params, uplo, n, datatype,
      self_working_copy_ptr + i * matrix_stride,
      lda, datatype,
      (char*)workdata_device_ptr + i * worksize_device, worksize_device,
      (char*)workdata_host_ptr + i * worksize_host, worksize_host,
      infos_ptr + i
    );
  }

  TORCH_CUSOLVER_CHECK(cusolverDnDestroyParams(params));
#else // USE_CUSOLVER_64_BIT
  int n_32 = cuda_int_cast(n, "n");
  int lda_32 = cuda_int_cast(lda, "lda");
  int lwork;
  at::cuda::solver::potrf_buffersize<scalar_t>(
    handle, uplo, n_32, nullptr, lda_32, &lwork);

   // allocate workspace storage
  auto& allocator = *at::cuda::getCUDADeviceAllocator();
  auto work_data = allocator.allocate(sizeof(scalar_t)*lwork * batch_size);
  scalar_t* work_data_ptr = static_cast<scalar_t*>(work_data.get());

  for (int64_t i = 0; i < batch_size; i++) {
    at::cuda::solver::potrf<scalar_t>(
      handle, uplo, n_32,
      self_working_copy_ptr + i * matrix_stride,
      lda_32,
      work_data_ptr + i * lwork,
      lwork,
      infos_ptr + i
    );
  }
#endif // USE_CUSOLVER_64_BIT
}

// Implementation of Cholesky decomposition using batched cusolverDn<T>potrfBatched
// Warning: cusolverDn<T>potrfBatched doesn't work quite well when matrix size or batch size is zero.
// If you write your own C++ extension and use this function, make sure you do a zero numel check for the input.
template<typename scalar_t>
inline static void apply_cholesky_cusolver_potrfBatched(const Tensor& self_working_copy, bool upper, const Tensor& infos) {
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const int n = cuda_int_cast(self_working_copy.size(-1), "n");
  const int lda = std::max<int>(1, n);

  const int batch_size = cuda_int_cast(batchCount(self_working_copy), "batch_size");

  // cusolver batched kernels require input be "device array of device pointers"
  Tensor self_working_copy_array = get_device_pointers<scalar_t>(self_working_copy);

  at::cuda::solver::potrfBatched<scalar_t>(
    handle, uplo, n,
    reinterpret_cast<scalar_t**>(self_working_copy_array.data_ptr()),
    lda, infos.data_ptr<int>(), batch_size);
}

void cholesky_helper_cusolver(const Tensor& input, bool upper, const Tensor& info) {
  if (input.numel() == 0) {
    return;
  }

  if (use_cusolver_potrf_batched_ && batchCount(input) > 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "cholesky_cusolver", [&] {
      apply_cholesky_cusolver_potrfBatched<scalar_t>(input, upper, info);
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "cholesky_cusolver", [&] {
      apply_cholesky_cusolver_potrf_looped<scalar_t>(input, upper, info);
    });
  }
}


template<typename scalar_t>
inline static void apply_cholesky_cusolver_potrs(Tensor& self_working_copy, const Tensor& A_column_major_copy, bool upper, Tensor& infos) {
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const int64_t n = self_working_copy.size(-2);
  const int64_t nrhs = self_working_copy.size(-1);
  const int64_t lda = std::max<int64_t>(1, n);
  const int64_t batch_size = batchCount(self_working_copy);
  const int64_t self_matrix_stride = matrixStride(self_working_copy);
  scalar_t* self_working_copy_ptr = self_working_copy.data_ptr<scalar_t>();

  const scalar_t* A_ptr = A_column_major_copy.data_ptr<scalar_t>();
  const int64_t A_matrix_stride = matrixStride(A_column_major_copy);
  const int64_t ldb = std::max<int64_t>(1, A_column_major_copy.size(-1));

  int* infos_ptr = infos.data_ptr<int>();

#ifdef USE_CUSOLVER_64_BIT
  cusolverDnParams_t params;
  cudaDataType datatype = at::cuda::solver::get_cusolver_datatype<scalar_t>();
  TORCH_CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  for (int64_t i = 0; i < batch_size; i++) {
    at::cuda::solver::xpotrs(
      handle, params, uplo, n, nrhs, datatype,
      A_ptr + i * A_matrix_stride,
      lda, datatype,
      self_working_copy_ptr + i * self_matrix_stride,
      ldb,
      infos_ptr
    );
  }

  TORCH_CUSOLVER_CHECK(cusolverDnDestroyParams(params));
#else // USE_CUSOLVER_64_BIT
  int n_32 = cuda_int_cast(n, "n");
  int nrhs_32 = cuda_int_cast(nrhs, "nrhs");
  int lda_32 = cuda_int_cast(lda, "lda");
  int ldb_32 = cuda_int_cast(ldb, "ldb");

  for (int64_t i = 0; i < batch_size; i++) {
    at::cuda::solver::potrs<scalar_t>(
      handle, uplo, n_32, nrhs_32,
      A_ptr + i * A_matrix_stride,
      lda_32,
      self_working_copy_ptr + i * self_matrix_stride,
      ldb_32,
      infos_ptr
    );
  }
#endif // USE_CUSOLVER_64_BIT
}


// This code path is only dispatched to if MAGMA is not linked in the pytorch build.
// cusolverDn<t>potrsBatched only supports nrhs == 1
template<typename scalar_t>
inline static void apply_cholesky_cusolver_potrsBatched(Tensor& self_working_copy, const Tensor& A_column_major_copy, bool upper, Tensor& infos) {
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const int64_t n = self_working_copy.size(-2);
  const int64_t nrhs = self_working_copy.size(-1);
  const int64_t lda = std::max<int64_t>(1, n);
  const int64_t batch_size = batchCount(self_working_copy);
  const int64_t self_matrix_stride = matrixStride(self_working_copy);
  scalar_t* self_working_copy_ptr = self_working_copy.data_ptr<scalar_t>();

  const scalar_t* A_ptr = A_column_major_copy.data_ptr<scalar_t>();
  const int64_t A_matrix_stride = matrixStride(A_column_major_copy);
  const int64_t ldb = std::max<int64_t>(1, A_column_major_copy.size(-1));

  int* infos_ptr = infos.data_ptr<int>();

  auto self_ptr_array = get_device_pointers<scalar_t>(self_working_copy);
  auto A_ptr_array = get_device_pointers<scalar_t>(A_column_major_copy);

  at::cuda::solver::potrsBatched(
    handle, uplo,
    cuda_int_cast(n, "n"),
    cuda_int_cast(nrhs, "nrhs"),
    reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr()),
    cuda_int_cast(lda, "lda"),
    reinterpret_cast<scalar_t**>(self_ptr_array.data_ptr()),
    cuda_int_cast(ldb, "ldb"),
    infos_ptr,
    cuda_int_cast(batch_size, "batch_size")
  );
}

Tensor _cholesky_solve_helper_cuda_cusolver(const Tensor& self, const Tensor& A, bool upper) {
  const int64_t batch_size = batchCount(self);
  at::Tensor infos = at::zeros({1}, self.options().dtype(at::kInt));
  at::Tensor self_working_copy = cloneBatchedColumnMajor(self);
  at::Tensor A_column_major_copy = cloneBatchedColumnMajor(A);

  const int64_t nrhs = self_working_copy.size(-1);

  // cusolverDn<t>potrsBatched only supports nrhs == 1
  if (batch_size > 1 && nrhs == 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_cuda_potrs_batched", [&] {
      apply_cholesky_cusolver_potrsBatched<scalar_t>(self_working_copy, A_column_major_copy, upper, infos);
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "cholesky_cuda_potrs", [&] {
      apply_cholesky_cusolver_potrs<scalar_t>(self_working_copy, A_column_major_copy, upper, infos);
    });
  }

  // info from potrs and potrsBatched only report if the i-th parameter is wrong, not about the matrix singularity, etc.
  // So we don't need to check it all the time.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.item().toInt() == 0);

  return self_working_copy;
}


void _cholesky_inverse_cusolver_potrs_based(Tensor& result, Tensor& infos, bool upper) {
  at::Tensor input_working_copy = cloneBatchedColumnMajor(result);
  at::Tensor infos_gpu = at::zeros({1}, result.options().dtype(at::kInt));
  result.fill_(0);
  result.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "cholesky_cuda_potri", [&] {
    apply_cholesky_cusolver_potrs<scalar_t>(result, input_working_copy, upper, infos_gpu);
  });

  // Debug only: info of cusolver potrs only check if the i-th parameter is wrong
  // Function argument `infos` is a CPU tensor, the following copy will cause a device-host sync.
  // infos.copy_(infos_gpu);
}

Tensor& cholesky_inverse_kernel_impl_cusolver(Tensor &result, Tensor& infos, bool upper) {
  _cholesky_inverse_cusolver_potrs_based(result, infos, upper);
  return result;
}


/*
  The geqrf function computes the QR decomposition of a m x n matrix A.

  Args:
  * `A` - [in] Tensor with matrices for QR decomposition,
          [out] Tensor containing R in the upper triangle of A
          and elementary reflectors below the main diagonal of A
  * `tau` - Tensor containing the magnitudes of the elementary reflectors
  * `m` - The number of rows of `input` to consider
  * `n` - The number of columns of `input` to consider (actual sizes of `input` could be larger)

  For further details, please see the cuSOLVER documentation for GEQRF.
*/
template <typename scalar_t>
static void apply_geqrf(const Tensor& A, const Tensor& tau) {
  int64_t m = A.size(-2);
  int64_t n = A.size(-1);
  int64_t lda = std::max<int64_t>(1, m);
  int64_t batch_size = batchCount(A);

  auto A_stride = matrixStride(A);
  auto tau_stride = tau.size(-1);

  auto A_data = A.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();

  auto infos = at::zeros({1}, A.options().dtype(at::kInt));
  auto infos_data = infos.data_ptr<int>();

  // get the optimal work size and allocate workspace tensor
#ifdef USE_CUSOLVER_64_BIT
  size_t worksize_device; // workspaceInBytesOnDevice
  size_t worksize_host; // workspaceInBytesOnHost
  cusolverDnParams_t params = NULL; // use default algorithm (currently it's the only option)
  at::cuda::solver::xgeqrf_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(),
      params,
      m,
      n,
      A_data,
      lda,
      tau_data,
      &worksize_device,
      &worksize_host);
#else
  int lwork;
  int m_32 = cuda_int_cast(m, "m");
  int n_32 = cuda_int_cast(n, "n");
  int lda_32 = cuda_int_cast(lda, "lda");
  at::cuda::solver::geqrf_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(), m_32, n_32, A_data, lda_32, &lwork);
#endif // USE_CUSOLVER_64_BIT

  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    scalar_t* A_working_ptr = &A_data[i * A_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();

#ifdef USE_CUSOLVER_64_BIT
    // allocate workspace storage on device and host
    auto& device_allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_device_data = device_allocator.allocate(worksize_device);
    auto& host_allocator = *at::getCPUAllocator();
    auto work_host_data = host_allocator.allocate(worksize_host);
    at::cuda::solver::xgeqrf<scalar_t>(
        handle,
        params,
        m,
        n,
        A_working_ptr,
        lda,
        tau_working_ptr,
        static_cast<scalar_t*>(work_device_data.get()),
        worksize_device,
        static_cast<scalar_t*>(work_host_data.get()),
        worksize_host,
        infos_data);
#else
    // allocate workspace storage on device
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t) * std::max<int>(1, lwork));
    at::cuda::solver::geqrf<scalar_t>(
        handle,
        m_32,
        n_32,
        A_working_ptr,
        lda_32,
        tau_working_ptr,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        infos_data);
#endif // USE_CUSOLVER_64_BIT
  }

  // info from geqrf only reports if the i-th parameter is wrong, not about the matrix singularity
  // so we don't need to check it all the time
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.item().toInt() == 0);
}

// This is a type dispatching helper function for 'apply_geqrf'
void geqrf_cusolver(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "geqrf_cuda", [&]{
    apply_geqrf<scalar_t>(input, tau);
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

  For further details, please see the cuSOLVER documentation for ORMQR and UNMQR.
*/
template <typename scalar_t>
static void apply_ormqr(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  auto side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  auto trans = transpose ? (input.is_complex() ? CUBLAS_OP_C : CUBLAS_OP_T) : CUBLAS_OP_N;

  auto input_data = input.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto other_data = other.data_ptr<scalar_t>();

  auto input_matrix_stride = matrixStride(input);
  auto other_matrix_stride = matrixStride(other);
  auto tau_stride = tau.size(-1);
  auto batch_size = batchCount(input);
  auto m = cuda_int_cast(other.size(-2), "m");
  auto n = cuda_int_cast(other.size(-1), "n");
  auto k = cuda_int_cast(tau.size(-1), "k");
  auto lda = std::max<int>(1, left ? m : n);
  auto ldc = std::max<int>(1, m);

  // get the optimal work size and allocate workspace tensor
  int lwork;
  at::cuda::solver::ormqr_bufferSize<scalar_t>(
    at::cuda::getCurrentCUDASolverDnHandle(), side, trans, m, n, k, input_data, lda, tau_data, other_data, ldc, &lwork);

  auto info = at::zeros({1}, input.options().dtype(at::kInt));
  auto info_data = info.data_ptr<int>();

  for (auto i = decltype(batch_size){0}; i < batch_size; i++) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* other_working_ptr = &other_data[i * other_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();

    // allocate workspace storage
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t)*lwork);

    at::cuda::solver::ormqr<scalar_t>(
      handle, side, trans, m, n, k,
      input_working_ptr,
      lda,
      tau_working_ptr,
      other_working_ptr,
      ldc,
      static_cast<scalar_t*>(work_data.get()),
      lwork,
      info_data
    );

    // info from ormqr only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);
  }
}

// This is a type dispatching helper function for 'apply_ormqr'
void ormqr_cusolver(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "orgmr_cuda", [&]{
    apply_ormqr<scalar_t>(input, tau, other, left, transpose);
  });
}

/*
  The orgqr function allows reconstruction of an orthogonal (or unitary) matrix Q,
  from a sequence of elementary reflectors, such as produced by the geqrf function.

  Args:
  * `self` - Tensor with the directions of the elementary reflectors below the diagonal,
              it will be overwritten with the result
  * `tau` - Tensor containing the magnitudes of the elementary reflectors

  For further details, please see the cuSOLVER documentation for ORGQR and UNGQR.
*/
template <typename scalar_t>
inline static void apply_orgqr(Tensor& self, const Tensor& tau) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto self_matrix_stride = matrixStride(self);
  auto batchsize = cuda_int_cast(batchCount(self), "batch size");
  auto m = cuda_int_cast(self.size(-2), "m");
  auto n = cuda_int_cast(self.size(-1), "n");
  auto k = cuda_int_cast(tau.size(-1), "k");
  auto tau_stride = std::max<int>(1, k);
  auto lda = std::max<int>(1, m);

  // LAPACK's requirement
  TORCH_INTERNAL_ASSERT(m >= n);
  TORCH_INTERNAL_ASSERT(n >= k);

  // cuSOLVER doesn't compute anything for this case, which is wrong
  // the result should be a matrix with 1 on the diagonal
  if (k == 0) {
    self.fill_(0);
    self.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);
    return;
  }

  // get the optimal work size and allocate workspace tensor
  int lwork;
  at::cuda::solver::orgqr_buffersize<scalar_t>(
    at::cuda::getCurrentCUDASolverDnHandle(), m, n, k, self_data, lda, tau_data, &lwork);

  auto info = at::zeros({1}, self.options().dtype(at::kInt));
  auto info_data = info.data_ptr<int>();

  for (auto i = decltype(batchsize){0}; i < batchsize; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();

    // allocate workspace storage
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t)*lwork);

    at::cuda::solver::orgqr<scalar_t>(
      handle, m, n, k,
      self_working_ptr,
      lda,
      tau_working_ptr,
      static_cast<scalar_t*>(work_data.get()),
      lwork,
      info_data
    );

    // info from orgqr only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);
  }
}

// This is a type dispatching helper function for 'apply_orgqr'
Tensor& orgqr_helper_cusolver(Tensor& result, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(result.scalar_type(), "orgqr_cuda", [&]{
    apply_orgqr<scalar_t>(result, tau);
  });
  return result;
}

template <typename scalar_t>
static void apply_syevd(Tensor& values, Tensor& vectors, Tensor& infos, bool upper, bool compute_eigenvectors) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cusolverEigMode_t jobz = compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  int64_t n = vectors.size(-1);
  int64_t lda = std::max<int64_t>(1, n);
  int64_t batch_size = batchCount(vectors);

  auto vectors_stride = matrixStride(vectors);
  auto values_stride = values.size(-1);

  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<int>();

  // get the optimal work size and allocate workspace tensor
#ifdef USE_CUSOLVER_64_BIT
  size_t worksize_device; // workspaceInBytesOnDevice
  size_t worksize_host; // workspaceInBytesOnHost
  cusolverDnParams_t params = NULL; // use default algorithm (currently it's the only option)
  at::cuda::solver::xsyevd_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(),
      params,
      jobz,
      uplo,
      n,
      vectors_data,
      lda,
      values_data,
      &worksize_device,
      &worksize_host);
#else
  int lwork;
  int n_32 = cuda_int_cast(n, "n");
  int lda_32 = cuda_int_cast(lda, "lda");
  at::cuda::solver::syevd_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(), jobz, uplo, n_32, vectors_data, lda_32, values_data, &lwork);
#endif // USE_CUSOLVER_64_BIT

  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    scalar_t* vectors_working_ptr = &vectors_data[i * vectors_stride];
    value_t* values_working_ptr = &values_data[i * values_stride];
    int* info_working_ptr = &infos_data[i];
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();

#ifdef USE_CUSOLVER_64_BIT
    // allocate workspace storage on device and host
    auto& device_allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_device_data = device_allocator.allocate(worksize_device);
    auto& host_allocator = *at::getCPUAllocator();
    auto work_host_data = host_allocator.allocate(worksize_host);
    at::cuda::solver::xsyevd<scalar_t>(
        handle,
        params,
        jobz,
        uplo,
        n,
        vectors_working_ptr,
        lda,
        values_working_ptr,
        static_cast<scalar_t*>(work_device_data.get()),
        worksize_device,
        static_cast<scalar_t*>(work_host_data.get()),
        worksize_host,
        info_working_ptr);
#else
    // allocate workspace storage on device
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t) * lwork);
    at::cuda::solver::syevd<scalar_t>(
        handle,
        jobz,
        uplo,
        n_32,
        vectors_working_ptr,
        lda_32,
        values_working_ptr,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        info_working_ptr);
#endif // USE_CUSOLVER_64_BIT
  }
}

template <typename scalar_t>
static void apply_syevj(Tensor& values, Tensor& vectors, Tensor& infos, bool upper, bool compute_eigenvectors) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cusolverEigMode_t jobz = compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  int n = cuda_int_cast(vectors.size(-1), "n");
  int lda = std::max<int>(1, n);
  auto batch_size = batchCount(vectors);

  auto vectors_stride = matrixStride(vectors);
  auto values_stride = values.size(-1);

  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<int>();

  // syevj_params controls the numerical accuracy of syevj
  // by default the tolerance is set to machine accuracy
  // the maximum number of iteration of Jacobi method by default is 100
  // cuSOLVER documentations says: "15 sweeps are good enough to converge to machine accuracy"
  // LAPACK has SVD routine based on similar Jacobi algorithm (gesvj) and there a maximum of 30 iterations is set
  // Let's use the default values for now
  syevjInfo_t syevj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));

  // get the optimal work size and allocate workspace tensor
  int lwork;
  at::cuda::solver::syevj_bufferSize<scalar_t>(
      at::cuda::getCurrentCUDASolverDnHandle(), jobz, uplo, n, vectors_data, lda, values_data, &lwork, syevj_params);

  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    scalar_t* vectors_working_ptr = &vectors_data[i * vectors_stride];
    value_t* values_working_ptr = &values_data[i * values_stride];
    int* info_working_ptr = &infos_data[i];
    auto handle = at::cuda::getCurrentCUDASolverDnHandle();

    // allocate workspace storage on device
    auto& allocator = *at::cuda::getCUDADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t) * lwork);
    at::cuda::solver::syevj<scalar_t>(
        handle,
        jobz,
        uplo,
        n,
        vectors_working_ptr,
        lda,
        values_working_ptr,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        info_working_ptr,
        syevj_params);
  }
  TORCH_CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
}

template <typename scalar_t>
static void apply_syevj_batched(Tensor& values, Tensor& vectors, Tensor& infos, bool upper, bool compute_eigenvectors) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;

  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cusolverEigMode_t jobz = compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  int n = cuda_int_cast(vectors.size(-1), "n");
  int lda = std::max<int>(1, n);
  int batch_size = cuda_int_cast(batchCount(vectors), "batch_size");

  auto vectors_data = vectors.data_ptr<scalar_t>();
  auto values_data = values.data_ptr<value_t>();
  auto infos_data = infos.data_ptr<int>();

  // syevj_params controls the numerical accuracy of syevj
  // by default the tolerance is set to machine accuracy
  // the maximum number of iteration of Jacobi method by default is 100
  // cuSOLVER documentations says: "15 sweeps are good enough to converge to machine accuracy"
  // LAPACK has SVD routine based on similar Jacobi algorithm (gesvj) and there a maximum of 30 iterations is set
  // Let's use the default values for now
  syevjInfo_t syevj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
  TORCH_CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, 1));

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();

  // get the optimal work size and allocate workspace tensor
  int lwork;
  at::cuda::solver::syevjBatched_bufferSize<scalar_t>(
      handle,
      jobz,
      uplo,
      n,
      vectors_data,
      lda,
      values_data,
      &lwork,
      syevj_params,
      batch_size);

  // allocate workspace storage on device
  auto& allocator = *at::cuda::getCUDADeviceAllocator();
  auto work_data = allocator.allocate(sizeof(scalar_t) * lwork);
  at::cuda::solver::syevjBatched<scalar_t>(
      handle,
      jobz,
      uplo,
      n,
      vectors_data,
      lda,
      values_data,
      static_cast<scalar_t*>(work_data.get()),
      lwork,
      infos_data,
      syevj_params,
      batch_size);
  TORCH_CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
}

static void linalg_eigh_cusolver_syevd(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, bool upper, bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(eigenvectors.scalar_type(), "linalg_eigh_cuda", [&] {
    apply_syevd<scalar_t>(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  });
}

static void linalg_eigh_cusolver_syevj(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, bool upper, bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(eigenvectors.scalar_type(), "linalg_eigh_cuda", [&] {
    apply_syevj<scalar_t>(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  });
}

void linalg_eigh_cusolver(Tensor& eigenvalues, Tensor& eigenvectors, Tensor& infos, bool upper, bool compute_eigenvectors) {
  // TODO: syevj_batched should be added here, but at least for CUDA 11.2 it contains a bug leading to incorrect results
  // See https://github.com/pytorch/pytorch/pull/53040#issuecomment-793626268 and https://github.com/cupy/cupy/issues/4847

  // syevj is better than syevd for float32 dtype and matrix sizes 32x32 - 512x512
  // See https://github.com/pytorch/pytorch/pull/53040#issuecomment-788264724
  if (eigenvectors.scalar_type() == at::kFloat && eigenvectors.size(-1) >= 32 && eigenvectors.size(-1) <= 512) {
    return linalg_eigh_cusolver_syevj(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  } else {
    return linalg_eigh_cusolver_syevd(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  }
}

#endif  // USE_CUSOLVER

}} // namespace at::native
