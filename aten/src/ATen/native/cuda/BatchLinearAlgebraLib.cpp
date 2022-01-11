#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/CUDASolver.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/cuda/BatchLinearAlgebraLib.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {

cublasOperation_t to_cublas(TransposeType trans) {
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
static void apply_lu_solve_batched_cublas(const Tensor& b, const Tensor& lu, const Tensor& pivots, TransposeType transpose) {
#ifndef CUDART_VERSION
  TORCH_CHECK(false, "lu_solve: cuBLAS backend for lu_solve is not available.")
#else
  const auto trans = to_cublas(transpose);

  auto pivots_data = pivots.data_ptr<int>();
  auto batch_size = cuda_int_cast(batchCount(lu), "batch_size");;
  auto m = cuda_int_cast(lu.size(-2), "m");
  auto nrhs = cuda_int_cast(b.size(-1), "nrhs");
  auto lda = cuda_int_cast(std::max<int>(1, m), "lda");
  int info = 0;

  Tensor lu_ptr_array = get_device_pointers<scalar_t>(lu);
  Tensor b_ptr_array = get_device_pointers<scalar_t>(b);
  auto lu_ptr_array_data = reinterpret_cast<scalar_t**>(lu_ptr_array.data_ptr());
  auto b_ptr_array_data = reinterpret_cast<scalar_t**>(b_ptr_array.data_ptr());

  auto handle = at::cuda::getCurrentCUDABlasHandle();
  at::cuda::blas::getrsBatched(handle, trans, m, nrhs, lu_ptr_array_data,
    lda, pivots_data, b_ptr_array_data, lda, &info, batch_size);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info == 0);
#endif
}

void lu_solve_batched_cublas(const Tensor& b, const Tensor& lu, const Tensor& pivots, TransposeType trans) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(lu.scalar_type(), "lu_solve_cublas", [&]{
    apply_lu_solve_batched_cublas<scalar_t>(b, lu, pivots, trans);
  });
}

template <typename scalar_t>
static void apply_triangular_solve(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular) {
  cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const auto trans = to_cublas(transpose);
  cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
  cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

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
  cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
  cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

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
  return at::ones(size_slice, self.options()).diag_embed().mT();
}

template <typename scalar_t>
inline static void _apply_single_inverse_helper(scalar_t* self_ptr, scalar_t* self_inv_ptr, int* ipiv_ptr, int* info_getrf_ptr, int* info_getrs_ptr, int n, int lda) {
  // self_inv_ptr should already be an identity matrix

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  at::cuda::solver::getrf<scalar_t>(handle, n, n, self_ptr, lda, ipiv_ptr, info_getrf_ptr);
  at::cuda::solver::getrs<scalar_t>(handle, n, n, self_ptr, lda, ipiv_ptr, self_inv_ptr, lda, info_getrs_ptr, CUBLAS_OP_N);
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

// call cusolver gesvd function to calculate svd
template<typename scalar_t>
inline static void _apply_svd_lib_gesvd(const Tensor& self, const Tensor& U, const Tensor& S, const Tensor& VT,
  const Tensor& infos, bool compute_uv, bool some,
  const bool calculate_all_batches,
  const std::vector<int64_t>& batches
) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto self_stride = matrixStride(self);
  auto U_stride = !compute_uv ? 1 : matrixStride(U);
  auto S_stride = S.size(-1);

  int _batchsize = cuda_int_cast(batchCount(self), "batch size");
  int m = cuda_int_cast(self.size(-2), "m");
  int n = cuda_int_cast(self.size(-1), "n");
  int lda = std::max<int>(1, m);
  int ldvt = std::max<int>(1, n);

  TORCH_INTERNAL_ASSERT(m >= n, "cusolver gesvd only supports matrix with sizes m >= n");

  char job = compute_uv ? (some ? 'S' : 'A') : 'N';
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();

  int lwork = -1;
  at::cuda::solver::gesvd_buffersize<scalar_t>(handle, m, n, &lwork);
  TORCH_INTERNAL_ASSERT(lwork >= 0, "gesvd_buffersize failed to get needed buffer size, got lwork = ", lwork);

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr_work = allocator.allocate(sizeof(scalar_t)*lwork);
  auto dataPtr_rwork = allocator.allocate(sizeof(value_t)*std::min(m, n));

  // VT_view shares the same memory space as VT tensor
  at::Tensor VT_view;
  if (compute_uv) {
    VT_view = VT.view({-1, n, n});
  }

  // This function `gesvd` mainly serves as a fallback of `gesvdj` when the latter failed to converge for some large matrices.
  // `gesvdj` returns matrix V in original form of V, while `gesvd` returns matrix V as Hermitian conjugate V^H.
  // With that, we can't directly write `gesvd` output into the memory of tensor VT, which has preset strides and conjugate flag
  // of V, not V^H.
  // One solution is to use an intermediate buffer vt_workspace to get V^H output from `gesvd`, then copy it to the result
  // tensor VT. The pytorch tensor copy implementation will correctly handle the strides and conjugate flag.
  // Note that gesvd returns V^H in column-major.
  // Without a transpose here, it will be V.conj() in the actual memory space.
  at::Tensor vt_workspace = at::empty({n, n}, VT.options()).conj();

  int batchsize = _batchsize;
  std::function<int(int)> get_batch_index = [&](int i) -> int { return i; };
  if (!calculate_all_batches) {
    batchsize = batches.size();
    get_batch_index = [&](int i) -> int { return batches[i]; };
  }

  for(int _i = 0; _i < batchsize; _i++){
    int i = get_batch_index(_i);

    at::cuda::solver::gesvd<scalar_t>(
      handle, job, job, m, n,
      self_data + i * self_stride,
      lda,
      S_data + i * S_stride,
      U_data + i * U_stride,
      lda,
      vt_workspace.data_ptr<scalar_t>(), // VT_data + i * VT_stride,
      ldvt,
      reinterpret_cast<scalar_t*>(dataPtr_work.get()),
      lwork,
      reinterpret_cast<value_t*>(dataPtr_rwork.get()),
      infos.data_ptr<int>() + i
    );

    if (compute_uv) {
      VT_view[i].copy_(vt_workspace);
    }
  }
}

// wrapper around _apply_svd_lib_gesvd that handles dtype dispatch
// note that gesvd returns V^H, which needs to be conjugated and transposed, we want this function to return V
inline static void apply_svd_lib_gesvd(const Tensor& self, const Tensor& U, const Tensor& S, const Tensor& VT,
  const Tensor& infos, bool compute_uv, bool some,
  const bool calculate_all_batches = true,
  const std::vector<int64_t>& batches = {}
) {
  if (self.numel() == 0) {
    return;
  }

  int m = cuda_int_cast(self.size(-2), "m");
  int n = cuda_int_cast(self.size(-1), "n");

  // cusolver gesvd only supports m >= n, we transpose the whole svd calculation if m < n
  // i.e.: A = U S V^H  --> A^H = V S U^H
  if (m < n) {
    apply_svd_lib_gesvd(self.transpose(-2, -1).conj(), VT, S, U, infos, compute_uv, some, calculate_all_batches, batches);
    return;
  }

  Tensor self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda_gesvd", [&] {
    _apply_svd_lib_gesvd<scalar_t>(self_working_copy, U, S, VT, infos, compute_uv, some, calculate_all_batches, batches);
  });
}

// call cusolver gesvdj function to calculate svd
template<typename scalar_t>
inline static void _apply_svd_lib_gesvdj(const Tensor& self, const Tensor& U, const Tensor& S, const Tensor& VT,
  const Tensor& infos, bool compute_uv, bool some
) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto VT_data = VT.data_ptr<scalar_t>();
  auto self_stride = matrixStride(self);
  auto U_stride = !compute_uv ? 1 : matrixStride(U);
  auto S_stride = S.size(-1);
  auto VT_stride = !compute_uv ? 1 : matrixStride(VT);

  int batchsize = cuda_int_cast(batchCount(self), "batch size");
  int m = cuda_int_cast(self.size(-2), "m");
  int n = cuda_int_cast(self.size(-1), "n");
  int lda = std::max<int>(1, m);
  int ldvt = std::max<int>(1, n);

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  int econ = some ? 1 : 0;

  // gesvdj_params controls the numerical accuracy of cusolver gesvdj iterations on GPU
  gesvdjInfo_t gesvdj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

  // Todo: expose the following two parameters to users
  // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, 1.0e-7));
  // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 15));

  int lwork = -1;
  at::cuda::solver::gesvdj_buffersize<scalar_t>(
    handle, jobz, econ, m, n, self_data, lda, S_data, U_data, lda, VT_data, ldvt, &lwork, gesvdj_params);
  TORCH_INTERNAL_ASSERT(lwork >= 0, "gesvdj_buffersize failed to get needed buffer size, got lwork = ", lwork);

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(sizeof(scalar_t)*lwork);

  for(int i = 0; i < batchsize; i++){
    at::cuda::solver::gesvdj<scalar_t>(
      handle, jobz, econ, m, n,
      self_data + i * self_stride,
      lda,
      S_data + i * S_stride,
      U_data + i * U_stride,
      lda,
      VT_data + i * VT_stride,
      ldvt,
      reinterpret_cast<scalar_t*>(dataPtr.get()),
      lwork,
      infos.data_ptr<int>() + i,
      gesvdj_params
    );
  }

  TORCH_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

// wrapper around _apply_svd_lib_gesvdj that handles dtype dispatch
// note that gesvdj returns V, which is what we want
inline static void apply_svd_lib_gesvdj(const Tensor& self, const Tensor& U, const Tensor& S, const Tensor& VT,
  const Tensor& infos, bool compute_uv, bool some
) {
  if (self.numel() == 0) {
    return;
  }

  Tensor self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda_gesvdj", [&] {
    _apply_svd_lib_gesvdj<scalar_t>(self_working_copy, U, S, VT, infos, compute_uv, some);
  });
}

// call cusolver gesvdj batched function to calculate svd
template<typename scalar_t>
inline static void _apply_svd_lib_gesvdjBatched(const Tensor& self, const Tensor& U, const Tensor& S, const Tensor& VT,
  const Tensor& infos, bool compute_uv
) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto self_data = self.data_ptr<scalar_t>();
  auto U_data = U.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto VT_data = VT.data_ptr<scalar_t>();

  int batchsize = cuda_int_cast(batchCount(self), "batch size");
  int m = cuda_int_cast(self.size(-2), "m");
  int n = cuda_int_cast(self.size(-1), "n");
  int lda = std::max<int>(1, m);
  int ldvt = std::max<int>(1, VT.size(-2));

  TORCH_INTERNAL_ASSERT(m <= 32 && n <= 32, "gesvdjBatched requires both matrix dimensions not greater than 32, but got "
                        "m = ", m, " n = ", n);

  // gesvdj_params controls the numerical accuracy of cusolver gesvdj iterations on GPU
  gesvdjInfo_t gesvdj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

  // Todo: expose the following two parameters to users
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

// wrapper around _apply_svd_lib_gesvdjBatched that handles dtype dispatch
// note that gesvdjBatched returns V, which is what we want
inline static void apply_svd_lib_gesvdjBatched(const Tensor& self, const Tensor& U, const Tensor& S, const Tensor& VT,
  const Tensor& infos, bool compute_uv
) {
  Tensor self_working_copy = cloneBatchedColumnMajor(self);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_cuda_gesvdjBatched", [&] {
    _apply_svd_lib_gesvdjBatched<scalar_t>(self_working_copy, U, S, VT, infos, compute_uv);
  });
}

// Check if gesvdj results converge. If not, return a vector that contains the non-converging batch index.
// If the returned vector is empty, all results are converged.
// This function will cause a device-host sync.
std::vector<int64_t> _check_gesvdj_convergence(const Tensor& infos, int64_t non_converging_info) {
  at::Tensor infos_cpu = infos.cpu();
  auto infos_cpu_data = infos_cpu.data_ptr<int>();

  std::vector<int64_t> res;

  for(int64_t i = 0; i < infos.numel(); i++) {
    int info_for_batch_i = infos_cpu_data[i];

    // From cusolver doc, if info < 0, the i-th function call parameter is wrong,
    // which means pytorch implementation of cusolver is wrong.
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info_for_batch_i >= 0);
    if (info_for_batch_i == non_converging_info) res.push_back(i);
  }

  return res;
}

// Depending on the number of non-converging batches,
// format the non-converging batches string as either (no leading or trailing whitespaces)
// batches 2, 3, 5  // or
// batches 2, 3, 5, 7, 11 and other 65535 batches
std::string _format_non_converging_batches(const std::vector<int64_t>& batches) {
  std::stringstream ss;
  const int too_long = 5;

  ss << "batches ";
  if (batches.size() <= too_long) {
    for (const auto i : c10::irange(batches.size() - 1)) {
      ss << batches[i] << ", ";
    }
    ss << batches.back();
  } else {
    for (const auto i : c10::irange(too_long)) {
      ss << batches[i] << ", ";
    }
    ss << "and other " << batches.size() - too_long << " batches";
  }

  return ss.str();
}

// entrance of calculations of `svd` using cusolver gesvdj and gesvdjBatched
// This function returns V, not V^T, not V^H.
std::tuple<Tensor, Tensor, Tensor> _svd_helper_cuda_lib(const Tensor& self, bool some, bool compute_uv) {
  const int64_t batch_size = batchCount(self);
  at::Tensor infos = at::zeros({batch_size}, self.options().dtype(at::kInt));
  const int64_t m = self.size(-2);
  const int64_t n = self.size(-1);

  Tensor U_working_copy, S_working_copy, VT_working_copy;
  std::tie(U_working_copy, S_working_copy, VT_working_copy) = \
    _create_U_S_VT(self, some, compute_uv, /* svd_use_cusolver = */ true);
  // U, S, V working copies are already column majored now

  // gesvdj computes V instead of V^H
  if (compute_uv) {
    VT_working_copy.transpose_(-2, -1);
  }

  // VT_working_copy is row-major, actually "V" now

  // heuristic for using `gesvdjBatched` over `gesvdj`
  if (m <= 32 && n <= 32 && batch_size > 1 && (!some || m == n)) {
    apply_svd_lib_gesvdjBatched(self, U_working_copy, S_working_copy, VT_working_copy, infos, compute_uv);
  } else {
    apply_svd_lib_gesvdj(self, U_working_copy, S_working_copy, VT_working_copy, infos, compute_uv, some);
  }

  // A device-host sync will be performed.
  std::vector<int64_t> gesvdj_convergence_check = _check_gesvdj_convergence(infos, std::min(m, n) + 1);

  if (!gesvdj_convergence_check.empty()) {
    // fall back to gesvd path for those non-converging batches

    TORCH_WARN_ONCE("During the SVD execution, ",
                    _format_non_converging_batches(gesvdj_convergence_check),
                    " failed to converge. ",
                    "A more accurate method will be used to calculate the SVD as a fallback.");
    apply_svd_lib_gesvd(self, U_working_copy, S_working_copy, VT_working_copy, infos, compute_uv, some,
      /* calculate_all_batches = */ false,
      /* batches = */ gesvdj_convergence_check
      );

    // A device-host sync will be performed.
    batchCheckErrors(infos, "svd_cuda");
  }

  // gesvdjBatched fails with illegal memory access and
  // gesvdj fails with CUSOLVER_STATUS_EXECUTION_FAILED
  // if matrices for U and VT are not allocated
  // even though the result of computation is not used we need to allocate this memory
  // now resize the temporary tensors to 0
  if (!compute_uv) {
    U_working_copy.resize_({0});
    VT_working_copy.resize_({0});
  }

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
static void apply_syevd(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
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
static void apply_syevj(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
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
static void apply_syevj_batched(const Tensor& values, const Tensor& vectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
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

static void linalg_eigh_cusolver_syevd(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(eigenvectors.scalar_type(), "linalg_eigh_cuda", [&] {
    apply_syevd<scalar_t>(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  });
}

static void linalg_eigh_cusolver_syevj(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(eigenvectors.scalar_type(), "linalg_eigh_cuda", [&] {
    apply_syevj<scalar_t>(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  });
}

static void linalg_eigh_cusolver_syevj_batched(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(eigenvectors.scalar_type(), "linalg_eigh_cuda", [&] {
    apply_syevj_batched<scalar_t>(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  });
}

void linalg_eigh_cusolver(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors) {
  if (use_cusolver_syevj_batched_ && batchCount(eigenvectors) > 1 && eigenvectors.size(-1) <= 32) {
    // Use syevjBatched for batched matrix opertion when matrix size <= 32
    // See https://github.com/pytorch/pytorch/pull/53040#issuecomment-788264724
    linalg_eigh_cusolver_syevj_batched(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  } else if (eigenvectors.scalar_type() == at::kFloat && eigenvectors.size(-1) >= 32 && eigenvectors.size(-1) <= 512) {
    // syevj is better than syevd for float32 dtype and matrix sizes 32x32 - 512x512
    // See https://github.com/pytorch/pytorch/pull/53040#issuecomment-788264724
    linalg_eigh_cusolver_syevj(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  } else {
    linalg_eigh_cusolver_syevd(eigenvalues, eigenvectors, infos, upper, compute_eigenvectors);
  }
}

// The 'apply_' word is used for templated by dtype functions that call an API routine
// underneath. Since the cusolver API has a slightly different structure we do not prepend
// apply_ to this function.
void lu_factor_looped_cusolver(const Tensor& self, const Tensor& pivots, const Tensor& infos, bool get_pivots, const bool use_magma_) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    self.scalar_type(),
    "lu_factor_cusolver",
    [&self,
     &pivots,
     &infos,
     &get_pivots]() {
    const auto m = cuda_int_cast(self.size(-2), "m");
    const auto n = cuda_int_cast(self.size(-1), "n");
    const auto lda = std::max<int>(1, m);
    const auto self_stride = matrixStride(self);
    const auto batch_size = batchCount(self);
    const auto self_data = self.data_ptr<scalar_t>();
    const auto infos_data = infos.data_ptr<int>();

    const auto pivots_data = get_pivots ? pivots.data_ptr<int>() : nullptr;
    const auto pivots_stride = get_pivots ? pivots.size(-1) : 0;

    const auto handle = at::cuda::getCurrentCUDASolverDnHandle();
    for (auto batch = decltype(batch_size){0}; batch < batch_size; ++batch) {
      at::cuda::solver::getrf<scalar_t>(
        handle, m, n,
        self_data + batch * self_stride,
        lda,
        get_pivots ? pivots_data + batch * pivots_stride : nullptr,
        infos_data + batch
      );
    }
  });

  // Necessary because cuSOLVER uses nan for outputs that correspond to 0 in MAGMA for non-pivoted LU.
  // See https://github.com/pytorch/pytorch/issues/53879 for more details.
  if (!get_pivots && use_magma_) {
    at::nan_to_num_(const_cast<Tensor&>(self), 0, std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity());
  }
}

void lu_solve_looped_cusolver(const Tensor& b, const Tensor& lu, const Tensor& pivots, TransposeType transpose) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(b.scalar_type(), "lu_solve_cusolver", [&] {
    const auto trans = to_cublas(transpose);
    int n = cuda_int_cast(lu.size(-2), "n");
    int nrhs = cuda_int_cast(b.size(-1), "nrhs");
    auto batch_size = batchCount(lu);
    auto info = at::zeros({1}, lu.options().dtype(kInt));
    auto info_data = info.data_ptr<int>();
    auto b_data = b.data_ptr<scalar_t>();
    auto lu_data = lu.data_ptr<scalar_t>();
    auto pivots_data = pivots.data_ptr<int>();
    auto pivots_stride = pivots.size(-1);
    auto lu_stride = matrixStride(lu);
    auto b_stride = matrixStride(b);
    int leading_dimension = cuda_int_cast(std::max<int>(1, n), "leading_dimension");

    auto handle = at::cuda::getCurrentCUDASolverDnHandle();
    for (auto batch = decltype(batch_size){0}; batch < batch_size; ++batch) {
      at::cuda::solver::getrs<scalar_t>(
        handle,
        n,
        nrhs,
        lu_data + batch * lu_stride,
        leading_dimension,
        pivots_data + batch * pivots_stride,
        b_data + batch * b_stride,
        leading_dimension,
        info_data,
        trans);

        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);
    }
  });
}

#endif  // USE_CUSOLVER

}} // namespace at::native
