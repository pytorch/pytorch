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
static void apply_lu_factor_batched_cublas(const Tensor& A, const Tensor& pivots, const Tensor& infos, bool get_pivots) {
#ifndef CUDART_VERSION
  TORCH_CHECK(false, "linalg.lu_factor: cuBLAS backend for linalg.lu_factor is not available.")
#else
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
#endif
}

void lu_factor_batched_cublas(const Tensor& A, const Tensor& pivots, const Tensor& infos, bool get_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "lu_factor_cublas", [&]{
    apply_lu_factor_batched_cublas<scalar_t>(A, pivots, infos, get_pivots);
  });
}

template <typename scalar_t>
static void apply_lu_solve_batched_cublas(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
#ifndef CUDART_VERSION
  TORCH_CHECK(false, "linalg.lu_solve: cuBLAS backend for linalg.lu_solve is not available.")
#else
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
#endif
}

void lu_solve_batched_cublas(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType trans) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_cublas", [&]{
    apply_lu_solve_batched_cublas<scalar_t>(LU, pivots, B, trans);
  });
}

namespace {

template <typename scalar_t>
void apply_ldl_factor_cusolver(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& info,
    bool upper) {
#ifndef USE_CUSOLVER
  TORCH_CHECK(
      false,
      "Calling torch.linalg.ldl_factor on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER support.");
#else
  auto batch_size = batchCount(A);
  auto n = cuda_int_cast(A.size(-2), "A.size(-2)");
  auto lda = cuda_int_cast(A.stride(-1), "A.stride(-1)");
  auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  auto a_data = A.data_ptr<scalar_t>();
  auto pivots_data = pivots.data_ptr<int>();
  auto info_data = info.data_ptr<int>();

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();

  int lwork = 0;
  at::cuda::solver::sytrf_bufferSize(handle, n, a_data, lda, &lwork);
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto work = allocator.allocate(sizeof(scalar_t) * lwork);

  for (const auto i : c10::irange(batch_size)) {
    auto* a_working_ptr = &a_data[i * a_stride];
    auto* pivots_working_ptr = &pivots_data[i * pivots_stride];
    auto* info_working_ptr = &info_data[i];
    at::cuda::solver::sytrf(
        handle,
        uplo,
        n,
        a_working_ptr,
        lda,
        pivots_working_ptr,
        reinterpret_cast<scalar_t*>(work.get()),
        lwork,
        info_working_ptr);
  }
#endif
}

template <typename scalar_t>
void apply_ldl_solve_cusolver(
    const Tensor& A,
    const Tensor& pivots,
    const Tensor& B,
    bool upper) {
#if !(defined(CUDART_VERSION) && defined(CUSOLVER_VERSION) && \
    CUSOLVER_VERSION >= 11102)
  TORCH_CHECK(
      false,
      "Calling torch.linalg.ldl_solve on a CUDA tensor requires compiling ",
      "PyTorch with cuSOLVER. Please use PyTorch built with cuSOLVER 11.1.2+ (CUDA 11.3.1+) support.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(A) > 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(pivots.unsqueeze(-1)) > 0);
  auto batch_size = batchCount(B);
  auto n = A.size(-2);
  auto nrhs = B.size(-1);
  auto lda = A.stride(-1);
  auto ldb = B.stride(-1);
  auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  auto a_stride = A.dim() > 2 ? A.stride(-3) : 0;
  auto b_stride = B.dim() > 2 ? B.stride(-3) : 0;
  auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;

  auto a_data = A.data_ptr<scalar_t>();
  auto b_data = B.data_ptr<scalar_t>();

  auto pivots_ = pivots.to(kLong);
  auto pivots_data = pivots_.data_ptr<int64_t>();

  // needed to run ldl_solve tests in parallel
  // see https://github.com/pytorch/pytorch/issues/82894 for examples of failures
  c10::cuda::device_synchronize();
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto datatype = at::cuda::solver::get_cusolver_datatype<scalar_t>();
  size_t worksize_device = 0;
  size_t worksize_host = 0;

  TORCH_CUSOLVER_CHECK(cusolverDnXsytrs_bufferSize(
      handle,
      uplo,
      n,
      nrhs,
      datatype,
      a_data,
      lda,
      pivots_data,
      datatype,
      b_data,
      ldb,
      &worksize_device,
      &worksize_host));

  // allocate workspace storage
  auto& device_allocator = *at::cuda::getCUDADeviceAllocator();
  auto workdata_device = device_allocator.allocate(worksize_device);
  void* workdata_device_ptr = workdata_device.get();

  auto& host_allocator = *at::getCPUAllocator();
  auto workdata_host = host_allocator.allocate(worksize_host);
  void* workdata_host_ptr = workdata_host.get();

  Tensor info = at::zeros({}, A.options().dtype(at::kInt));
  for (const auto i : c10::irange(batch_size)) {
    auto* a_working_ptr = &a_data[i * a_stride];
    auto* b_working_ptr = &b_data[i * b_stride];
    auto* pivots_working_ptr = &pivots_data[i * pivots_stride];
    TORCH_CUSOLVER_CHECK(cusolverDnXsytrs(
        handle,
        uplo,
        n,
        nrhs,
        datatype,
        a_working_ptr,
        lda,
        pivots_working_ptr,
        datatype,
        b_working_ptr,
        ldb,
        workdata_device_ptr,
        worksize_device,
        workdata_host_ptr,
        worksize_host,
        info.data_ptr<int>()));
  }

  // info from sytrs only reports if the i-th parameter is wrong
  // so we don't need to check it all the time
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);
#endif
}

} // anonymous namespace

void ldl_factor_cusolver(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian) {
  if (LD.is_complex()) {
    TORCH_CHECK(
        !hermitian,
        "torch.linalg.ldl_factor: complex tensors with hermitian=True flag are not supported with cuSOLVER backend. ",
        "Currently preferred backend is ",
        at::globalContext().linalgPreferredBackend(),
        ", please set 'default' or 'magma' backend with torch.backends.cuda.preferred_linalg_library");
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_factor_looped_cusolver", [&] {
        apply_ldl_factor_cusolver<scalar_t>(LD, pivots, info, upper);
      });
}

void ldl_solve_cusolver(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LD.scalar_type(), "ldl_solve_looped_cusolver", [&] {
        apply_ldl_solve_cusolver<scalar_t>(LD, pivots, B, upper);
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

// call cusolver gesvd function to calculate svd
template<typename scalar_t>
inline static void apply_svd_cusolver_gesvd(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
  const Tensor& infos, bool full_matrices, bool compute_uv,
  const bool calculate_all_batches,
  const std::vector<int64_t>& batches
) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  auto A_data = A.data_ptr<scalar_t>();
  auto S_data = S.data_ptr<value_t>();
  auto A_stride = matrixStride(A);
  auto S_stride = S.size(-1);

  int m = cuda_int_cast(A.size(-2), "m");
  int n = cuda_int_cast(A.size(-1), "n");
  auto k = std::min(m, n);
  int lda = std::max<int>(1, m);
  int ldvh = std::max<int>(1, n);

  TORCH_INTERNAL_ASSERT(m >= n, "cusolver gesvd only supports matrix with sizes m >= n");

  char job = compute_uv ? (full_matrices ? 'A' : 'S') : 'N';
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();

  int lwork = -1;
  at::cuda::solver::gesvd_buffersize<scalar_t>(handle, m, n, &lwork);
  TORCH_INTERNAL_ASSERT(lwork >= 0, "gesvd_buffersize failed to get needed buffer size, got lwork = ", lwork);

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  const auto dataPtr_work = allocator.allocate(sizeof(scalar_t)*lwork);
  const auto dataPtr_rwork = allocator.allocate(sizeof(value_t)*std::min(m, n));

  // nb. We can do this .view() because V is a batch of F-contig matrices
  const auto V_view = compute_uv ? V.view({-1, n, V.size(-1)})
                                 : Tensor{};
  // V is F-contig. Since this function computes Vh, we need an auxiliary F-conj-transposed matrix to hold Vh
  const auto Vh_workspace = compute_uv ?  at::empty({n, full_matrices ? n : k},
                                              A.options().memory_format(at::MemoryFormat::Contiguous)).conj()
                                       : Tensor{};
  const auto Vh_ptr = compute_uv ? Vh_workspace.data_ptr<scalar_t>()
                                 : nullptr;

  const auto U_stride = compute_uv ? matrixStride(U) : 0;
  const auto U_ptr = compute_uv ? U.data_ptr<scalar_t>() : nullptr;

  int batchsize = calculate_all_batches ? cuda_int_cast(batchCount(A), "batch size")
                                        : batches.size();


  for(int _i = 0; _i < batchsize; _i++){
    int i = calculate_all_batches ? _i : batches[_i];

    at::cuda::solver::gesvd<scalar_t>(
      handle, job, job, m, n,
      A_data + i * A_stride,
      lda,
      S_data + i * S_stride,
      compute_uv ? U_ptr + i * U_stride : nullptr,
      lda,
      compute_uv ? Vh_ptr : nullptr,
      ldvh,
      reinterpret_cast<scalar_t*>(dataPtr_work.get()),
      lwork,
      reinterpret_cast<value_t*>(dataPtr_rwork.get()),
      infos.data_ptr<int>() + i
    );

    if (compute_uv) {
      V_view[i].copy_(Vh_workspace);
    }
  }
}

// We'll copy A inside svd_cusolver_gesvd
inline static void svd_cusolver_gesvd(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
  const Tensor& infos, bool full_matrices, bool compute_uv,
  const bool calculate_all_batches = true,
  const std::vector<int64_t>& batches = {}
) {
  // We need to pass a copy of A, as it will be overwritten
  // gesvd just knows how to handle m >= n, so in the other case we need to transpose A
  const auto not_A_H = A.size(-2) >= A.size(-1);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda_gesvd", [&] {
    apply_svd_cusolver_gesvd<scalar_t>(cloneBatchedColumnMajor(not_A_H ? A : A.mH()),
                                       not_A_H ? U : V,
                                       S,
                                       not_A_H ? V : U,
                                       infos,
                                       full_matrices, compute_uv, calculate_all_batches, batches);
  });
}

// call cusolver gesvdj function to calculate svd
template<typename scalar_t>
inline static void apply_svd_cusolver_gesvdj(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
  const Tensor& infos, bool full_matrices, bool compute_uv) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  int m = cuda_int_cast(A.size(-2), "m");
  int n = cuda_int_cast(A.size(-1), "n");
  int k = std::min(m, n);

  // Need to pass allocated memory to the function, otherwise it fails
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr_U = !compute_uv ? allocator.allocate(sizeof(scalar_t)* m * k) : c10::DataPtr{};
  auto dataPtr_V = !compute_uv ? allocator.allocate(sizeof(scalar_t)* n * k) : c10::DataPtr{};

  auto A_data = A.data_ptr<scalar_t>();
  auto U_data = compute_uv ? U.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_U.get());
  auto S_data = S.data_ptr<value_t>();
  auto V_data = compute_uv ? V.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_V.get());
  auto A_stride = matrixStride(A);
  auto U_stride = compute_uv ? matrixStride(U) : 0;
  auto S_stride = S.size(-1);
  auto V_stride = compute_uv ? matrixStride(V) : 0;

  int batchsize = cuda_int_cast(batchCount(A), "batch size");
  int lda = A.stride(-1);
  int ldu = compute_uv ? U.stride(-1) : m;
  int ldv = compute_uv ? V.stride(-1) : n;

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  int econ = full_matrices ? 0 : 1;

  // gesvdj_params controls the numerical accuracy of cusolver gesvdj iterations on GPU
  gesvdjInfo_t gesvdj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

  // Todo: expose the following two parameters to users
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, std::numeric_limits<scalar_t>::epsilon()));
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, cusolver_gesvdj_max_sweeps));

  int lwork = -1;
  at::cuda::solver::gesvdj_buffersize<scalar_t>(
    handle, jobz, econ, m, n, A_data, lda, S_data, U_data, ldu, V_data, ldv, &lwork, gesvdj_params);
  TORCH_INTERNAL_ASSERT(lwork >= 0, "gesvdj_buffersize failed to get needed buffer size, got lwork = ", lwork);

  auto dataPtr = allocator.allocate(sizeof(scalar_t)*lwork);

  for(int i = 0; i < batchsize; i++){
    at::cuda::solver::gesvdj<scalar_t>(
      handle, jobz, econ, m, n,
      A_data + i * A_stride,
      lda,
      S_data + i * S_stride,
      U_data + i * U_stride,
      ldu,
      V_data + i * V_stride,
      ldv,
      reinterpret_cast<scalar_t*>(dataPtr.get()),
      lwork,
      infos.data_ptr<int>() + i,
      gesvdj_params
    );

    // // The following code can be used to check or report the gesvdj residual.
    // // Note: this will introduce a device-host sync and may negatively affect the performance
    // double residual = 0;
    // TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjGetResidual(handle, gesvdj_params, &residual));
    // printf("gesvdj residual = %.6e\n", residual);
  }

  TORCH_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

// wrapper around apply_svd_cusolver_gesvdj that handles dtype dispatch
// note that gesvdj returns V, which is what we want
// Need to pass a copy of A, since A will be rewritten inside the function call
inline static void svd_cusolver_gesvdj(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V, const Tensor& infos, bool full_matrices, bool compute_uv) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda_gesvdj", [&] {
    apply_svd_cusolver_gesvdj<scalar_t>(A, U, S, V, infos, full_matrices, compute_uv);
  });
}

// call cusolver gesvdj batched function to calculate svd
template<typename scalar_t>
inline static void apply_svd_cusolver_gesvdjBatched(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
  const Tensor& infos, bool compute_uv
) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  int m = cuda_int_cast(A.size(-2), "m");
  int n = cuda_int_cast(A.size(-1), "n");
  int k = std::min(m, n);
  int batchsize = cuda_int_cast(batchCount(A), "batch size");

  // Need to pass allocated memory to the function, otherwise it fails
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr_U = !compute_uv ? allocator.allocate(sizeof(scalar_t) * batchsize * m * k) : c10::DataPtr{};
  auto dataPtr_V = !compute_uv ? allocator.allocate(sizeof(scalar_t) * batchsize * n * k) : c10::DataPtr{};

  auto A_data = A.data_ptr<scalar_t>();
  auto U_data = compute_uv ? U.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_U.get());
  auto S_data = S.data_ptr<value_t>();
  auto V_data = compute_uv ? V.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_V.get());

  int lda = A.stride(-1);
  int ldu = compute_uv ? U.stride(-1) : m;
  int ldv = compute_uv ? V.stride(-1) : n;

  TORCH_INTERNAL_ASSERT(m <= 32 && n <= 32, "gesvdjBatched requires both matrix dimensions not greater than 32, but got "
                        "m = ", m, " n = ", n);

  // gesvdj_params controls the numerical accuracy of cusolver gesvdj iterations on GPU
  gesvdjInfo_t gesvdj_params;
  TORCH_CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

  // Todo: expose the following two parameters to users
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, std::numeric_limits<scalar_t>::epsilon()));
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, cusolver_gesvdj_max_sweeps));
  TORCH_CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(gesvdj_params, 1));

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  at::cuda::solver::gesvdjBatched<scalar_t>(
    handle, jobz, m, n, A_data, lda, S_data, U_data, ldu, V_data, ldv,
    infos.data_ptr<int>(), gesvdj_params, batchsize
  );

  TORCH_CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

inline static void svd_cusolver_gesvdjBatched(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V, const Tensor& infos, bool compute_uv) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda_gesvdjBatched", [&] {
    apply_svd_cusolver_gesvdjBatched<scalar_t>(A, U, S, V, infos, compute_uv);
  });
}

template<typename scalar_t>
inline static void apply_svd_cusolver_gesvdaStridedBatched(const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
    const Tensor& infos, bool full_matrices, bool compute_uv) {
  using value_t = typename c10::scalar_value_type<scalar_t>::type;
  int m = cuda_int_cast(A.size(-2), "m");
  int n = cuda_int_cast(A.size(-1), "n");
  TORCH_INTERNAL_ASSERT(m >= n, "cusolver gesvdaStridedBatched requires m >= n");
  int batchsize = cuda_int_cast(batchCount(A), "batch size");

  int lda = A.stride(-1);
  int ldu = compute_uv ? U.stride(-1) : m;
  int ldv = compute_uv ? V.stride(-1) : n;

  auto A_stride = matrixStride(A);
  auto S_stride = S.size(-1);
  auto rank = S_stride; // number of singular values
  auto U_stride = compute_uv ? matrixStride(U) : ldu * rank;  // The strides for "empty matrices" are needed to satisfy cusolver.
  auto V_stride = compute_uv ? matrixStride(V) : ldv * rank;

  // Need to pass allocated memory to the function, otherwise it fails
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto dataPtr_U = !compute_uv ? allocator.allocate(sizeof(scalar_t) * batchsize * m * n) : c10::DataPtr{};
  auto dataPtr_V = !compute_uv ? allocator.allocate(sizeof(scalar_t) * batchsize * n * n) : c10::DataPtr{};

  auto A_data = A.data_ptr<scalar_t>();
  auto U_data = compute_uv ? U.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_U.get());
  auto S_data = S.data_ptr<value_t>();
  auto V_data = compute_uv ? V.data_ptr<scalar_t>() : reinterpret_cast<scalar_t*>(dataPtr_V.get());

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  auto jobz = compute_uv ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

  int lwork = -1;
  at::cuda::solver::gesvdaStridedBatched_buffersize<scalar_t>(
    handle, jobz, rank, m, n, A_data, lda, A_stride, S_data, S_stride, U_data, ldu, U_stride, V_data, ldv, V_stride,
    &lwork, batchsize);
  TORCH_INTERNAL_ASSERT(lwork >= 0, "gesvdaStridedBatched_buffersize failed to get needed buffer size, got lwork = ", lwork);
  auto workspace = allocator.allocate(sizeof(scalar_t)*lwork);

  // The residual Frobenius norm is always returned in double.
  // cuSOLVER remark: if the user is confident on the accuracy of singular values and singular vectors,
  //   for example, certain conditions hold (required singular value is far from zero),
  //   then the performance can be improved by passing a null pointer to h_RnrmF, i.e. no computation of residual norm.
  // Comment: calculation of Frobenius norm is expensive and doesn't affect accuracy of the result

  at::cuda::solver::gesvdaStridedBatched<scalar_t>(
    handle, jobz, rank, m, n, A_data, lda, A_stride, S_data, S_stride, U_data, ldu, U_stride, V_data, ldv, V_stride,
    reinterpret_cast<scalar_t*>(workspace.get()),
    lwork, infos.data_ptr<int>(),
    nullptr,  // cuSOLVER h_RnrmF is not calculated: reinterpret_cast<double*>(residual_frobenius_norm.get()),
    batchsize);
}

// We'll copy A inside svd_cusolver_gesvdaStridedBatched
inline static void svd_cusolver_gesvdaStridedBatched(
    const Tensor& A, const Tensor& U, const Tensor& S, const Tensor& V,
    const Tensor& infos, bool full_matrices, bool compute_uv) {
  // We need to pass a copy of A, as it will be overwritten
  // gesvdaStridedBatched just knows how to handle m >= n, so in the other case we need to transpose A
  const auto not_A_H = A.size(-2) >= A.size(-1);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(A.scalar_type(), "svd_cuda_gesvdaStridedBatched", [&] {
    apply_svd_cusolver_gesvdaStridedBatched<scalar_t>(
      cloneBatchedColumnMajor(not_A_H ? A : A.mH()),
      not_A_H ? U : V,
      S,
      not_A_H ? V : U,
      infos, full_matrices, compute_uv);
  });
}

// Check convergence of gesvdj/gesvdjBatched/gesvdaStridedBatched results.
// If not converged, return a vector that contains indices of the non-converging batches.
// If the returned vector is empty, all the matrices are converged.
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

    // In our use case, gesvdj, gesvdjBatched, and gesvdaStridedBatched have the same notations for `info`.
    if (info_for_batch_i == non_converging_info) res.push_back(i);

    // However, it is not the same for gesvd, though we don't use this function to check gesvd convergence either.
    // If it's implemented some day in the future, this needs to be handled carefully.
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

// This function returns V, not V^H.
void svd_cusolver(const Tensor& A,
                  const bool full_matrices,
                  const bool compute_uv,
                  const c10::optional<c10::string_view>& driver,
                  const Tensor& U,
                  const Tensor& S,
                  const Tensor& V,
                  const Tensor& info) {
  // Here U and V are F-contig whenever they are defined (i.e. whenever compute_uv=true)
  const auto batch_size = batchCount(A);
  const auto m = A.size(-2);
  const auto n = A.size(-1);
  const auto k = std::min(m, n);

  static const char* check_svd_doc = "Check doc at https://pytorch.org/docs/stable/generated/torch.linalg.svd.html";

  // The default heuristic is to use gesvdj driver
  const auto driver_v = driver.value_or("gesvdj");

  if (driver_v == "gesvd") {
    svd_cusolver_gesvd(A, U, S, V, info, full_matrices, compute_uv);
  } else if (driver_v == "gesvdj") {
    if (m <= 32 && n <= 32 && batch_size > 1 && (full_matrices || m == n)) {
      svd_cusolver_gesvdjBatched(cloneBatchedColumnMajor(A), U, S, V, info, compute_uv);
    } else {
      // gesvdj driver may be numerically unstable for large sized matrix
      svd_cusolver_gesvdj(cloneBatchedColumnMajor(A), U, S, V, info, full_matrices, compute_uv);
    }
  } else if (driver_v == "gesvda") {
    // cuSOLVER: gesvdaStridedBatched is preferred for "tall skinny" (m > n) matrices
    // We do a transpose here to make it also work for (m < n) matrices.
    svd_cusolver_gesvdaStridedBatched(A, U, S, V, info, full_matrices, compute_uv);
  } else {
    TORCH_CHECK(false, "torch.linalg.svd: unknown svd driver ", driver_v, " in svd_cusolver computation. ", check_svd_doc);
  }

  // Need convergence check
  if (driver_v != "gesvd") {
    // A device-host sync will be performed.
    // Todo: implement the svd_ex variant to not check result convergence, thus removing the device-host sync
    const auto svd_non_converging_batches = _check_gesvdj_convergence(info, k + 1);

    if (!svd_non_converging_batches.empty()) {
      TORCH_WARN_ONCE("torch.linalg.svd: During SVD computation with the selected cusolver driver, ",
                      _format_non_converging_batches(svd_non_converging_batches),
                      " failed to converge. ",
                      (driver.has_value()
                        ?  "It is recommended to redo this SVD with another driver. "
                        : "A more accurate method will be used to compute the SVD as a fallback. "),
                      check_svd_doc);

      // We'll do the fallback if user doesn't specify a driver and the default heuristic doesn't converge well.
      // However, if user manually chooses a driver, should we just do a warning or a hard crash?
      if (!driver.has_value()) {
        svd_cusolver_gesvd(A, U, S, V, info, full_matrices, compute_uv, false, svd_non_converging_batches);
      }
    }
  }

  // `info` will be checked later at `TORCH_IMPL_FUNC(_linalg_svd_out)` function.
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
void lu_factor_looped_cusolver(const Tensor& self, const Tensor& pivots, const Tensor& infos, bool get_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    self.scalar_type(),
    "lu_factor_cusolver",
    [&self,
     &pivots,
     &infos,
     get_pivots]() {
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
  // https://github.com/pytorch/pytorch/issues/53879#issuecomment-830633572
  if (!get_pivots) {
    // nan_to_num does not work for complex inputs
    // https://github.com/pytorch/pytorch/issues/59247
    if (self.is_complex()) {
      self.copy_(at::where(self.eq(self), self,  at::scalar_tensor(0., self.options())));
    } else {
      at::nan_to_num_(const_cast<Tensor&>(self), 0, std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());
    }
  }
}

void lu_solve_looped_cusolver(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_cusolver", [&] {
    const auto trans = to_cublas(transpose);
    int n = cuda_int_cast(LU.size(-2), "n");
    int nrhs = cuda_int_cast(B.size(-1), "nrhs");
    auto batch_size = batchCount(B);
    auto info = at::zeros({1}, LU.options().dtype(kInt));
    auto info_data = info.data_ptr<int>();
    auto b_data = B.data_ptr<scalar_t>();
    auto lu_data = LU.data_ptr<scalar_t>();
    auto pivots_data = pivots.data_ptr<int>();
    auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;
    auto lu_stride = LU.dim() > 2 ? LU.stride(-3) : 0;
    auto b_stride = matrixStride(B);
    int leading_dimension = cuda_int_cast(std::max<int>(1, n), "leading_dimension");

    // lu and pivots tensors can be broadcast to b
    // here we construct a helper indexing tensor to linearly index into lu and pivots
    IntArrayRef lu_batch_shape(LU.sizes().data(), LU.dim() - 2);
    IntArrayRef b_batch_shape(B.sizes().data(), B.dim() - 2);
    BroadcastLinearIndices lu_index(
        batchCount(LU), lu_batch_shape, b_batch_shape);

    auto handle = at::cuda::getCurrentCUDASolverDnHandle();
    for (auto batch = decltype(batch_size){0}; batch < batch_size; ++batch) {
      int64_t lu_index_i = lu_index(batch);
      at::cuda::solver::getrs<scalar_t>(
        handle,
        n,
        nrhs,
        lu_data + lu_index_i * lu_stride,
        leading_dimension,
        pivots_data + lu_index_i * pivots_stride,
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
