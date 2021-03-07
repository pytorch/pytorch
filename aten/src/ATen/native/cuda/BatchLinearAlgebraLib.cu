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

#ifdef USE_CUSOLVER

namespace at {
namespace native {

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

}} // namespace at::native

#endif  // USE_CUSOLVER
