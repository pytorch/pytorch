#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/cuda/BatchLinearAlgebraLib.h>

#ifdef USE_CUSOLVER

namespace at {
namespace native {

template<>
void cusolver_LU<double>(int m, int n, double* dA, int ldda, int* ipiv, int* info) {
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  Tensor buffer = at::empty({lwork}, at::device(at::kCUDA).dtype(at::kDouble));
  Tensor devInfo = at::empty({1}, at::device(at::kCUDA).dtype(at::kInt));
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrf(handle, m, n, dA, ldda, buffer.data_ptr<double>(), ipiv, devInfo.data_ptr<int>()));
  *info = devInfo.item<int>();
}

template<>
void cusolver_LU<float>(int m, int n, float* dA, int ldda, int* ipiv, int* info) {
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  int lwork;
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, m, n, dA, ldda, &lwork));
  Tensor buffer = at::empty({lwork}, at::device(at::kCUDA).dtype(at::kFloat));
  Tensor devInfo = at::empty({1}, at::device(at::kCUDA).dtype(at::kInt));
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrf(handle, m, n, dA, ldda, buffer.data_ptr<float>(), ipiv, devInfo.data_ptr<int>()));
  *info = devInfo.item<int>();
}

template<>
void cusolver_getrs<double>(int n, int nrhs, double* dA, int lda, int* ipiv, double* ret, int ldb, int* info) {
  Tensor dinfo = at::empty({1}, at::device(at::kCUDA).dtype(at::kInt));
  TORCH_CUSOLVER_CHECK(cusolverDnDgetrs(at::cuda::getCurrentCUDASolverDnHandle(), CUBLAS_OP_N, n, nrhs, dA, lda, ipiv, ret, ldb, dinfo.data_ptr<int>()));
  *info = dinfo.item<int>();
}

template<>
void cusolver_getrs<float>(int n, int nrhs, float* dA, int lda, int* ipiv, float* ret, int ldb, int* info) {
  Tensor dinfo = at::empty({1}, at::device(at::kCUDA).dtype(at::kInt));
  TORCH_CUSOLVER_CHECK(cusolverDnSgetrs(at::cuda::getCurrentCUDASolverDnHandle(), CUBLAS_OP_N, n, nrhs, dA, lda, ipiv, ret, ldb, dinfo.data_ptr<int>()));
  *info = dinfo.item<int>();
}

template<>
void cublas_LU_batched<double>(
    int _m, int n, double** dA_array, int ldda,
    int* ipiv_array, int* info_array, int batchsize){
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasDgetrfBatched(handle, n, dA_array, ldda, ipiv_array, info_array, batchsize));
}

template<>
void cublas_LU_batched<float>(
    int _m, int n, float** dA_array, int ldda,
    int* ipiv_array, int* info_array, int batchsize){
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSgetrfBatched(handle, n, dA_array, ldda, ipiv_array, info_array, batchsize));
}

template<>
void cublas_getri_batched<double>(
    int _m, int n, double** dA_array, int ldda,
    int* ipiv_array, int* info_array, int batchsize, double** dC_array){
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasDgetriBatched(handle, n, dA_array, ldda, ipiv_array, dC_array, n, info_array, batchsize));
}

template<>
void cublas_getri_batched<float>(
    int _m, int n, float** dA_array, int ldda,
    int* ipiv_array, int* info_array, int batchsize, float** dC_array){
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSgetriBatched(handle, n, dA_array, ldda, ipiv_array, dC_array, n, info_array, batchsize));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ inverse ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename scalar_t>
static void apply_batched_inverse_lib(Tensor& self, Tensor& self_inv, Tensor& infos) {
  const int batch_size = cuda_int_cast(batchCount(self), "batchCount");
  const int n = cuda_int_cast(self.size(-2), "self.size(-2)");

  const bool use_loop_launch_ = use_loop_launch(batch_size, n);

  if (use_loop_launch_) {
    self_inv = at::eye({n}, self.options()).expand_as(self).clone();
    self_inv.unsafeGetTensorImpl()->set_stride(self.dim()-2, 1); // These two lines set self_inv to column-major
    self_inv.unsafeGetTensorImpl()->set_stride(self.dim()-1, n);
  }

  auto self_data = self.data_ptr<scalar_t>();
  auto self_mat_stride = matrixStride(self);
  auto self_inv_data = self_inv.data_ptr<scalar_t>();
  auto self_inv_mat_stride = matrixStride(self_inv);

  scalar_t** self_array;
  scalar_t** self_inv_array;

  ALLOCATE_ARRAY(self_array, scalar_t*, batch_size);
  ALLOCATE_ARRAY(self_inv_array, scalar_t*, batch_size);

  // Set up the created arrays
  for (int64_t i = 0; i < batch_size; i++) {
    self_array[i] = &self_data[i * self_mat_stride];
    self_inv_array[i] = &self_inv_data[i * self_inv_mat_stride];
  }

  Tensor ipiv_array = at::zeros({batch_size * n}, self.options().dtype(at::kInt));

  if (use_loop_launch_) {
    infos = infos.cpu();
    int* p_ipiv_array = ipiv_array.data_ptr<int>();
    int* p_infos = infos.data_ptr<int>();
    for (int64_t i = 0; i < batch_size; i++) {
      _apply_single_inverse_helper<scalar_t>(self_array[i], self_inv_array[i], p_ipiv_array + i * n, p_infos + i, n);
    }
  } else {
    cublas_LU_batched<scalar_t>(n, n, self_array, n,
      ipiv_array.data_ptr<int>(), infos.data_ptr<int>(), batch_size);

    cublas_getri_batched<scalar_t>(n, n, self_array, n,
      ipiv_array.data_ptr<int>(), infos.data_ptr<int>(), batch_size, self_inv_array);
  }
}

template <typename scalar_t>
static void apply_single_inverse_lib(const Tensor& self, Tensor& self_inv, int64_t& info) {
  int n = cuda_int_cast(self.size(-2), "self.size(-2)");

  Tensor ipiv = at::empty({n}, self.options().dtype(at::kInt));
  self_inv = at::eye(n, self.options());
  self_inv.unsafeGetTensorImpl()->set_stride(0, 1); // These two lines set self_inv to column-major
  self_inv.unsafeGetTensorImpl()->set_stride(1, n);

  int info_tmp = 0;
  _apply_single_inverse_helper<scalar_t>(self.data_ptr<scalar_t>(), self_inv.data_ptr<scalar_t>(), ipiv.data_ptr<int>(), &info_tmp, n);
  info = info_tmp;
}

template <typename scalar_t>
inline static void _apply_single_inverse_helper(scalar_t* self_ptr, scalar_t* self_inv_ptr, int* ipiv_ptr, int* info_ptr, int n) {
  // self_inv_ptr should already be an identity matrix
  cusolver_LU<scalar_t>(n, n, self_ptr, n, ipiv_ptr, info_ptr);
  cusolver_getrs<scalar_t>(n, n, self_ptr, n, ipiv_ptr, self_inv_ptr, n, info_ptr);
}

Tensor _inverse_helper_cuda_lib(const Tensor& self) {
  auto self_working_copy = cloneBatchedColumnMajor(self);
  if (self.dim() > 2) {
    auto self_inv_working_copy = at::empty_like(self_working_copy);
    Tensor infos = at::zeros({batchCount(self)}, self.options().dtype(kInt));
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "inverse_cuda", [&]{
      apply_batched_inverse_lib<scalar_t>(
        self_working_copy, self_inv_working_copy, infos);
    });
    batchCheckErrors(infos, "inverse_cuda");
    return self_inv_working_copy;
  } else {
    Tensor self_inv;
    int64_t info = 0;
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "inverse_cuda", [&]{
      apply_single_inverse_lib<scalar_t>(self_working_copy, self_inv, info);
    });
    singleCheckErrors(info, "inverse_cuda");
    return self_inv;
  }
}

}} // namespace at::native

#endif  // USE_CUSOLVER

#undef ALLOCATE_ARRAY