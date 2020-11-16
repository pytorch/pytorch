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
inline static void _apply_single_inverse_helper(scalar_t* self_ptr, scalar_t* self_inv_ptr, int* ipiv_ptr, int* info_ptr, int n) {
  // self_inv_ptr should already be an identity matrix

  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  at::cuda::solver::getrf<scalar_t>(handle, n, n, self_ptr, n, ipiv_ptr, info_ptr);
  at::cuda::solver::getrs<scalar_t>(handle, n, n, self_ptr, n, ipiv_ptr, self_inv_ptr, n, info_ptr + 1);
}

template <typename scalar_t>
static void apply_batched_inverse_lib(Tensor& self, Tensor& self_inv, Tensor& infos) {
  const int batch_size = cuda_int_cast(batchCount(self), "batchCount");
  const int n = cuda_int_cast(self.size(-2), "self.size(-2)");

  auto self_data = self.data_ptr<scalar_t>();
  auto self_mat_stride = matrixStride(self);
  auto self_inv_data = self_inv.data_ptr<scalar_t>();
  auto self_inv_mat_stride = matrixStride(self_inv);

  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();

  if (use_loop_launch(batch_size, n)) {
    int* p_infos = infos.data_ptr<int>();
    auto main_stream = at::cuda::getCurrentCUDAStream();

    for (int64_t i = 0; i < batch_size; i++) {
      auto stream = at::cuda::getStreamFromPool();
      at::cuda::CUDAStreamGuard guard(stream);

      at::cuda::CUDAEvent can_start;
      can_start.record(main_stream);
      can_start.block(main_stream);

      auto dataPtr = allocator.allocate(sizeof(int) * n);
      int* pivot = reinterpret_cast<int*>(dataPtr.get());
      _apply_single_inverse_helper<scalar_t>(
        &self_data[i * self_mat_stride], &self_inv_data[i * self_inv_mat_stride], pivot, p_infos + i * 2, n);

      at::cuda::CUDAEvent finished;
      finished.record(stream);
      finished.block(main_stream);
    }
  } else {
    // cublas batched kernels require input be "device array of device pointers"
    Tensor self_array = at::arange(
      reinterpret_cast<long>(self_data),
      reinterpret_cast<long>(&self_data[(batch_size-1) * self_mat_stride]) + 1,
      static_cast<long>(self_mat_stride * sizeof(scalar_t)), self.options().dtype(at::kLong));
    Tensor self_inv_array = at::arange(
      reinterpret_cast<long>(self_inv_data),
      reinterpret_cast<long>(&self_inv_data[(batch_size-1) * self_inv_mat_stride]) + 1,
      static_cast<long>(self_inv_mat_stride * sizeof(scalar_t)), self.options().dtype(at::kLong));

    auto dataPtr = allocator.allocate(sizeof(int)*batch_size*n);
    int* ipiv_array = reinterpret_cast<int*>(dataPtr.get());

    at::cuda::blas::getrfBatched<scalar_t>(n, reinterpret_cast<scalar_t**>(self_array.data_ptr()), n,
      ipiv_array, infos.data_ptr<int>(), batch_size);

    at::cuda::blas::getriBatched<scalar_t>(n, reinterpret_cast<scalar_t**>(self_array.data_ptr()), n,
      ipiv_array, infos.data_ptr<int>(), batch_size, reinterpret_cast<scalar_t**>(self_inv_array.data_ptr()));
  }
}

template <typename scalar_t>
static void apply_single_inverse_lib(const Tensor& self, Tensor& self_inv, Tensor& info) {
  int n = cuda_int_cast(self.size(-2), "self.size(-2)");

  Tensor ipiv = at::empty({n}, self.options().dtype(at::kInt));

  _apply_single_inverse_helper<scalar_t>(
    self.data_ptr<scalar_t>(), self_inv.data_ptr<scalar_t>(), ipiv.data_ptr<int>(), info.data_ptr<int>(), n);
}

Tensor _inverse_helper_cuda_lib(const Tensor& self) {
  Tensor self_working_copy = cloneBatchedColumnMajor(self);
  Tensor self_inv_working_copy = column_major_identity_matrix_like(self_working_copy);
  const int batch_size = cuda_int_cast(batchCount(self), "batchCount");

  if (self.dim() > 2 && batch_size > 1) {
    Tensor infos = at::zeros({batchCount(self) * 2}, self.options().dtype(kInt));
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "inverse_cuda", [&]{
      apply_batched_inverse_lib<scalar_t>(
        self_working_copy, self_inv_working_copy, infos);
    });
    batchCheckErrors(infos, "inverse_cuda", false, 2);
  } else {
    Tensor info = at::zeros({2}, self.options().dtype(at::kInt));
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "inverse_cuda", [&]{
      apply_single_inverse_lib<scalar_t>(self_working_copy, self_inv_working_copy, info);
    });
    batchCheckErrors(info, "inverse_cuda", false, 2);
  }

  return self_inv_working_copy;
}

}} // namespace at::native

#endif  // USE_CUSOLVER
