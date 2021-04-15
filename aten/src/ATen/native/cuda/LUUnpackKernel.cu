#include <ATen/native/LinearAlgebraUtils.h>

#include <ATen/native/cuda/Loops.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/CUDAContext.h>

namespace at {
namespace native {

namespace {

template <int n_threads, int n_elems_per_thread, typename func_t>
C10_LAUNCH_BOUNDS_2(n_threads, n_elems_per_thread)
__global__ void _elementwise_kernel(int total_n_elems, func_t f) {
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  int idx = total_work_block * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int i = 0; i < n_elems_per_thread; ++i) {
    if (idx < total_n_elems) {
      f(idx);
      idx += n_threads;
    }
  }
}

template <int n_threads, int n_elems_per_thread, typename func_t>
static void _launch_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
    total_n_elems >= 0 && total_n_elems <= std::numeric_limits<int32_t>::max()
  );

  dim3 block(n_threads);
  constexpr int total_work_block = n_threads * n_elems_per_thread;
  dim3 grid((total_n_elems + total_work_block - 1) / total_work_block);

  auto stream = at::cuda::getCurrentCUDAStream();
  _elementwise_kernel<n_threads, n_elems_per_thread, func_t>
    <<<grid, block, 0, stream>>>(total_n_elems, f);
  AT_CUDA_CHECK(cudaGetLastError());
}

void _unpack_pivots_internal_kernel(
  TensorIterator& iter,
  int64_t dim_size
) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unpack_pivots_internal_kernel(sub_iter, dim_size);
    }
    return;
  }

  auto offset_calculator = make_offset_calculator<2>(iter);

  char* unpacked_pivots_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ pivots_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  auto loop = [=]C10_DEVICE(int i) {
    auto offsets = offset_calculator.get(i);

    auto* unpacked_pivots_data = reinterpret_cast<int32_t*>(
      unpacked_pivots_ptr + offsets[0]);
    auto* __restrict__ pivots_data = reinterpret_cast<int32_t*>(
      pivots_ptr + offsets[1]);

    // QUESTION: can we mix 64bit offsets with 32bit Iterator indexing?
    for (int64_t i = 0; i < dim_size; ++i) {
      thrust::swap(
        unpacked_pivots_data[i],
        unpacked_pivots_data[pivots_data[i]]
      );
    }
  };

  _launch_kernel<num_threads, thread_work_size>(iter.numel(), loop);
}

void unpack_pivots_cuda_kernel(
  TensorIterator& iter,
  int64_t dim_size
) {
  _unpack_pivots_internal_kernel(iter, dim_size);
}

}

REGISTER_DISPATCH(unpack_pivots_stub, &unpack_pivots_cuda_kernel);

}} // namespace at::native
