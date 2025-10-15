#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/ceil_div.h>

namespace at::native {
template <int Alignment, typename index_t>
__global__ void vectorized_gather_kernel(char * out, char * inp, index_t * idx, int num_ind, int64_t slice_size, int64_t ind_dim_size, int64_t inp_stride, int64_t out_stride, bool allow_neg_indices) {
    int64_t ind = idx[blockIdx.x];
    if (allow_neg_indices) {
        ind = (ind < 0) ? ind + ind_dim_size : ind;
    }
    CUDA_KERNEL_ASSERT(ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds");
    int32_t off = (blockDim.x * blockIdx.y + threadIdx.x) * Alignment; // off is guaranteed to be within int32 limits
    if (off >= slice_size) return;
    auto vec = at::native::memory::ld_vec<Alignment>(inp + ind * inp_stride + off);
    at::native::memory::st_vec<Alignment>(out + blockIdx.x * (int32_t)out_stride + off, vec);  // out offset is guaranteed to be within int32 limits
}



template <int64_t Alignment, typename index_t>
void vectorized_gather_kernel_launch(char * out, char * inp, index_t * idx, int num_ind,
                                     int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices){

  constexpr int64_t max_num_threads=256;
  auto num_threads = at::round_up(
      at::ceil_div(slice_size_in_bytes, Alignment),
      static_cast<int64_t>(C10_WARP_SIZE));
  dim3 grid = {static_cast<uint32_t>(num_ind), static_cast<uint32_t>(at::ceil_div(slice_size_in_bytes, max_num_threads * Alignment)), 1};
  auto block = std::min(max_num_threads, num_threads);
  vectorized_gather_kernel<Alignment, index_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(out, inp, idx, num_ind, slice_size_in_bytes,
  ind_dim_size, inp_stride_bytes, out_stride_bytes, allow_neg_indices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// explicit template instantiation
template void vectorized_gather_kernel_launch<16, int64_t>(char * out, char * inp, int64_t * idx, int num_ind, int64_t slice_size_in_bytes,
int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices);
template void vectorized_gather_kernel_launch<16, int32_t>(char * out, char * inp, int32_t * idx, int num_ind, int64_t slice_size_in_bytes,
int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes, bool allow_neg_indices);

}
