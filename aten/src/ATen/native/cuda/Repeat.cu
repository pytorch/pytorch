#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Repeat.h>

__global__ static void compute_cuda_kernel(int64_t *repeat_ptr, int64_t *cumsum_ptr, int64_t *result_ptr, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (blockDim.x * gridDim.x) / C10_WARP_SIZE;
    int warp_id = idx / C10_WARP_SIZE;
    int tid_in_warp = idx % C10_WARP_SIZE;
    for (int64_t i = warp_id; i < size; i += stride) {
        int64_t end = cumsum_ptr[i];
        int64_t repeat = repeat_ptr[i];
        int64_t start = end - repeat;
        for(int64_t j = start + tid_in_warp; j < end; j += C10_WARP_SIZE) {
            result_ptr[j] = i;
        }
    }
}

static void compute_cuda(int64_t *repeat_ptr, int64_t *cumsum_ptr, int64_t *result_ptr, int64_t size) {
    int64_t block = 512;
    int64_t warps_per_block = block / C10_WARP_SIZE;
    int64_t grid = std::min<int64_t>((size + warps_per_block - 1) / warps_per_block, 2048L);

    compute_cuda_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(repeat_ptr, cumsum_ptr, result_ptr, size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace at { namespace native {

Tensor repeat_interleave_cuda(const Tensor &repeat) {
    return repeat_interleave_common<compute_cuda>(repeat);
}

}}
