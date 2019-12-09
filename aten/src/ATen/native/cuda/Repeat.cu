#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/Repeat.h>

__global__ static void compute_cuda_kernel(int64_t *repeat_ptr, int64_t *cumsum_ptr, int64_t *result_ptr, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < size; i += stride) {
        int64_t end = cumsum_ptr[i];
        int64_t repeat = repeat_ptr[i];
        int64_t start = end - repeat;
        for(int64_t j = start; j < end; j++) {
            result_ptr[j] = i;
        }
    }
}

static void compute_cuda(int64_t *repeat_ptr, int64_t *cumsum_ptr, int64_t *result_ptr, int64_t size) {
    int64_t block = 512;
    int64_t grid = std::min<int64_t>((size + block - 1) / block, 2048L);
    compute_cuda_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(repeat_ptr, cumsum_ptr, result_ptr, size);
}

namespace at { namespace native {

Tensor repeat_interleave_cuda(const Tensor &repeat) {
    return repeat_interleave_common<compute_cuda>(repeat);
}

}}
