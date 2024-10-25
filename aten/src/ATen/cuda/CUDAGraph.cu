#include <ATen/cuda/CUDAGraph.h>

namespace at::cuda {

// the best way to speed this up would be to minimize the number of
// global memory stalls.  We can pass three arrays: nodes, offsets,
// and pValues, to avoid filling in the other fields.  Right now,
// dynamic_graph_updater_kernel kernel takes about 5.0 microseconds on
// average, but dynamic_graph_updater_kernel2 takes about 2.8
// microseconds on average. There might be other potential speed ups,
// but this is pretty good.
template <size_t BLOCK_SIZE, bool VOLTA_OR_LATER>
__global__ void __launch_bounds__(BLOCK_SIZE) dynamic_graph_updater_kernel(
    __grid_constant__ const KernelUpdateSOA<VOLTA_OR_LATER> update_soa) {
  __shared__ const void* pointer_indirection[BLOCK_SIZE];
  __shared__ alignas(cudaGraphKernelNodeUpdate) char
      shared_updates_wrong_type[BLOCK_SIZE * sizeof(cudaGraphKernelNodeUpdate)];
  cudaGraphKernelNodeUpdate* shared_updates =
      (cudaGraphKernelNodeUpdate*)shared_updates_wrong_type;

  // for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i <
  // update_soa.num_updates; i += gridDim.x * blockDim.x) {
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < update_soa.num_updates) {
    pointer_indirection[threadIdx.x] = update_soa.new_pointers[i];
    shared_updates[threadIdx.x] = cudaGraphKernelNodeUpdate{
        .node = update_soa.device_nodes[i],
        .field = cudaGraphKernelNodeFieldParam,
        .updateData = {
            .param = {
                .pValue = &pointer_indirection[threadIdx.x],
                .offset = update_soa.param_offsets[i],
                .size = sizeof(void*),
            },
        },
    };

    cudaError_t error =
        cudaGraphKernelNodeUpdatesApply(&shared_updates[threadIdx.x], 1);
    CUDA_KERNEL_ASSERT_MSG(
        error == cudaSuccess,
        "cudaGraphKernelNodeUpdatesApply did not succeed");
  }
}

template<bool VOLTA_OR_LATER>
void dynamic_graph_updater(const KernelUpdateSOA<VOLTA_OR_LATER>& update_soa) {
  constexpr size_t BLOCK_SIZE = std::min<size_t>(64, KernelUpdateSOA<VOLTA_OR_LATER>::MAX_NUM_UPDATES);
  dynamic_graph_updater_kernel<BLOCK_SIZE, VOLTA_OR_LATER>
      <<<(update_soa.num_updates + BLOCK_SIZE - 1) / BLOCK_SIZE,
         BLOCK_SIZE,
         0,
         getCurrentCUDAStream()>>>(update_soa);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void dynamic_graph_updater(const KernelUpdateSOA<false>& update_soa);
template void dynamic_graph_updater(const KernelUpdateSOA<true>& update_soa);

} // namespace at::cuda
