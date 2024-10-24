#include <ATen/cuda/CUDAGraph.h>

namespace at::cuda {

__global__ void dynamic_graph_updater_kernel(cudaGraphKernelNodeUpdate* updates, size_t num_updates) {
    constexpr size_t buf_sz = 1024;
    __shared__ const void* pointer_indirection[buf_sz];
    for (size_t off = 0; off < num_updates; off += buf_sz) {
        int local_num_updates = min(buf_sz, num_updates - off);
        for (int i = 0; i < local_num_updates; i++) {
            cudaGraphKernelNodeUpdate& this_update = updates[i + off];
            if (this_update.field == cudaGraphKernelNodeFieldParam) {
                CUDA_KERNEL_ASSERT_MSG(this_update.updateData.param.size == sizeof(void*), "Dynamic graph updater only supports pointers");
                // assume pValue points directly to the data
                pointer_indirection[i] = this_update.updateData.param.pValue;
                // pointer_indirection[i] now points to the data
                // &pointer_indirection[i] now points to a void* that points to the data
                this_update.updateData.param.pValue = &pointer_indirection[i];
                // now the update will read *&pointer_indirection[i], which gives the original pValue, and splat that into the correct kernel arg
            }
        }
        cudaError_t error = cudaGraphKernelNodeUpdatesApply(&updates[off], local_num_updates);
        CUDA_KERNEL_ASSERT_MSG(error == cudaSuccess, "cudaGraphKernelNodeUpdatesApply did not succeed");
    }
}

void dynamic_graph_updater(cudaGraphKernelNodeUpdate* updates, size_t num_updates) {
	dynamic_graph_updater_kernel<<<1, 1, 0, getCurrentCUDAStream()>>>(updates, num_updates);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// the best way to speed this up would be to minimize the number of
// global memory stalls.  We can pass three arrays: nodes, offsets,
// and pValues, to avoid filling in the other fields.  Right now,
// dynamic_graph_updater_kernel kernel takes about 5.0 microseconds on
// average, but dynamic_graph_updater_kernel2 takes about 2.8
// microseconds on average. There might be other potential speed ups,
// but this is pretty good.
template <size_t BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE) dynamic_graph_updater_kernel2(
    __grid_constant__ const KernelUpdateSOA update_soa) {
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
    shared_updates[threadIdx.x].node = update_soa.device_nodes[i];
    shared_updates[threadIdx.x].field = cudaGraphKernelNodeFieldParam;
    shared_updates[threadIdx.x].updateData.param.pValue =
        &pointer_indirection[threadIdx.x];
    shared_updates[threadIdx.x].updateData.param.offset =
        update_soa.param_offsets[i];
    shared_updates[threadIdx.x].updateData.param.size = sizeof(void*);

    cudaError_t error =
        cudaGraphKernelNodeUpdatesApply(&shared_updates[threadIdx.x], 1);
    CUDA_KERNEL_ASSERT_MSG(
        error == cudaSuccess,
        "cudaGraphKernelNodeUpdatesApply did not succeed");
  }
}

void dynamic_graph_updater2(const KernelUpdateSOA& update_soa) {
  constexpr size_t BLOCK_SIZE = 64;
  dynamic_graph_updater_kernel2<BLOCK_SIZE>
      <<<(update_soa.num_updates + BLOCK_SIZE - 1) / BLOCK_SIZE,
         BLOCK_SIZE,
         0,
         getCurrentCUDAStream()>>>(update_soa);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace at::cuda
