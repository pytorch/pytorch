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
                if (this_update.updateData.param.size != sizeof(void*)) {
                    printf("dynamic graph updater only supports pointers\n");
                    __trap();
                }
                // assume pValue points directly to the data
                pointer_indirection[i] = this_update.updateData.param.pValue;
                // pointer_indirection[i] now points to the data
                // &pointer_indirection[i] now points to a void* that points to the data
                this_update.updateData.param.pValue = &pointer_indirection[i];
                // now the update will read *&pointer_indirection[i], which gives the original pValue, and splat that into the correct kernel arg
            }
        }
        cudaError_t error = cudaGraphKernelNodeUpdatesApply(&updates[off], local_num_updates);
        if (error != cudaSuccess) {
            printf("cudaGraphKernelNodeUpdatesApply returned error %d\n", error);
            __trap();
        }
    }
}

void dynamic_graph_updater(cudaGraphKernelNodeUpdate* updates, size_t num_updates) {
	dynamic_graph_updater_kernel<<<1, 1, 0, getCurrentCUDAStream()>>>(updates, num_updates);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}