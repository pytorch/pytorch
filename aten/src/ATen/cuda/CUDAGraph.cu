#include <ATen/cuda/CUDAGraph.h>

namespace at::cuda {


__global__ void dynamicGraphUpdaterKernel(cudaGraphKernelNodeUpdate* updates, size_t numUpdates) {
    constexpr size_t bufSz = 1024;
    __shared__ const void* pointerIndirection[bufSz];
    for (size_t off = 0; off < numUpdates; off += bufSz) {
        int localNumUpdates = min(bufSz, numUpdates - off);
        for (int i = 0; i < localNumUpdates; i++) {
            cudaGraphKernelNodeUpdate& thisUpdate = updates[i + off];
            if (thisUpdate.field == cudaGraphKernelNodeFieldParam) {
                if (thisUpdate.updateData.param.size != sizeof(void*)) {
                    printf("dynamic graph updater only supports pointers\n");
                    __trap();
                }
                // assume pValue points directly to the data
                pointerIndirection[i] = thisUpdate.updateData.param.pValue;
                // pointerIndirection[i] now points to the data
                // &pointerIndirection[i] now points to a void* that points to the data
                thisUpdate.updateData.param.pValue = &pointerIndirection[i];
                // now the update will read *&pointerIndirection[i], which gives the original pValue, and splat that into the correct kernel arg
            }
        }
        cudaError_t error = cudaGraphKernelNodeUpdatesApply(&updates[off], localNumUpdates);
        if (error != cudaSuccess) {
            printf("cudaGraphKernelNodeUpdatesApply returned error %d\n", error);
            __trap();
        }
    }
}

void dynamicGraphUpdater(cudaGraphKernelNodeUpdate* updates, size_t numUpdates) {
	dynamicGraphUpdaterKernel<<<1, 1, 0, getCurrentCUDAStream()>>>(updates, numUpdates);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}