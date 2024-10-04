#include <ATen/cuda/CUDAGraph.h>

namespace at::cuda {


__global__ void dynamicGraphUpdaterKernel(cudaGraphKernelNodeUpdate* updates, size_t numUpdates) {
    constexpr size_t bufSz = 1024;
    if (numUpdates > bufSz) {
      printf("too many updates\n"); __trap();
    }
    printf("i am the dynamic graph updater and i am doing %llu updates\n", numUpdates);
    __shared__ const void* pointerIndirection[bufSz];
    for (int i = 0; i < numUpdates; i++) {
    	printf("update %d\n", i);
    	cudaGraphKernelNodeUpdate myUpdate = updates[i];
        if (updates[i].field == cudaGraphKernelNodeFieldParam) {
        	printf("it's a node field param\n");
            if (updates[i].updateData.param.size != sizeof(void*)) {
                printf("this only supports pointers\n"); __trap();
            }
            printf("the node is %llu\n", (size_t)updates[i].node);
            printf("the size is %llu\n", updates[i].updateData.param.size);
            printf("the offset is %llu\n", updates[i].updateData.param.offset);
            printf("the pointer value is %llu\n", (size_t)updates[i].updateData.param.pValue);
            // assume pValue points directly to the data
            const void* data = updates[i].updateData.param.pValue;
            // pointerIndirection[i] now points to the data
            // &pointerIndirection[i] now points to a void* that points to the data
            myUpdate.updateData.param.pValue = &data;
            // now the update will read *&pointerIndirection[i], which gives the original pValue, and splat that into the correct kernel arg
        }
        cudaError_t error = cudaGraphKernelNodeUpdatesApply(&myUpdate, 1);
    	printf("the error was %d\n", error);

    	if (false){
    		myUpdate.field = cudaGraphKernelNodeFieldGridDim;
    		myUpdate.updateData.gridDim = dim3(2, 1, 1);
    		cudaError_t error2 = cudaGraphKernelNodeUpdatesApply(&myUpdate, 1);
    		printf("the error for the grid dim thingy was %d\n", error2); // still gives cudaErrorInvalidValue
    	}
    }

}

void dynamicGraphUpdater(cudaGraphKernelNodeUpdate* updates, size_t numUpdates) {
	dynamicGraphUpdaterKernel<<<1, 1, 0, getCurrentCUDAStream()>>>(updates, numUpdates);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}