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


#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


__global__ void myFirstKernel(cudaGraphDeviceNode_t secondKernelDevNode, cudaGraphConditionalHandle loopHandle, int* iterationCount) {
    int i = ++*iterationCount;

    __shared__ int newArg;
    newArg = 10000 + *iterationCount;
    constexpr size_t numUpdates = 2;
    cudaGraphKernelNodeUpdate myUpdate[numUpdates] = {
        {
            .node = secondKernelDevNode,
            .field = cudaGraphKernelNodeFieldGridDim,
            .updateData = {
                .gridDim = dim3(i, 1, 1)
            }
        },
        {
            .node = secondKernelDevNode,
            .field = cudaGraphKernelNodeFieldParam,
            .updateData = {
                .param = {
                    .pValue = &newArg,
                    .offset = sizeof(int),
                    .size = sizeof(int),
                }
            }
        }
    };
    cudaError_t error = cudaGraphKernelNodeUpdatesApply(myUpdate, numUpdates);
    
    printf("First kernel, iteration %d, set middle arg of second kernel to %d and gridDim to %d and the error was %d\n", i, newArg, i, error);
    if (i == 3) {
        printf("i==3 so turning off the loop handle (this will be the last loop)\n");
        cudaGraphSetConditional(loopHandle, 0);
    }
}

__global__ void mySecondKernel(int arg1, int arg2, int arg3, int* iterationCount) {
    printf("Second kernel, iteration %d, gridDim.x %d, args %d %d %d\n", *iterationCount, gridDim.x, arg1, arg2, arg3);
}

int graphTestMain() {
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    int* iterationCount;
    CUDA_CHECK(cudaMalloc(&iterationCount, sizeof(int)));
    CUDA_CHECK(cudaMemset(iterationCount, 0, sizeof(int)));

    cudaGraphConditionalHandle loopHandle;
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&loopHandle, graph, true, cudaGraphCondAssignDefault));
    cudaGraphNodeParams whileNodeParams = {
        .type = cudaGraphNodeTypeConditional,
        .conditional = {
            .handle = loopHandle,
            .type = cudaGraphCondTypeWhile,
            .size = 1,
        }
    };
    cudaGraphNode_t whileNode;
    CUDA_CHECK(cudaGraphAddNode(&whileNode, graph, nullptr, 0, &whileNodeParams));
    cudaGraph_t withinLoop = whileNodeParams.conditional.phGraph_out[0];

    int a = 1001;
    int b = 1002;
    int c = 1003;
    void* secondKernelArgs[] = {&a, &b, &c, &iterationCount};
    cudaKernelNodeParams secondKernelParams = {
        .func = (void*)mySecondKernel,
        .gridDim = dim3(1, 1, 1),
        .blockDim = dim3(1, 1, 1),
        .sharedMemBytes = 0,
        .kernelParams = secondKernelArgs,
        .extra = nullptr
    };
    cudaGraphNode_t secondKernelNode;
    CUDA_CHECK(cudaGraphAddKernelNode(&secondKernelNode, withinLoop, nullptr, 0, &secondKernelParams));

    cudaKernelNodeAttrValue attr_value = {
        .deviceUpdatableKernelNode = {
            .deviceUpdatable = 1,
            .devNode = nullptr,
        }
    };
    CUDA_CHECK(cudaGraphKernelNodeSetAttribute(secondKernelNode, cudaKernelNodeAttributeDeviceUpdatableKernelNode, &attr_value));
    cudaGraphDeviceNode_t devNode = attr_value.deviceUpdatableKernelNode.devNode;

    void* firstKernelArgs[] = {&devNode, &loopHandle, &iterationCount};
    cudaKernelNodeParams firstKernelParams = {
        .func = (void*)myFirstKernel,
        .gridDim = dim3(1, 1, 1),
        .blockDim = dim3(1, 1, 1),
        .sharedMemBytes = 0,
        .kernelParams = firstKernelArgs,
        .extra = nullptr
    };
    cudaGraphNode_t firstKernelNode;
    CUDA_CHECK(cudaGraphAddKernelNode(&firstKernelNode, withinLoop, nullptr, 0, &firstKernelParams));

    CUDA_CHECK(cudaGraphAddDependencies(withinLoop, &firstKernelNode, &secondKernelNode, 1));

    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphLaunch(graphExec, cudaStreamDefault));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    return 0;
}

}