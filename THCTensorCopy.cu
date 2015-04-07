#include "THCApply.cuh"

static inline int curGPU() {
  int curDev;
  THCudaCheck(cudaGetDevice(&curDev));
  return curDev;
}

THC_API void
THCudaTensor_copy(THCState* state, THCudaTensor* dst, THCudaTensor* src) {
  long totalElements = THCudaTensor_nElement(state, dst);

  THArgCheck(totalElements == THCudaTensor_nElement(state, src), 2,
             "sizes do not match");

  if (THCudaTensor_nDimension(state, dst) == 0) {
    // Zero-dim tensor; copy nothing
    return;
  }

  // We can memcpy the memory if:
  // -both tensors are contiguous; or,
  // -there is only one element to copy; or,
  // -FIXME: if both tensors have matching size and stride arrays, and no
  // holes within (in other words, there is some permutation that can be applied
  // to the size/strides such that the resulting tensor is contiguous).
  bool srcContig = THCudaTensor_isContiguous(state, src);
  bool dstContig = THCudaTensor_isContiguous(state, dst);
  bool memcpyEligible = (srcContig && dstContig) || (totalElements == 1);

  if (memcpyEligible) {
    THCudaCheck(cudaMemcpyAsync(THCudaTensor_data(state, dst),
                                THCudaTensor_data(state, src),
                                totalElements * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                THCState_getCurrentStream(state)));
  } else {
    int oldDev = curGPU();
    int srcDev = THCudaTensor_getDevice(state, src);
    int dstDev = THCudaTensor_getDevice(state, dst);

    if (srcDev == dstDev) {
      if (oldDev != srcDev) {
        THCudaCheck(cudaSetDevice(srcDev));
      }

      bool succ =
        THCudaTensor_pointwiseApply2(state, dst, src, CopyOp<float>());
      THArgCheck(succ, 2, CUTORCH_DIM_WARNING);
    } else { // multi-gpu
      // empirically, running the kernel on the device that holds the
      // non-contiguous tensor is faster by 5-10x
      int copyDev   = dstContig ? srcDev : dstDev;
      int remoteDev = dstContig ? dstDev : srcDev;

      // synchronize remote device before copy
      cudaEvent_t dataReady;
      THCudaCheck(cudaSetDevice(remoteDev));
      THCudaCheck(cudaEventCreate(&dataReady));
      THCudaCheck(cudaEventRecord(
                    dataReady,
                    THCState_getDeviceStream(state, remoteDev, THCState_getCurrentStreamIndex(state))));
      THCudaCheck(cudaSetDevice(copyDev));
      THCudaCheck(cudaStreamWaitEvent(
                    THCState_getDeviceStream(state, copyDev, THCState_getCurrentStreamIndex(state)),
                    dataReady, 0));
      THCudaCheck(cudaEventDestroy(dataReady));

      bool succ =
        THCudaTensor_pointwiseApply2(state, dst, src, CopyOp<float>());
      THArgCheck(succ, 2, CUTORCH_DIM_WARNING);

      // synchronize remote device after copy
      cudaEvent_t doneCopying;
      THCudaCheck(cudaEventCreate(&doneCopying));
      THCudaCheck(cudaEventRecord(
                    doneCopying,
                    THCState_getDeviceStream(state, copyDev, THCState_getCurrentStreamIndex(state))));
      THCudaCheck(cudaSetDevice(remoteDev));
      THCudaCheck(cudaStreamWaitEvent(
                    THCState_getDeviceStream(state, remoteDev, THCState_getCurrentStreamIndex(state)),
                    doneCopying, 0));
      THCudaCheck(cudaEventDestroy(doneCopying));
    }

    if (curGPU() != oldDev) {
      THCudaCheck(cudaSetDevice(oldDev));
    }
  }

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}
