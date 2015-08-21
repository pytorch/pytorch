#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorCopy.cu"
#else

THC_API void
THCTensor_(copy)(THCState* state, THCTensor* dst, THCTensor* src) {
  long totalElements = THCTensor_(nElement)(state, dst);

  THArgCheck(totalElements == THCTensor_(nElement)(state, src), 2,
             "sizes do not match");

  if (THCTensor_(nDimension)(state, dst) == 0) {
    // Zero-dim tensor; copy nothing
    return;
  }

  // We can memcpy the memory if:
  // -both tensors are contiguous; or,
  // -there is only one element to copy; or,
  // -FIXME: if both tensors have matching size and stride arrays, and no
  // holes within (in other words, there is some permutation that can be applied
  // to the size/strides such that the resulting tensor is contiguous).
  bool srcContig = THCTensor_(isContiguous)(state, src);
  bool dstContig = THCTensor_(isContiguous)(state, dst);
  bool memcpyEligible = (srcContig && dstContig) || (totalElements == 1);

  int srcDev = THCTensor_(getDevice)(state, src);
  int dstDev = THCTensor_(getDevice)(state, dst);
  int oldDev = curGPU();

  // We always perform the copy on the source device, using the
  // current stream on the source device.
  // If the copy is on the default stream, then we fully synchronize
  // both src and dst's default streams for completion of the
  // copy. We have to explicitly do this for non-contig copies.
  // This mimics the behavior of cross-device cudaMemcpyAsync on
  // the default stream.
  // If the copy is not on the default stream, then it is up to the
  // user to add needed synchronization on the dst device, since the
  // stream on the dst device that wishes to synchronize may not be
  // the same index as the one on the src device.
  int copyStreamIndex =
    THCState_getCurrentStreamIndex(state);
  cudaStream_t copyStream =
    THCState_getDeviceStream(state, srcDev, copyStreamIndex);

  if (srcDev != dstDev && copyStreamIndex == 0) {
    // This is a cross-device copy on the default stream. We perform a
    // two-way barrier between both devices' default streams before
    // the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are
    // handled, so that no one is operating on the dst memory when
    // we perform the copy.
    // src waits on dst barrier (src already waits on src)
    cudaEvent_t dstReady;
    THCudaCheck(cudaSetDevice(dstDev));
    THCudaCheck(cudaEventCreateWithFlags(&dstReady, cudaEventDisableTiming));
    THCudaCheck(cudaEventRecord(dstReady, NULL));

    THCudaCheck(cudaSetDevice(srcDev));
    THCudaCheck(cudaStreamWaitEvent(NULL, dstReady, 0));
    THCudaCheck(cudaEventDestroy(dstReady));
  } else if (srcDev != oldDev) {
    THCudaCheck(cudaSetDevice(srcDev));
  }

  // We are now on srcDev
  if (memcpyEligible) {
    // Perform the copy
    THCudaCheck(cudaMemcpyAsync(THCTensor_(data)(state, dst),
                                THCTensor_(data)(state, src),
                                totalElements * sizeof(real),
                                cudaMemcpyDeviceToDevice,
                                copyStream));
  } else {
#ifdef THC_REAL_IS_FLOAT
    // Non-contiguous copy

    // We avoid creating temporary memory copies if possible.
    // If both src and dst are on the same device, or if they are on
    // different devices and p2p access is enabled, perform the copy
    // by a pointwise copy kernel.
    // Otherwise, we'll have to make contiguous (which will in fact
    // invoke copy() again), and then perform the copy.
    // FIXME: might want to consider only running the pointwise kernel
    // if both src and dst innermost dimensions are contiguous. If
    // they are not, then taking the hit of the memory allocation/free
    // might be worth it to avoid non-coalesced reads or writes.

    // A device always has access to itself, so this also handles the
    // case srcDev == dstDev
    if (THCState_getPeerToPeerAccess(state, srcDev, dstDev)) {
      // Make sure we have the current stream set in THCState, since
      // pointwise uses that
      if (srcDev != oldDev) {
        THCState_setStream(state, srcDev, copyStreamIndex);
      }

      bool succ =
        THCudaTensor_pointwiseApply2(state, dst, src, CopyOp<float>());
      THArgCheck(succ, 2, CUTORCH_DIM_WARNING);

      // Restore prior THCState stream
      if (srcDev != oldDev) {
        THCState_setStream(state, oldDev, copyStreamIndex);
      }
    } else {
      // GPUs can't access each other directly; fall back to
      // newContiguous and memcpy
      THCudaTensor* srcContig = THCudaTensor_newContiguous(state, src);
      THCudaTensor* dstContig = dst;

      if (!THCudaTensor_isContiguous(state, dst)) {
        // We are copying over the contents of dst, so we don't need
        // to preserve its values. We just need a destination tensor
        // the same size as dst.

        // Allocate the tensor on the new device
        THCudaCheck(cudaSetDevice(dstDev));

        dstContig = THCudaTensor_new(state);
        THCudaTensor_resizeAs(state, dstContig, dst);

        THCudaCheck(cudaSetDevice(srcDev));
      }

      THCudaCheck(cudaMemcpyAsync(THCudaTensor_data(state, dstContig),
                                  THCudaTensor_data(state, srcContig),
                                  totalElements * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  copyStream));

      THCudaTensor_free(state, srcContig);

      if (dst != dstContig) {
        THCudaTensor_freeCopyTo(state, dstContig, dst);
      }
    }
#else
#define STRINGIFY(x) #x
      THError("Non-contiguous copy not implemented for Cuda%sTensor", STRINGIFY(Real));
#undef STRINGIFY
#endif
  }

  if (srcDev != dstDev && copyStreamIndex == 0) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on srcDev, record default stream event
    cudaEvent_t srcReady;
    THCudaCheck(cudaEventCreateWithFlags(&srcReady, cudaEventDisableTiming));
    THCudaCheck(cudaEventRecord(srcReady, NULL));

    THCudaCheck(cudaSetDevice(dstDev));
    THCudaCheck(cudaStreamWaitEvent(NULL, srcReady, 0));
    THCudaCheck(cudaEventDestroy(srcReady));

    // We are now on dstDev (right above). Restore prior device from dst
    if (dstDev != oldDev) {
      THCudaCheck(cudaSetDevice(oldDev));
    }
  } else {
    // We are still on srcDev. Restore prior device from src
    if (srcDev != oldDev) {
      THCudaCheck(cudaSetDevice(oldDev));
    }
  }

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

// conversions are mediated by the CPU
// yes, this is slow; feel free to write CUDA kernels for this
#define THC_CUDA_TENSOR_IMPLEMENT_COPY(TYPEC,TYPECUDA)                  \
  void THCTensor_(copyCuda##TYPEC)(THCState *state, THCTensor *self, struct THCuda##TYPECUDA##Tensor *src)  \
  {                                                                     \
    if(THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {                       \
      THCTensor_(copy)(state, self, (THCTensor*) src);   /* cast just removes compiler warning */ \
    } else {                                                            \
      THArgCheck(THCTensor_(nElement)(state, self) == THCuda##TYPECUDA##Tensor_nElement(state, src), 2, "size does not match");    \
      THLongStorage *size = THCuda##TYPECUDA##Tensor_newSizeOf(state, src);   \
      TH##TYPEC##Tensor *buffer1 = TH##TYPEC##Tensor_newWithSize(size, NULL); \
      THTensor *buffer2  = THTensor_(newWithSize)(size, NULL);          \
      TH##TYPEC##Tensor_copyCuda(state, buffer1, src);                  \
      THTensor_(copy##TYPEC)(buffer2, buffer1);                         \
      THCTensor_(copyCPU)(state, self, buffer2);                        \
      THLongStorage_free(size);                                         \
      TH##TYPEC##Tensor_free(buffer1);                                  \
      THTensor_(free)(buffer2);                                         \
    }                                                                   \
  }

THC_CUDA_TENSOR_IMPLEMENT_COPY(Byte,Byte)
THC_CUDA_TENSOR_IMPLEMENT_COPY(Char,Char)
THC_CUDA_TENSOR_IMPLEMENT_COPY(Short,Short)
THC_CUDA_TENSOR_IMPLEMENT_COPY(Int,Int)
THC_CUDA_TENSOR_IMPLEMENT_COPY(Long,Long)
THC_CUDA_TENSOR_IMPLEMENT_COPY(Float,)  // i.e. float
THC_CUDA_TENSOR_IMPLEMENT_COPY(Double,Double)

#endif
