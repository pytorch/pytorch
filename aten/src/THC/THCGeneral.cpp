#include <THC/THCGeneral.h>
#include <TH/TH.h>
#include <THC/THCAllocator.h>
#include <THC/THCCachingHostAllocator.h>
#include <THC/THCGeneral.hpp>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <stdlib.h>
#include <stdint.h>

/* Size of scratch space available in global memory per each SM + stream */
#define MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM 4 * sizeof(float)

/* Minimum amount of scratch space per device. Total scratch memory per
 * device is either this amount, or the # of SMs * the space per SM defined
 * above, whichever is greater.*/
#define MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE 32768 * sizeof(float)

/* Maximum number of P2P connections (if there are more than 9 then P2P is
 * enabled in groups of 8). */
#define THC_CUDA_MAX_PEER_SIZE 8

void THCState_free(THCState* state)
{
  free(state);
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device);

THCState* THCState_alloc(void)
{
  THCState* state = (THCState*) calloc(1, sizeof(THCState));
  return state;
}

void THCudaInit(THCState* state)
{
  if (!state->cudaHostAllocator) {
    state->cudaHostAllocator = getTHCCachingHostAllocator();
  }

  // We want to throw if there are no GPUs
  int numDevices = static_cast<int>(c10::cuda::device_count_ensure_non_zero());
  state->numDevices = numDevices;

  c10::cuda::CUDACachingAllocator::init(numDevices);

  int device = 0;
  THCudaCheck(cudaGetDevice(&device));

  state->resourcesPerDevice = (THCCudaResourcesPerDevice*)
    calloc(numDevices, sizeof(THCCudaResourcesPerDevice));

  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  state->p2pAccessEnabled = (int**) calloc(numDevices, sizeof(int*));
  for (int i = 0; i < numDevices; ++i) {
    state->p2pAccessEnabled[i] = (int*) calloc(numDevices, sizeof(int));
    for (int j = 0; j < numDevices; ++j)
      if (i == j)
        state->p2pAccessEnabled[i][j] = 1;
      else
        state->p2pAccessEnabled[i][j] = -1;
  }

  for (int i = 0; i < numDevices; ++i) {
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);
    THCudaCheck(cudaSetDevice(i));

    /* The scratch space that we want to have available per each device is
       based on the number of SMs available per device. We guarantee a
       minimum of 128kb of space per device, but to future-proof against
       future architectures that may have huge #s of SMs, we guarantee that
       we have at least 16 bytes for each SM. */
    int numSM = at::cuda::getDeviceProperties(i)->multiProcessorCount;
    size_t sizePerStream =
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE >= numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM ?
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE :
      numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
    res->scratchSpacePerStream = sizePerStream;
  }

  /* Restore to previous device */
  THCudaCheck(cudaSetDevice(device));
}

void THCudaShutdown(THCState* state)
{

  int deviceCount = 0;
  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));
  THCudaCheck(cudaGetDeviceCount(&deviceCount));

  /* cleanup p2p access state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    free(state->p2pAccessEnabled[dev]);
  }
  free(state->p2pAccessEnabled);

  free(state->resourcesPerDevice);
  c10::cuda::CUDACachingAllocator::emptyCache();
  if (state->cudaHostAllocator == getTHCCachingHostAllocator()) {
    THCCachingHostAllocator_emptyCache();
  }

  THCudaCheck(cudaSetDevice(prevDev));
}

int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess)
{
  if (dev < 0 || dev >= state->numDevices) {
    THError("%d is not a device", dev);
  }
  if (devToAccess < 0 || devToAccess >= state->numDevices) {
    THError("%d is not a device", devToAccess);
  }
  if (state->p2pAccessEnabled[dev][devToAccess] == -1) {
    int prevDev = 0;
    THCudaCheck(cudaGetDevice(&prevDev));
    THCudaCheck(cudaSetDevice(dev));

    int access = 0;
    THCudaCheck(cudaDeviceCanAccessPeer(&access, dev, devToAccess));
    if (access) {
      cudaError_t err = cudaDeviceEnablePeerAccess(devToAccess, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        // ignore and clear the error if access was already enabled
        cudaGetLastError();
      } else {
        THCudaCheck(err);
      }
      state->p2pAccessEnabled[dev][devToAccess] = 1;
    } else {
      state->p2pAccessEnabled[dev][devToAccess] = 0;
    }

    THCudaCheck(cudaSetDevice(prevDev));
  }
  return state->p2pAccessEnabled[dev][devToAccess];
}

c10::Allocator* THCState_getCudaHostAllocator(THCState* state)
{
  return state->cudaHostAllocator;
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  return &(state->resourcesPerDevice[device]);
}

size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state)
{
  int device = -1;
  THCudaCheck(cudaGetDevice(&device));
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  return res->scratchSpacePerStream;
}

void __THCudaCheck(cudaError_t err, const char *file, const int line)
{
  if(err != cudaSuccess)
  {
    static int alreadyFailed = 0;
    if(!alreadyFailed) {
      fprintf(stderr, "THCudaCheck FAIL file=%s line=%i error=%i : %s\n", file, line, err, cudaGetErrorString(err));
      alreadyFailed = 1;
    }
    _THError(file, line, "cuda runtime error (%d) : %s", err,
             cudaGetErrorString(err));
  }
}

void __THCudaCheckWarn(cudaError_t err, const char *file, const int line)
{
  if(err != cudaSuccess)
  {
    fprintf(stderr, "THCudaCheckWarn FAIL file=%s line=%i error=%i : %s\n", file, line, err, cudaGetErrorString(err));
  }
}

void __THCublasCheck(cublasStatus_t status, const char *file, const int line)
{
  if(status != CUBLAS_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case CUBLAS_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case CUBLAS_STATUS_INVALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case CUBLAS_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

#ifndef __HIP_PLATFORM_HCC__
      case CUBLAS_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case CUBLAS_STATUS_EXECUTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;
#endif

      case CUBLAS_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "cublas runtime error : %s", errmsg);
  }
}

void __THCusparseCheck(cusparseStatus_t status, const char *file, const int line)
{
  if(status != CUSPARSE_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case CUSPARSE_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case CUSPARSE_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case CUSPARSE_STATUS_INVALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case CUSPARSE_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

      case CUSPARSE_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case CUSPARSE_STATUS_EXECUTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;

      case CUSPARSE_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        errmsg = "the matrix type is not supported by this function";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "cusparse runtime error : %s", errmsg);
  }
}

void* THCudaMalloc(THCState *state, size_t size)
{
  return c10::cuda::CUDACachingAllocator::raw_alloc(size);
}

void THCudaFree(THCState *state, void* ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

at::DataPtr THCudaHostAlloc(THCState *state, size_t size)
{
  THCudaCheck(cudaGetLastError());
  c10::Allocator* allocator = state->cudaHostAllocator;
  return allocator->allocate(size);
}

void THCudaHostRecord(THCState *state, void *ptr) {
  if (state->cudaHostAllocator == getTHCCachingHostAllocator()) {
    THCCachingHostAllocator_recordEvent(ptr, at::cuda::getCurrentCUDAStream());
  }
}

#undef MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
#undef MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE

#include <THC/THCStorage.cpp>
#include <THC/THCAllocator.cpp>
