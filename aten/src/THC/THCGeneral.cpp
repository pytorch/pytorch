#include "THCGeneral.h"
#include "TH.h"
#include "THCAllocator.h"
#include "THCCachingHostAllocator.h"
#include "THCStream.h"
#include "THCThreadLocal.h"
#include "THCTensorRandom.h"
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

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device);

THCState* THCState_alloc(void)
{
  THCState* state = (THCState*) malloc(sizeof(THCState));
  memset(state, 0, sizeof(THCState));
  return state;
}

void THCState_free(THCState* state)
{
  free(state);
}

static cudaError_t cudaMallocWrapper(void* ctx, void** devPtr, size_t size, cudaStream_t stream)
{
  return cudaMalloc(devPtr, size);
}

static cudaError_t cudaFreeWrapper(void* ctx, void* devPtr)
{
  return cudaFree(devPtr);
}

static THCDeviceAllocator defaultDeviceAllocator = {
  &cudaMallocWrapper,
  NULL,
  &cudaFreeWrapper,
  NULL,
  NULL,
  NULL
};

void THCudaInit(THCState* state)
{
  if (!state->cudaDeviceAllocator) {
    state->cudaDeviceAllocator = &defaultDeviceAllocator;
  }
  if (!state->cudaHostAllocator) {
    state->cudaHostAllocator = &THCudaHostAllocator;
  }
  if (!state->cudaUVAAllocator) {
    state->cudaUVAAllocator = &THCUVAAllocator;
  }

  int numDevices = 0;
  THCudaCheck(cudaGetDeviceCount(&numDevices));
  state->numDevices = numDevices;

  int device = 0;
  THCudaCheck(cudaGetDevice(&device));

  /* Start in the default stream on the current device */
  state->currentStreams = (THCThreadLocal*) malloc(numDevices * sizeof(THCThreadLocal));
  for (int i = 0; i < numDevices; ++i) {
    state->currentStreams[i] = THCThreadLocal_alloc();
  }
  state->currentPerDeviceBlasHandle = THCThreadLocal_alloc();
  state->currentPerDeviceSparseHandle = THCThreadLocal_alloc();

  state->resourcesPerDevice = (THCCudaResourcesPerDevice*)
    malloc(numDevices * sizeof(THCCudaResourcesPerDevice));
  memset(state->resourcesPerDevice, 0, numDevices * sizeof(THCCudaResourcesPerDevice));

  state->deviceProperties =
    (struct cudaDeviceProp*)malloc(numDevices * sizeof(struct cudaDeviceProp));

  state->rngState = (THCRNGState*)malloc(sizeof(THCRNGState));
  THCRandom_init(state, numDevices, device);

  // By default, all direct p2p kernel access (besides copy) is disallowed,
  // since direct access without knowing whether or not a certain operation
  // should be cross-GPU leads to synchronization errors. The user can choose
  // to disable this functionality, however.
  state->p2pKernelAccessEnabled = 0;

  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  state->p2pAccessEnabled = (int**) malloc(sizeof(int*) * numDevices);
  for (int i = 0; i < numDevices; ++i) {
    state->p2pAccessEnabled[i] = (int*) malloc(sizeof(int) * numDevices);
    for (int j = 0; j < numDevices; ++j)
      if (i == j)
        state->p2pAccessEnabled[i][j] = 1;
      else if (j / THC_CUDA_MAX_PEER_SIZE != i / THC_CUDA_MAX_PEER_SIZE)
        state->p2pAccessEnabled[i][j] = 0;
      else
        state->p2pAccessEnabled[i][j] = -1;
  }

  for (int i = 0; i < numDevices; ++i) {
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);
    THCudaCheck(cudaSetDevice(i));
    THCudaCheck(cudaGetDeviceProperties(&state->deviceProperties[i], i));

    // Allocate space for the default stream
    res->streams = (THCStream**) malloc(sizeof(THCStream*));
    res->streams[0] = THCStream_defaultStream(i);

    /* The scratch space that we want to have available per each device is
       based on the number of SMs available per device. We guarantee a
       minimum of 128kb of space per device, but to future-proof against
       future architectures that may have huge #s of SMs, we guarantee that
       we have at least 16 bytes for each SM. */
    int numSM = state->deviceProperties[i].multiProcessorCount;
    size_t sizePerStream =
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE >= numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM ?
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE :
      numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
    res->scratchSpacePerStream = sizePerStream;
  }

  /* Restore to previous device */
  THCudaCheck(cudaSetDevice(device));

  // Unlike CUDA streams, there is no NULL cuBLAS handle. The default THC
  // cuBLAS handle is the first user BLAS handle. Note that the actual BLAS
  // handles are created lazily.
  state->numUserBlasHandles = 1;
  state->numUserSparseHandles = 1;

  state->heapSoftmax = 3e8; // 300MB, adjusted upward dynamically
  state->heapDelta = 0;
}

void THCudaShutdown(THCState* state)
{
  THCRandom_shutdown(state);

  free(state->rngState);
  free(state->deviceProperties);

  int deviceCount = 0;
  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));
  THCudaCheck(cudaGetDeviceCount(&deviceCount));

  /* cleanup p2p access state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    free(state->p2pAccessEnabled[dev]);
  }
  free(state->p2pAccessEnabled);

  /* cleanup per-device state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    THCudaCheck(cudaSetDevice(dev));
    THCCudaResourcesPerDevice* res = &(state->resourcesPerDevice[dev]);
    /* Free all streams */
    for (int i = 0; i <= state->numUserStreams; ++i) {
      THCStream_free(res->streams[i]);
    }
    /* Free user defined BLAS handles */
    for (int i = 0; i < res->numBlasHandles; ++i) {
      THCublasCheck(cublasDestroy(res->blasHandles[i]));
    }
    /* Free user defined sparse handles */
    for (int i = 0; i < res->numSparseHandles; ++i) {
      THCusparseCheck(cusparseDestroy(res->sparseHandles[i]));
    }
    /* Free per-stream scratch space; starts at 0 because there is space for
       the default stream as well*/
    if (res->devScratchSpacePerStream) {
      for (int stream = 0; stream <= state->numUserStreams; ++stream) {
        THCudaCheck(THCudaFree(state, res->devScratchSpacePerStream[stream]));
      }
    }

    free(res->streams);
    free(res->blasHandles);
    free(res->sparseHandles);
    free(res->devScratchSpacePerStream);
    THCStream_free((THCStream*)THCThreadLocal_get(state->currentStreams[dev]));
    THCThreadLocal_free(state->currentStreams[dev]);
  }
  free(state->resourcesPerDevice);
  if (state->cudaDeviceAllocator->emptyCache) {
    state->cudaDeviceAllocator->emptyCache(state->cudaDeviceAllocator->state);
  }
  if (state->cudaHostAllocator == &THCCachingHostAllocator) {
    THCCachingHostAllocator_emptyCache();
  }
  free(state->currentStreams);
  THCThreadLocal_free(state->currentPerDeviceBlasHandle);

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

void THCState_setPeerToPeerAccess(THCState* state, int dev, int devToAccess,
                                  int enable)
{
  /* This will perform device bounds checking for us */
  int prevEnabled = THCState_getPeerToPeerAccess(state, dev, devToAccess);

  if (enable != prevEnabled) {
    /* If we're attempting to enable p2p access but p2p access isn't */
    /* supported, throw an error */
    if (enable) {
      int access = 0;
      THCudaCheck(cudaDeviceCanAccessPeer(&access, dev, devToAccess));

      if (!access) {
        THError("p2p access not supported for %d accessing %d",
                dev, devToAccess);
      }
    }

    state->p2pAccessEnabled[dev][devToAccess] = enable;

    int prevDev = 0;
    THCudaCheck(cudaGetDevice(&prevDev));
    THCudaCheck(cudaSetDevice(dev));

    /* This should be in sync with the current access state */
    if (enable) {
      THCudaCheck(cudaDeviceEnablePeerAccess(devToAccess, 0));
    } else {
      THCudaCheck(cudaDeviceDisablePeerAccess(devToAccess));
    }

    THCudaCheck(cudaSetDevice(prevDev));
  }
}

int THCState_getKernelPeerToPeerAccessEnabled(THCState* state) {
  return state->p2pKernelAccessEnabled;
}

void THCState_setKernelPeerToPeerAccessEnabled(THCState* state, int val) {
  state->p2pKernelAccessEnabled = val;
}

struct cudaDeviceProp* THCState_getCurrentDeviceProperties(THCState* state)
{
  int curDev = -1;
  THCudaCheck(cudaGetDevice(&curDev));

  return &(state->deviceProperties[curDev]);
}

struct cudaDeviceProp* THCState_getDeviceProperties(THCState* state, int device)
{
  THAssert(device >= 0 && device < state->numDevices);
  return &(state->deviceProperties[device]);
}

struct THCRNGState* THCState_getRngState(THCState *state)
{
  return state->rngState;
}

THAllocator* THCState_getCudaHostAllocator(THCState* state)
{
  return state->cudaHostAllocator;
}

THAllocator* THCState_getCudaUVAAllocator(THCState* state)
{
  return state->cudaUVAAllocator;
}

THC_API THCDeviceAllocator* THCState_getDeviceAllocator(THCState* state)
{
  return state->cudaDeviceAllocator;
}

void THCState_setDeviceAllocator(THCState* state, THCDeviceAllocator* allocator)
{
  state->cudaDeviceAllocator = allocator;
}

int THCState_isCachingAllocatorEnabled(THCState* state) {
  return state->cudaHostAllocator == &THCCachingHostAllocator;
}

int THCState_getNumDevices(THCState *state)
{
  return state->numDevices;
}

static void THCState_initializeScratchSpace(THCState* state, int dev)
{
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);
  if (res->devScratchSpacePerStream) {
    return;
  }
  size_t size = (state->numUserStreams + 1) * sizeof(void*);
  void** scratch = (void**)malloc(size);
  for (int i = 0; i <= state->numUserStreams; ++i) {
    THCudaCheck(THCudaMalloc(state, &scratch[i], res->scratchSpacePerStream));
  }
  res->devScratchSpacePerStream = scratch;
}

void THCState_reserveStreams(THCState* state, int numStreams, int nonBlocking)
{
  if (numStreams <= state->numUserStreams)
  {
    return;
  }

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  /* Otherwise, we have to allocate a new set of streams and stream data */
  for (int dev = 0; dev < state->numDevices; ++dev) {
    THCudaCheck(cudaSetDevice(dev));
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);

    /* +1 for the default stream as well */
    THCStream** newStreams = (THCStream**)realloc(res->streams, (numStreams + 1) * sizeof(THCStream*));
    THAssert(newStreams);

    THCState_initializeScratchSpace(state, dev);
    void** newScratchSpace = (void**)realloc(res->devScratchSpacePerStream, (numStreams + 1) * sizeof(void*));
    THAssert(newScratchSpace);

    /* Allocate new stream resources */
    size_t scratchSpaceSize = THCState_getDeviceScratchSpaceSize(state, dev);
    unsigned int flags =
      nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;

    for (int stream = state->numUserStreams + 1; stream <= numStreams; ++stream) {
      newStreams[stream] = THCStream_new(flags);
      newScratchSpace[stream] = NULL;
      THCudaCheck(THCudaMalloc(state, &newScratchSpace[stream], scratchSpaceSize));
    }

    res->streams = newStreams;
    res->devScratchSpacePerStream = newScratchSpace;
  }

  state->numUserStreams = numStreams;

  THCudaCheck(cudaSetDevice(prevDev));
}

void THCState_reserveDeviceBlasHandles(THCState* state, int device, int numBlasHandles)
{
  int prevDev = -1;
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  if (numBlasHandles <= res->numBlasHandles) {
    return;
  }

  THCudaCheck(cudaGetDevice(&prevDev));
  THCudaCheck(cudaSetDevice(device));

  size_t size = numBlasHandles * sizeof(cublasHandle_t);
  cublasHandle_t* handles = (cublasHandle_t*) realloc(res->blasHandles, size);
  for (int i = res->numBlasHandles; i < numBlasHandles; ++i) {
    handles[i] = NULL;
    THCublasCheck(cublasCreate(&handles[i]));
  }
  res->blasHandles = handles;
  res->numBlasHandles = numBlasHandles;

  THCudaCheck(cudaSetDevice(prevDev));
}

void THCState_reserveDeviceSparseHandles(THCState* state, int device, int numSparseHandles)
{
  int prevDev = -1;
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  if (numSparseHandles <= res->numSparseHandles) {
    return;
  }

  THCudaCheck(cudaGetDevice(&prevDev));
  THCudaCheck(cudaSetDevice(device));

  size_t size = numSparseHandles * sizeof(cusparseHandle_t);
  cusparseHandle_t* handles = (cusparseHandle_t*) realloc(res->sparseHandles, size);
  for (int i = res->numSparseHandles; i < numSparseHandles; ++i) {
    handles[i] = NULL;
    THCusparseCheck(cusparseCreate(&handles[i]));
  }
  res->sparseHandles = handles;
  res->numSparseHandles = numSparseHandles;

  THCudaCheck(cudaSetDevice(prevDev));
}

void THCState_reserveBlasHandles(THCState* state, int numBlasHandles)
{
  // cuBLAS handles are created lazily from THCState_getDeviceBlasHandle
  // to avoid initializing unused devices
  if (numBlasHandles > state->numUserBlasHandles)
  {
    state->numUserBlasHandles = numBlasHandles;
  }
}

void THCState_reserveSparseHandles(THCState* state, int numSparseHandles)
{
  // cuBLAS handles are created lazily from THCState_getDeviceSparseHandle
  // to avoid initializing unused devices
  if (numSparseHandles > state->numUserSparseHandles)
  {
    state->numUserSparseHandles = numSparseHandles;
  }
}

int THCState_getNumStreams(THCState* state)
{
  return state->numUserStreams;
}

int THCState_getNumBlasHandles(THCState* state)
{
  return state->numUserBlasHandles;
}

int THCState_getNumSparseHandles(THCState* state)
{
  return state->numUserSparseHandles;
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

cudaStream_t THCState_getDeviceStream(THCState *state, int device, int streamIndex)
{
  if (streamIndex > state->numUserStreams || streamIndex < 0)
  {
    THError("%d is not a stream", streamIndex);
  }
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  THCStream* stream = res->streams[streamIndex];
  return stream->stream;
}

cublasHandle_t THCState_getDeviceBlasHandle(THCState *state, int device, int handle)
{
  if (handle <= 0 || handle > state->numUserBlasHandles) {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserBlasHandles);
  }
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  THCState_reserveDeviceBlasHandles(state, device, handle);
  return res->blasHandles[handle - 1];
}

cusparseHandle_t THCState_getDeviceSparseHandle(THCState *state, int device, int handle)
{
  if (handle <= 0 || handle > state->numUserSparseHandles) {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserSparseHandles);
  }
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  THCState_reserveDeviceSparseHandles(state, device, handle);
  return res->sparseHandles[handle - 1];
}

static THCStream* THCState_getStreamOnDevice(THCState* state, int device)
{
  THCThreadLocal local = state->currentStreams[device];
  THCStream* stream = (THCStream*)THCThreadLocal_get(local);
  if (!stream) {
    stream = THCStream_defaultStream(device);
    THCStream_retain(stream);
    THCThreadLocal_set(local, stream);
  }
  return stream;
}

static void THCState_setStreamOnDevice(THCState *state, int device, THCStream *stream)
{
  THAssert(stream);
  if (stream->device != device) {
    THError("invalid stream; expected stream for device %d, but was on %d",
        device, stream->device);
  }
  THCStream_retain(stream);
  THCThreadLocal local = state->currentStreams[device];
  THCStream_free((THCStream*)THCThreadLocal_get(local));
  THCThreadLocal_set(local, stream);
}

cudaStream_t THCState_getCurrentStreamOnDevice(THCState *state, int device)
{
  THCStream* stream = THCState_getStreamOnDevice(state, device);
  THAssert(stream);
  return stream->stream;
}

cudaStream_t THCState_getCurrentStream(THCState *state)
{
  /* This is called at the point of kernel execution.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    int device;
    THCudaCheck(cudaGetDevice(&device));
    return THCState_getCurrentStreamOnDevice(state, device);
  } else {
    /* assume default stream */
    return NULL;
  }
}

cublasHandle_t THCState_getCurrentBlasHandle(THCState *state)
{
  /* This is called at the point of kernel execution.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    int device;
    THCudaCheck(cudaGetDevice(&device));

    int handle = THCState_getCurrentBlasHandleIndex(state);
    return THCState_getDeviceBlasHandle(state, device, handle);
  }
  THError("THCState and blasHandles must be set as there is no default blasHandle");
  return NULL;
}

cusparseHandle_t THCState_getCurrentSparseHandle(THCState *state)
{
  /* This is called at the point of kernel execution.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    int device;
    THCudaCheck(cudaGetDevice(&device));

    int handle = THCState_getCurrentSparseHandleIndex(state);
    return THCState_getDeviceSparseHandle(state, device, handle);
  }
  THError("THCState and sparseHandles must be set as there is no default sparseHandle");
  return NULL;
}

int THCState_getCurrentStreamIndex(THCState *state)
{
  THCStream* stream = THCState_getStream(state);

  int device;
  THCudaCheck(cudaGetDevice(&device));
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  for (int i = 0; i <= state->numUserStreams; ++i) {
    if (res->streams[i] == stream) {
      return i;
    }
  }

  return -1;
}

int THCState_getCurrentBlasHandleIndex(THCState *state)
{
  void* value = THCThreadLocal_get(state->currentPerDeviceBlasHandle);
  if (value == NULL) {
    return 1;
  }
  return (int) (intptr_t) value;
}

int THCState_getCurrentSparseHandleIndex(THCState *state)
{
  void* value = THCThreadLocal_get(state->currentPerDeviceSparseHandle);
  if (value == NULL) {
    return 1;
  }
  return (int) (intptr_t) value;
}

THCStream* THCState_getStream(THCState *state)
{
  int device;
  THCudaCheck(cudaGetDevice(&device));
  return THCState_getStreamOnDevice(state, device);
}

void THCState_setStream(THCState *state, THCStream *stream)
{
  int device;
  THCudaCheck(cudaGetDevice(&device));
  THCState_setStreamOnDevice(state, device, stream);
}

void THCState_setCurrentStreamIndex(THCState *state, int streamIndex)
{
  if (streamIndex < 0 || streamIndex > state->numUserStreams) {
    THError("%d is not a valid stream, valid range is: (0, %d)", streamIndex,
        state->numUserStreams);
  }

  int device;
  for (device = 0; device < state->numDevices; ++device) {
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
    THCState_setStreamOnDevice(state, device, res->streams[streamIndex]);
  }
}

void THCState_setCurrentBlasHandleIndex(THCState *state, int handle)
{
  if (handle > state->numUserBlasHandles || handle <= 0)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserBlasHandles);
  }
  THCThreadLocal_set(state->currentPerDeviceBlasHandle, (void*)(intptr_t)handle);
}

void THCState_setCurrentSparseHandleIndex(THCState *state, int handle)
{
  if (handle > state->numUserSparseHandles || handle <= 0)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserSparseHandles);
  }
  THCThreadLocal_set(state->currentPerDeviceSparseHandle, (void*)(intptr_t)handle);
}

void* THCState_getCurrentDeviceScratchSpace(THCState* state)
{
  int device = -1;
  THCudaCheck(cudaGetDevice(&device));
  int stream = THCState_getCurrentStreamIndex(state);
  if (stream < 0) {
    // new stream API
    return NULL;
  }
  return THCState_getDeviceScratchSpace(state, device, stream);
}

void* THCState_getDeviceScratchSpace(THCState* state, int dev, int stream)
{
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);
  if (stream > state->numUserStreams || stream < 0) {
    THError("%d is not a stream", stream);
  }
  THCState_initializeScratchSpace(state, dev);
  return res->devScratchSpacePerStream[stream];
}

size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state)
{
  int device = -1;
  THCudaCheck(cudaGetDevice(&device));
  return THCState_getDeviceScratchSpaceSize(state, device);
}

size_t THCState_getDeviceScratchSpaceSize(THCState* state, int device)
{
  THCCudaResourcesPerDevice* res =
    THCState_getDeviceResourcePtr(state, device);

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

      case CUBLAS_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case CUBLAS_STATUS_EXECUTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;

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

void THCSetGCHandler(THCState *state, void (*cutorchGCFunction_)(void *data), void *data )
{
  state->cutorchGCFunction = cutorchGCFunction_;
  state->cutorchGCData = data;
}

cudaError_t THCudaMalloc(THCState *state, void** ptr, size_t size)
{
  THCudaCheck(cudaGetLastError());
  cudaStream_t stream = THCState_getCurrentStream(state);
  THCDeviceAllocator* allocator = state->cudaDeviceAllocator;
  cudaError_t err = allocator->malloc(allocator->state, ptr, size, stream);
  if (state->cutorchGCFunction != NULL && err != cudaSuccess) {
    cudaGetLastError(); // reset OOM error
    (state->cutorchGCFunction)(state->cutorchGCData);
    err = allocator->malloc(allocator->state, ptr, size, stream);
  }
  return err;
}

cudaError_t THCudaFree(THCState *state, void *ptr)
{
  THCDeviceAllocator* allocator = state->cudaDeviceAllocator;
  return allocator->free(allocator->state, ptr);
}

void* THCudaHostAlloc(THCState *state, size_t size)
{
  THCudaCheck(cudaGetLastError());
  THAllocator* allocator = state->cudaHostAllocator;
  return allocator->malloc(NULL, size);
}

void THCudaHostFree(THCState *state, void *ptr)
{
  THAllocator* allocator = state->cudaHostAllocator;
  return allocator->free(NULL, ptr);
}

void THCudaHostRecord(THCState *state, void *ptr)
{
  if (state->cudaHostAllocator == &THCCachingHostAllocator) {
    THCStream* stream = THCState_getStream(state);
    THCCachingHostAllocator_recordEvent(ptr, stream);
  }
}

cudaError_t THCudaMemGetInfo(THCState *state,  size_t* freeBytes, size_t* totalBytes)
{
  size_t largestBlock = 0;
  return THCudaMemGetInfoCached(state, freeBytes, totalBytes, &largestBlock);
}

cudaError_t THCudaMemGetInfoCached(THCState *state,  size_t* freeBytes, size_t* totalBytes, size_t* largestBlock)
{
  size_t cachedBytes = 0;
  THCDeviceAllocator* allocator = state->cudaDeviceAllocator;

  *largestBlock = 0;
  /* get info from CUDA first */
  cudaError_t ret = cudaMemGetInfo(freeBytes, totalBytes);
  if (ret!= cudaSuccess)
    return ret;

  int device;
  ret = cudaGetDevice(&device);
  if (ret!= cudaSuccess)
    return ret;

  /* not always true - our optimistic guess here */
  *largestBlock = *freeBytes;

  if (allocator->cacheInfo != NULL)
    allocator->cacheInfo(allocator->state, device, &cachedBytes, largestBlock);

  /* Adjust resulting free bytes number. largesBlock unused for now */
  *freeBytes += cachedBytes;
  return cudaSuccess;
}

#undef MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
#undef MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE

#include "THCStorage.c"
#include "THCAllocator.c"

/* from THCHalf.h */

half THC_float2half(float f)
{
#if CUDA_VERSION < 9000
  half h;
  TH_float2halfbits(&f, &h.x);
  return h;
#else
  __half_raw h_raw;
  TH_float2halfbits(&f, &h_raw.x);
  return half(h_raw);
#endif
}

float  THC_half2float(half h)
{
  float f;
#if CUDA_VERSION < 9000
  TH_halfbits2float(&h.x, &f);
#else
  __half_raw h_raw(h);
  TH_halfbits2float(&h_raw.x, &f);
#endif
  return f;
}
