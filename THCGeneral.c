#include "THCGeneral.h"
#include "TH.h"
#include "THCTensorRandom.h"
#include "THCBlas.h"
#include "THCAllocator.h"
#include <stdlib.h>

/* Size of scratch space available in global memory per each SM + stream */
#define GLOBAL_SCRATCH_SPACE_PER_SM_STREAM 4 * sizeof(float)

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device);

void THCudaInit(THCState* state)
{
  state->cutorchGCFunction = NULL;
  state->cutorchGCData = NULL;

  int count = 0;
  THCudaCheck(cudaGetDeviceCount(&count));

  int device = 0;
  THCudaCheck(cudaGetDevice(&device));

  state->rngState = (THCRNGState*)malloc(sizeof(THCRNGState));
  THCRandom_init(state, count, device);

  THCAllocator_init(state);

  state->numDevices = count;
  state->deviceProperties =
    (struct cudaDeviceProp*)malloc(count * sizeof(struct cudaDeviceProp));

  state->numUserStreams = 0;
  state->numUserBlasHandles = 0;

  /* Enable P2P access between all pairs, if possible */
  THCudaEnablePeerToPeerAccess(state);

  state->resourcesPerDevice = (THCCudaResourcesPerDevice*)
    malloc(count * sizeof(THCCudaResourcesPerDevice));
  for (int i = 0; i < count; ++i) {
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);

    THCudaCheck(cudaSetDevice(i));
    THCudaCheck(cudaGetDeviceProperties(&state->deviceProperties[i], i));
    /* Stream index 0 will be the default stream for convenience; by
       default no user streams are reserved */
    res->streams = NULL;
    res->blasHandles = NULL;

    /* The scratch space that we want to have available per each device is
       based on the number of SMs available per device */
    int numSM = state->deviceProperties[i].multiProcessorCount;
    size_t sizePerStream = numSM * GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
    res->scratchSpacePerStream = sizePerStream;

    /* Allocate scratch space for each stream */
    res->devScratchSpacePerStream = (void**) malloc(sizeof(void*));
    THCudaCheck(THCudaMalloc(state, &res->devScratchSpacePerStream[0],
                           sizePerStream));
  }

  /* Restore to previous device */
  THCudaCheck(cudaSetDevice(device));

  /* Start in the default stream on the current device */
  state->currentPerDeviceStream = 0;
  state->currentStream = NULL;

  /* There is no such thing as a default cublas handle.
     To maintain consistency with streams API, handle 0 is always NULL and we
     start counting at 1
   */
  THCState_reserveBlasHandles(state, 1);
  state->currentPerDeviceBlasHandle = 1;
  state->currentBlasHandle = THCState_getDeviceBlasHandle(state, device, 1);

  state->heapSoftmax = 3e8; // 300MB, adjusted upward dynamically
  state->heapDelta = 0;
}

void THCudaShutdown(THCState* state)
{
  THCRandom_shutdown(state);
  THCAllocator_shutdown(state);

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
    /* Free Torch-defined streams (0 is the default stream) */
    for (int stream = 1; stream <= state->numUserStreams; ++stream) {
      THCudaCheck(cudaStreamDestroy(
                    THCState_getDeviceStream(state, dev, stream)));
    }
    /* Free Torch-defined handles (0 is NULL for consistency with streams API) */
    for (int handle = 1; handle <= state->numUserBlasHandles; ++handle) {
      THCublasCheck(cublasDestroy(
                      THCState_getDeviceBlasHandle(state, dev, handle)));
    }
    /* Free per-stream scratch space; starts at 0 because there is space for
       the default stream as well*/
    for (int stream = 0; stream <= state->numUserStreams; ++stream) {
      THCudaCheck(THCudaFree(state, THCState_getDeviceScratchSpace(state, dev, stream)));
    }

    free(state->resourcesPerDevice[dev].streams);
    free(state->resourcesPerDevice[dev].blasHandles);
    free(state->resourcesPerDevice[dev].devScratchSpacePerStream);
  }
  free(state->resourcesPerDevice);

  THCudaCheck(cudaSetDevice(prevDev));
}

void THCudaEnablePeerToPeerAccess(THCState* state)
{
  /* By default, all direct p2p kernel access (besides copy) is disallowed, */
  /* since direct access without knowing whether or not a certain operation */
  /* should be cross-GPU leads to synchronization errors. The user can choose */
  /* to disable this functionality, however. */
  state->p2pKernelAccessEnabled = 0;

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  int numDevices = -1;
  THCudaCheck(cudaGetDeviceCount(&numDevices));

  state->p2pAccessEnabled = (int**) malloc(sizeof(int*) * numDevices);
  for (int i = 0; i < numDevices; ++i) {
    state->p2pAccessEnabled[i] = (int*) malloc(sizeof(int) * numDevices);
  }

  /* Build a table of all allowed p2p accesses, to avoid checking the p2p
     status at runtime. */
  for (int i = 0; i < numDevices; ++i) {
    THCudaCheck(cudaSetDevice(i));

    for (int j = 0; j < numDevices; ++j) {
      /* Presume no access by default */
      state->p2pAccessEnabled[i][j] = 0;

      if (i == j) {
        /* A GPU can access itself */
        state->p2pAccessEnabled[i][j] = 1;
      } else {
        int access = 0;
        THCudaCheck(cudaDeviceCanAccessPeer(&access, i, j));

        if (access) {
          cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
          if (err == cudaErrorPeerAccessAlreadyEnabled) {
            /* Any future call to cudaGetLastError will now return an error, */
            /* even though we've already dealt with this specific error here. */
            /* Call cudaGetLastError once to reset the last error state. */
            cudaGetLastError();
            continue;
          }

          /* In case there are unknown errors returned from the above */
          THCudaCheck(err);

          /* Access could be enabled */
          state->p2pAccessEnabled[i][j] = 1;
        }
      }
    }
  }

  /* Restore previous device before continuing */
  THCudaCheck(cudaSetDevice(prevDev));
}

int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess)
{
  int numDevices = 0;
  THCudaCheck(cudaGetDeviceCount(&numDevices));
  if (dev < 0 || dev >= numDevices) {
    THError("%d is not a device", dev);
  }

  if (devToAccess < 0 || dev >= numDevices) {
    THError("%d is not a device", devToAccess);
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

int THCState_getNumDevices(THCState *state)
{
  return state->numDevices;
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

    /* +1 for the default stream as well */
    cudaStream_t* newStreams =
      (cudaStream_t*) malloc((numStreams + 1) * sizeof(cudaStream_t));

    void** newScratchSpace =
      (void**) malloc((numStreams + 1) * sizeof(void*));

    /* Copy over old stream data
       (0 is default stream, 1 ... numUserStreams are rest) */
    for (int stream = 0; stream <= state->numUserStreams; ++stream) {
      newStreams[stream] =
        THCState_getDeviceStream(state, dev, stream);
      newScratchSpace[stream] =
        THCState_getDeviceScratchSpace(state, dev, stream);
    }

    /* Allocate new stream resources */
    size_t scratchSpaceSize = THCState_getDeviceScratchSpaceSize(state, dev);
    unsigned int flags =
      nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;

    for (int stream = state->numUserStreams + 1; stream <= numStreams; ++stream) {
      newStreams[stream] = NULL;
      THCudaCheck(cudaStreamCreateWithFlags(newStreams + stream, flags));
      newScratchSpace[stream] = NULL;
      THCudaCheck(THCudaMalloc(state, &newScratchSpace[stream], scratchSpaceSize));
    }

    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);
    free(res->streams);
    res->streams = newStreams;
    free(res->devScratchSpacePerStream);
    res->devScratchSpacePerStream = newScratchSpace;
  }

  state->numUserStreams = numStreams;

  THCudaCheck(cudaSetDevice(prevDev));
}

void THCState_reserveBlasHandles(THCState* state, int numBlasHandles)
{
  if (numBlasHandles <= state->numUserBlasHandles)
  {
    return;
  }

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  /* Otherwise, we have to allocate a new set of blasHandles */
  for (int dev = 0; dev < state->numDevices; ++dev) {
    THCudaCheck(cudaSetDevice(dev));

    /* +1 to be consistent with stream API, blas handle 0 is NULL and unused */
    cublasHandle_t* newBlasHandles =
      (cublasHandle_t*) malloc((numBlasHandles + 1) * sizeof(cublasHandle_t));

    /* Copy over old blasHandles
       (0 is NULL, 1 ... numUserBlasHandles are rest) */
    newBlasHandles[0] = NULL;
    for (int hndl = 1; hndl <= state->numUserBlasHandles; ++hndl) {
      newBlasHandles[hndl] = THCState_getDeviceBlasHandle(state, dev, hndl);
    }

    /* Allocate new handles */
    for (int hndl = state->numUserBlasHandles + 1; hndl <= numBlasHandles; ++hndl) {
      newBlasHandles[hndl] = NULL;
      THCublasCheck(cublasCreate(newBlasHandles + hndl));
    }

    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, dev);
    free(res->blasHandles);
    res->blasHandles = newBlasHandles;
  }

  state->numUserBlasHandles = numBlasHandles;

  THCudaCheck(cudaSetDevice(prevDev));
}

int THCState_getNumStreams(THCState* state)
{
  return state->numUserStreams;
}

int THCState_getNumBlasHandles(THCState* state)
{
  return state->numUserBlasHandles;
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

cudaStream_t THCState_getDeviceStream(THCState *state, int device, int stream)
{
  if (stream > state->numUserStreams || stream < 0)
  {
    THError("%d is not a stream", stream);
  }
  return (THCState_getDeviceResourcePtr(state, device)->streams == NULL) ? 0
    : THCState_getDeviceResourcePtr(state, device)->streams[stream];
}

cublasHandle_t THCState_getDeviceBlasHandle(THCState *state, int device, int handle)
{
  if (handle <= 0 || handle > state->numUserBlasHandles)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserBlasHandles);
  }
  return THCState_getDeviceResourcePtr(state, device)->blasHandles[handle];
}

cudaStream_t THCState_getCurrentStream(THCState *state)
{
  /* This is called at the point of kernel execution.
     For some debugging code or improperly instrumented kernels,
     `state` is null */
  if (state) {
    return state->currentStream;
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
    if (state->currentBlasHandle <= 0) {
      THError("%d is not a valid handle, valid range is: (1, %d)",
              state->currentBlasHandle, state->numUserBlasHandles);
    }
    return state->currentBlasHandle;
  }
  THError("THCState and blasHandles must be set as there is no default blasHandle");
  return NULL;
}

int THCState_getCurrentStreamIndex(THCState *state)
{
  return state->currentPerDeviceStream;
}

int THCState_getCurrentBlasHandleIndex(THCState *state)
{
  if (state->currentPerDeviceBlasHandle <= 0)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            state->currentPerDeviceBlasHandle, state->numUserBlasHandles);
  }
  return state->currentPerDeviceBlasHandle;
}

void THCState_setStream(THCState *state, int device, int stream)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  if (stream > state->numUserStreams || stream < 0)
  {
    THError("%d is not a stream", stream);
  }
  state->currentStream =
    THCState_getDeviceStream(state, device, stream);
  state->currentPerDeviceStream = stream;
  THCublasCheck(cublasSetStream(state->currentBlasHandle,
                                state->currentStream));
}

void THCState_setBlasHandle(THCState *state, int device, int handle)
{  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  if (handle > state->numUserBlasHandles || handle <= 0)
  {
    THError("%d is not a valid handle, valid range is: (1, %d)",
            handle, state->numUserBlasHandles);
  }
  state->currentBlasHandle =
    THCState_getDeviceBlasHandle(state, device, handle);
  state->currentPerDeviceBlasHandle = handle;
  THCublasCheck(cublasSetStream(state->currentBlasHandle, state->currentStream));
}

void THCState_setStreamForCurrentDevice(THCState *state, int stream)
{
  if (state->currentPerDeviceStream != stream)
  {
    int device = -1;
    THCudaCheck(cudaGetDevice(&device));
    THCState_setStream(state, device, stream);
  }
}

void THCState_setBlasHandleForCurrentDevice(THCState *state, int handle)
{
  if (state->currentPerDeviceBlasHandle != handle)
  {
    int device = -1;
    THCudaCheck(cudaGetDevice(&device));
    THCState_setBlasHandle(state, device, handle);
  }
}

void* THCState_getCurrentDeviceScratchSpace(THCState* state)
{
  int device = -1;
  THCudaCheck(cudaGetDevice(&device));
  int stream = THCState_getCurrentStreamIndex(state);

  return THCState_getDeviceScratchSpace(state, device, stream);
}

void* THCState_getDeviceScratchSpace(THCState* state, int device, int stream)
{
  THCCudaResourcesPerDevice* res =
    THCState_getDeviceResourcePtr(state, device);

  if (stream > state->numUserStreams || stream < 0)
  {
    THError("%d is not a stream", stream);
  }

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

static long heapSize = 0; // not thread-local
static const long heapMaxDelta = 1e6;
static const double heapSoftmaxGrowthThresh = 0.8; // grow softmax if >80% max after GC
static const double heapSoftmaxGrowthFactor = 1.4; // grow softmax by 40%

void THCSetGCHandler(THCState *state, void (*cutorchGCFunction_)(void *data), void *data )
{
  state->cutorchGCFunction = cutorchGCFunction_;
  state->cutorchGCData = data;
}

cudaError_t THCudaMalloc(THCState *state, void** ptr, size_t size)
{
  THCudaCheck(cudaGetLastError());
  cudaError_t err = cudaMalloc(ptr, size);
  if (state->cutorchGCFunction != NULL && err != cudaSuccess) {
    cudaGetLastError(); // reset OOM error
    (state->cutorchGCFunction)(state->cutorchGCData);
    err = cudaMalloc(ptr, size);
  }
  return err;
}

cudaError_t THCudaFree(THCState *state, void *ptr)
{
  cudaError_t err = cudaFree(ptr);
  return err;
}

static long applyHeapDelta(THCState *state) {
  long newHeapSize = THAtomicAddLong(&heapSize, state->heapDelta) + state->heapDelta;
  state->heapDelta = 0;
  return newHeapSize;
}

// Here we maintain a dynamic softmax threshold for THC-allocated storages.
// When THC heap size goes above this softmax, the GC hook is triggered.
// If heap size is above 80% of the softmax after GC, then the softmax is
// increased.
static void maybeTriggerGC(THCState *state, long curHeapSize) {
  if (state->cutorchGCFunction != NULL && curHeapSize > state->heapSoftmax) {
    (state->cutorchGCFunction)(state->cutorchGCData);

    // ensure heapSize is accurate before updating heapSoftmax
    long newHeapSize = applyHeapDelta(state);

    if (newHeapSize > state->heapSoftmax * heapSoftmaxGrowthThresh) {
      state->heapSoftmax = state->heapSoftmax * heapSoftmaxGrowthFactor;
    }
  }
}

void THCHeapUpdate(THCState *state, long size) {
  state->heapDelta += size;
  // batch updates to global heapSize to minimize thread contention
  if (labs(state->heapDelta) < heapMaxDelta) {
    return;
  }

  long newHeapSize = applyHeapDelta(state);
  if (size > 0) {
    maybeTriggerGC(state, newHeapSize);
  }
}

#undef GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
