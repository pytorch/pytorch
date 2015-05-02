#include "THCGeneral.h"
#include "TH.h"
#include "THCTensorRandom.h"
#include "THCBlas.h"

void THCudaInit(THCState* state)
{
  int count = 0;
  THCudaCheck(cudaGetDeviceCount(&count));

  int device = 0;
  THCudaCheck(cudaGetDevice(&device));

  state->rngState = (THCRNGState*)malloc(sizeof(THCRNGState));
  THCRandom_init(state, count, device);

  state->blasState = (THCBlasState*)malloc(sizeof(THCBlasState));
  THCudaBlas_init(state, count, device);

  state->numDevices = count;
  state->deviceProperties =
    (struct cudaDeviceProp*)malloc(count * sizeof(struct cudaDeviceProp));

  THCState_setDeviceMode(state, THCStateDeviceModeManual);

  state->numUserStreams = 0;
  state->streamsPerDevice =
    (cudaStream_t**)malloc(count * sizeof(cudaStream_t*));

  /* Enable P2P access between all pairs, if possible */
  THCudaEnablePeerToPeerAccess(state);

  for (int i = 0; i < count; ++i)
  {
    THCudaCheck(cudaSetDevice(i));
    THCudaCheck(cudaGetDeviceProperties(&state->deviceProperties[i], i));
    /* Stream index 0 will be the default stream for convenience; by
       default no user streams are reserved */
    state->streamsPerDevice[i] =
      (cudaStream_t*)malloc(sizeof(cudaStream_t));
    state->streamsPerDevice[i][0] = NULL;
  }

  /* Restore to previous device */
  THCudaCheck(cudaSetDevice(device));

  /* Start in the default stream on the current device */
  state->currentPerDeviceStream = 0;
  state->currentStream = NULL;
}

void THCudaShutdown(THCState* state)
{
  THCRandom_shutdown(state);
  THCudaBlas_shutdown(state);
  free(state->blasState);
  free(state->rngState);
  free(state->deviceProperties);

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  for (int dev = 0; dev < state->numDevices; ++dev) {
    THCudaCheck(cudaSetDevice(dev));

    /* Free Torch-defined streams (0 is the default stream) */
    for (int stream = 1; stream <= state->numUserStreams; ++stream) {
      THCudaCheck(cudaStreamDestroy(state->streamsPerDevice[dev][stream]));
    }

    free(state->streamsPerDevice[dev]);
  }

  free(state->streamsPerDevice);
  THCudaCheck(cudaSetDevice(prevDev));
}

void THCudaEnablePeerToPeerAccess(THCState* state)
{
  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  int numDevices = -1;
  THCudaCheck(cudaGetDeviceCount(&numDevices));

  for (int i = 0; i < numDevices; ++i) {
    THCudaCheck(cudaSetDevice(i));

    for (int j = 0; j < numDevices; ++j) {
      if (i != j) {
        int can = 0;
        THCudaCheck(cudaDeviceCanAccessPeer(&can, i, j));

        if (can) {
          cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
          if (err == cudaErrorPeerAccessAlreadyEnabled) {
            // Any future call to cudaGetLastError will now return an error,
            // even though we've already dealt with this specific error here.
            // Call cudaGetLastError once to reset the last error state.
            cudaGetLastError();

            continue;
          }

          THCudaCheck(err);
        }
      }
    }
  }

  /* Restore previous device before continuing */
  THCudaCheck(cudaSetDevice(prevDev));
}

int THCState_getNumDevices(THCState *state)
{
  return state->numDevices;
}

THCStateDeviceMode THCState_getDeviceMode(THCState* state)
{
  return state->deviceMode;
}

void THCState_setDeviceMode(THCState* state, THCStateDeviceMode mode)
{
  state->deviceMode = mode;
}

void THCState_setDevice(THCState *state, int device)
{
  int curDev;
  THCudaCheck(cudaGetDevice(&curDev));
  if (device != curDev) {
    THCudaCheck(cudaSetDevice(device));
    THCRandom_setGenerator(state, device);
    THCudaBlas_setHandle(state, device);

    /* The stream is per device, so update the stream as well */
    THCState_setStream(state, device, THCState_getCurrentStreamIndex(state));
  }
}

void THCState_reserveStreams(THCState* state, int numStreams)
{
  if (numStreams <= state->numUserStreams)
  {
    return;
  }

  int prevDev = -1;
  THCudaCheck(cudaGetDevice(&prevDev));

  /* Otherwise, we have to allocate a new set of streams */
  cudaStream_t** newStreams =
    (cudaStream_t**) malloc(state->numDevices * sizeof(cudaStream_t*));
  for (int dev = 0; dev < state->numDevices; ++dev) {
    THCudaCheck(cudaSetDevice(dev));

    /* +1 for the default stream as well */
    newStreams[dev] =
      (cudaStream_t*) malloc((numStreams + 1) * sizeof(cudaStream_t));

    /* Copy over old streams
       (0 is default stream, 1 ... numUserStreams are rest) */
    for (int stream = 0; stream <= state->numUserStreams; ++stream) {
      newStreams[dev][stream] = state->streamsPerDevice[dev][stream];
    }

    /* Allocate new streams */
    for (int stream = state->numUserStreams + 1; stream <= numStreams; ++stream) {
      newStreams[dev][stream] = NULL;
      THCudaCheck(cudaStreamCreate(&newStreams[dev][stream]));
    }
  }

  cudaStream_t** oldStreams = state->streamsPerDevice;
  state->streamsPerDevice = newStreams;
  state->numUserStreams = numStreams;

  for (int dev = 0; dev < state->numDevices; ++dev) {
    free(oldStreams[dev]);
  }
  free(oldStreams);

  THCudaCheck(cudaSetDevice(prevDev));
}

int THCState_getNumStreams(THCState* state)
{
  return state->numUserStreams;
}

cudaStream_t THCState_getDeviceStream(THCState *state, int device, int stream)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  /* Stream 0 is the default stream, 1 ... `numUserStreams` are Torch streams */
  if (stream > state->numUserStreams || stream < 0)
  {
    THError("%d is not a stream", stream);
  }

  return state->streamsPerDevice[device][stream];
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

int THCState_getCurrentStreamIndex(THCState *state)
{
  return state->currentPerDeviceStream;
}

void THCState_setStream(THCState *state, int device, int stream)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  /* Stream 0 is the default stream, 1 ... `numUserStreams` are Torch streams */
  if (stream > state->numUserStreams || stream < 0)
  {
    THError("%d is not a stream", stream);
  }

  state->currentStream = state->streamsPerDevice[device][stream];
  state->currentPerDeviceStream = stream;
  THCudaBlas_setStream(state, device, state->currentStream);
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

void __THCudaCheck(cudaError_t err, const char *file, const int line)
{
  if(err != cudaSuccess)
  {
    THError("%s(%i) : cuda runtime error : %s",
            file, line, cudaGetErrorString(err));
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

    THError("%s(%i) : cublas runtime error : %s",
            file, line, errmsg);
  }
}

void THCudaGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size)
{
  const int nThreadPerBlock = 256;
  long nBlockPerGrid = size / nThreadPerBlock;
  long nBlockPerColumn = 0L;
  long nBlockPerRow = 0L;

  if(size % nThreadPerBlock)
    nBlockPerGrid++;

  if(nBlockPerGrid <= 65535)
  {
    nBlockPerRow = nBlockPerGrid;
    nBlockPerColumn = 1;
  }
  else if(nBlockPerGrid <= (65355L * 65355L))
  {
    unsigned int uiSqrt = (unsigned int)(sqrt((float)nBlockPerGrid));
    nBlockPerRow = uiSqrt;
    nBlockPerColumn = uiSqrt;
    while((nBlockPerRow * nBlockPerColumn) < nBlockPerGrid)
      nBlockPerRow++;
  }
  else
    THError("too large vector for Cuda, sorry");

  *nBlockPerColumn_ = (int)nBlockPerColumn;
  *nBlockPerRow_ = (int)nBlockPerRow;
  *nThreadPerBlock_ = (int)nThreadPerBlock;
}
