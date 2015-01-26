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

  int i,j;
  for(i=0; i < count; ++i)
  {
    THCudaCheck(cudaSetDevice(i));
    for (j=0; j < count; ++j)
    {
      if(i != j)
      {
        int can = 0;
        THCudaCheck(cudaDeviceCanAccessPeer(&can, i, j));
        if(can)
          THCudaCheck(cudaDeviceEnablePeerAccess(j, 0));
      }
    }
  }
  THCudaCheck(cudaSetDevice(device));
}

void THCudaShutdown(THCState* state)
{
  THCRandom_shutdown(state);
  free(state->blasState);
  free(state->rngState);
  THCudaBlas_shutdown(state);
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
