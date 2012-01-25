#ifndef THC_GENERAL_INC
#define THC_GENERAL_INC

#include "THGeneral.h"

#include "cuda.h"
#include "cublas.h"
//#include "cuda_runtime_api.h"

TH_API void THCudaInit(void);
TH_API void THCudaShutdown(void);

#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)
#define THCublasCheck()  __THCublasCheck( __FILE__, __LINE__)

TH_API void __THCudaCheck(cudaError_t err, const char *file, const int line);
TH_API void __THCublasCheck(const char *file, const int line);

TH_API void THCudaGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size);

#endif
