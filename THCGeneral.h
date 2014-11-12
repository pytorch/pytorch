#ifndef THC_GENERAL_INC
#define THC_GENERAL_INC

#include "THGeneral.h"
#undef log1p

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#ifdef __cplusplus
# define THC_EXTERNC extern "C"
#else
# define THC_EXTERNC extern
#endif

#ifdef WIN32
# ifdef THC_EXPORTS
#  define THC_API THC_EXTERNC __declspec(dllexport)
# else
#  define THC_API THC_EXTERNC __declspec(dllimport)
# endif
#else
# define THC_API THC_EXTERNC
#endif

THC_API void THCudaBlas_init(int num_devices, int current_device);
THC_API void THCudaBlas_shutdown();
THC_API void THCudaBlas_setHandle(int device);

struct THCudaRNGState;  /* Random number generator state. */

/* Global state to be held in the cutorch table. */
typedef struct THCudaState
{
  struct THCudaRNGState* rngState;
} THCudaState;

THC_API void THCudaInit(THCudaState* state);
THC_API void THCudaShutdown(THCudaState* state);

#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)
#define THCublasCheck(err)  __THCublasCheck(err,  __FILE__, __LINE__)

THC_API void __THCudaCheck(cudaError_t err, const char *file, const int line);
THC_API void __THCublasCheck(cublasStatus_t status, const char *file, const int line);

THC_API void THCudaGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size);

#endif
