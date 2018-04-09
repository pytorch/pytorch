#ifndef THCRTS_GENERAL_INC
#define THCRTS_GENERAL_INC

#include "THGeneral.h"
#include "THAllocator.h"
#include "THCThreadLocal.h"

#include "cuda.h"
#include "cuda_runtime.h"

#ifdef __cplusplus
# define THC_EXTERNC extern "C"
#else
# define THC_EXTERNC extern
#endif

#ifdef _WIN32
# ifdef THC_EXPORTS
#  define THC_API THC_EXTERNC __declspec(dllexport)
#  define THC_CLASS __declspec(dllexport)
# else
#  define THC_API THC_EXTERNC __declspec(dllimport)
#  define THC_CLASS __declspec(dllimport)
# endif
#else
# define THC_API THC_EXTERNC
# define THC_CLASS
#endif

typedef struct _THCDeviceAllocator {
   cudaError_t (*malloc)( void*, void**, size_t,         cudaStream_t);
   cudaError_t (*realloc)(void*, void**, size_t, size_t, cudaStream_t);
   cudaError_t (*free)(void*, void*);
   cudaError_t (*emptyCache)(void*);
   cudaError_t  (*cacheInfo)(void*, int, size_t*, size_t*);
   void* state;
} THCDeviceAllocator;


#define THCAssertSameGPU(expr) if (!expr) THError("arguments are located on different GPUs")
#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)
#define THCudaCheckWarn(err)  __THCudaCheckWarn(err, __FILE__, __LINE__)

THC_API void __THCudaCheck(cudaError_t err, const char *file, const int line);
THC_API void __THCudaCheckWarn(cudaError_t err, const char *file, const int line);

#endif
