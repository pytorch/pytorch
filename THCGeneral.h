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

#ifndef THAssert
#define THAssert(exp)                                                   \
  do {                                                                  \
    if (!(exp)) {                                                       \
      THError("assert(%s) failed in file %s, line %d", #exp, __FILE__, __LINE__); \
    }                                                                   \
  } while(0)
#endif

struct THCRNGState;  /* Random number generator state. */
struct THCBlasState;

typedef enum THCStateDeviceMode {
  THCStateDeviceModeManual,
  THCStateDeviceModeAuto
} THCStateDeviceMode;

/* Global state to be held in the cutorch table. */
typedef struct THCState
{
  struct THCRNGState* rngState;
  struct THCBlasState* blasState;
  struct cudaDeviceProp* deviceProperties;
  /* Convenience reference to the current stream in use */
  cudaStream_t currentStream;
  /* Set of all allocated streams. streamsPerDevice[dev][0] is NULL, which
     specifies the per-device default stream. */
  cudaStream_t** streamsPerDevice;

  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;
  /* Number of Torch defined streams available, indices 1 ... numStreams */
  int numUserStreams;
  /* Index of the current selected per-device stream. Actual CUDA stream changes
     based on the current device, since streams are per-device */
  int currentPerDeviceStream;
  /* in DeviceModeAuto, cutorch can set the device based on the location of data tensors */
  THCStateDeviceMode deviceMode;
} THCState;

THC_API void THCudaBlas_init(THCState *state, int num_devices, int current_device);
THC_API void THCudaBlas_shutdown(THCState *state);
THC_API void THCudaBlas_reset(THCState *state, int device);
THC_API void THCudaBlas_setHandle(THCState *state, int device);
THC_API void THCudaBlas_setStream(THCState *state, int device, cudaStream_t stream);

THC_API void THCudaInit(THCState* state);
THC_API void THCudaShutdown(THCState* state);
THC_API void THCudaEnablePeerToPeerAccess(THCState* state);

/* State manipulators and accessors */
THC_API int THCState_getNumDevices(THCState* state);
THC_API void THCState_setDevice(THCState* state, int device);
THC_API THCStateDeviceMode THCState_getDeviceMode(THCState* state);
THC_API void THCState_setDeviceMode(THCState* state, THCStateDeviceMode mode);
THC_API void THCState_reserveStreams(THCState* state, int numStreams);
THC_API int THCState_getNumStreams(THCState* state);
THC_API void THCState_resetStreams(THCState* state, int device);

THC_API cudaStream_t THCState_getDeviceStream(THCState *state, int device, int stream);
THC_API cudaStream_t THCState_getCurrentStream(THCState *state);
THC_API int THCState_getCurrentStreamIndex(THCState *state);
THC_API void THCState_setStream(THCState *state, int device, int stream);
THC_API void THCState_setStreamForCurrentDevice(THCState *state, int stream);

#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)
#define THCublasCheck(err)  __THCublasCheck(err,  __FILE__, __LINE__)

THC_API void __THCudaCheck(cudaError_t err, const char *file, const int line);
THC_API void __THCublasCheck(cublasStatus_t status, const char *file, const int line);

THC_API void THCudaGetGridSize(int *nBlockPerColumn_, int *nBlockPerRow_, int *nThreadPerBlock_, long size);

#endif
