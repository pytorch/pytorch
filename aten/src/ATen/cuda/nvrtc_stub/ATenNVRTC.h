#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>
#include <cuda.h>
#include <nvrtc.h>

namespace at { namespace cuda {


// NOTE [ USE OF NVRTC AND DRIVER API ]
//
// ATen does not directly link to either libnvrtc or libcuda because they
// require libcuda to be installed, yet we want our GPU build to work on CPU
// machines as long as CUDA is not initialized.
//
// Normal CUDA code in torch uses the cuda runtime libraries which can be
// installed even if the driver is not installed, but sometimes we specifically
// need to use the driver API (e.g., to load JIT compiled code).
// To accomplish this, we lazily link libcaffe2_nvrtc which provides a struct
// at::cuda::NVRTC that contains function pointers to all of the apis we need.
//
// IT IS AN ERROR TO TRY TO CALL ANY nvrtc* or cu* FUNCTION DIRECTLY.
// INSTEAD USE, e.g.
//   detail::getCUDAHooks().nvrtc().cuLoadModule(...)
// oe
//   globalContext().getNVRTC().cuLoadModule(...)
//
// If a function is missing add it to the list in ATen/cuda/nvrtc_stub/ATenNVRTC.h.

#define AT_FORALL_NVRTC(_)                    \
  _(nvrtcVersion)                                \
  _(nvrtcCreateProgram)                          \
  _(nvrtcDestroyProgram)                         \
  _(nvrtcGetPTXSize)                             \
  _(nvrtcGetPTX)                                 \
  _(cuModuleLoadData)                            \
  _(cuModuleGetFunction)                         \
  _(cuOccupancyMaxActiveBlocksPerMultiprocessor) \
  _(cuGetErrorString)                            \
  _(nvrtcGetErrorString)                         \
  _(nvrtcGetProgramLogSize)                      \
  _(nvrtcGetProgramLog)                          \
  _(cuLaunchKernel)                              \
  _(nvrtcCompileProgram)                         \
  _(cuCtxGetCurrent)                             \
  _(cuModuleUnload)                              \
  _(cuDevicePrimaryCtxGetState)

extern "C" typedef struct NVRTC {
#define CREATE_MEMBER(name) decltype(&name) name;
  AT_FORALL_NVRTC(CREATE_MEMBER)
#undef CREATE_MEMBER
} NVRTC;

extern "C" AT_CUDA_API NVRTC* load_nvrtc();

}} // at::cuda
