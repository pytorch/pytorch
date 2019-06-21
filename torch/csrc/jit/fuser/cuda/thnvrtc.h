#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <cuda.h>
#include <nvrtc.h>

// See [USE OF NVRTC AND DRIVER API]

#define TORCH_FORALL_NVRTC(_)                    \
  _(nvrtcVersion)                                \
  _(nvrtcCreateProgram)                          \
  _(nvrtcDestroyProgram)                         \
  _(nvrtcGetPTXSize)                             \
  _(nvrtcGetPTX)                                 \
  _(cuModuleLoadData)                            \
  _(cuModuleGetFunction)                         \
  _(nvrtcGetErrorString)                         \
  _(nvrtcGetProgramLogSize)                      \
  _(nvrtcGetProgramLog)                          \
  _(cuLaunchKernel)                              \
  _(nvrtcCompileProgram)                         \
  _(cuCtxGetCurrent)                             \
  _(cuModuleUnload)                              \
  _(cuDevicePrimaryCtxGetState)

#define TORCH_FORALL_NVRTC_ONLY(_)               \
  _(cuOccupancyMaxActiveBlocksPerMultiprocessor) \
  _(cuGetErrorString)                            \

extern "C" typedef struct THNVRTC {
#define CREATE_MEMBER(name) decltype(&name) name;
  TORCH_FORALL_NVRTC(CREATE_MEMBER)
#ifndef __HIP_PLATFORM_HCC__
  TORCH_FORALL_NVRTC_ONLY(CREATE_MEMBER)
#endif
#undef CREATE_MEMBER
} THNVRTC;

extern "C" TORCH_API THNVRTC* torch_load_nvrtc();
