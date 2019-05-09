#pragma once

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

extern "C" typedef struct THNVRTC {
#define CREATE_MEMBER(name) decltype(&name) name;
  TORCH_FORALL_NVRTC(CREATE_MEMBER)
#undef CREATE_MEMBER
} THNVRTC;

extern "C" THNVRTC* torch_load_nvrtc();
