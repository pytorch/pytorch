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
// or
//   globalContext().getNVRTC().cuLoadModule(...)
//
// If a function is missing add it to the list in ATen/cuda/nvrtc_stub/ATenNVRTC.h
// and edit ATen/cuda/detail/LazyNVRTC.cpp accordingly (e.g., via one of the stub
// macros).

#if !defined(USE_ROCM)

#define AT_FORALL_NVRTC_BASE(_)                  \
  _(nvrtcVersion)                                \
  _(nvrtcAddNameExpression)                      \
  _(nvrtcCreateProgram)                          \
  _(nvrtcDestroyProgram)                         \
  _(nvrtcGetPTXSize)                             \
  _(nvrtcGetPTX)                                 \
  _(nvrtcCompileProgram)                         \
  _(nvrtcGetErrorString)                         \
  _(nvrtcGetProgramLogSize)                      \
  _(nvrtcGetProgramLog)                          \
  _(nvrtcGetLoweredName)                         \
  _(cuModuleLoadData)                            \
  _(cuModuleLoadDataEx)                          \
  _(cuModuleGetFunction)                         \
  _(cuOccupancyMaxActiveBlocksPerMultiprocessor) \
  _(cuGetErrorString)                            \
  _(cuLaunchKernel)                              \
  _(cuLaunchCooperativeKernel)                   \
  _(cuCtxGetCurrent)                             \
  _(cuCtxSetCurrent)                             \
  _(cuModuleUnload)                              \
  _(cuDevicePrimaryCtxGetState)                  \
  _(cuDevicePrimaryCtxRetain)                    \
  _(cuLinkCreate)                                \
  _(cuLinkAddData)                               \
  _(cuLinkComplete)                              \
  _(cuFuncSetAttribute)                          \
  _(cuFuncGetAttribute)                          \

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
#define AT_FORALL_NVRTC_EXTENDED(_)              \
  AT_FORALL_NVRTC_BASE(_)                        \
  _(cuTensorMapEncodeTiled)
#else
#define AT_FORALL_NVRTC_EXTENDED(_)              \
  AT_FORALL_NVRTC_BASE(_)
#endif

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
#define AT_FORALL_NVRTC(_) \
  AT_FORALL_NVRTC_EXTENDED(_)  \
  _(nvrtcGetCUBINSize)     \
  _(nvrtcGetCUBIN)
#else
#define AT_FORALL_NVRTC(_) \
  AT_FORALL_NVRTC_EXTENDED(_)
#endif

#else

// NOTE [ ATen NVRTC Stub and HIP ]
//
// ATen's NVRTC stub library, caffe2_nvrtc, provides dynamic loading of both
// NVRTC and driver APIs. While the former is not yet supported for HIP, the
// later is supported and needed (e.g., in CUDAHooks::getDeviceWithPrimaryContext()
// used by tensor.pin_memory()).
//
// The macro below strips out certain unsupported operations on HIP from the full
// list above.
//
// HIP doesn't have
//   cuGetErrorString  (maps to non-functional hipGetErrorString___)
//
// HIP from ROCm 3.5 on renamed hipOccupancyMaxActiveBlocksPerMultiprocessor
// to hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.
#if TORCH_HIP_VERSION < 305
#define HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR hipOccupancyMaxActiveBlocksPerMultiprocessor
#else
#define HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR cuOccupancyMaxActiveBlocksPerMultiprocessor
#endif

#define AT_FORALL_NVRTC(_)                        \
  _(nvrtcVersion)                                 \
  _(nvrtcCreateProgram)                           \
  _(nvrtcAddNameExpression)                       \
  _(nvrtcDestroyProgram)                          \
  _(nvrtcGetPTXSize)                              \
  _(nvrtcGetPTX)                                  \
  _(cuModuleLoadData)                             \
  _(cuModuleGetFunction)                          \
  _(HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR) \
  _(nvrtcGetErrorString)                          \
  _(nvrtcGetProgramLogSize)                       \
  _(nvrtcGetProgramLog)                           \
  _(cuLaunchKernel)                               \
  _(nvrtcCompileProgram)                          \
  _(cuCtxGetCurrent)                              \
  _(nvrtcGetLoweredName)                          \
  _(cuModuleUnload)                               \
  _(cuDevicePrimaryCtxGetState)

#endif

extern "C" typedef struct NVRTC {
#define CREATE_MEMBER(name) decltype(&name) name;
  AT_FORALL_NVRTC(CREATE_MEMBER)
#undef CREATE_MEMBER
} NVRTC;

extern "C" TORCH_CUDA_CPP_API NVRTC* load_nvrtc();
}} // at::cuda
