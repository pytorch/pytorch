#include <ATen/cuda/detail/LazyNVRTC.h>

#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/DynamicLibrary.h>
#include <stdexcept>

namespace at {
namespace cuda {
namespace detail {
namespace _stubs {

at::DynamicLibrary& getCUDALibrary() {
#if defined(_WIN32)
  static at::DynamicLibrary lib("nvcuda.dll");
#else
  static at::DynamicLibrary lib("libcuda.so.1");
#endif
  return lib;
}

at::DynamicLibrary& getNVRTCLibrary() {
  constexpr auto major = CUDA_VERSION / 1000;
  constexpr auto minor = ( CUDA_VERSION / 10 ) % 10;
#if defined(_WIN32)
  auto libname = std::string("nvrtc64_") + std::to_string(major) + std::to_string(minor) + "_0.dll";
  std::string alt_libname;
#else
  static auto lib_version = std::to_string(major) + "." + std::to_string(minor);
  static auto libname = std::string("libnvrtc.so.") + lib_version;
#ifdef NVRTC_SHORTHASH
  static auto alt_libname = std::string("libnvrtc-") + C10_STRINGIZE(NVRTC_SHORTHASH) + ".so." + lib_version;
#else
  std::string alt_libname;
#endif
#endif
  static at::DynamicLibrary lib(libname.c_str(), alt_libname.empty() ? nullptr : alt_libname.c_str());
  return lib;
}

#define _STUB_1(LIB, NAME, RETTYPE, ARG1)                                            \
RETTYPE NAME(ARG1 a1) {                                                              \
  auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); \
  if (!fn)                                                                           \
    throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     \
  lazyNVRTC.NAME = fn;                                                               \
  return fn(a1);                                                                     \
}

#define _STUB_2(LIB, NAME, RETTYPE, ARG1, ARG2)                                      \
RETTYPE NAME(ARG1 a1, ARG2 a2) {                                                     \
  auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); \
  if (!fn)                                                                           \
    throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     \
  lazyNVRTC.NAME = fn;                                                               \
  return fn(a1, a2);                                                                 \
}

#define _STUB_3(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3)                                \
RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3) {                                            \
  auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); \
  if (!fn)                                                                           \
    throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     \
  lazyNVRTC.NAME = fn;                                                               \
  return fn(a1, a2, a3);                                                             \
}

#define _STUB_4(LIB, NAME, RETTYPE, ARG1, ARG2, ARG3, ARG4)                          \
RETTYPE NAME(ARG1 a1, ARG2 a2, ARG3 a3, ARG4 a4) {                                   \
  auto fn = reinterpret_cast<decltype(&NAME)>(get## LIB ## Library().sym(__func__)); \
  if (!fn)                                                                           \
    throw std::runtime_error("Can't get " C10_STRINGIZE(NAME) );                     \
  lazyNVRTC.NAME = fn;                                                               \
  return fn(a1, a2, a3, a4);                                                         \
}

#define CUDA_STUB1(NAME, A1) _STUB_1(CUDA, NAME, CUresult CUDAAPI, A1)
#define CUDA_STUB2(NAME, A1, A2) _STUB_2(CUDA, NAME, CUresult CUDAAPI, A1, A2)
#define CUDA_STUB3(NAME, A1, A2, A3) _STUB_3(CUDA, NAME, CUresult CUDAAPI, A1, A2, A3)
#define CUDA_STUB4(NAME, A1, A2, A3, A4) _STUB_4(CUDA, NAME, CUresult CUDAAPI, A1, A2, A3, A4)

#define NVRTC_STUB1(NAME, A1) _STUB_1(NVRTC, NAME, nvrtcResult, A1)
#define NVRTC_STUB2(NAME, A1, A2) _STUB_2(NVRTC, NAME, nvrtcResult, A1, A2)
#define NVRTC_STUB3(NAME, A1, A2, A3) _STUB_3(NVRTC, NAME, nvrtcResult, A1, A2, A3)

NVRTC_STUB2(nvrtcVersion, int*, int*);
NVRTC_STUB2(nvrtcAddNameExpression, nvrtcProgram, const char * const);

nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int numHeaders,
                               const char * const *headers,
                               const char * const *includeNames) {
  auto fn = reinterpret_cast<decltype(&nvrtcCreateProgram)>(getNVRTCLibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get nvrtcCreateProgram");
  lazyNVRTC.nvrtcCreateProgram = fn;
  return fn(prog, src, name, numHeaders, headers, includeNames);
}

NVRTC_STUB1(nvrtcDestroyProgram, nvrtcProgram *);
NVRTC_STUB2(nvrtcGetPTXSize, nvrtcProgram, size_t *);
NVRTC_STUB2(nvrtcGetPTX, nvrtcProgram, char *);
#if CUDA_VERSION >= 11010
NVRTC_STUB2(nvrtcGetCUBINSize, nvrtcProgram, size_t *);
NVRTC_STUB2(nvrtcGetCUBIN, nvrtcProgram, char *);
#endif
NVRTC_STUB3(nvrtcCompileProgram, nvrtcProgram, int, const char * const *);
_STUB_1(NVRTC, nvrtcGetErrorString, const char *, nvrtcResult);
NVRTC_STUB2(nvrtcGetProgramLogSize,nvrtcProgram, size_t*);
NVRTC_STUB2(nvrtcGetProgramLog, nvrtcProgram, char *);
NVRTC_STUB3(nvrtcGetLoweredName, nvrtcProgram, const char *, const char **);

CUDA_STUB2(cuModuleLoadData, CUmodule *, const void *);
CUDA_STUB3(cuModuleGetFunction, CUfunction *, CUmodule, const char *);
CUDA_STUB4(cuOccupancyMaxActiveBlocksPerMultiprocessor, int *, CUfunction, int, size_t);
CUDA_STUB2(cuGetErrorString, CUresult, const char **);
CUDA_STUB1(cuCtxGetCurrent, CUcontext *);
CUDA_STUB1(cuModuleUnload, CUmodule);
CUDA_STUB3(cuDevicePrimaryCtxGetState, CUdevice, unsigned int *, int *);
CUDA_STUB4(cuLinkCreate, unsigned int, CUjit_option *, void **, CUlinkState *);
CUDA_STUB3(cuLinkComplete, CUlinkState, void **, size_t *);

// Irregularly shaped functions
CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) {
  auto fn = reinterpret_cast<decltype(&cuLaunchKernel)>(getCUDALibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get cuLaunchKernel");
  lazyNVRTC.cuLaunchKernel = fn;
  return fn(f,
            gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
            sharedMemBytes, hStream, kernelParams, extra);
}

CUresult CUDAAPI cuModuleLoadDataEx(CUmodule *module,
                                    const void *image,
                                    unsigned int numOptions,
                                    CUjit_option *options,
                                    void **optionValues) {
  auto fn = reinterpret_cast<decltype(&cuModuleLoadDataEx)>(getCUDALibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get cuModuleLoadDataEx");
  lazyNVRTC.cuModuleLoadDataEx = fn;
  return fn(module, image, numOptions, options, optionValues);
}

CUresult CUDAAPI
cuLinkAddData(CUlinkState state,
              CUjitInputType type,
              void *data,
              size_t size,
              const char *name,
              unsigned int numOptions,
              CUjit_option *options,
              void **optionValues) {
  auto fn = reinterpret_cast<decltype(&cuLinkAddData)>(getCUDALibrary().sym(__func__));
  if (!fn)
    throw std::runtime_error("Can't get cuLinkAddData");
  lazyNVRTC.cuLinkAddData = fn;
  return fn(state, type, data, size, name, numOptions, options, optionValues);
}

} // namespace _stubs

NVRTC lazyNVRTC = {
#define _REFERENCE_MEMBER(name) _stubs::name,
  AT_FORALL_NVRTC(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
};
} // namespace detail
} // namespace cuda
} // namespace at
