#pragma once

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.

#ifdef USE_CUDA

// FIXME: Currently, CPU and CUDA backend are mutually exclusive.
// This is a temporary workaround. We need a better way to support
// multi devices.

#include <cuda.h>
#include <cuda_runtime_api.h>

#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)                    \
  do {                                                     \
    const cudaError_t code = EXPR;                         \
    const char* msg = cudaGetErrorString(code);            \
    if (code != cudaSuccess) {                             \
      throw std::runtime_error(                            \
          std::string("CUDA error: ") + std::string(msg)); \
    }                                                      \
  } while (0)

namespace torch::aot_inductor {

using DeviceStreamType = cudaStream_t;

} // namespace torch::aot_inductor

#elif defined(USE_XPU)
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sstream>
#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)                                   \
  do {                                                                    \
    const ze_result_t status = EXPR;                                      \
    if (status != ZE_RESULT_SUCCESS) {                                    \
      std::stringstream ss;                                               \
      ss << "L0 runtime error: " << std::hex << std::uppercase << status; \
      throw std::runtime_error(ss.str());                                 \
    }                                                                     \
  } while (0)

namespace torch::aot_inductor {

using DeviceStreamType = sycl::queue*;

} // namespace torch::aot_inductor

#else

#define AOTI_RUNTIME_DEVICE_CHECK(EXPR)            \
  bool ok = EXPR;                                  \
  if (!ok) {                                       \
    throw std::runtime_error("CPU runtime error"); \
  }

namespace torch::aot_inductor {

using DeviceStreamType = void*;

} // namespace torch::aot_inductor

#endif // USE_CUDA
