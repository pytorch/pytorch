#pragma once
#include <level_zero/ze_api.h>

#include <c10/util/Exception.h>
#include <c10/xpu/XPUMacros.h>

#define C10_XPU_DRIVER_CHECK(EXPR)                          \
  do {                                                      \
    ze_result_t __err = EXPR;                               \
    if (__err != ZE_RESULT_SUCCESS) {                       \
      TORCH_CHECK(false, "XPU driver error code: ", __err); \
    }                                                       \
  } while (0)

namespace c10::xpu {

#define C10_LIBXPU_DRIVER_API_REQUIRED(_) \
  _(zeModuleCreate)                       \
  _(zeKernelCreate)                       \
  _(zeKernelGetProperties)                \
  _(zeMemGetAllocProperties)              \
  _(zeModuleBuildLogGetString)            \
  _(zeModuleBuildLogDestroy)              \
  _(zeDeviceGetProperties)                \
  _(zeDeviceGetMemoryProperties)

struct DriverAPI {
#define DECLARE_MEMBER(name) decltype(&name) name##_;
  C10_LIBXPU_DRIVER_API_REQUIRED(DECLARE_MEMBER)
#undef DECLARE_MEMBER

  static C10_XPU_API DriverAPI* get();
};

} // namespace c10::xpu
