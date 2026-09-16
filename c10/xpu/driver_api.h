#pragma once
#include <level_zero/ze_api.h>

#include <c10/util/Exception.h>
#include <c10/xpu/XPUMacros.h>

namespace c10::xpu {

#define C10_LIBXPU_DRIVER_API_REQUIRED(_) \
  _(zeModuleCreate)                       \
  _(zeKernelCreate)                       \
  _(zeKernelGetProperties)                \
  _(zeMemGetAllocProperties)              \
  _(zeModuleBuildLogGetString)            \
  _(zeModuleBuildLogDestroy)

struct DriverAPI {
#define DECLARE_MEMBER(name) decltype(&name) name##_;
  C10_LIBXPU_DRIVER_API_REQUIRED(DECLARE_MEMBER)
#undef DECLARE_MEMBER

  static C10_XPU_API DriverAPI* get();
};

} // namespace c10::xpu
