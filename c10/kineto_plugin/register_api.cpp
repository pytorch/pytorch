#include "register_api.h"

#include <libkineto.h>

#include <iostream>

#ifdef C10_XPU_BUILD_KINETO_PLUGIN 
#include <c10/kineto_plugin/xpu/XPUActivityProfiler.h>
#endif

namespace c10 {
namespace kineto_plugin {

#define REGISTER_KINETO_PLUGIN(PROFILER_TYPE)                 \
  libkineto::api().registerProfilerFactory(                   \
      []() -> std::unique_ptr<libkineto::IActivityProfiler> { \
        return std::make_unique<PROFILER_TYPE>();             \
      });

void registerKinetoPluginProfiler() {
#ifdef C10_XPU_BUILD_KINETO_PLUGIN
  REGISTER_KINETO_PLUGIN(xpu::XPUActivityProfiler);
#endif
}

}
}
