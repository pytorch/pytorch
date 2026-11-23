#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

#include <c10/core/Allocator.h>
#include <ATen/native/mkldnn/IDeepRegistration.h>
#include <ATen/native/mkldnn/detail/PrimitiveCache.h>
#include <ATen/native/mkldnn/detail/Tensor.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace at::native::mkldnn {

// Ported from ideep::engine::cpu_engine.
Engine& Engine::cpu_engine() {
  static Engine cpu_engine(kind::cpu, 0);
  return cpu_engine;
}

namespace {

// Ported from ideep::EnvSetter.
struct EnvSetter {
  // oneDNN will only accept runtime flags which start with "DNNL_/ONEDNN_" from
  // version v2.5. If user setting runtime flags start with MKLDNN_, we need to
  // keep it works for a while before we finally deprecated it. This is a
  // compatibility layer for runtime flags start with MKLDNN_
  EnvSetter() {
    for (auto name : mkldnn_runtime_flags) {
      query_and_set_env(name.c_str());
    }
  }

  void query_and_set_env(std::string name) {
    std::string value;
    if (getenv_user(name, value)) {
      std::string dnnl_name = "DNNL_";
      dnnl_name += std::string(name);
#ifdef _WIN32
      SetEnvironmentVariable(dnnl_name.c_str(), value.c_str());
#else
      setenv(dnnl_name.c_str(), value.c_str(), 1);
#endif
    }
  }

  bool getenv_user(std::string name, std::string& value) {
    std::string name_str = "MKLDNN_" + std::string(name);
    size_t value_length = 0;
    const char* p = getenv(name_str.c_str());
    value_length = p == nullptr ? 0 : strlen(p);
    if (value_length > 0) {
      value += std::string(p);
      return true;
    }
    return false;
  }

  // current runtime flags in mkldnn
  const std::vector<std::string> mkldnn_runtime_flags = {
      "VERBOSE",
      "ITT_TASK_LEVEL",
      "PRIMITIVE_CACHE_CAPACITY",
      "SC_STACK_SIZE",
      "SC_SOFT_STACK_LIMIT",
      "JIT_PROFILE",
      "VERBOSE_TIMESTAMP",
      "DEFAULT_FPMATH_MODE",
      "MAX_CPU_ISA",
      "CPU_ISA_HINTS"};
};

// Ported from ideep::RegisterEngineAllocator.
struct RegisterEngineAllocator {
  RegisterEngineAllocator(
      Engine& eng,
      const std::function<void*(size_t)>& malloc,
      const std::function<void(void*)>& free) {
    // change runtime flag start with "MKLDNN_" to "DNNL_"
    EnvSetter env_setter;
    eng.set_allocator(malloc, free);
  }
};

} // namespace

} // namespace at::native::mkldnn

static at::native::mkldnn::RegisterEngineAllocator cpu_alloc(
  at::native::mkldnn::Engine::cpu_engine(),
  [](size_t size) {
    return c10::GetAllocator(c10::DeviceType::CPU)->raw_allocate(size);
  },
  [](void* p) {
    c10::GetAllocator(c10::DeviceType::CPU)->raw_deallocate(p);
  }
);

namespace at::native::mkldnn{
void clear_computation_cache() {
  // Reset computation_cache for forward convolutions
  // As it also caches max number of OpenMP workers
#ifndef __aarch64__
  utils::computation_cache<dnnl::convolution_forward::primitive_desc>::t_store()
      .clear();
#endif
}

} // namespace  at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
