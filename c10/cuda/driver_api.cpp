#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <iostream>
namespace c10 {
namespace cuda {

namespace {
DriverAPI create_driver_api() {
  void* handle_0 = dlopen("libcuda.so", RTLD_LAZY | RTLD_NOLOAD);
  TORCH_INTERNAL_ASSERT(handle_0);
  void* handle_1 = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  try {
    TORCH_INTERNAL_ASSERT(handle_1);
  } catch (...) {
    nvml_is_available = false;
  }

/*
#define OPEN_LIBRARIES(name, n)               \
  void* handle_##n = dlopen(name, RTLD_LAZY); \
  TORCH_INTERNAL_ASSERT(handle_##n);

  C10_FORALL_DRIVER_LIBRARIES(OPEN_LIBRARIES)
#undef OPEN_LIBRARIES
*/
  DriverAPI r{};

#define LOOKUP_ENTRY(name, n)                              \
  r.name##_ = ((decltype(&name))dlsym(handle_##n, #name)); \
  try {
    TORCH_INTERNAL_ASSERT(r.name##_)
  C10_FORALL_DRIVER_API(LOOKUP_ENTRY)
  } catch (...) {
    nvml_is_available = false;
  }
#undef LOOKUP_ENTRY

/*
#define LOOKUP_LIBCUDA_ENTRY(name) \
  r.name##_ = ((decltype(&name))dlsym(handle_0, #name)); 
  TORCH_INTERNAL_ASSERT(r.name##_)
  C10_LIBCUDA_DRIVER_API(LOOKUP_LIBCUDA_ENTRY)
#undef LOOKUP_LIBCUDA_ENTRY

if (nvml_is_available) {
#define LOOKUP_NVML_ENTRY(name) \
  r.name##_ = ((decltype(&name))dlsym(handle_1, #name)); 
  TORCH_INTERNAL_ASSERT(r.name##_)
  C10_NVML_DRIVER_API(LOOKUP_NVML_ENTRY)
#undef LOOKUP_NVML_ENTRY
}
*/
  return r;
}
} // namespace

DriverAPI* DriverAPI::get() {
  static DriverAPI singleton = create_driver_api();
  return &singleton;
}

} // namespace cuda
} // namespace c10

#endif
