#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <iostream>
namespace c10 {
namespace cuda {

namespace {
DriverAPI create_driver_api() {
#define OPEN_LIBRARIES(name, n)               \
  void* handle_##n = dlopen(name, RTLD_LAZY); \
  TORCH_INTERNAL_ASSERT(handle_##n);

  C10_FORALL_DRIVER_LIBRARIES(OPEN_LIBRARIES)
#undef OPEN_LIBRARIES
  DriverAPI r{};

#define LOOKUP_ENTRY(name, n)                              \
  r.name##_ = ((decltype(&name))dlsym(handle_##n, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_)
  C10_FORALL_DRIVER_API(LOOKUP_ENTRY)
#undef LOOKUP_ENTRY
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
