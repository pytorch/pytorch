#if !defined(USE_ROCM) && !defined(USE_BAZEL) && !defined(FBCODE_CAFFE2) && \
    !defined(OVRSOURCE) && !defined(_WIN32)

#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <iostream>
namespace c10 {
namespace cuda {

namespace {
DriverAPI create_driver_api() {
  void* handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_NOLOAD);
  TORCH_INTERNAL_ASSERT(handle);
  DriverAPI r;

#define LOOKUP_ENTRY(name)                             \
  r.name##_ = ((decltype(&name))dlsym(handle, #name)); \
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
