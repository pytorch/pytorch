#if defined(PYTORCH_C10_DRIVER_API_SUPPORTED)

#include <c10/cuda/driver_api.h>

#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>

namespace c10::cuda {

namespace {

DriverAPI create_driver_api() {
#ifndef USE_ROCM
  c10::string_view libName{"libcuda.so.1"};
#else
  c10::string_view libName{"libamdhip64.so"};
#endif
  void* handle_0 = dlopen(libName.data(), RTLD_LAZY | RTLD_NOLOAD);
  TORCH_CHECK(handle_0, "Can't open libs", dlerror());
#ifndef USE_ROCM
  void* handle_1 = DriverAPI::get_nvml_handle();
#endif
  DriverAPI r{};

#define LOOKUP_LIBCUDA_ENTRY(name)                       \
  r.name##_ = ((decltype(&name))dlsym(handle_0, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name, ": ", dlerror())
  C10_LIBCUDA_DRIVER_API(LOOKUP_LIBCUDA_ENTRY)
#undef LOOKUP_LIBCUDA_ENTRY

#define LOOKUP_LIBCUDA_ENTRY(name)                       \
  r.name##_ = ((decltype(&name))dlsym(handle_0, #name)); \
  dlerror();
  C10_LIBCUDA_DRIVER_API_12030(LOOKUP_LIBCUDA_ENTRY)
#undef LOOKUP_LIBCUDA_ENTRY

#ifndef USE_ROCM
  if (handle_1) {
#define LOOKUP_NVML_ENTRY(name)                          \
  r.name##_ = ((decltype(&name))dlsym(handle_1, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name, ": ", dlerror())
    C10_NVML_DRIVER_API(LOOKUP_NVML_ENTRY)
#undef LOOKUP_NVML_ENTRY
  }
#endif
  return r;
}
} // namespace

#ifndef USE_ROCM
void* DriverAPI::get_nvml_handle() {
  static void* nvml_hanle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  return nvml_hanle;
}
#endif
C10_EXPORT DriverAPI* DriverAPI::get() {
  static DriverAPI singleton = create_driver_api();
  return &singleton;
}

} // namespace c10::cuda

#endif
