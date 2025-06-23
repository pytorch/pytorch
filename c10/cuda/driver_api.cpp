#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

namespace c10::cuda {

namespace {

DriverAPI create_driver_api() {
  void* handle_1 = DriverAPI::get_nvml_handle();
  DriverAPI r{};

#define LOOKUP_LIBCUDA_ENTRY(name)                                  \
  r.name##_ = reinterpret_cast<decltype(&name)>(get_symbol(#name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name)
  C10_LIBCUDA_DRIVER_API(LOOKUP_LIBCUDA_ENTRY)
  C10_LIBCUDA_DRIVER_API_12030(LOOKUP_LIBCUDA_ENTRY)
#undef LOOKUP_LIBCUDA_ENTRY

  if (handle_1) {
#define LOOKUP_NVML_ENTRY(name)                          \
  r.name##_ = ((decltype(&name))dlsym(handle_1, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name, ": ", dlerror())
    C10_NVML_DRIVER_API(LOOKUP_NVML_ENTRY)
#undef LOOKUP_NVML_ENTRY
  }
  return r;
}
} // namespace

void* DriverAPI::get_nvml_handle() {
  static void* nvml_hanle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  return nvml_hanle;
}

C10_EXPORT DriverAPI* DriverAPI::get() {
  static DriverAPI singleton = create_driver_api();
  return &singleton;
}

void* get_symbol(const char* symbol) {
  cudaDriverEntryPointQueryResult driver_result{};
  void* entry_point = nullptr;
  C10_CUDA_CHECK(cudaGetDriverEntryPoint(
      symbol, &entry_point, cudaEnableDefault, &driver_result));
  TORCH_CHECK(
      driver_result == cudaDriverEntryPointSuccess,
      "Could not find CUDA driver entry point for ",
      symbol);
  return entry_point;
}

} // namespace c10::cuda

#endif
