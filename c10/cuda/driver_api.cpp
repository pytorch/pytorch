#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
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

typedef cudaError_t (*VersionedGetEntryPoint)(
    const char*,
    void**,
    unsigned int,
    unsigned long long, // NOLINT(*)
    cudaDriverEntryPointQueryResult*);
typedef cudaError_t (*GetEntryPoint)(
    const char*,
    void**,
    unsigned long long, // NOLINT(*)
    cudaDriverEntryPointQueryResult*);

void* get_symbol(const char* symbol, int cuda_version) {
  // We link to the libcudart.so already, so can search for it in the current
  // context
  static GetEntryPoint driver_entrypoint_fun = reinterpret_cast<GetEntryPoint>(
      dlsym(RTLD_DEFAULT, "cudaGetDriverEntryPoint"));
  static VersionedGetEntryPoint driver_entrypoint_versioned_fun =
      reinterpret_cast<VersionedGetEntryPoint>(
          dlsym(RTLD_DEFAULT, "cudaGetDriverEntryPointByVersion"));

  cudaDriverEntryPointQueryResult driver_result{};
  void* entry_point = nullptr;
  if (driver_entrypoint_versioned_fun != nullptr) {
    // Found versioned entrypoint function
    cudaError_t result = driver_entrypoint_versioned_fun(
        symbol, &entry_point, cuda_version, cudaEnableDefault, &driver_result);
    TORCH_CHECK(
        result == cudaSuccess,
        "Error calling cudaGetDriverEntryPointByVersion");
  } else {
    TORCH_CHECK(
        driver_entrypoint_fun != nullptr,
        "Error finding the CUDA Runtime-Driver interop.");
    // Versioned entrypoint function not found
    cudaError_t result = driver_entrypoint_fun(
        symbol, &entry_point, cudaEnableDefault, &driver_result);
    TORCH_CHECK(result == cudaSuccess, "Error calling cudaGetDriverEntryPoint");
  }
  TORCH_CHECK(
      driver_result == cudaDriverEntryPointSuccess,
      "Could not find CUDA driver entry point for ",
      symbol);
  return entry_point;
}

} // namespace c10::cuda

#endif
