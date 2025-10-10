#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

namespace c10::cuda {

namespace {

void* get_symbol(const char* name, int version);

DriverAPI create_driver_api() {
  void* handle_1 = DriverAPI::get_nvml_handle();
  DriverAPI r{};

#define LOOKUP_LIBCUDA_ENTRY_WITH_VERSION_REQUIRED(name, version)            \
  r.name##_ = reinterpret_cast<decltype(&name)>(get_symbol(#name, version)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name);
  C10_LIBCUDA_DRIVER_API_REQUIRED(LOOKUP_LIBCUDA_ENTRY_WITH_VERSION_REQUIRED)
#undef LOOKUP_LIBCUDA_ENTRY_WITH_VERSION_REQUIRED

// Users running drivers between 12.0 and 12.3 will not have these symbols,
// they would be resolved into nullptr, but we guard their usage at runtime
// to ensure safe fallback behavior.
#define LOOKUP_LIBCUDA_ENTRY_WITH_VERSION_OPTIONAL(name, version) \
  r.name##_ = reinterpret_cast<decltype(&name)>(get_symbol(#name, version));
  C10_LIBCUDA_DRIVER_API_OPTIONAL(LOOKUP_LIBCUDA_ENTRY_WITH_VERSION_OPTIONAL)
#undef LOOKUP_LIBCUDA_ENTRY_WITH_VERSION_OPTIONAL

  if (handle_1) {
#define LOOKUP_NVML_ENTRY(name)                          \
  r.name##_ = ((decltype(&name))dlsym(handle_1, #name)); \
  TORCH_INTERNAL_ASSERT(r.name##_, "Can't find ", #name, ": ", dlerror())
    C10_NVML_DRIVER_API(LOOKUP_NVML_ENTRY)
#undef LOOKUP_NVML_ENTRY
  }

  if (handle_1) {
#define LOOKUP_NVML_ENTRY_OPTIONAL(name) \
  r.name##_ = ((decltype(&name))dlsym(handle_1, #name));
    C10_NVML_DRIVER_API_OPTIONAL(LOOKUP_NVML_ENTRY_OPTIONAL)
#undef LOOKUP_NVML_ENTRY_OPTIONAL
  }
  return r;
}

void* get_symbol(const char* name, int version) {
  void* out = nullptr;
  cudaDriverEntryPointQueryResult qres{};

  // CUDA 12.5+ supports version-based lookup
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12050)
  if (auto st = cudaGetDriverEntryPointByVersion(
          name, &out, version, cudaEnableDefault, &qres);
      st == cudaSuccess && qres == cudaDriverEntryPointSuccess && out) {
    return out;
  }
#endif

  // As of CUDA 13, this API is deprecated.
#if defined(CUDA_VERSION) && (CUDA_VERSION < 13000)
  // This fallback to the old API to try getting the symbol again.
  if (auto st = cudaGetDriverEntryPoint(name, &out, cudaEnableDefault, &qres);
      st == cudaSuccess && qres == cudaDriverEntryPointSuccess && out) {
    return out;
  }
#endif

  // If the symbol cannot be resolved, report and return nullptr;
  // the caller is responsible for checking the pointer.
  LOG(INFO) << "Failed to resolve symbol " << name;
  return nullptr;
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

} // namespace c10::cuda

#endif
