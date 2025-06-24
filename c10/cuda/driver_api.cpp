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

void* get_symbol(const char* name) {
  int runtime_ver = 0, driver_ver = 0;
  C10_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_ver));
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_ver));

  void* out = nullptr;
  cudaDriverEntryPointQueryResult qres{};
  if (auto st = cudaGetDriverEntryPoint(name, &out, cudaEnableDefault, &qres);
      st == cudaSuccess && qres == cudaDriverEntryPointSuccess && out) {
    return out;
  }

  unsigned int req_ver = std::min(runtime_ver, driver_ver);
  if (auto st = cudaGetDriverEntryPointByVersion(
          name, &out, req_ver, cudaEnableDefault, &qres);
      st == cudaSuccess && qres == cudaDriverEntryPointSuccess && out) {
    return out;
  }

  // If the symbol cannot be resolved, issue a warning and return nullptr;
  // the caller is responsible for checking the pointer.
  LOG(WARNING) << "Failed to resolve symbol " << name << " with runtime_ver "
               << runtime_ver << " and driver_ver " << driver_ver;
  return nullptr;
}

} // namespace c10::cuda

#endif
