#pragma once
#include <cuda.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>
#include <cuda_runtime.h>

#include <c10/util/Exception.h>
#include <mutex>
#include <unordered_map>

#define C10_CUDA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                     \
    CUresult __err = EXPR;                                                 \
    if (__err != CUDA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      CUresult get_error_str_err [[maybe_unused]] =                        \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      if (get_error_str_err != CUDA_SUCCESS) {                             \
        TORCH_CHECK(false, "CUDA driver error: unknown error");            \
      } else {                                                             \
        TORCH_CHECK(false, "CUDA driver error: ", err_str);                \
      }                                                                    \
    }                                                                      \
  } while (0)

#define C10_LIBCUDA_DRIVER_API(_)   \
  _(cuDeviceGetAttribute)           \
  _(cuMemAddressReserve)            \
  _(cuMemRelease)                   \
  _(cuMemMap)                       \
  _(cuMemAddressFree)               \
  _(cuMemSetAccess)                 \
  _(cuMemUnmap)                     \
  _(cuMemCreate)                    \
  _(cuMemGetAllocationGranularity)  \
  _(cuMemExportToShareableHandle)   \
  _(cuMemImportFromShareableHandle) \
  _(cuMemsetD32Async)               \
  _(cuStreamWriteValue32)           \
  _(cuGetErrorString)

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12030)
#define C10_LIBCUDA_DRIVER_API_12030(_) \
  _(cuMulticastAddDevice)               \
  _(cuMulticastBindMem)                 \
  _(cuMulticastCreate)
#else
#define C10_LIBCUDA_DRIVER_API_12030(_)
#endif

#define C10_NVML_DRIVER_API(_)            \
  _(nvmlInit_v2)                          \
  _(nvmlDeviceGetHandleByPciBusId_v2)     \
  _(nvmlDeviceGetNvLinkRemoteDeviceType)  \
  _(nvmlDeviceGetNvLinkRemotePciInfo_v2)  \
  _(nvmlDeviceGetComputeRunningProcesses) \
  _(nvmlSystemGetCudaDriverVersion_v2)

namespace c10::cuda {

struct DriverAPI {
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBCUDA_DRIVER_API(CREATE_MEMBER)
  C10_LIBCUDA_DRIVER_API_12030(CREATE_MEMBER)
  C10_NVML_DRIVER_API(CREATE_MEMBER)
#undef CREATE_MEMBER
  
  // Legacy API - maintains backward compatibility
  static DriverAPI* get();
  static void* get_nvml_handle();
  
  // New get() API - calls cudaGetDriverEntryPoint directly and returns function pointer
  template<typename FuncType>
  FuncType* get_entry_point(const char* symbol, unsigned int version = 11000) {
    std::lock_guard<std::mutex> lock(entry_point_mutex_);
    
    // Check cache first
    auto it = loaded_entry_points_.find(symbol);
    if (it != loaded_entry_points_.end()) {
      return reinterpret_cast<FuncType*>(it->second);
    }
    
    // Load the function using cudaGetDriverEntryPoint
    void* func_ptr = load_driver_entry_point(symbol, version);
    
    // Cache the result (even if nullptr)
    loaded_entry_points_[symbol] = func_ptr;
    
    return reinterpret_cast<FuncType*>(func_ptr);
  }
  
  // Convenience methods for specific CUDA driver APIs using the new get() approach
  decltype(::cuDeviceGetAttribute)* cuDeviceGetAttribute_() {
    return get_entry_point<decltype(::cuDeviceGetAttribute)>("cuDeviceGetAttribute", 11000);
  }
  
  decltype(::cuMemAddressReserve)* cuMemAddressReserve_() {
    return get_entry_point<decltype(::cuMemAddressReserve)>("cuMemAddressReserve", 10020);
  }
  
  decltype(::cuMemRelease)* cuMemRelease_() {
    return get_entry_point<decltype(::cuMemRelease)>("cuMemRelease", 10020);
  }
  
  decltype(::cuMemMap)* cuMemMap_() {
    return get_entry_point<decltype(::cuMemMap)>("cuMemMap", 10020);
  }
  
  decltype(::cuMemAddressFree)* cuMemAddressFree_() {
    return get_entry_point<decltype(::cuMemAddressFree)>("cuMemAddressFree", 10020);
  }
  
  decltype(::cuMemSetAccess)* cuMemSetAccess_() {
    return get_entry_point<decltype(::cuMemSetAccess)>("cuMemSetAccess", 10020);
  }
  
  decltype(::cuMemUnmap)* cuMemUnmap_() {
    return get_entry_point<decltype(::cuMemUnmap)>("cuMemUnmap", 10020);
  }
  
  decltype(::cuMemCreate)* cuMemCreate_() {
    return get_entry_point<decltype(::cuMemCreate)>("cuMemCreate", 10020);
  }
  
  decltype(::cuMemGetAllocationGranularity)* cuMemGetAllocationGranularity_() {
    return get_entry_point<decltype(::cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity", 10020);
  }
  
  decltype(::cuMemExportToShareableHandle)* cuMemExportToShareableHandle_() {
    return get_entry_point<decltype(::cuMemExportToShareableHandle)>("cuMemExportToShareableHandle", 10020);
  }
  
  decltype(::cuMemImportFromShareableHandle)* cuMemImportFromShareableHandle_() {
    return get_entry_point<decltype(::cuMemImportFromShareableHandle)>("cuMemImportFromShareableHandle", 10020);
  }
  
  decltype(::cuMemsetD32Async)* cuMemsetD32Async_() {
    return get_entry_point<decltype(::cuMemsetD32Async)>("cuMemsetD32Async", 11000);
  }
  
  decltype(::cuStreamWriteValue32)* cuStreamWriteValue32_() {
#if (CUDA_VERSION >= 12000)
    return get_entry_point<decltype(::cuStreamWriteValue32)>("cuStreamWriteValue32", 12000);
#else
    return get_entry_point<decltype(::cuStreamWriteValue32)>("cuStreamWriteValue32", 11000);
#endif
  }
  
  decltype(::cuGetErrorString)* cuGetErrorString_() {
    return get_entry_point<decltype(::cuGetErrorString)>("cuGetErrorString", 11000);
  }

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12030)
  decltype(::cuMulticastAddDevice)* cuMulticastAddDevice_() {
    return get_entry_point<decltype(::cuMulticastAddDevice)>("cuMulticastAddDevice", 12030);
  }
  
  decltype(::cuMulticastBindMem)* cuMulticastBindMem_() {
    return get_entry_point<decltype(::cuMulticastBindMem)>("cuMulticastBindMem", 12030);
  }
  
  decltype(::cuMulticastCreate)* cuMulticastCreate_() {
    return get_entry_point<decltype(::cuMulticastCreate)>("cuMulticastCreate", 12030);
  }
#endif
  
  // Utility methods
  bool is_entry_point_available(const char* symbol, unsigned int version = 11000) {
    return get_entry_point<void>(symbol, version) != nullptr;
  }
  
  void clear_entry_point_cache() {
    std::lock_guard<std::mutex> lock(entry_point_mutex_);
    loaded_entry_points_.clear();
  }
  
  size_t get_cache_size() {
    std::lock_guard<std::mutex> lock(entry_point_mutex_);
    return loaded_entry_points_.size();
  }

private:
  // Helper function for loading driver entry points
  static void* load_driver_entry_point(const char* symbol, unsigned int version);
  
  // Thread-safe cache for loaded entry points
  mutable std::mutex entry_point_mutex_;
  mutable std::unordered_map<std::string, void*> loaded_entry_points_;
};

// Convenience macros for using the new get() API
#define C10_CUDA_DRIVER_API() c10::cuda::DriverAPI::get()

// Macro to call a driver API function with automatic loading via get_entry_point
#define C10_CALL_CUDA_DRIVER_API(funcName, ...) \
  ([&]() { \
    auto* func_ptr = C10_CUDA_DRIVER_API()->funcName##_(); \
    TORCH_CHECK(func_ptr != nullptr, \
                "Failed to load CUDA driver API: ", #funcName); \
    return func_ptr(__VA_ARGS__); \
  })()

// Macro to get a function pointer without calling it
#define C10_GET_CUDA_DRIVER_API(funcName) \
  C10_CUDA_DRIVER_API()->funcName##_()

// Macro to check if an API is available
#define C10_IS_CUDA_DRIVER_API_AVAILABLE(funcName) \
  (C10_CUDA_DRIVER_API()->funcName##_() != nullptr)

} // namespace c10::cuda
