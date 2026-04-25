#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <atomic>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <utility>

#if defined(USE_ROCM)
#include <rocblas/rocblas.h>
#endif

/**
 * Note [hipblaslt handles]
 * ~~~~~~~~~~~~~~~~~~~~~~~~
 * The cublas documentation states:
 * cuBLAS handle (cublasHandle_t) encapsulates a cuBLASLt handle.
 * Any valid cublasHandle_t can be used in place of cublasLtHandle_t with a simple cast.
 *
 * hipblaslt does not behave in this way.
 * A hipblas handle does not encapsulate a hipblaslt handle.
 *
 * To work around this difference in behavior, a separate handle pool is available for ROCm builds.
 * For CUDA builds, getCurrentCUDABlasLtHandle will alias for getCurrentCUDABlasHandle,
 * whereas for ROCm builds, it is a distinct function.
 *
 * Additionally, hipblaslt cannot share a single handle across multiple streams.
 * On ROCm, getCurrentCUDABlasLtHandle returns a handle unique to each (device, stream)
 * pair, rather than just per-device like the cublas handle pool.
 *
 * The workspace pools are separate for ROCm. On CUDA, the env var
 * TORCH_CUBLASLT_UNIFIED_WORKSPACE can be used to opt-in to unifying the workspace pools.
 */

namespace at::cuda {

namespace {
// -1 means no override; use env var / default
std::atomic<int64_t> cublas_workspace_override{-1};
std::atomic<int64_t> cublaslt_workspace_override{-1};
} // namespace

namespace {

#if defined(USE_ROCM)
void createCublasLtHandle(cublasLtHandle_t *handle) {
  TORCH_CUDABLAS_CHECK(cublasLtCreate(handle));
}

void destroyCublasLtHandle(cublasLtHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
    cublasLtDestroy(handle);
#endif
}

using CuBlasLtPoolType = DeviceThreadHandlePool<cublasLtHandle_t, createCublasLtHandle, destroyCublasLtHandle>;

// ugly hack until hipblasSetWorkspace exists
static hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status error) {
    switch(error) {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    TORCH_CHECK(false, "HIPBLAS_STATUS_INVALID_ENUM");
}

static hipblasStatus_t hipblasSetWorkspace_replacement(hipblasHandle_t handle, void* addr, size_t size) {
    return rocBLASStatusToHIPStatus(rocblas_set_workspace((rocblas_handle)handle, addr, size));
}

// hipify mappings file correctly maps this but the function doesn't exist yet
#define hipblasSetWorkspace hipblasSetWorkspace_replacement

#endif

void createCublasHandle(cublasHandle_t *handle) {
  TORCH_CUDABLAS_CHECK(cublasCreate(handle));
}

void destroyCublasHandle(cublasHandle_t handle) {
// this is because of something dumb in the ordering of
// destruction. Sometimes atexit, the cuda context (or something)
// would already be destroyed by the time this gets destroyed. It
// happens in fbcode setting. @colesbury and @soumith decided to not destroy
// the handle as a workaround.
//   - Comments of @soumith copied from cuDNN handle pool implementation
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
  cublasDestroy(handle);
#endif
}

using CuBlasPoolType = DeviceThreadHandlePool<cublasHandle_t, createCublasHandle, destroyCublasHandle>;

} // namespace

WorkspaceMap& cublas_stream_to_workspace() {
  thread_local WorkspaceMap instance;
  return instance;
}

WorkspaceMap& cublaslt_stream_to_workspace() {
  thread_local WorkspaceMap instance;
  return instance;
}

void clearCublasWorkspaces() {
  cublas_stream_to_workspace().clear();
  cublaslt_stream_to_workspace().clear();
}

void clearCublasWorkspacesForStream(cudaStream_t stream) {
  void* stream_ptr = static_cast<void*>(stream);
  std::erase_if(cublas_stream_to_workspace(), [stream_ptr](const auto& entry) {
    return entry.first.second == stream_ptr;
  });
  std::erase_if(cublaslt_stream_to_workspace(), [stream_ptr](const auto& entry) {
    return entry.first.second == stream_ptr;
  });
}

size_t parseChosenWorkspaceSize() {
  auto val = c10::utils::get_env("CUBLAS_WORKSPACE_CONFIG");
#ifdef USE_ROCM
  if (!val) {
    val = c10::utils::get_env("HIPBLAS_WORKSPACE_CONFIG");
  }
  if (!val) {
    // for extra convenience
    val = c10::utils::get_env("ROCBLAS_WORKSPACE_CONFIG");
  }
  /* 32MiB default, 128MiB for gfx94x/gfx95x */
  const bool gfx94_95 = at::detail::getCUDAHooks().isGPUArch({"gfx94", "gfx95"});
  const size_t default_size = gfx94_95 ? 1024 * 128 * 1024 : 1024 * 32 * 1024;
#else
  /* :4096:2:16:8 default, 32MiB for Hopper and Blackwell */
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool use32mb = properties != nullptr && (properties->major == 9 || properties->major == 10 || properties->major == 12);
  const size_t default_size = use32mb ? 4096 * 8 * 1024 : 4096 * 1024 * 2 + 16 * 1024 * 8;
#endif

  if (val) {
    size_t total_size = 0;
    const std::string& config(val.value());
    std::regex exp(":([0-9]+):([0-9]+)");
    std::sregex_iterator next(config.begin(), config.end(), exp);
    std::sregex_iterator end;
    if (next == end) {
      TORCH_WARN("Could not parse CUBLAS_WORKSPACE_CONFIG, using default workspace size of ", default_size, " bytes.");
      return default_size;
    }
    while (next != end) {
      std::smatch match = *next;
      TORCH_CHECK(match.size() == 3, "Expected CUBLAS_WORKSPACE_SPACE_CONFIG match of size 3 (Format :SIZE:COUNT)");
      size_t curr_size = std::stoull(match.str(1));
      size_t count = std::stoull(match.str(2));
      total_size += curr_size * 1024 * count;
      next++;
    }
    return total_size;
  } else {
    return default_size;
  }
}

#define TORCH_CUBLASLT_UNIFIED_WORKSPACE "TORCH_CUBLASLT_UNIFIED_WORKSPACE"
#ifndef USE_ROCM
inline bool unified_cublas_and_lt_workspaces() {
  static auto unified_env_var = c10::utils::check_env(TORCH_CUBLASLT_UNIFIED_WORKSPACE);
#if !defined(FBCODE)
  static bool unified = (unified_env_var == std::nullopt) || (unified_env_var == true);
#else
  static bool unified = unified_env_var == true;
#endif
  return unified;
}
#endif

size_t parseCUDABlasLtWorkspaceSize() {
  auto val = c10::utils::get_env("CUBLASLT_WORKSPACE_SIZE");
#ifdef USE_ROCM
  if (!val.has_value()) {
    // accept either env var
    val = c10::utils::get_env("HIPBLASLT_WORKSPACE_SIZE");
  }
  size_t workspace_size = 76*1024; /* Use 76 MB for hipBLASLt */
#else
  /* use CUDABlas default workspace size if unified */
  /* otherwise, use default size in KiB according to #73328 */
  size_t workspace_size = unified_cublas_and_lt_workspaces() ? parseChosenWorkspaceSize() / 1024 : 1024;
#endif

  if (val.has_value()) {
    try {
      workspace_size = std::stoi(val.value());
    } catch (std::invalid_argument const&) {
      TORCH_WARN(
          "invalid CUBLASLT_WORKSPACE_SIZE,",
          " using default workspace size of ",
          workspace_size,
          " KiB.");
    } catch (std::out_of_range const&) {
      TORCH_WARN(
          "CUBLASLT_WORKSPACE_SIZE out of range,",
          " using default workspace size of ",
          workspace_size,
          " KiB.");
    }
  }
  return workspace_size * 1024;
}

size_t getChosenWorkspaceSize() {
  int64_t ov = cublas_workspace_override.load(std::memory_order_relaxed);
  if (ov >= 0) {
    return static_cast<size_t>(ov);
  }
  static size_t pool_size = parseChosenWorkspaceSize();
  return pool_size;
}

void setChosenWorkspaceSize(size_t size) {
  cublas_workspace_override.store(static_cast<int64_t>(size), std::memory_order_relaxed);
}

void setCUDABlasLtWorkspaceSize(size_t size) {
  cublaslt_workspace_override.store(static_cast<int64_t>(size), std::memory_order_relaxed);
}

void resetChosenWorkspaceSize() {
  cublas_workspace_override.store(-1, std::memory_order_relaxed);
}

void resetCUDABlasLtWorkspaceSize() {
  cublaslt_workspace_override.store(-1, std::memory_order_relaxed);
}

size_t getCUDABlasLtWorkspaceSize() {
  int64_t ov = cublaslt_workspace_override.load(std::memory_order_relaxed);
  const size_t pool_size = [&] {
    if (ov >= 0) {
      return static_cast<size_t>(ov);
    }
    static size_t parsed_pool_size = parseCUDABlasLtWorkspaceSize();
    return parsed_pool_size;
  }();
#ifndef USE_ROCM
  if (unified_cublas_and_lt_workspaces()) {
    size_t cublasWorkspaceSize = getChosenWorkspaceSize();
    if (cublasWorkspaceSize < pool_size) {
      TORCH_WARN_ONCE("Requested unified CUBLASLT workspace size of ", pool_size,
                      " bytes exceeds CUBLAS workspace size of ", cublasWorkspaceSize,
                      " bytes. Please increase CUBLAS workspace size",
                      " via CUBLAS_WORKSPACE_CONFIG or decrease requested"
                      " CUBLASLT_WORKSPACE_SIZE. Otherwise CUBLASLT workspace"
                      " size will be limited to the CUBLAS workspace size.");
      return cublasWorkspaceSize;
    }
  }
#endif
  return pool_size;
}

at::DataPtr getNewWorkspace() {
  return c10::cuda::CUDACachingAllocator::get()->allocate(getChosenWorkspaceSize());
}

at::DataPtr getNewCUDABlasLtWorkspace() {
  return c10::cuda::CUDACachingAllocator::get()->allocate(getCUDABlasLtWorkspaceSize());
}

void setCublasWorkspace(cublasHandle_t handle, c10::cuda::CUDAStream stream) {
  c10::DeviceIndex device = stream.device_index();
  cudaStream_t _stream = stream;
  auto key = std::make_pair(static_cast<int>(device), static_cast<void *>(_stream));

  auto& workspace_map = cublas_stream_to_workspace();

  size_t workspace_size = getChosenWorkspaceSize();

  auto workspace_it = workspace_map.find(key);
  if (workspace_it != workspace_map.end() && workspace_it->second.second >= workspace_size) {
    TORCH_CUDABLAS_CHECK(cublasSetWorkspace(
        handle, workspace_it->second.first.get(), workspace_size));
    return;
  }

  auto [it, _] = workspace_map.emplace(key, std::make_pair(getNewWorkspace(), workspace_size));
  TORCH_CUDABLAS_CHECK(
      cublasSetWorkspace(handle, it->second.first.get(), workspace_size));
}

void* getCUDABlasLtWorkspace() {
#ifndef USE_ROCM
  if (unified_cublas_and_lt_workspaces()) {
    c10::DeviceIndex device = c10::cuda::current_device();
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t _stream = stream;
    auto key = std::make_pair(static_cast<int>(device), static_cast<void *>(_stream));
    auto& workspace_map = at::cuda::cublas_stream_to_workspace();
    auto workspace_it = workspace_map.find(key);
    if (workspace_it != workspace_map.end()) {
      return workspace_it->second.first.mutable_get();
    }
    auto [it, _] = workspace_map.emplace(key, std::make_pair(getNewWorkspace(), getChosenWorkspaceSize()));
    return it->second.first.mutable_get();
  }
#endif
  c10::DeviceIndex device = c10::cuda::current_device();
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaStream_t _stream = stream;
  auto key = std::make_pair(static_cast<int>(device), static_cast<void *>(_stream));

  auto& workspace_map = cublaslt_stream_to_workspace();

  auto workspace_it = workspace_map.find(key);
  if (workspace_it != workspace_map.end()) {
    return workspace_it->second.first.mutable_get();
  }

  auto [it, _] = workspace_map.emplace(key, std::make_pair(getNewCUDABlasLtWorkspace(), getCUDABlasLtWorkspaceSize()));
  return it->second.first.mutable_get();
}

cublasHandle_t getCurrentCUDABlasHandle(bool setup) {
  c10::DeviceIndex device = c10::cuda::current_device();

#if !defined(USE_ROCM)
  CUcontext pctx = nullptr;
  at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx);
  if (C10_UNLIKELY(!pctx)) {
    TORCH_WARN_ONCE("Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context...");
    at::globalContext().getNVRTC().cuDevicePrimaryCtxRetain(&pctx, device);
    at::globalContext().getNVRTC().cuCtxSetCurrent(pctx);
  }
#endif

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.

  // Use a leaky singleton for the pool following standard practice around
  // singletons: https://isocpp.org/wiki/faq/ctors#construct-on-first-use-v2
  static auto pool = std::shared_ptr<CuBlasPoolType>(
      new CuBlasPoolType(), [](CuBlasPoolType* p) {
        // Leak the memory.
      });
  thread_local std::unique_ptr<CuBlasPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);

  if (!setup) {
    return handle;
  }

  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
  // We explicitly set the cublas workspace even though CUDA 12.2+ fixed the
  // issue where memory usage increased during graph capture.
  // original issue: https://github.com/pytorch/pytorch/pull/83461
  // This is because in CUDA 12.2+, the use of cudaMallocAsync in cublas
  // will allocate memory dynamically (even if they're cheap) outside
  // PyTorch's CUDA caching allocator. It's possible that CCA used up
  // all the memory and cublas's cudaMallocAsync will return OOM
  setCublasWorkspace(handle, stream);

#if !defined(USE_ROCM)
  // On CUDA >= 11, and architecture >= Ampere, cuBLAS can use TF32 to speedup
  // FP32 data type calculations based on the value of the allow_tf32 flag.
  // To enable TF32, set the math mode of the handle to CUBLAS_TF32_TENSOR_OP_MATH.
  if (!NoTF32Guard::should_disable_tf32() &&
      at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) == at::Float32Precision::TF32) {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
#else
  hipblasAtomicsMode_t hipblas_mode;
  if (at::globalContext().deterministicAlgorithms()) {
    hipblas_mode = HIPBLAS_ATOMICS_NOT_ALLOWED;
  } else {
    hipblas_mode = HIPBLAS_ATOMICS_ALLOWED;
  }
  TORCH_CUDABLAS_CHECK(hipblasSetAtomicsMode(handle, hipblas_mode));
#endif
  return handle;
}

cublasLtHandle_t getCurrentCUDABlasLtHandle() {
#ifdef USE_ROCM
  c10::DeviceIndex device = c10::cuda::current_device();

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.

  // Use a leaky singleton for the pool following standard practice around
  // singletons: https://isocpp.org/wiki/faq/ctors#construct-on-first-use-v2
  static auto pool = std::shared_ptr<CuBlasLtPoolType>(
      new CuBlasLtPoolType(), [](CuBlasLtPoolType* p) {
        // Leak the memory.
      });
  thread_local std::unique_ptr<CuBlasLtPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  // hipblaslt cannot share a single handle across multiple streams,
  // so reserve a handle unique to each (device, stream) pair.
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaStream_t _stream = stream;
  auto handle = myPoolWindow->reserve(device, static_cast<void*>(_stream));
  return handle;
#else
  return reinterpret_cast<cublasLtHandle_t>(getCurrentCUDABlasHandle(/*setup=*/false));
#endif
}

} // namespace at::cuda
