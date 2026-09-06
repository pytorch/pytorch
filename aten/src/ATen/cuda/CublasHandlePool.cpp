#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <regex>
#include <shared_mutex>
#include <string>
#include <tuple>
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
 * The workspace pools are separate for ROCm. When workspace caching is enabled on CUDA,
 * TORCH_CUBLASLT_UNIFIED_WORKSPACE can be used to unify the workspace pools.
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

enum class WorkspaceMode { Cached, Explicit, Default };

} // namespace

WorkspaceMapWithMutex& cublas_handle_stream_to_workspace() {
  static auto& instance = *new WorkspaceMapWithMutex;
  return instance;
}

WorkspaceMapWithMutex& cublaslt_handle_stream_to_workspace() {
  static auto& instance = *new WorkspaceMapWithMutex;
  return instance;
}

void clearCublasWorkspaces() {
  {
    auto& workspace = cublas_handle_stream_to_workspace();
    std::unique_lock<std::shared_mutex> lock(workspace.mutex);
    workspace.map.clear();
  }
  {
    auto& workspace = cublaslt_handle_stream_to_workspace();
    std::unique_lock<std::shared_mutex> lock(workspace.mutex);
    workspace.map.clear();
  }
}

void clearCublasWorkspacesForStream(cudaStream_t stream) {
  void* stream_ptr = static_cast<void*>(stream);
  {
    auto& workspace = cublas_handle_stream_to_workspace();
    std::unique_lock<std::shared_mutex> lock(workspace.mutex);
    std::erase_if(workspace.map, [stream_ptr](const auto& entry) {
      return std::get<1>(entry.first) == stream_ptr;
    });
  }
  {
    auto& workspace = cublaslt_handle_stream_to_workspace();
    std::unique_lock<std::shared_mutex> lock(workspace.mutex);
    std::erase_if(workspace.map, [stream_ptr](const auto& entry) {
      return std::get<1>(entry.first) == stream_ptr;
    });
  }
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
  /* 32MiB default, 128MiB for gfx942/gfx950/gfx1250 */
  const bool gfx942_950_1250 = at::detail::getCUDAHooks().isGPUArch({"gfx942", "gfx950", "gfx1250"});
  const size_t default_size = gfx942_950_1250 ? 1024 * 128 * 1024 : 1024 * 32 * 1024;
#else
  /* :4096:2:16:8 default, 32MiB for Hopper and Blackwell */
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool use32mb = properties != nullptr &&
      (properties->major == 9 || properties->major == 10 ||
       properties->major == 11 || properties->major == 12);
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
#define TORCH_CUBLAS_WORKSPACE_CACHE "TORCH_CUBLAS_WORKSPACE_CACHE"
bool isCUDABlasWorkspaceCachingEnabled() {
#ifndef USE_ROCM
  // default false
  static bool cache =
      c10::utils::check_env(TORCH_CUBLAS_WORKSPACE_CACHE) == true;
  return cache;
#else
  // default true
  static bool cache =
      c10::utils::check_env(TORCH_CUBLAS_WORKSPACE_CACHE) != false;
  return cache;
#endif
}

#ifndef USE_ROCM
inline bool unified_cublas_and_lt_workspaces() {
  if (!isCUDABlasWorkspaceCachingEnabled()) {
    return false;
  }
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
  size_t workspace_size = 1024; /* default size in KiB according to #73328 */
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

at::DataPtr allocateCUDABlasWorkspace(size_t size) {
  return c10::cuda::CUDACachingAllocator::get()->allocate(size);
}

void setWorkspaceForHandle(cublasHandle_t handle, c10::cuda::CUDAStream stream) {
  cudaStream_t _stream = stream;
  auto key = std::make_tuple(static_cast<void *>(handle), static_cast<void *>(_stream));

  auto& workspace = cublas_handle_stream_to_workspace();

  size_t workspace_size = getChosenWorkspaceSize();

  // Fast path: check if workspace already exists and is large enough
  {
    std::shared_lock<std::shared_mutex> lock(workspace.mutex);
    auto workspace_it = workspace.map.find(key);
    if (workspace_it != workspace.map.end() && workspace_it->second.second >= workspace_size) {
      TORCH_CUDABLAS_CHECK(cublasSetWorkspace(
          handle, workspace_it->second.first.get(), workspace_size));
      return;
    }
  }

  // Slow path: allocate workspace outside the lock
  auto new_workspace = allocateCUDABlasWorkspace(workspace_size);

  // Insert with lock, replacing any undersized entry
  {
    std::unique_lock<std::shared_mutex> lock(workspace.mutex);
    auto workspace_it = workspace.map.find(key);
    if (workspace_it == workspace.map.end() ||
        workspace_it->second.second < workspace_size) {
      workspace.map.insert_or_assign(
          key, std::make_pair(std::move(new_workspace), workspace_size));
      workspace_it = workspace.map.find(key);
    }
    TORCH_CUDABLAS_CHECK(
        cublasSetWorkspace(handle, workspace_it->second.first.get(), workspace_size));
  }
}

void* getCUDABlasLtWorkspace(size_t workspace_size) {
#ifndef USE_ROCM
  if (unified_cublas_and_lt_workspaces()) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle(/*setup=*/false);
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaStream_t _stream = stream;
    auto key = std::make_tuple(static_cast<void *>(handle), static_cast<void *>(_stream));
    auto& workspace = at::cuda::cublas_handle_stream_to_workspace();
    {
      std::shared_lock<std::shared_mutex> lock(workspace.mutex);
      auto workspace_it = workspace.map.find(key);
      if (workspace_it != workspace.map.end() &&
          workspace_it->second.second >= workspace_size) {
        return workspace_it->second.first.mutable_get();
      }
    }
    // First use for this handle+stream pair — allocate and insert directly.
    // No need to call cublasSetWorkspace; Lt passes workspace explicitly.
    size_t allocation_size = std::max(workspace_size, getChosenWorkspaceSize());
    auto new_workspace = allocateCUDABlasWorkspace(allocation_size);
    {
      std::unique_lock<std::shared_mutex> lock(workspace.mutex);
      auto workspace_it = workspace.map.find(key);
      if (workspace_it == workspace.map.end() ||
          workspace_it->second.second < workspace_size) {
        workspace.map.insert_or_assign(
            key, std::make_pair(std::move(new_workspace), allocation_size));
        workspace_it = workspace.map.find(key);
      }
      return workspace_it->second.first.mutable_get();
    }
  }
#endif
  cublasLtHandle_t handle = getCurrentCUDABlasLtHandle();
  auto stream = c10::cuda::getCurrentCUDAStream();
  cudaStream_t _stream = stream;
  auto key = std::make_tuple(static_cast<void *>(handle), static_cast<void *>(_stream));

  auto& workspace = cublaslt_handle_stream_to_workspace();

  // Fast path: check if workspace already exists and is large enough
  {
    std::shared_lock<std::shared_mutex> lock(workspace.mutex);
    auto workspace_it = workspace.map.find(key);
    if (workspace_it != workspace.map.end() && workspace_it->second.second >= workspace_size) {
      return workspace_it->second.first.mutable_get();
    }
  }

  // Slow path: allocate workspace outside the lock
  auto new_workspace = allocateCUDABlasWorkspace(workspace_size);

  // Insert with lock, replacing any undersized entry
  {
    std::unique_lock<std::shared_mutex> lock(workspace.mutex);
    auto workspace_it = workspace.map.find(key);
    if (workspace_it == workspace.map.end() ||
        workspace_it->second.second < workspace_size) {
      workspace.map.insert_or_assign(
          key, std::make_pair(std::move(new_workspace), workspace_size));
      workspace_it = workspace.map.find(key);
    }
    return workspace_it->second.first.mutable_get();
  }
}

void* getCUDABlasLtWorkspace() {
  return getCUDABlasLtWorkspace(getCUDABlasLtWorkspaceSize());
}

static void setupCUDABlasHandle(
    cublasHandle_t handle,
    c10::cuda::CUDAStream stream,
    void* workspace,
    size_t workspace_size,
    WorkspaceMode workspace_mode) {
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
  // ATen explicitly sets the cuBLAS workspace even though CUDA 12.2+ fixed the
  // issue where memory usage increased during graph capture. Public handles
  // use the cuBLAS default workspace instead when caching is disabled.
  // original issue: https://github.com/pytorch/pytorch/pull/83461
  // This is because in CUDA 12.2+, the use of cudaMallocAsync in cublas
  // will allocate memory dynamically (even if they're cheap) outside
  // PyTorch's CUDA caching allocator. It's possible that CCA used up
  // all the memory and cublas's cudaMallocAsync will return OOM
  switch (workspace_mode) {
    case WorkspaceMode::Cached:
      setWorkspaceForHandle(handle, stream);
      break;
    case WorkspaceMode::Explicit:
      TORCH_CUDABLAS_CHECK(
          cublasSetWorkspace(handle, workspace, workspace_size));
      break;
    case WorkspaceMode::Default:
      // cublasSetStream above resets the handle to cuBLAS's default workspace.
      break;
  }

#if !defined(USE_ROCM)
  // On CUDA >= 11, and architecture >= Ampere, cuBLAS can use TF32 to speedup
  // FP32 data type calculations based on the value of the allow_tf32 flag.
  // To enable TF32, set the math mode of the handle to CUBLAS_TF32_TENSOR_OP_MATH.
  if (!NoTF32Guard::should_disable_fp32_reduced_precision() &&
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
}

static cublasHandle_t getCurrentCUDABlasHandleImpl(
    void* workspace,
    size_t workspace_size,
    WorkspaceMode workspace_mode,
    bool setup) {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

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
  cublasHandle_t handle = myPoolWindow->reserve(device);

  if (!setup) {
    return handle;
  }

  setupCUDABlasHandle(
      handle,
      c10::cuda::getCurrentCUDAStream(),
      workspace,
      workspace_size,
      workspace_mode);
  return handle;
}

cublasHandle_t getCurrentCUDABlasHandle(bool setup) {
  WorkspaceMode workspace_mode = isCUDABlasWorkspaceCachingEnabled()
      ? WorkspaceMode::Cached
      : WorkspaceMode::Default;
  return getCurrentCUDABlasHandleImpl(nullptr, 0, workspace_mode, setup);
}

CUDABlasHandleWithWorkspace::CUDABlasHandleWithWorkspace(
    cublasHandle_t handle,
    cudaStream_t stream,
    at::DataPtr workspace,
    bool restore_default_workspace)
    : handle_(handle),
      stream_(stream),
      workspace_(std::move(workspace)),
      restore_default_workspace_(restore_default_workspace) {}

CUDABlasHandleWithWorkspace::CUDABlasHandleWithWorkspace(
    CUDABlasHandleWithWorkspace&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      stream_(std::exchange(other.stream_, nullptr)),
      workspace_(std::move(other.workspace_)),
      restore_default_workspace_(
          std::exchange(other.restore_default_workspace_, false)) {}

CUDABlasHandleWithWorkspace::~CUDABlasHandleWithWorkspace() {
  if (!restore_default_workspace_) {
    return;
  }
  const cublasStatus_t status = cublasSetStream(handle_, stream_);
  if (C10_UNLIKELY(status != CUBLAS_STATUS_SUCCESS)) {
    // The handle may still refer to this allocation. Retain it rather than
    // leaving a dangling workspace pointer in a handle returned by the public
    // API. Destructors cannot report this failure by throwing.
    (void)workspace_.release_context();
    TORCH_WARN_ONCE(
        "Failed to restore the cuBLAS default workspace: ",
        at::cuda::blas::_cublasGetErrorEnum(status),
        ". Retaining the eager workspace to keep the handle binding valid.");
  }
}

CUDABlasHandleWithWorkspace getCurrentCUDABlasHandleWithWorkspace() {
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (isCUDABlasWorkspaceCachingEnabled()) {
    return CUDABlasHandleWithWorkspace(
        getCurrentCUDABlasHandle(), stream, {}, false);
  }
  size_t workspace_size = getChosenWorkspaceSize();
  auto workspace = allocateCUDABlasWorkspace(workspace_size);
  void* workspace_ptr = workspace.get();
  auto handle = getCurrentCUDABlasHandleImpl(
      nullptr, 0, WorkspaceMode::Default, false);
  CUDABlasHandleWithWorkspace scoped_handle(
      handle, stream, std::move(workspace), true);
  setupCUDABlasHandle(
      handle,
      stream,
      workspace_ptr,
      workspace_size,
      WorkspaceMode::Explicit);
  return scoped_handle;
}

cublasLtHandle_t getCurrentCUDABlasLtHandle() {
#ifdef USE_ROCM
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

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
