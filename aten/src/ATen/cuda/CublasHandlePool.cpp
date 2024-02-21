#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>

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
 */

namespace at::cuda {

namespace {

#if defined(USE_ROCM) && ROCM_VERSION >= 50700
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
#endif

std::map<std::tuple<void *, void *>, at::DataPtr>& cublas_handle_stream_to_workspace() {
  static auto& instance = *new std::map<std::tuple<void *, void *>, at::DataPtr>;
  return instance;
}

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

void clearCublasWorkspaces() {
  #if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12200
      cublas_handle_stream_to_workspace().clear();
  #endif
}

size_t parseChosenWorkspaceSize() {
  const char * val = getenv("CUBLAS_WORKSPACE_CONFIG");
  /* :4096:2:16:8 default, 32MiB for Hopper */
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  const bool sm90 = properties != nullptr && properties->major == 9 && properties->minor == 0;
  const size_t default_size = sm90 ? 4096 * 8 * 1024 : 4096 * 1024 * 2 + 16 * 1024 * 8;

  if (val) {
    size_t total_size = 0;
    const std::string config(val);
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
      size_t curr_size = (size_t) std::stoi(match.str(1));
      size_t count = (size_t) std::stoi(match.str(2));
      total_size += curr_size * 1024 * count;
      next++;
    }
    return total_size;
  } else {
    return default_size;
  }
}

size_t getChosenWorkspaceSize() {
  size_t pool_size = parseChosenWorkspaceSize();
  return pool_size;
}

at::DataPtr getNewWorkspace() {
  return c10::cuda::CUDACachingAllocator::get()->allocate(getChosenWorkspaceSize());
}

cublasHandle_t getCurrentCUDABlasHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

#if !defined(USE_ROCM)
  CUcontext pctx = nullptr;
  at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx);
  if (C10_UNLIKELY(!pctx)) {
    // workaround for corner case where a primary context exists but is not
    // the current context, seen in multithreaded use-cases
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
  auto stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12200
  // cuBLAS should not need an explicitly allocated workspace after CUDA 12.2
  // to avoid increasing memory usage during graph captures
  // original issue: https://github.com/pytorch/pytorch/pull/83461
  cudaStream_t _stream = stream;
  auto key = std::make_tuple(static_cast<void *>(handle), static_cast<void *>(_stream));
  auto workspace_it = cublas_handle_stream_to_workspace().find(key);
  if (workspace_it == cublas_handle_stream_to_workspace().end()) {
    workspace_it = cublas_handle_stream_to_workspace().insert(workspace_it, {key, getNewWorkspace()});
  }
  TORCH_CUDABLAS_CHECK(cublasSetWorkspace(handle, workspace_it->second.get(), getChosenWorkspaceSize()));
#endif
#if !defined(USE_ROCM)
  // On CUDA >= 11, and architecture >= Ampere, cuBLAS can use TF32 to speedup
  // FP32 data type calculations based on the value of the allow_tf32 flag.
  // To enable TF32, set the math mode of the handle to CUBLAS_TF32_TENSOR_OP_MATH.
  if (!NoTF32Guard::should_disable_tf32() && at::globalContext().allowTF32CuBLAS()) {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
#endif
#if defined(USE_ROCM)
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

#if (!defined(USE_ROCM) && !defined(_MSC_VER)) || (defined(USE_ROCM) && ROCM_VERSION >= 50700)
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

  auto handle = myPoolWindow->reserve(device);
  return handle;
#else
  return reinterpret_cast<cublasLtHandle_t>(getCurrentCUDABlasHandle());
#endif
}
#endif

} // namespace at::cuda
