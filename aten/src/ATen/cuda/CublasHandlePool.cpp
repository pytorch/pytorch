#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/DeviceThreadHandles.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <regex>

namespace at { namespace cuda {

namespace {

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
  cublas_handle_stream_to_workspace().clear();
}

size_t parseChosenWorkspaceSize() {
  const char * val = getenv("CUBLAS_WORKSPACE_CONFIG");
  if (val) {
    size_t total_size = 0;
    const std::string config(val);
    std::regex exp(":([0-9]+):([0-9]+)");
    std::sregex_iterator next(config.begin(), config.end(), exp);
    std::sregex_iterator end;
    if (next == end) {
      TORCH_WARN("Could not parse CUBLAS_WORKSPACE_CONFIG, using default workspace size of 4096.");
      return 4096 * 1024;
    }
    while (next != end) {
      std::smatch match = *next;
      TORCH_CHECK(match.size() == 3, "Expected CUBLAS_WORKSPACE_SPACE_CONFIG match of size 3 (Format :SIZE:COUNT)");
      size_t curr_size = (size_t) std::stoi(match.str(1));
      total_size += curr_size * 1024;
      next++;
    }
    return total_size;
  } else /* :4096:8 */ {
    return 4096 * 1024;
  }
}

size_t getChosenWorkspaceSize() {
  static size_t pool_size = parseChosenWorkspaceSize();
  return pool_size;
}

at::DataPtr getNewWorkspace() {
  return c10::cuda::CUDACachingAllocator::get()->allocate(getChosenWorkspaceSize());
}

cublasHandle_t getCurrentCUDABlasHandle() {
  int device;
  AT_CUDA_CHECK(cudaGetDevice(&device));

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
#if !defined(USE_ROCM) && CUDA_VERSION >= 11000
  // cublasSetWorkspace not available on CUDA 10.2
  cudaStream_t _stream = stream;
  auto key = std::make_tuple(static_cast<void *>(handle), static_cast<void *>(_stream));
  auto workspace_it = cublas_handle_stream_to_workspace().find(key);
  if (workspace_it == cublas_handle_stream_to_workspace().end()) {
    workspace_it = cublas_handle_stream_to_workspace().insert(workspace_it, {key, getNewWorkspace()});
  }
  TORCH_CUDABLAS_CHECK(cublasSetWorkspace(handle, workspace_it->second.get(), getChosenWorkspaceSize()));
#endif
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  // On CUDA >= 11, and architecture >= Ampere, cuBLAS can use TF32 to speedup
  // FP32 data type calculations based on the value of the allow_tf32 flag.
  // To enable TF32, set the math mode of the handle to CUBLAS_TF32_TENSOR_OP_MATH.
  if (!NoTF32Guard::should_disable_tf32() && at::globalContext().allowTF32CuBLAS()) {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
#endif
#if defined(USE_ROCM) && ROCM_VERSION >= 30800
  rocblas_atomics_mode rocblas_mode;
  if (at::globalContext().deterministicAlgorithms()) {
    rocblas_mode = rocblas_atomics_not_allowed;
  } else {
    rocblas_mode = rocblas_atomics_allowed;
  }
  TORCH_CUDABLAS_CHECK(rocblas_set_atomics_mode(handle, rocblas_mode));
#endif
  return handle;
}

}} // namespace at::cuda
