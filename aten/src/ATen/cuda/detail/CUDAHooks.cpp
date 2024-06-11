#include <ATen/cuda/detail/CUDAHooks.h>

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/core/Vitals.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <ATen/MapAllocator.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>
#include <torch/csrc/CudaIPCTypes.h>

#if AT_CUDNN_ENABLED()
#include <ATen/cudnn/cudnn-wrapper.h>
#endif

#if AT_MAGMA_ENABLED()
#include <magma_v2.h>
#endif

#if defined(USE_ROCM)
#include <miopen/version.h>
#endif

#ifndef USE_ROCM
#include <ATen/cuda/detail/LazyNVRTC.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <cstddef>
#include <functional>
#include <memory>

namespace c10::cuda::_internal {
void setHasPrimaryContext(bool (*func)(DeviceIndex));
}

namespace at::cuda::detail {

const at::cuda::NVRTC& nvrtc();
DeviceIndex current_device();

static void (*magma_init_fn)() = nullptr;

void set_magma_init_fn(void (*fn)()) {
  magma_init_fn = fn;
}

namespace {
bool _hasPrimaryContext(DeviceIndex device_index) {
  TORCH_CHECK(device_index >= 0 && device_index < at::cuda::device_count(),
              "hasPrimaryContext expects a valid device index, but got device_index=", device_index);
  unsigned int ctx_flags;
  // In standalone tests of cuDevicePrimaryCtxGetState, I've seen the "active" argument end up with weird
  // (garbage-looking nonzero) values when the context is not active, unless I initialize it to zero.
  int ctx_is_active = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}

// Register hasPrimaryContext back to c10::cuda
struct _Initializer {
  _Initializer() {
      c10::cuda::_internal::setHasPrimaryContext(_hasPrimaryContext);
  }
  ~_Initializer() {
      c10::cuda::_internal::setHasPrimaryContext(nullptr);
  }
} initializer;
} // anonymous namespace

// Sets the CUDA_MODULE_LOADING environment variable
// if it's not set by the user.
void maybe_set_cuda_module_loading(const std::string &def_value) {
  auto value = std::getenv("CUDA_MODULE_LOADING");
  if (!value) {
#ifdef _WIN32
    auto env_var = "CUDA_MODULE_LOADING=" + def_value;
    _putenv(env_var.c_str());
#else
    setenv("CUDA_MODULE_LOADING", def_value.c_str(), 1);
#endif
  }
}

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
void CUDAHooks::initCUDA() const {
  C10_LOG_API_USAGE_ONCE("aten.init.cuda");
  // Force the update to enable unit testing. This code get executed before unit tests
  // have a chance to enable vitals.
  at::vitals::VitalsAPI.setVital("CUDA", "used", "true", /* force = */ true);

  maybe_set_cuda_module_loading("LAZY");
  const auto num_devices = c10::cuda::device_count_ensure_non_zero();
  c10::cuda::CUDACachingAllocator::init(num_devices);
  at::cuda::detail::init_p2p_access_cache(num_devices);

#if AT_MAGMA_ENABLED()
  TORCH_INTERNAL_ASSERT(magma_init_fn != nullptr, "Cannot initialize magma, init routine not set");
  magma_init_fn();
#endif
}

const Generator& CUDAHooks::getDefaultCUDAGenerator(DeviceIndex device_index) const {
  return at::cuda::detail::getDefaultCUDAGenerator(device_index);
}

Device CUDAHooks::getDeviceFromPtr(void* data) const {
  return at::cuda::getDeviceFromPtr(data);
}

bool CUDAHooks::isPinnedPtr(const void* data) const {
  // First check if driver is broken/missing, in which case PyTorch CPU
  // functionalities should still work, we should report `false` here.
  if (!at::cuda::is_available()) {
    return false;
  }
  // cudaPointerGetAttributes grabs context on the current device, so we set
  // device to one that already has context, if exists.
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = getDeviceIndexWithPrimaryContext();
  if (primary_ctx_device_index.has_value()) {
    device_guard.reset_device(at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
  }
  cudaPointerAttributes attr;
  // We do not believe that CUDA needs mutable access to the data
  // here.
  cudaError_t err = cudaPointerGetAttributes(&attr, data);
#if !defined(USE_ROCM)
  if (err == cudaErrorInvalidValue) {
    (void)cudaGetLastError(); // clear CUDA error
    return false;
  }
  AT_CUDA_CHECK(err);
#else
  // HIP throws hipErrorUnknown here
  if (err != cudaSuccess) {
    (void)cudaGetLastError(); // clear HIP error
    return false;
  }
#endif
  return attr.type == cudaMemoryTypeHost;
}

bool CUDAHooks::hasCUDA() const {
  return at::cuda::is_available();
}

bool CUDAHooks::hasMAGMA() const {
#if AT_MAGMA_ENABLED()
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::hasCuDNN() const {
  return AT_CUDNN_ENABLED();
}

bool CUDAHooks::hasCuSOLVER() const {
#if defined(CUDART_VERSION) && defined(CUSOLVER_VERSION)
  return true;
#elif AT_ROCM_ENABLED()
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::hasCuBLASLt() const {
#if defined(CUDART_VERSION)
  return true;
#elif AT_ROCM_ENABLED()
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::hasROCM() const {
  // Currently, this is same as `compiledWithMIOpen`.
  // But in future if there are ROCm builds without MIOpen,
  // then `hasROCM` should return true while `compiledWithMIOpen`
  // should return false
  return AT_ROCM_ENABLED();
}

#if defined(USE_DIRECT_NVRTC)
static std::pair<std::unique_ptr<at::DynamicLibrary>, at::cuda::NVRTC*> load_nvrtc() {
  return std::make_pair(nullptr, at::cuda::load_nvrtc());
}
#elif !defined(USE_ROCM)
static std::pair<std::unique_ptr<at::DynamicLibrary>, at::cuda::NVRTC*> load_nvrtc() {
  return std::make_pair(nullptr, &at::cuda::detail::lazyNVRTC);
}
#else
static std::pair<std::unique_ptr<at::DynamicLibrary>, at::cuda::NVRTC*> load_nvrtc() {
#if defined(_WIN32)
  std::string libcaffe2_nvrtc = "caffe2_nvrtc.dll";
#elif defined(__APPLE__)
  std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.dylib";
#else
  std::string libcaffe2_nvrtc = "libcaffe2_nvrtc.so";
#endif
  std::unique_ptr<at::DynamicLibrary> libnvrtc_stub(
      new at::DynamicLibrary(libcaffe2_nvrtc.c_str()));
  auto fn = (at::cuda::NVRTC * (*)()) libnvrtc_stub->sym("load_nvrtc");
  return std::make_pair(std::move(libnvrtc_stub), fn());
}
#endif

const at::cuda::NVRTC& nvrtc() {
  // must hold onto DynamicLibrary otherwise it will unload
  static auto handle = load_nvrtc();
  return *handle.second;
}

const at::cuda::NVRTC& CUDAHooks::nvrtc() const {
  return at::cuda::detail::nvrtc();
}

DeviceIndex current_device() {
  c10::DeviceIndex device = 0;
  cudaError_t err = c10::cuda::GetDevice(&device);
  if (err == cudaSuccess) {
    return device;
  }
  return -1;
}

DeviceIndex CUDAHooks::current_device() const {
  return at::cuda::detail::current_device();
}

bool CUDAHooks::hasPrimaryContext(DeviceIndex device_index) const {
  return _hasPrimaryContext(device_index);
}

Allocator* CUDAHooks::getPinnedMemoryAllocator() const {
  return at::cuda::getPinnedMemoryAllocator();
}

Allocator* CUDAHooks::getCUDADeviceAllocator() const {
  return at::cuda::getCUDADeviceAllocator();
}

bool CUDAHooks::compiledWithCuDNN() const {
  return AT_CUDNN_ENABLED();
}

bool CUDAHooks::compiledWithMIOpen() const {
  return AT_ROCM_ENABLED();
}

bool CUDAHooks::supportsDilatedConvolutionWithCuDNN() const {
#if AT_CUDNN_ENABLED()
  // NOTE: extra parenthesis around numbers disable clang warnings about
  // dead code
  return true;
#else
  return false;
#endif
}

bool CUDAHooks::supportsDepthwiseConvolutionWithCuDNN() const {
#if AT_CUDNN_ENABLED()
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // Check for Volta cores
  if (prop->major >= 7) {
    return true;
  } else {
    return false;
  }
#else
  return false;
#endif
}

bool CUDAHooks::supportsBFloat16ConvolutionWithCuDNNv8() const {
#if AT_CUDNN_ENABLED()
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // Check for Volta cores
  if (prop->major >= 8) {
    return true;
  } else {
    return false;
  }
#else
  return false;
#endif
}

long CUDAHooks::versionCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_VERSION;
#else
  AT_ERROR("Cannot query CuDNN version if ATen_cuda is not built with CuDNN");
#endif
}

long CUDAHooks::versionCUDART() const {
#ifdef CUDART_VERSION
  return CUDART_VERSION;
#else
  TORCH_CHECK(
    false,
    "Cannot query CUDART version because CUDART is not available");
#endif
}

bool CUDAHooks::hasCUDART() const {
#ifdef CUDART_VERSION
  return true;
#else
  return false;
#endif
}

std::string CUDAHooks::showConfig() const {
  std::ostringstream oss;

  int runtimeVersion;
  cudaRuntimeGetVersion(&runtimeVersion);

  auto printCudaStyleVersion = [&](int v) {
#ifdef USE_ROCM
    // HIP_VERSION value format was changed after ROCm v4.2 to include the patch number
    if(v < 500) {
      // If major=xx, minor=yy then format -> xxyy
      oss << (v / 100) << "." << (v % 10);
    }
    else {
      // If major=xx, minor=yy & patch=zzzzz then format -> xxyyzzzzz
      oss << (v / 10000000) << "." << (v / 100000 % 100) << "." << (v % 100000);
    }
#else
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
#endif
  };

#if !defined(USE_ROCM)
  oss << "  - CUDA Runtime ";
#else
  oss << "  - HIP Runtime ";
#endif
  printCudaStyleVersion(runtimeVersion);
  oss << "\n";

  // TODO: Make HIPIFY understand CUDART_VERSION macro
#if !defined(USE_ROCM)
  if (runtimeVersion != CUDART_VERSION) {
    oss << "  - Built with CUDA Runtime ";
    printCudaStyleVersion(CUDART_VERSION);
    oss << "\n";
  }
  oss << "  - NVCC architecture flags: " << NVCC_FLAGS_EXTRA << "\n";
#endif

#if !defined(USE_ROCM)
#if AT_CUDNN_ENABLED()


  auto printCudnnStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 100 % 10);
    if (v % 100 != 0) {
      oss << "." << (v % 100);
    }
  };

  size_t cudnnVersion = cudnnGetVersion();
  oss << "  - CuDNN ";
  printCudnnStyleVersion(cudnnVersion);
  size_t cudnnCudartVersion = cudnnGetCudartVersion();
  if (cudnnCudartVersion != CUDART_VERSION) {
    oss << "  (built against CUDA ";
    printCudaStyleVersion(cudnnCudartVersion);
    oss << ")";
  }
  oss << "\n";
  if (cudnnVersion != CUDNN_VERSION) {
    oss << "    - Built with CuDNN ";
    printCudnnStyleVersion(CUDNN_VERSION);
    oss << "\n";
  }
#endif
#else
  // TODO: Check if miopen has the functions above and unify
  oss << "  - MIOpen " << MIOPEN_VERSION_MAJOR << "." << MIOPEN_VERSION_MINOR << "." << MIOPEN_VERSION_PATCH << "\n";
#endif

#if AT_MAGMA_ENABLED()
  oss << "  - Magma " << MAGMA_VERSION_MAJOR << "." << MAGMA_VERSION_MINOR << "." << MAGMA_VERSION_MICRO << "\n";
#endif

  return oss.str();
}

double CUDAHooks::batchnormMinEpsilonCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_BN_MIN_EPSILON;
#else
  AT_ERROR(
      "Cannot query CUDNN_BN_MIN_EPSILON if ATen_cuda is not built with CuDNN");
#endif
}

int64_t CUDAHooks::cuFFTGetPlanCacheMaxSize(DeviceIndex device_index) const {
  return at::native::detail::cufft_get_plan_cache_max_size_impl(device_index);
}

void CUDAHooks::cuFFTSetPlanCacheMaxSize(DeviceIndex device_index, int64_t max_size) const {
  at::native::detail::cufft_set_plan_cache_max_size_impl(device_index, max_size);
}

int64_t CUDAHooks::cuFFTGetPlanCacheSize(DeviceIndex device_index) const {
  return at::native::detail::cufft_get_plan_cache_size_impl(device_index);
}

void CUDAHooks::cuFFTClearPlanCache(DeviceIndex device_index) const {
  at::native::detail::cufft_clear_plan_cache_impl(device_index);
}

int CUDAHooks::getNumGPUs() const {
  return at::cuda::device_count();
}

void CUDAHooks::deviceSynchronize(DeviceIndex device_index) const {
  at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
  c10::cuda::device_synchronize();
}

void CUDAHooks::getIpcHandleSize(size_t& ipc_memory_handle_size,
                                 size_t& ipc_event_handle_size) const{
  ipc_memory_handle_size = CUDA_IPC_HANDLE_SIZE;
  ipc_event_handle_size = CUDA_IPC_HANDLE_SIZE;
}

void CUDAHooks::StorageShareDevice(const c10::Storage& storage,
                                   ptrdiff_t& offset_bytes,
                                   std::unique_ptr<char[]>& new_memory_handle,
                                   std::unique_ptr<char[]>& new_event_handle,
                                   std::unique_ptr<char[]>& new_ref_counter,
                                   uint64_t& new_ref_counter_offset,
                                   bool& new_event_sync_required) const {

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t base_size;
  void* base_ptr = c10::cuda::CUDACachingAllocator::getBaseAllocation(
      storage.mutable_data(), &base_size);
  offset_bytes = (char*)storage.data() - (char*)base_ptr;

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  cudaIpcMemHandle_t handle;
  C10_CUDA_CHECK(cudaIpcGetMemHandle(&handle, base_ptr));
  std::memcpy(new_memory_handle.get(), (char*)&handle, sizeof(cudaIpcMemHandle_t));

  // Put Storage Data behind new ref counting context
  // See Note [CUDA IPC Refcounting implementation explained]
  at::DataPtr sent_data_ptr = torch::GetNewRefCountedSentData(
      storage.mutable_data(), storage.device());
  auto old_data_ptr = storage.set_data_ptr(std::move(sent_data_ptr));
  auto sent_data =
      static_cast<torch::CudaIPCSentData*>(storage.data_ptr().get_context());
  sent_data->set_original_ptr(std::move(old_data_ptr));
  std::memcpy(new_ref_counter.get(), (sent_data->handle()).c_str(), sizeof(sent_data->handle()));
  new_ref_counter_offset = sent_data->offset();

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  cudaIpcEventHandle_t ipc_event_handle;
  if (sent_data->event_sync_required_) {
    C10_CUDA_CHECK(
        cudaIpcGetEventHandle(&ipc_event_handle, sent_data->event_));
  }
  std::memcpy(new_event_handle.get(), (char*)&ipc_event_handle, sizeof(cudaIpcEventHandle_t));
  new_event_sync_required = sent_data->event_sync_required_;
}

void CUDAHooks::StorageNewSharedDevice(const c10::DeviceIndex& device,
                                       bool& event_sync_required,
                                       std::string& s_ipc_event_handle,
                                       std::string& s_handle,
                                       std::string& ref_counter_handle,
                                       ptrdiff_t& ref_counter_offset,
                                       ptrdiff_t& storage_offset_bytes,
                                       c10::DataPtr& data_ptr) const {
  if (event_sync_required) {
    auto ipc_event_handle = reinterpret_cast<const cudaIpcEventHandle_t*>(
        s_ipc_event_handle.c_str());
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    cudaEvent_t event;
    cudaIpcOpenEventHandle(&event, *ipc_event_handle);
    C10_CUDA_CHECK(
        cudaStreamWaitEvent(c10::cuda::getCurrentCUDAStream(device), event, 0));
    cudaEventDestroy(event);
  }
  if (s_handle.empty()) return;

  std::shared_ptr<void> basePtr = c10::cuda::CUDACachingAllocator::getIpcDevPtr(s_handle);
  // Offset the basePtr to reconstruct the real storage
  // devPtr = basePtr + storage_offset
  void* devPtr = basePtr.get();
  devPtr = (char*)devPtr + storage_offset_bytes;

  struct IpcDeleterContext {
    std::string ref_counter_handle;
    ptrdiff_t ref_counter_offset{};
    c10::DeviceIndex device{-1};
    torch::CudaIPCReceivedData received_data;
  };

  auto ctx = std::make_unique<IpcDeleterContext>();
  ctx->ref_counter_handle = std::move(ref_counter_handle);
  ctx->ref_counter_offset = ref_counter_offset;
  ctx->device = device;
  ctx->received_data.shared_ptr_ = std::move(basePtr);

  auto cur_device = at::cuda::current_device();

  data_ptr = c10::DataPtr(
      devPtr,
      ctx.release(),
      +[](void* ctx_) {
        std::unique_ptr<IpcDeleterContext> ctx(
            static_cast<IpcDeleterContext*>(ctx_));
        ctx->received_data.shared_ptr_.reset();

        // Sync default stream to make sure all operations related to the
        // storage is finished (otherwise another process may reuse memory and
        // corrupt data)

        // Ideally all shared memory reference counting could be replaced by
        // sending untriggered CUDA event from the producer to consumer and
        // using this event as the criteria of memory release. However, CUDA
        // (atm 10.1) does not support the creation of untriggered events and
        // performance impact of having thousands of shared events is unknown.

        // TODO: Instead of cudaStreamSynchronize it is possible to add Stream
        // Callback and release counter inside of it (need to check performance
        // impact)
        at::cuda::stream_synchronize(
            c10::cuda::getCurrentCUDAStream(ctx->device));

        // We don't want to break existing code, so resource deletion is best
        // effort basis. Exception expected if producer process terminated
        // before consumer released data.
        int flags =
            at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE;
        try {
          auto sptr = at::RefcountedMapAllocator::makeDataPtr(
              ctx->ref_counter_handle.c_str(),
              flags,
              sizeof(int64_t) * torch::CUDA_IPC_REF_COUNTER_FILE_SIZE,
              nullptr);
          *(static_cast<int64_t*>(sptr.get()) + ctx->ref_counter_offset) -= 1;
        } catch (c10::Error& err) {
          // Already warned inside of producer process
        }
      },
      at::Device(at::DeviceType::CUDA, cur_device));
}

void CUDAHooks::getIpcRefCounterFileSize(int64_t& ipc_ref_counter_file_size) const {
  ipc_ref_counter_file_size = torch::CUDA_IPC_REF_COUNTER_FILE_SIZE;
}

// Sigh, the registry doesn't support namespaces :(
using at::CUDAHooksRegistry;
using at::RegistererCUDAHooksRegistry;

REGISTER_CUDA_HOOKS(CUDAHooks);

} // namespace at::cuda::detail
