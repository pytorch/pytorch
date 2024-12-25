#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Macros.h>
#include <c10/util/WaitCounter.h>

#include <limits>

namespace c10::cuda {

namespace {
// returns -1 on failure
int32_t driver_version() {
  int driver_version = -1;
  C10_CUDA_IGNORE_ERROR(cudaDriverGetVersion(&driver_version));
  return driver_version;
}

int device_count_impl(bool fail_if_no_driver) {
  int count = 0;
  auto err = C10_CUDA_ERROR_HANDLED(c10::cuda::GetDeviceCount(&count));
  if (err == cudaSuccess) {
    return count;
  }
  // Clear out the error state, so we don't spuriously trigger someone else.
  // (This shouldn't really matter, since we won't be running very much CUDA
  // code in this regime.)
  [[maybe_unused]] cudaError_t last_err = cudaGetLastError();
  switch (err) {
    case cudaErrorNoDevice:
      // Zero devices is ok here
      count = 0;
      break;
    case cudaErrorInsufficientDriver: {
      auto version = driver_version();
      if (version <= 0) {
        if (!fail_if_no_driver) {
          // No CUDA driver means no devices
          count = 0;
          break;
        }
        TORCH_CHECK(
            false,
            "Found no NVIDIA driver on your system. Please check that you "
            "have an NVIDIA GPU and installed a driver from "
            "http://www.nvidia.com/Download/index.aspx");
      } else {
        TORCH_CHECK(
            false,
            "The NVIDIA driver on your system is too old (found version ",
            version,
            "). Please update your GPU driver by downloading and installing "
            "a new version from the URL: "
            "http://www.nvidia.com/Download/index.aspx Alternatively, go to: "
            "https://pytorch.org to install a PyTorch version that has been "
            "compiled with your version of the CUDA driver.");
      }
    } break;
    case cudaErrorInitializationError:
      TORCH_CHECK(
          false,
          "CUDA driver initialization failed, you might not "
          "have a CUDA gpu.");
      break;
    case cudaErrorUnknown:
      TORCH_CHECK(
          false,
          "CUDA unknown error - this may be due to an "
          "incorrectly set up environment, e.g. changing env "
          "variable CUDA_VISIBLE_DEVICES after program start. "
          "Setting the available devices to be zero.");
      break;
#if C10_ASAN_ENABLED
    case cudaErrorMemoryAllocation:
      // In ASAN mode, we know that a cudaErrorMemoryAllocation error will
      // pop up if compiled with NVCC (clang-cuda is fine)
      TORCH_CHECK(
          false,
          "Got 'out of memory' error while trying to initialize CUDA. "
          "CUDA with nvcc does not work well with ASAN and it's probably "
          "the reason. We will simply shut down CUDA support. If you "
          "would like to use GPUs, turn off ASAN.");
      break;
#endif // C10_ASAN_ENABLED
    default:
      TORCH_CHECK(
          false,
          "Unexpected error from cudaGetDeviceCount(). Did you run "
          "some cuda functions before calling NumCudaDevices() "
          "that might have already set an error? Error ",
          err,
          ": ",
          cudaGetErrorString(err));
  }
  return count;
}
} // namespace

DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl(/*fail_if_no_driver=*/false);
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many CUDA devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("CUDA initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  // Zero gpus doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No CUDA GPUs are available");
  TORCH_INTERNAL_ASSERT(
      count <= std::numeric_limits<DeviceIndex>::max(),
      "Too many CUDA devices, DeviceIndex overflowed");
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&cur_device));
  return cur_device;
}

void set_device(DeviceIndex device) {
  C10_CUDA_CHECK(c10::cuda::SetDevice(device));
}

void device_synchronize() {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kCUDA);
  }
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.cuda_device_synchronize);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

// this function has to be called from callers performing cuda synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing CUDA operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_WARN("called a synchronizing CUDA operation");
  }
}

std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // check current device first
  auto current_device_index = current_device();
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(at::cuda::device_count())) {
    if (device_index == current_device_index)
      continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return std::nullopt;
}

namespace _internal {
bool dummyHasPrimaryContext([[maybe_unused]] DeviceIndex device_index) {
  TORCH_CHECK(false, "Should never been called");
}
static bool (*hasPrimaryContext)(DeviceIndex) = dummyHasPrimaryContext;

// Private api to be called from CUDAHooks.cpp
C10_CUDA_API void setHasPrimaryContext(bool (*func)(DeviceIndex)) {
  hasPrimaryContext = func ? func : dummyHasPrimaryContext;
}
} // namespace _internal

bool hasPrimaryContext(DeviceIndex device_index) {
  return _internal::hasPrimaryContext(device_index);
}

// Wrappers for raw CUDA device management functions
cudaError_t GetDeviceCount(int* dev_count) {
  return cudaGetDeviceCount(dev_count);
}

// This is a codepath for CUDA 12 that comes with a critical change in behavior
// of `cudaSetDevice`. Unlike to previous CUDA versions that allocate context
// lazily CUDA 12.x eagerly allocates primary context the moment `cudaSetDevice`
// is called. This can lead to dramatic consequences and pollute the device
// memory in distributed runs. To avoid unnecessary context creation a new
// function called `MaybeSetDevice` was introduced. This function is to be
// called in device guard destructor and at the exit of torch.cuda.device
// context manager. The behavior of `MaybeSetDevice` is quite simple, it calls
// to `cudaSetDevice` if context already exist or if context was not allocated
// on targeted device it simply saves the device index. This way we can keep
// PyTorch backward compatible for applications like this:
//
// ```
// import torch
// x = torch.empty(1, device=“cuda:1”) # no CUDA context on cuda:0 after this
// call y = torch.empty(1, device=“cuda”) # CUDA context is created on cuda:0
// ```
#if CUDA_VERSION >= 12000
thread_local static DeviceIndex targetDeviceIndex = -1;

cudaError_t GetDevice(DeviceIndex* device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return cudaSuccess;
  }
  int tmp_device = -1;
  auto err = cudaGetDevice(&tmp_device);
  if (err == cudaSuccess) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<DeviceIndex>::max(),
        "cudaGetDevice returns invalid device ",
        tmp_device);
    *device = static_cast<DeviceIndex>(tmp_device);
  }
  return err;
}

cudaError_t SetDevice(DeviceIndex device) {
  TORCH_CHECK(device >= 0, "device id must be positive!", device);
  targetDeviceIndex = -1;
  int cur_device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  if (device == cur_device) {
    return cudaSuccess;
  }
  return cudaSetDevice(device);
}

cudaError_t MaybeSetDevice(DeviceIndex device) {
  if (hasPrimaryContext(device)) {
    return c10::cuda::SetDevice(device);
  }
  targetDeviceIndex = device;
  return cudaSuccess;
}

// This function always initializes the CUDA context
// on to_device
DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&tmp_device));
    cur_device = static_cast<DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  C10_CUDA_CHECK(cudaSetDevice(to_device));
  return cur_device;
}

// This function does not initialize the CUDA context
// on to_device if it does not already exist
DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  int tmp_cur_device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&tmp_cur_device));
  TORCH_INTERNAL_ASSERT(
      tmp_cur_device >= 0 &&
          tmp_cur_device <= std::numeric_limits<DeviceIndex>::max(),
      "cudaGetDevice returns invalid device ",
      tmp_cur_device);
  auto cur_device = static_cast<DeviceIndex>(tmp_cur_device);
  if (to_device == tmp_cur_device) {
    return cur_device;
  }
  if (hasPrimaryContext(to_device)) {
    C10_CUDA_CHECK(cudaSetDevice(to_device));
  } else {
    targetDeviceIndex = to_device;
  }
  return cur_device;
}

void SetTargetDevice() {
  if (targetDeviceIndex >= 0) {
    C10_CUDA_CHECK(c10::cuda::SetDevice(targetDeviceIndex));
  }
}
#else
cudaError_t GetDevice(DeviceIndex* device) {
  int tmp_device = -1;
  auto err = cudaGetDevice(&tmp_device);
  if (err == cudaSuccess) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<DeviceIndex>::max(),
        "cudaGetDevice returns invalid device ",
        tmp_device);
    *device = static_cast<DeviceIndex>(tmp_device);
  }
  return err;
}

cudaError_t SetDevice(DeviceIndex device) {
  TORCH_CHECK(device >= 0, "device id must be positive!", device);
  int cur_device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  if (device == cur_device) {
    return cudaSuccess;
  }
  return cudaSetDevice(device);
}

cudaError_t MaybeSetDevice(DeviceIndex device) {
  return c10::cuda::SetDevice(device);
}

DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  DeviceIndex cur_device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&cur_device));
  if (to_device == cur_device) {
    return cur_device;
  }
  C10_CUDA_CHECK(cudaSetDevice(to_device));
  return cur_device;
}

DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  return c10::cuda::ExchangeDevice(to_device);
}

void SetTargetDevice() {
  // no-op on CUDA version < 12.x
}
#endif

} // namespace c10::cuda
