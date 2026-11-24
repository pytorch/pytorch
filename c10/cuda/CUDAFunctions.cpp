#include <c10/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
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
    }
    case cudaErrorInitializationError:
      TORCH_CHECK(
          false,
          "CUDA driver initialization failed, you might not "
          "have a CUDA gpu.");
    case cudaErrorUnknown:
      TORCH_CHECK(
          false,
          "CUDA unknown error - this may be due to an "
          "incorrectly set up environment, e.g. changing env "
          "variable CUDA_VISIBLE_DEVICES after program start. "
          "Setting the available devices to be zero.");
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
#endif // C10_ASAN_ENABLED
#if defined(_WIN32) && CUDA_VERSION >= 13000
    // Workaround for CUDA-13.0 error handling on Windows, see
    // https://github.com/pytorch/pytorch/issues/162333#issuecomment-3267929585
    case cudaErrorNotSupported:
      if (!fail_if_no_driver) {
        TORCH_WARN(
            "cudaGetDeviceCount() returned cudaErrorNotSupported, "
            "likely using older driver or on CPU machine");
        count = 0;
        break;
      }
#endif
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

void set_device(DeviceIndex device, const bool force) {
  C10_CUDA_CHECK(c10::cuda::SetDevice(device, force));
}

void device_synchronize() {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (interp) [[unlikely]] {
    (*interp)->trace_gpu_device_synchronization(c10::kCUDA);
  }
  STATIC_SCOPED_WAIT_COUNTER(pytorch.wait_counter.cuda_device_synchronize);
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

void stream_wait_stream(CUDAStream waiter, CUDAStream source) {
  TORCH_CHECK(waiter.device() == source.device());
  if (waiter.stream() == source.stream()) {
    return;
  }
  CUDAGuard guard(source.device());
  const auto source_info = captureInfoMayInitCtx(source);
  const auto waiter_info = captureInfoMayInitCtx(waiter);
  TORCH_CHECK(
      source_info.status == waiter_info.status &&
          source_info.status != CaptureStatus::Invalidated &&
          (source_info.status == CaptureStatus::None ||
           (source_info.id == waiter_info.id &&
            source_info.graph == waiter_info.graph)),
      "Cannot establish a stream dependency across CUDA graph capture "
      "boundaries; order the operation outside capture first");
  CUDAEvent event;
  event.record(source);
  event.block(waiter);
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
static bool dummyHasPrimaryContext([[maybe_unused]] DeviceIndex device_index) {
  TORCH_CHECK(false, "Should never been called");
}
static bool (*hasPrimaryContext)(DeviceIndex) = dummyHasPrimaryContext;

// Private api to be called from CUDAHooks.cpp
// NOLINTNEXTLINE(misc-use-internal-linkage)
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

// CUDA 12+ and HIP 7.14+ eagerly create primary context when cudaSetDevice
// (hipSetDevice) is called. To avoid unnecessary context creation when
// restoring a device, MaybeSetDevice is used by device guard destructors
// and on exit from torch.cuda.device context managers.
//
// MaybeSetDevice first queries the runtime's current device. If it is already
// the requested device, it clears the deferred device index without calling
// cudaSetDevice. Otherwise, it calls cudaSetDevice and clears the deferred
// index if the requested device has a primary context, or only saves the
// requested index if it does not. GetDevice reports the deferred index when one
// is set. This preserves lazy context creation for applications like this:
//
// ```
// import torch
// x = torch.empty(1, device="cuda:1") # no CUDA context on cuda:0 after this
// call y = torch.empty(1, device="cuda")   # CUDA context is created on cuda:0
// ```
#if CUDA_VERSION >= 12000
thread_local static DeviceIndex deferredDeviceIndex = -1;

cudaError_t GetDevice(DeviceIndex* device) {
  if (deferredDeviceIndex >= 0) {
    *device = deferredDeviceIndex;
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

// Clear the deferred device index and call cudaSetDevice if force is true or
// the requested device differs from the runtime's current device.
cudaError_t SetDevice(DeviceIndex device, const bool force) {
  TORCH_CHECK(
      device >= 0, "device id must be non-negative!", static_cast<int>(device));
  deferredDeviceIndex = -1;
  if (force) {
    return cudaSetDevice(device);
  }
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
  deferredDeviceIndex = device;
  return cudaSuccess;
}

// Return the deferred device index, or the runtime's current device if none is
// deferred, and clear the deferred index. Call cudaSetDevice unless no index
// was deferred and to_device is already current; that early return does not
// initialize a context.
DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  auto cur_device = deferredDeviceIndex;
  deferredDeviceIndex = -1;
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

// Return the runtime's current device. If to_device is already current, return
// without changing the deferred index. Otherwise, call cudaSetDevice if
// to_device has a primary context, or save to_device as the deferred index if
// it does not. Calling cudaSetDevice here leaves any deferred index unchanged.
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
    deferredDeviceIndex = to_device;
  }
  return cur_device;
}

void SetTargetDevice() {
  if (deferredDeviceIndex >= 0) {
    C10_CUDA_CHECK(c10::cuda::SetDevice(deferredDeviceIndex));
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

cudaError_t SetDevice(DeviceIndex device, const bool force) {
  TORCH_CHECK(
      device >= 0, "device id must be non-negative!", static_cast<int>(device));
  if (force) {
    return cudaSetDevice(device);
  }
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
  // No deferred device switching on CUDA < 12 without ROCm.
}
#endif

} // namespace c10::cuda
