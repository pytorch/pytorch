#include <c10/cuda/CUDAException.h>

#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/util/Exception.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <utility>

namespace c10::cuda {

namespace {

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
// This size matches the one used by cuLogsDumpToMemory, however we could shrink
// it as in our case this size is per-thread, and it only needs to be large
// enough to contain the messages printed by a single CUDA function call.
constexpr size_t kCUDAErrorLogBufferSize = 25600;

struct CUDAErrorLogBuffer {
  std::array<char, kCUDAErrorLogBufferSize> data{};
  size_t length{0};
};

thread_local CUDAErrorLogBuffer cuda_error_log_buffer;

void CUDA_CB cuda_error_log_callback(
    void* /*data*/,
    CUlogLevel /*logLevel*/,
    char* message,
    size_t length) noexcept {
  auto& buffer = cuda_error_log_buffer;
  const auto copy_size = std::min(length, buffer.data.size() - buffer.length);
  if (copy_size > 0) {
    std::memcpy(buffer.data.data() + buffer.length, message, copy_size);
  }
  buffer.length += copy_size;
  if (length > 0 && copy_size == length && buffer.length < buffer.data.size() &&
      message[length - 1] != '\n') {
    buffer.data[buffer.length++] = '\n';
  }
}

class CUDAErrorLogCallbackRegistration {
 public:
  CUDAErrorLogCallbackRegistration() noexcept {
    DriverAPI* api = nullptr;
    try {
      api = DriverAPI::get();
    } catch (...) {
      // This can happen if no suitable CUDA driver is found. In that case,
      // silently disable CUDA error log reporting.
      return;
    }
    if (api->cuLogsRegisterCallback_ && api->cuLogsUnregisterCallback_ &&
        api->cuLogsRegisterCallback_(
            cuda_error_log_callback,
            /*userData=*/nullptr,
            &callback_handle_) == CUDA_SUCCESS) {
      unregister_callback_ = api->cuLogsUnregisterCallback_;
    }
  }

  CUDAErrorLogCallbackRegistration(const CUDAErrorLogCallbackRegistration&) =
      delete;
  CUDAErrorLogCallbackRegistration& operator=(
      const CUDAErrorLogCallbackRegistration&) = delete;
  CUDAErrorLogCallbackRegistration(CUDAErrorLogCallbackRegistration&&) = delete;
  CUDAErrorLogCallbackRegistration& operator=(
      CUDAErrorLogCallbackRegistration&&) = delete;

  ~CUDAErrorLogCallbackRegistration() noexcept {
    if (unregister_callback_) {
      (void)unregister_callback_(callback_handle_);
    }
  }

 private:
  decltype(&cuLogsUnregisterCallback) unregister_callback_{nullptr};
  CUlogsCallbackHandle callback_handle_{nullptr};
};
#endif

} // namespace

CUDAErrorLogCapture::CUDAErrorLogCapture() noexcept {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
  static const CUDAErrorLogCallbackRegistration callback_registration
      [[maybe_unused]];
  cuda_error_log_buffer.length = 0;
#endif
}

std::string CUDAErrorLogCapture::get_error_log_suffix() noexcept {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
  auto& buffer = cuda_error_log_buffer;
  if (buffer.length == 0) {
    return {};
  }
  std::string error_log{
      "\nThe CUDA driver logged these messages, which may provide useful details:\n"};
  error_log.append(buffer.data.data(), buffer.length);
  return error_log;
#else
  return {};
#endif
}

void c10_cuda_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const uint32_t line_number,
    const bool include_device_assertions,
    CUDAErrorLogCapture* error_log) {
  const auto cuda_error = static_cast<cudaError_t>(err);
#ifndef STRIP_ERROR_MESSAGES
  const auto error_log_suffix = cuda_error != cudaSuccess && error_log
      ? error_log->get_error_log_suffix()
      : std::string{};
#endif
  const auto cuda_kernel_failure = include_device_assertions
      ? c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().has_failed()
      : false;

  if (C10_LIKELY(cuda_error == cudaSuccess && !cuda_kernel_failure)) {
    return;
  }

  [[maybe_unused]] auto error_unused = cudaGetLastError();

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("CUDA error: ");
  const char* error_string = cudaGetErrorString(cuda_error);
  check_message.append(error_string);
  check_message.append(c10::cuda::get_cuda_error_help(cuda_error));
  check_message.append(c10::cuda::get_cuda_async_error_suffix(cuda_error));
  check_message.append(error_log_suffix);
  check_message.push_back('\n');
  if (include_device_assertions) {
    check_message.append(c10_retrieve_device_side_assertion_info());
  } else {
    check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
  }
#endif
  throw c10::AcceleratorError(
      {.function = function_name, .file = filename, .line = line_number},
      err,
      std::move(check_message));
}

} // namespace c10::cuda
