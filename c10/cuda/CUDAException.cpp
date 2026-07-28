#include <c10/cuda/CUDAException.h>

#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/util/Exception.h>

#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#endif

#include <string>
#include <utility>

namespace c10::cuda {

CUDAErrorLogCapture::CUDAErrorLogCapture() noexcept {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
  try {
    auto* api = CUDAErrorLogAPI::get();
    if (api->cuLogsCurrent_ && api->cuLogsDumpToMemory_) {
      CUlogIterator iterator;
      if (api->cuLogsCurrent_(&iterator, 0) == CUDA_SUCCESS) {
        iterator_ = iterator;
        enabled_ = true;
      }
    }
  } catch (...) {
  }
#endif
}

std::string CUDAErrorLogCapture::get_error_log_suffix() {
#if !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED) && \
    defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
  if (!enabled_) {
    return {};
  }

  char buffer[25600];
  size_t size = sizeof(buffer);
  auto iterator = static_cast<CUlogIterator>(iterator_);
  try {
    auto* api = CUDAErrorLogAPI::get();
    if (api->cuLogsDumpToMemory_(&iterator, buffer, &size, 0) ==
            CUDA_SUCCESS &&
        size > 0 && size <= sizeof(buffer)) {
      std::string error_log{
          "\nThe CUDA driver logged these messages, which may provide useful details:\n"};
      error_log.append(buffer, size);
      return error_log;
    }
  } catch (...) {
  }
#endif
  return {};
}

namespace {

void c10_cuda_check_implementation_internal(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const uint32_t line_number,
    const bool include_device_assertions,
    CUDAErrorLogCapture* error_log) {
  const auto cuda_error = static_cast<cudaError_t>(err);
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
  if (error_log) {
    check_message.append(error_log->get_error_log_suffix());
  }
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

} // namespace

void c10_cuda_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const uint32_t line_number,
    const bool include_device_assertions) {
  c10_cuda_check_implementation_internal(
      err,
      filename,
      function_name,
      line_number,
      include_device_assertions,
      nullptr);
}

void c10_cuda_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const uint32_t line_number,
    const bool include_device_assertions,
    CUDAErrorLogCapture& error_log) {
  c10_cuda_check_implementation_internal(
      err,
      filename,
      function_name,
      line_number,
      include_device_assertions,
      &error_log);
}

} // namespace c10::cuda
