#pragma once

#include <c10/util/Exception.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <exception>
#include <string>
#include <system_error>
#include <utility>

namespace c10::xpu {

[[noreturn]] inline void c10_xpu_check_implementation(
    const std::error_code& code,
    const std::string& message,
    c10::SourceLocation location) {
  throw c10::AcceleratorError(
      location,
      static_cast<int32_t>(code.value()),
      std::string("SYCL error [") + code.category().name() + ":" +
          std::to_string(code.value()) + "]: " + message);
}


template <typename Func>
inline decltype(auto) withSyclErrorTranslation(
    Func&& func,
  c10::SourceLocation location = c10::SourceLocation::current()) {
  std::error_code code;
  std::string message;
  try {
    return std::forward<Func>(func)();
  } catch (const sycl::exception& error) {
    code = error.code();
    message = error.what();
  }
  c10_xpu_check_implementation(code, message, location);
}


static inline sycl::async_handler asyncHandler =
    [](const sycl::exception_list& errors) {
      for (const auto& error : errors) {
        std::rethrow_exception(error);
      }
    };

} // namespace c10::xpu

#define C10_XPU_CHECK(...)                                           \
  ::c10::xpu::withSyclErrorTranslation(                               \
      [&]() -> decltype(auto) { return (__VA_ARGS__); },              \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)})
