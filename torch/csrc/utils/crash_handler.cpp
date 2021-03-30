#include <iostream>

#ifdef __linux__
#include <client/linux/handler/exception_handler.h>
#endif

#include <c10/util/Exception.h>
#include <torch/csrc/utils/crash_handler.h>

namespace torch {
namespace crash_handler {

#ifdef __linux__

bool dumpCallback(
    const google_breakpad::MinidumpDescriptor& descriptor,
    void* context,
    bool succeeded) {
  if (succeeded) {
    std::cerr << "Wrote minidump to " << descriptor.path() << std::endl;
  }
  return succeeded;
}

static std::unique_ptr<google_breakpad::ExceptionHandler> handler;
static std::string minidump_directory;

TORCH_API void _enable_minidump_collection(const std::string& dir) {
  minidump_directory = dir;
  handler = std::make_unique<google_breakpad::ExceptionHandler>(
      google_breakpad::MinidumpDescriptor(minidump_directory),
      nullptr,
      dumpCallback,
      nullptr,
      true,
      -1);
}
#else
TORCH_API void _enable_minidump_collection(const std::string& dir) {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux platforms");
}
#endif

TORCH_API const std::string& _get_minidump_directory() {
  if (handler == nullptr) {
    AT_ERROR(
        "Minidump handler is uninintialized, make sure to call _enable_minidump_collection first");
  }
  return minidump_directory;
}

} // namespace crash_handler
} // namespace torch
