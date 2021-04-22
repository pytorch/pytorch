#include <iostream>

#ifdef ADD_BREAKPAD_SIGNAL_HANDLER
#include <breakpad/client/linux/handler/exception_handler.h>
#endif

#include <c10/util/Exception.h>
#include <torch/csrc/utils/crash_handler.h>

namespace torch {
namespace crash_handler {

#ifdef ADD_BREAKPAD_SIGNAL_HANDLER

bool dumpCallback(
    const google_breakpad::MinidumpDescriptor& descriptor,
    void* context,
    bool succeeded) {
  if (succeeded) {
    std::cerr << "Wrote minidump to " << descriptor.path() << std::endl;
  }
  return succeeded;
}

static std::unique_ptr<google_breakpad::ExceptionHandler> handler; // NOLINT
static std::string minidump_directory; // NOLINT

void _enable_minidump_collection(const std::string& dir) {
  minidump_directory = dir;
  handler = std::make_unique<google_breakpad::ExceptionHandler>(
      google_breakpad::MinidumpDescriptor(minidump_directory),
      nullptr,
      dumpCallback,
      nullptr,
      true,
      -1);
}

void _disable_minidump_collection() {
  handler.reset();
}

const std::string& _get_minidump_directory() {
  if (handler == nullptr) {
    AT_ERROR(
        "Minidump handler is uninintialized, make sure to call _enable_minidump_collection first");
  }
  return minidump_directory;
}
bool is_enabled() {
  return handler != nullptr;
}
void write_minidump() {
  TORCH_CHECK(handler != nullptr,"Minidump handler is uninintialized, make sure to call _enable_minidump_collection first");
  handler->WriteMinidump();
}
#else
void _enable_minidump_collection(const std::string& dir) {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux platforms");
}

void _disable_minidump_collection() {
  // Purposefully do nothing
}

const std::string& _get_minidump_directory() {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux platforms");
}

bool is_enabled() {
  return false;
}

void write_minidump() {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux platforms");
}
#endif

} // namespace crash_handler
} // namespace torch
