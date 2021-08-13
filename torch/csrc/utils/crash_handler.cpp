#include <cstdlib>
#include <cstring>
#include <iostream>

// #ifdef ADD_BREAKPAD_SIGNAL_HANDLER
// #include <breakpad/client/linux/handler/exception_handler.h>

#ifdef ADD_BREAKPAD_SIGNAL_HANDLER
#ifdef __linux__
#include <breakpad/client/linux/handler/exception_handler.h>
#include <csignal>
#elif __APPLE__
#include <breakpad/src/client/mac/handler/exception_handler.h>
#else
#error unsupported platform
#endif
#endif

#include <c10/util/Exception.h>
#include <torch/csrc/utils/crash_handler.h>

namespace torch {
namespace crash_handler {

#ifdef ADD_BREAKPAD_SIGNAL_HANDLER

static std::unique_ptr<google_breakpad::ExceptionHandler> handler; // NOLINT
static std::string minidump_directory; // NOLINT
static bool enabled_for_exceptions = false; // NOLINT

#if __linux__
bool dump_callback(
    const google_breakpad::MinidumpDescriptor& descriptor,
    void* context,
    bool succeeded) {
  if (succeeded) {
    std::cerr << "Wrote minidump to " << descriptor.path() << std::endl;
  }
  return succeeded;
}
#elif __APPLE__

bool dump_callback(
    const char* dump_dir,
    const char* minidump_id,
    void* context,
    bool succeeded) {
  if (succeeded) {
    std::cerr << "Wrote minidump to " << dump_dir << "/" << minidump_id
              << ".dmp" << std::endl;
  }
  return succeeded;
}

#else
#error unsupported platform
#endif

void enable_minidumps(const std::string& dir) {
  minidump_directory = dir;
// The constructor here registers the actual signal handler
#ifdef __linux__
  handler = std::make_unique<google_breakpad::ExceptionHandler>(
      google_breakpad::MinidumpDescriptor(minidump_directory),
      nullptr,
      dump_callback,
      nullptr,
      true,
      -1);
#elif __APPLE__
  handler = std::make_unique<google_breakpad::ExceptionHandler>(
      /*dump_path=*/minidump_directory.c_str(),
      /*filter=*/nullptr,
      /*callback=*/dump_callback,
      /*callback_context=*/nullptr,
      /*install_handler=*/true,
      /*port_name=*/nullptr);
#else
#error unsupported platform
#endif
}

void disable_minidumps() {
  handler.reset();
}

const std::string& get_minidump_directory() {
  if (handler == nullptr) {
    AT_ERROR(
        "Minidump handler is uninintialized, make sure to call enable_minidumps first");
  }
  return minidump_directory;
}

bool is_enabled_on_exceptions() {
  if (handler == nullptr) {
    return false;
  }

  return enabled_for_exceptions;
}

void write_minidump() {
  TORCH_CHECK(
      handler != nullptr,
      "Minidump handler is uninintialized, make sure to call enable_minidumps first");
  handler->WriteMinidump();
}

void enable_minidumps_on_exceptions() {
  if (handler == nullptr) {
    AT_ERROR(
        "Minidump handler is uninintialized, make sure to call enable_minidumps first");
  }
  enabled_for_exceptions = true;
}

#else
// On unspported systems we can't do anything, so stub out everything.
void enable_minidumps(const std::string& dir) {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux/MacOS");
}

void disable_minidumps() {
  // Purposefully do nothing
}

const std::string& get_minidump_directory() {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux/MacOS");
}

bool is_enabled_on_exceptions() {
  return false;
}

void write_minidump() {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux/MacOS");
}

void enable_minidumps_on_exceptions() {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux/MacOS");
}

#endif

} // namespace crash_handler
} // namespace torch
