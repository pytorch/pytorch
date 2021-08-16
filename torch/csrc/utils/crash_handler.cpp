#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef ADD_BREAKPAD_SIGNAL_HANDLER
#ifdef __linux__
#include <breakpad/src/client/linux/handler/exception_handler.h>
#include <csignal>
#elif __APPLE__
#include <breakpad/src/client/mac/handler/exception_handler.h>
#elif _WIN32
#include <breakpad/src/client/windows/handler/exception_handler.h>
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
static STRING_TYPE minidump_directory; // NOLINT
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
#elif _WIN32
bool dump_callback(
    const wchar_t* dump_path,
    const wchar_t* minidump_id,
    void* context,
    EXCEPTION_POINTERS* exinfo,
    MDRawAssertionInfo* assertion,
    bool succeeded) {
  if (succeeded) {
    std::wcerr << "Wrote minidump to " << dump_path << "\\"
               << minidump_id << ".dmp" << std::endl;
  }
  return succeeded;
}
#endif

void enable_minidumps(const STRING_TYPE& dir) {
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
#elif _WIN32
  handler = std::make_unique<google_breakpad::ExceptionHandler>(
      /*dump_path=*/minidump_directory.c_str(),
      /*filter=*/nullptr,
      /*callback=*/dump_callback,
      /*callback_context=*/nullptr,
      /*handler_types=*/
      google_breakpad::ExceptionHandler::HandlerType::HANDLER_ALL);
#endif
}

void disable_minidumps() {
  handler.reset();
}

const STRING_TYPE& get_minidump_directory() {
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
void enable_minidumps(const STRING_TYPE& dir) {
  AT_ERROR("Compiled without minidump support");
}

void disable_minidumps() {
  // Purposefully do nothing
}

const STRING_TYPE& get_minidump_directory() {
  AT_ERROR("Compiled without minidump support");
}

bool is_enabled_on_exceptions() {
  return false;
}

void write_minidump() {
  AT_ERROR("Compiled without minidump support");
}

void enable_minidumps_on_exceptions() {
  AT_ERROR("Compiled without minidump support");
}

#endif

} // namespace crash_handler
} // namespace torch
