#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef ADD_BREAKPAD_SIGNAL_HANDLER
#include <breakpad/client/linux/handler/exception_handler.h>
#include <csignal>
#endif

#include <c10/util/Exception.h>
#include <torch/csrc/utils/crash_handler.h>

namespace torch {
namespace crash_handler {

#ifdef ADD_BREAKPAD_SIGNAL_HANDLER

static std::unique_ptr<google_breakpad::ExceptionHandler> handler; // NOLINT
static std::string minidump_directory; // NOLINT
static bool enabled_for_exceptions = false; // NOLINT

bool dump_callback(
    const google_breakpad::MinidumpDescriptor& descriptor,
    void* context,
    bool succeeded) {
  if (succeeded) {
    std::cerr << "Wrote minidump to " << descriptor.path() << std::endl;
  }
  return succeeded;
}

void enable_minidumps(const std::string& dir) {
  minidump_directory = dir;
  // The constructor here registers the actual signal handler
  handler = std::make_unique<google_breakpad::ExceptionHandler>(
      google_breakpad::MinidumpDescriptor(minidump_directory),
      nullptr,
      dump_callback,
      nullptr,
      true,
      -1);
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
      "Minidump collection is currently only implemented for Linux platforms");
}

void disable_minidumps() {
  // Purposefully do nothing
}

const std::string& get_minidump_directory() {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux platforms");
}

bool is_enabled_on_exceptions() {
  return false;
}

void write_minidump() {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux platforms");
}

void enable_minidumps_on_exceptions() {
  AT_ERROR(
      "Minidump collection is currently only implemented for Linux platforms");
}

#endif

} // namespace crash_handler
} // namespace torch
