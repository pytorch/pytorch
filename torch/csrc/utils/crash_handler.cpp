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

namespace {
bool check_env(char* name) {
  char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return value[0] == '1';
}
} // namespace

#ifdef ADD_BREAKPAD_SIGNAL_HANDLER

namespace {
bool should_enable_minidump_handler() {
  if (check_env("TORCH_DISABLE_MINIDUMPS")) {
    // If disabled in the environment, don't do anything
    return false;
  }

  // Test these signals, if any of them already have a handler then we don't
  // want to override it with breakpad's signal handler. It should match the
  // list of breakpad signal handlers here:
  // https://chromium.googlesource.com/breakpad/breakpad/+/refs/heads/chrome_43/src/client/linux/handler/exception_handler.cc#121
  std::array<int, 5> signals{
      SIGSEGV,
      SIGABRT,
      SIGFPE,
      SIGILL,
      SIGBUS
  };

  for (int signal : signals) {
    // NOLINTNEXTLINE
    struct sigaction action;
    action.sa_handler = nullptr;
    sigaction(signal, nullptr, &action);

    if (action.sa_handler != nullptr) {
      // A signal handler was already set, don't override it and give up
      return false;
    }
  }

  // Nothing was blocking it, so enable the handler by default
  return true;
}

bool initialize_minidumps() {
  if (!should_enable_minidump_handler()) {
    return false;
  }
  enable_minidumps("/tmp/pytorch_crashes");
  return true;
}
} // namespace

static std::unique_ptr<google_breakpad::ExceptionHandler> handler; // NOLINT
static std::string minidump_directory; // NOLINT
static bool enabled_for_exceptions = false; // NOLINT
static bool is_enabled = initialize_minidumps(); // NOLINT

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
        "Minidump handler is uninintialized, make sure to call _enable_minidump_collection first");
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
      "Minidump handler is uninintialized, make sure to call _enable_minidump_collection first");
  handler->WriteMinidump();
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
#endif

} // namespace crash_handler
} // namespace torch
