#pragma once

#include <torch/csrc/jit/codegen/cuda/utils.h>

// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdio.h>
#include <chrono>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace inst {

//! An optional record of selected timestamped operations, events and counters
//!
//! This class is not intended to be used directly. Instead, the operations
//! to be traced are marked (for example using the FUSER_PERF_SCOPE macro)
//!
//! In order to enable tracing, the `PYTORCH_CUDA_FUSER_TRACE` environment
//! variable is set to point to a trace file (ex `test.trace`). The file name
//! may be a relative or an absolute path.
//!
//! The trace uses the Chrome Tracing (Catapult) format, which is a well
//! documented JSON based format supported by multiple tools:
//! https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md
//!
//! An easy way to view traces is to type `about://tracing` in Chrome or
//! Chromium.
//!
class Trace : public NonCopyable {
 public:
  using Clock = std::chrono::steady_clock;

 public:
  static Trace* instance() {
    static Trace trace;
    return &trace;
  }

  void beginEvent(const char* name) {
    if (log_file_ != nullptr) {
      logEvent('B', name);
    }
  }

  void endEvent(const char* name) {
    if (log_file_ != nullptr) {
      logEvent('E', name);
    }
  }

 private:
  Trace();
  ~Trace();

  void logEvent(char ph, const char* name, char sep = ',');

 private:
  FILE* log_file_ = nullptr;
  Clock::time_point start_timestamp_;
};

//! \internal Automatic scope for a perf marker
//!   (normally used through the FUSER_PERF_SCOPE macro)
class TraceScope : public NonCopyable {
 public:
  explicit TraceScope(const char* event_name) : event_name_(event_name) {
    Trace::instance()->beginEvent(event_name_);
  }

  ~TraceScope() {
    Trace::instance()->endEvent(event_name_);
  }

 private:
  const char* event_name_ = nullptr;
};

#define FUSER_MACRO_CONCAT2(a, b) a##b
#define FUSER_MACRO_CONCAT(a, b) FUSER_MACRO_CONCAT2(a, b)
#define FUSER_ANONYMOUS(prefix) FUSER_MACRO_CONCAT(prefix, __COUNTER__)

//! Defines a scope we want to measure and record in a perf trace
//!
//! \param name The name of the scope, normally a simple string literal
//!
#define FUSER_PERF_SCOPE(name) \
  torch::jit::fuser::cuda::inst::TraceScope FUSER_ANONYMOUS(_perf_scope_)(name)

} // namespace inst
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
