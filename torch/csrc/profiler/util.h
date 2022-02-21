#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <c10/macros/Macros.h>
#include <ATen/record_function.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/source_range.h>

#ifndef _WIN32
#include <ctime>
#endif
#if defined(C10_IOS) && defined(C10_MOBILE)
#include <sys/time.h> // for gettimeofday()
#endif

namespace torch {
namespace profiler {
namespace impl {

inline int64_t getTime(bool allow_monotonic = false) {
#if defined(C10_IOS) && defined(C10_MOBILE)
  // clock_gettime is only available on iOS 10.0 or newer. Unlike OS X, iOS
  // can't rely on CLOCK_REALTIME, as it is defined no matter if clock_gettime
  // is implemented or not
  struct timeval now;
  gettimeofday(&now, NULL);
  return static_cast<int64_t>(now.tv_sec) * 1000000000 +
      static_cast<int64_t>(now.tv_usec) * 1000;
#elif defined(_WIN32) || defined(__MACH__)
  using namespace std::chrono;
  using clock = std::conditional<
      high_resolution_clock::is_steady,
      high_resolution_clock,
      steady_clock>::type;
  return duration_cast<nanoseconds>(clock::now().time_since_epoch()).count();
#else
  // clock_gettime is *much* faster than std::chrono implementation on Linux
  struct timespec t {};
  auto mode = CLOCK_REALTIME;
  if (allow_monotonic) {
    mode = CLOCK_MONOTONIC;
  }
  clock_gettime(mode, &t);
  return static_cast<int64_t>(t.tv_sec) * 1000000000 +
      static_cast<int64_t>(t.tv_nsec);
#endif
}

std::string getNvtxStr(
    const char* name,
    int64_t sequence_nr,
    const std::vector<std::vector<int64_t>>& shapes);

struct TORCH_API FileLineFunc {
  std::string filename;
  size_t line;
  std::string funcname;
};

TORCH_API std::vector<FileLineFunc> prepareCallstack(
    const std::vector<jit::StackEntry>& cs);
TORCH_API std::vector<std::string> callstackStr(
    const std::vector<FileLineFunc>& cs);
TORCH_API std::string stacksToStr(
    const std::vector<std::string>& stacks,
    const char* delim);
TORCH_API std::vector<std::vector<int64_t>> inputSizes(
    const at::RecordFunction& fn);
TORCH_API std::string shapesToStr(
    const std::vector<std::vector<int64_t>>& shapes);
TORCH_API std::string dtypesToStr(const std::vector<std::string>& types);
TORCH_API std::vector<std::string> inputTypes(const at::RecordFunction& fn);

std::unordered_map<std::string, c10::IValue> TORCH_API
saveExtraArgs(const at::RecordFunction& fn);

uint64_t TORCH_API computeFlops(
    const std::string& op_name,
    const std::unordered_map<std::string, c10::IValue>& extra_args);

} // namespace impl
} // namespace profiler
} // namespace torch

namespace torch {
namespace autograd {
namespace profiler {
using torch::profiler::impl::getTime;
using torch::profiler::impl::computeFlops;
} // namespace profiler
} // namespace autograd
} // namespace torch
