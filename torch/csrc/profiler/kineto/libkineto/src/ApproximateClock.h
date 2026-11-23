/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* The contents of this file was borrowed c10/util/ApproximateClock.h */

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <type_traits>

#if defined(__i386__) || defined(__x86_64__) || defined(__amd64__)
#define KINETO_RDTSC
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__CUDACC__) || defined(__HIPCC__)
#undef KINETO_RDTSC
#elif defined(__clang__)
// `__rdtsc` is available by default.
// NB: This has to be first, because Clang will also define `__GNUC__`
#elif defined(__GNUC__)
#include <x86intrin.h>
#else
#undef KINETO_RDTSC
#endif
#elif defined(__aarch64__) && !defined(__CUDACC__) && !defined(__HIPCC__)
#define KINETO_ARMTSC
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define KINETO_UNUSED __pragma(warning(suppress : 4100 4101))
#else
#define KINETO_UNUSED __attribute__((__unused__))
#endif //_MSC_VER

namespace libkineto {

using time_t = int64_t;
using steady_clock_t = std::conditional_t<
    std::chrono::high_resolution_clock::is_steady,
    std::chrono::high_resolution_clock,
    std::chrono::steady_clock>;

inline time_t getTime([[maybe_unused]] bool allow_monotonic = false) {
#if defined(_WIN32) || defined(__MACH__)
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             steady_clock_t::now().time_since_epoch())
      .count();
#else
  // clock_gettime is *much* faster than std::chrono implementation on Linux
  struct timespec t{};
  auto mode = CLOCK_REALTIME;
  if (allow_monotonic) {
    mode = CLOCK_MONOTONIC;
  }
  clock_gettime(mode, &t);
  return (static_cast<time_t>(t.tv_sec) * 1000000000) +
      static_cast<time_t>(t.tv_nsec);
#endif
}

#if defined(KINETO_ARMTSC)
inline uint64_t getArmApproximateTime() {
  uint64_t val;
  __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
  return val;
}
#endif

// We often do not need to capture true wall times. If a fast mechanism such
// as TSC is available we can use that instead and convert back to epoch time
// during post processing. This greatly reduce the clock's contribution to
// profiling.
//   http://btorpey.github.io/blog/2014/02/18/clock-sources-in-linux/
//   https://quick-bench.com/q/r8opkkGZSJMu9wM_XTbDouq-0Io
// TODO: We should use
// `https://github.com/google/benchmark/blob/main/src/cycleclock.h`
inline auto getApproximateTime() {
#if defined(KINETO_RDTSC)
  return static_cast<uint64_t>(__rdtsc());
#elif defined(KINETO_ARMTSC)
  return getArmApproximateTime();
#else
  return getTime();
#endif
}

using approx_time_t = decltype(getApproximateTime());
static_assert(
    std::is_same_v<approx_time_t, int64_t> ||
        std::is_same_v<approx_time_t, uint64_t>,
    "Expected either int64_t (`getTime`) or uint64_t (some TSC reads).");

std::function<time_t(approx_time_t)>& get_time_converter();

// Convert `getCount` results to Nanoseconds since unix epoch.
class ApproximateClockToUnixTimeConverter final {
 public:
  ApproximateClockToUnixTimeConverter();
  std::function<time_t(approx_time_t)> makeConverter();

  struct UnixAndApproximateTimePair {
    time_t t_;
    approx_time_t approx_t_;
  };
  static UnixAndApproximateTimePair measurePair();

 private:
  static constexpr size_t replicates = 1001;
  using time_pairs = std::array<UnixAndApproximateTimePair, replicates>;
  time_pairs measurePairs();

  time_pairs start_times_;
};
} // namespace libkineto
