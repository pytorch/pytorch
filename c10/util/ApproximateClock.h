// Copyright 2023-present Facebook. All Rights Reserved.

#pragma once

#include <c10/macros/Export.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <type_traits>

#if defined(C10_IOS) && defined(C10_MOBILE)
#include <sys/time.h> // for gettimeofday()
#endif

#if defined(__i386__) || defined(__x86_64__) || defined(__amd64__)
// Use the builtin rather than `__rdtsc`, whose headers (<x86intrin.h>,
// <intrin.h>) pull in every x86 intrinsic header and add over a second of
// parsing to each TU that includes this one.
#if !defined(__CUDACC__) && !defined(__HIPCC__) && defined(__has_builtin)
#if __has_builtin(__builtin_ia32_rdtsc)
#define C10_RDTSC __builtin_ia32_rdtsc
#endif
#endif
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)) && \
    !defined(_M_ARM64EC) && !defined(__CUDACC__)
// MSVC defines none of the macros above and has no __has_builtin. Declare the
// intrinsic as <intrin.h> does instead of including it, for the same reason.
extern "C" unsigned __int64 __rdtsc();
#define C10_RDTSC __rdtsc
#elif defined(__aarch64__) && !defined(__CUDACC__) && !defined(__HIPCC__)
#define C10_ARMTSC
#elif defined(__riscv) && (__riscv_xlen == 64) && !defined(__CUDACC__) && \
    !defined(__HIPCC__)
#define C10_RISCVTSC
#endif

namespace c10 {

using time_t = int64_t;
using steady_clock_t = std::conditional_t<
    std::chrono::high_resolution_clock::is_steady,
    std::chrono::high_resolution_clock,
    std::chrono::steady_clock>;

inline time_t getTimeSinceEpoch() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

inline time_t getTime(bool allow_monotonic = false) {
#if defined(C10_IOS) && defined(C10_MOBILE)
  // clock_gettime is only available on iOS 10.0 or newer. Unlike OS X, iOS
  // can't rely on CLOCK_REALTIME, as it is defined no matter if clock_gettime
  // is implemented or not
  struct timeval now;
  gettimeofday(&now, NULL);
  return static_cast<time_t>(now.tv_sec) * 1000000000 +
      static_cast<time_t>(now.tv_usec) * 1000;
#elif defined(_WIN32) || defined(__MACH__)
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
  return static_cast<time_t>(t.tv_sec) * 1000000000 +
      static_cast<time_t>(t.tv_nsec);
#endif
}

#if defined(C10_ARMTSC)
inline uint64_t getArmApproximateTime() {
  uint64_t val;
  __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
  return val;
}
#endif

#if defined(C10_RISCVTSC)
inline uint64_t getRiscvApproximateTime() {
  uint64_t val;
  // rdtime reads the constant-frequency `time` CSR (user-readable on Linux,
  // as cntvct_el0 is on aarch64).
  __asm__ __volatile__("rdtime %0" : "=r"(val));
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
#if defined(C10_RDTSC)
  return static_cast<uint64_t>(C10_RDTSC());
#elif defined(C10_ARMTSC)
  return getArmApproximateTime();
#elif defined(C10_RISCVTSC)
  return getRiscvApproximateTime();
#else
  return getTime();
#endif
}

using approx_time_t = decltype(getApproximateTime());
static_assert(
    std::is_same_v<approx_time_t, int64_t> ||
        std::is_same_v<approx_time_t, uint64_t>,
    "Expected either int64_t (`getTime`) or uint64_t (some TSC reads).");

// Convert `getCount` results to Nanoseconds since unix epoch.
class C10_API ApproximateClockToUnixTimeConverter final {
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

} // namespace c10
