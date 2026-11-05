/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ApproximateClock.h"
#include <fmt/format.h>
#include <algorithm>

namespace libkineto {

ApproximateClockToUnixTimeConverter::ApproximateClockToUnixTimeConverter()
    : start_times_(measurePairs()) {}

ApproximateClockToUnixTimeConverter::UnixAndApproximateTimePair
ApproximateClockToUnixTimeConverter::measurePair() {
  // Take a measurement on either side to avoid an ordering bias.
  auto fast_0 = getApproximateTime();
  auto wall = std::chrono::system_clock::now();
  auto fast_1 = getApproximateTime();

  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
      wall.time_since_epoch());

  // `x + (y - x) / 2` is a more numerically stable average than `(x + y) / 2`.
  return {t.count(), fast_0 + ((fast_1 - fast_0) / 2)};
}

ApproximateClockToUnixTimeConverter::time_pairs
ApproximateClockToUnixTimeConverter::measurePairs() {
  static constexpr auto n_warmup = 5;
  for (int i = 0; i < n_warmup; i++) {
    getApproximateTime();
    static_cast<void>(steady_clock_t::now());
  }

  time_pairs out;
  for (auto& i : out) {
    i = measurePair();
  }
  return out;
}

std::function<time_t(approx_time_t)> ApproximateClockToUnixTimeConverter::
    makeConverter() {
  auto end_times = measurePairs();

  // Compute the real time that passes for each tick of the approximate clock.
  std::array<long double, replicates> scale_factors{};
  for (size_t i = 0; i < replicates; i++) {
    auto delta_ns = end_times[i].t_ - start_times_[i].t_;
    auto delta_approx = end_times[i].approx_t_ - start_times_[i].approx_t_;
    scale_factors[i] =
        static_cast<double>(delta_ns) / static_cast<double>(delta_approx);
  }
  std::ranges::sort(scale_factors);
  long double scale_factor = scale_factors[(replicates / 2) + 1];

  // We shift all times by `t0` for better numerics. Double precision only has
  // 16 decimal digits of accuracy, so if we blindly multiply times by
  // `scale_factor` we may suffer from precision loss. The choice of `t0` is
  // mostly arbitrary; we just need a factor that is the correct order of
  // magnitude to bring the intermediate values closer to zero. We are not,
  // however, guaranteed that `t0_approx` is *exactly* the getApproximateTime
  // equivalent of `t0`; it is only an estimate that we have to fine tune.
  auto t0 = start_times_[0].t_;
  auto t0_approx = start_times_[0].approx_t_;
  std::array<double, replicates> t0_correction{};
  for (size_t i = 0; i < replicates; i++) {
    auto dt = start_times_[i].t_ - t0;
    auto dt_approx =
        static_cast<double>(start_times_[i].approx_t_ - t0_approx) *
        scale_factor;
    t0_correction[i] =
        static_cast<double>(dt - static_cast<time_t>(dt_approx)); // NOLINT
  }
  t0 += static_cast<time_t>(
      t0_correction[t0_correction.size() / 2 + 1]); // NOLINT

  return [=](approx_time_t t_approx) {
    // See above for why this is more stable than `A * t_approx + B`.
    return t_approx > t0_approx
        ? static_cast<time_t>(
              static_cast<double>(t_approx - t0_approx) * scale_factor) +
            t0
        : 0;
  };
}

// Sets the timestamp converter. If nothing is set then the converter just
// returns the input. For this reason, until we add profiler impl of passing in
// TSC converter we just need to guard the callback itself
std::function<time_t(approx_time_t)>& get_time_converter() {
  static std::function<time_t(approx_time_t)> _time_converter =
      [](approx_time_t t) { return t; };
  return _time_converter;
}

} // namespace libkineto
