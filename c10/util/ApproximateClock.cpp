#include <c10/util/ApproximateClock.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

namespace c10 {

ApproximateClockToUnixTimeConverter::ApproximateClockToUnixTimeConverter()
    : start_times_(measurePairs()) {}

ApproximateClockToUnixTimeConverter::UnixAndApproximateTimePair
ApproximateClockToUnixTimeConverter::measurePair() {
  // Take a measurement on either side to avoid an ordering bias.
  auto fast_0 = getApproximateTime();
  auto wall = std::chrono::system_clock::now();
  auto fast_1 = getApproximateTime();

  TORCH_INTERNAL_ASSERT(fast_1 >= fast_0, "getCount is non-monotonic.");
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
      wall.time_since_epoch());

  // `x + (y - x) / 2` is a more numerically stable average than `(x + y) / 2`.
  return {t.count(), fast_0 + (fast_1 - fast_0) / 2};
}

ApproximateClockToUnixTimeConverter::time_pairs
ApproximateClockToUnixTimeConverter::measurePairs() {
  static constexpr auto n_warmup = 5;
  for ([[maybe_unused]] const auto _ : c10::irange(n_warmup)) {
    getApproximateTime();
    static_cast<void>(steady_clock_t::now());
  }

  time_pairs out;
  for (const auto i : c10::irange(out.size())) {
    out[i] = measurePair();
  }
  return out;
}

std::function<time_t(approx_time_t)> ApproximateClockToUnixTimeConverter::
    makeConverter() {
  auto end_times = measurePairs();

  // Compute the real time that passes for each tick of the approximate clock.
  std::array<long double, replicates> scale_factors{};
  for (const auto i : c10::irange(replicates)) {
    auto delta_ns = end_times[i].t_ - start_times_[i].t_;
    auto delta_approx = end_times[i].approx_t_ - start_times_[i].approx_t_;
    scale_factors[i] = (double)delta_ns / (double)delta_approx;
  }
  std::sort(scale_factors.begin(), scale_factors.end());
  long double scale_factor = scale_factors[replicates / 2 + 1];

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
  for (const auto i : c10::irange(replicates)) {
    auto dt = start_times_[i].t_ - t0;
    auto dt_approx =
        (double)(start_times_[i].approx_t_ - t0_approx) * scale_factor;
    t0_correction[i] = dt - (time_t)dt_approx; // NOLINT
  }
  t0 += t0_correction[t0_correction.size() / 2 + 1]; // NOLINT

  return [=](approx_time_t t_approx) {
    // See above for why this is more stable than `A * t_approx + B`.
    return (time_t)((double)(t_approx - t0_approx) * scale_factor) + t0;
  };
}

} // namespace c10
