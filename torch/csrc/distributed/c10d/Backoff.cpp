#include <torch/csrc/distributed/c10d/Backoff.hpp>

#include <c10/util/Exception.h>
#include <stdexcept>
// @allow-raw-throw

namespace c10d {
namespace {
constexpr std::chrono::milliseconds kZeroInterval{0};

std::random_device::result_type randSeed() {
  std::random_device rd;
  return rd();
}
} // namespace

ExponentialBackoffWithJitter::ExponentialBackoffWithJitter()
    : gen_(randSeed()) {}

std::chrono::milliseconds ExponentialBackoffWithJitter::nextBackoff() {
  TORCH_CHECK(
      initialInterval != kZeroInterval,
      "ExponentialBackoffWithJitter requires non-zero initial interval");
  TORCH_CHECK(
      initialInterval <= maxInterval,
      "ExponentialBackoffWithJitter requires initialInterval <= maxInterval");
  TORCH_CHECK(
      randomizationFactor < 1 && randomizationFactor >= 0,
      "ExponentialBackoffWithJitter requires randomization factor (0,1]");
  TORCH_CHECK(
      multiplier >= 1.0,
      "ExponentialBackoffWithJitter requires multiplier >=1");

  // detect initial setup
  if (currentInterval_ == kZeroInterval) {
    currentInterval_ = initialInterval;
  }

  // sample current interval
  std::chrono::milliseconds randomization{static_cast<int64_t>(
      randomizationFactor * static_cast<double>(currentInterval_.count()))};
  std::chrono::milliseconds minSampleInterval =
      currentInterval_ - randomization;
  std::chrono::milliseconds maxSampleInterval =
      currentInterval_ + randomization;

  std::uniform_int_distribution<int64_t> dist(
      minSampleInterval.count(), maxSampleInterval.count());
  std::chrono::milliseconds backoffInterval{dist(gen_)};

  // update current interval
  currentInterval_ = std::chrono::milliseconds(static_cast<int64_t>(
      static_cast<double>(currentInterval_.count()) * multiplier));

  if (currentInterval_ > maxInterval) {
    currentInterval_ = maxInterval;
  }

  return backoffInterval;
}

void ExponentialBackoffWithJitter::reset() {
  currentInterval_ = kZeroInterval;
}

FixedBackoff::FixedBackoff(std::chrono::milliseconds interval)
    : interval_(interval) {}

std::chrono::milliseconds FixedBackoff::nextBackoff() {
  return interval_;
}

void FixedBackoff::reset() {}
} // namespace c10d
