// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace common
{
/**
 * @brief A timepoint relative to the system clock epoch.
 *
 * This is used for marking the beginning and end of an operation.
 */
class SystemTimestamp
{
public:
  /**
   * @brief Initializes a system timestamp pointing to the start of the epoch.
   */
  SystemTimestamp() noexcept : nanos_since_epoch_{0} {}

  /**
   * @brief Initializes a system timestamp from a duration.
   *
   * @param time_since_epoch Time elapsed since the beginning of the epoch.
   */
  template <class Rep, class Period>
  explicit SystemTimestamp(const std::chrono::duration<Rep, Period> &time_since_epoch) noexcept
      : nanos_since_epoch_{static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count())}
  {}

  /**
   * @brief Initializes a system timestamp based on a point in time.
   *
   * @param time_point A point in time.
   */
  /*implicit*/ SystemTimestamp(const std::chrono::system_clock::time_point &time_point) noexcept
      : SystemTimestamp{time_point.time_since_epoch()}
  {}

  /**
   * @brief Returns a time point for the time stamp.
   *
   * @return A time point corresponding to the time stamp.
   */
  operator std::chrono::system_clock::time_point() const noexcept
  {
    return std::chrono::system_clock::time_point{
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::nanoseconds{nanos_since_epoch_})};
  }

  /**
   * @brief Returns the nanoseconds since the beginning of the epoch.
   *
   * @return Elapsed nanoseconds since the beginning of the epoch for this timestamp.
   */
  std::chrono::nanoseconds time_since_epoch() const noexcept
  {
    return std::chrono::nanoseconds{nanos_since_epoch_};
  }

  /**
   * @brief Compare two steady time stamps.
   *
   * @return true if the two time stamps are equal.
   */
  bool operator==(const SystemTimestamp &other) const noexcept
  {
    return nanos_since_epoch_ == other.nanos_since_epoch_;
  }

  /**
   * @brief Compare two steady time stamps for inequality.
   *
   * @return true if the two time stamps are not equal.
   */
  bool operator!=(const SystemTimestamp &other) const noexcept
  {
    return nanos_since_epoch_ != other.nanos_since_epoch_;
  }

private:
  int64_t nanos_since_epoch_;
};

/**
 * @brief A timepoint relative to the monotonic clock epoch
 *
 * This is used for calculating the duration of an operation.
 */
class SteadyTimestamp
{
public:
  /**
   * @brief Initializes a monotonic timestamp pointing to the start of the epoch.
   */
  SteadyTimestamp() noexcept : nanos_since_epoch_{0} {}

  /**
   * @brief Initializes a monotonic timestamp from a duration.
   *
   * @param time_since_epoch Time elapsed since the beginning of the epoch.
   */
  template <class Rep, class Period>
  explicit SteadyTimestamp(const std::chrono::duration<Rep, Period> &time_since_epoch) noexcept
      : nanos_since_epoch_{static_cast<int64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count())}
  {}

  /**
   * @brief Initializes a monotonic timestamp based on a point in time.
   *
   * @param time_point A point in time.
   */
  /*implicit*/ SteadyTimestamp(const std::chrono::steady_clock::time_point &time_point) noexcept
      : SteadyTimestamp{time_point.time_since_epoch()}
  {}

  /**
   * @brief Returns a time point for the time stamp.
   *
   * @return A time point corresponding to the time stamp.
   */
  operator std::chrono::steady_clock::time_point() const noexcept
  {
    return std::chrono::steady_clock::time_point{
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds{nanos_since_epoch_})};
  }

  /**
   * @brief Returns the nanoseconds since the beginning of the epoch.
   *
   * @return Elapsed nanoseconds since the beginning of the epoch for this timestamp.
   */
  std::chrono::nanoseconds time_since_epoch() const noexcept
  {
    return std::chrono::nanoseconds{nanos_since_epoch_};
  }

  /**
   * @brief Compare two steady time stamps.
   *
   * @return true if the two time stamps are equal.
   */
  bool operator==(const SteadyTimestamp &other) const noexcept
  {
    return nanos_since_epoch_ == other.nanos_since_epoch_;
  }

  /**
   * @brief Compare two steady time stamps for inequality.
   *
   * @return true if the two time stamps are not equal.
   */
  bool operator!=(const SteadyTimestamp &other) const noexcept
  {
    return nanos_since_epoch_ != other.nanos_since_epoch_;
  }

private:
  int64_t nanos_since_epoch_;
};

class DurationUtil
{
public:
  template <class Rep, class Period>
  static std::chrono::duration<Rep, Period> AdjustWaitForTimeout(
      std::chrono::duration<Rep, Period> timeout,
      std::chrono::duration<Rep, Period> indefinite_value) noexcept
  {
    // Do not call now() when this duration is max value, now() may have a expensive cost.
    if (timeout == (std::chrono::duration<Rep, Period>::max)())
    {
      return indefinite_value;
    }

    // std::future<T>::wait_for, std::this_thread::sleep_for, and std::condition_variable::wait_for
    // may use steady_clock or system_clock.We need make sure now() + timeout do not overflow.
    auto max_timeout = std::chrono::duration_cast<std::chrono::duration<Rep, Period>>(
        (std::chrono::steady_clock::time_point::max)() - std::chrono::steady_clock::now());
    if (timeout >= max_timeout)
    {
      return indefinite_value;
    }
    max_timeout = std::chrono::duration_cast<std::chrono::duration<Rep, Period>>(
        (std::chrono::system_clock::time_point::max)() - std::chrono::system_clock::now());
    if (timeout >= max_timeout)
    {
      return indefinite_value;
    }

    return timeout;
  }
};

}  // namespace common
OPENTELEMETRY_END_NAMESPACE
