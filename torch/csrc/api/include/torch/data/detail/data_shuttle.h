#pragma once

#include <torch/data/detail/queue.h>
#include <torch/types.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <atomic>
#include <chrono>
#include <utility>

namespace torch {
namespace data {
namespace detail {

/// Encapsulates the full life cycle of DataLoader jobs.
///
/// When a new job is enqueued to the `DataShuttle`, a counter for in-flight
/// jobs is bumped. This job is said to be "in-flight" until its result is
/// popped. Worker threads dequeue jobs as soon as they are available. When a
/// worker finishes a job, it enqueues the result. Only when the main thread
/// dequeues a result is the count of in-flight jobs decremented. When the main
/// thread attempts to dequeue a job but no jobs are in-flight, that means the
/// epoch is complete and `pop_result` returns an empty optional.
template <typename Job, typename Result>
class DataShuttle {
 public:
  DataShuttle() : in_flight_jobs_(0) { }
  DataShuttle(DataShuttle&&) = default;
  DataShuttle& operator=(DataShuttle&&) = default;
  DataShuttle(const DataShuttle&) = delete;
  DataShuttle& operator=(const DataShuttle&) = delete;

  /// Pushes a new job. Called by the main thread.
  void push_job(Job job) {
    new_jobs_.push(std::move(job));
    ++in_flight_jobs_;
  }

  /// Pushes the result of a job. Called by worker threads.
  void push_result(Result result) {
    AT_ASSERT(in_flight_jobs_ > 0);
    results_.push(std::move(result));
  }

  /// Returns the next job, blocking until there is one available. Called by
  /// worker threads.
  Job pop_job() {
    return new_jobs_.pop();
  }

  /// Returns the result of a job, or nullopt if all jobs were exhausted. Called
  /// by the main thread.
  optional<Result> pop_result(
      optional<std::chrono::milliseconds> timeout = nullopt) {
    if (in_flight_jobs_ > 0) {
      auto result = results_.pop(timeout);
      AT_ASSERT(in_flight_jobs_ > 0);
      --in_flight_jobs_;
      return result;
    }
    return nullopt;
  }

  /// Discards any jobs that are not yet in flight, and waits for all in-flight
  /// jobs to finish, discarding their result.
  void drain() {
    // AT_ASSERT(in_flight_jobs_ == new_jobs_.size() + results_.size());
    // Clear all inputs so that no further jobs are scheduled.
    const auto number_cleared = new_jobs_.clear();
    AT_ASSERT(in_flight_jobs_ >= number_cleared);
    in_flight_jobs_ -= number_cleared;
    // Remove any outstanding results.
    while (in_flight_jobs_ > 0) {
      pop_result();
    }
  }

  /// Returns the number of jobs that are still in progress.
  /// When this number is zero, an epoch is finished.
  size_t in_flight_jobs() const noexcept {
    return in_flight_jobs_;
  }

 private:
  /// The queue for jobs that are not yet in flight.
  Queue<Job> new_jobs_;
  /// The number of in-flight jobs.
  std::atomic<size_t> in_flight_jobs_;
  /// The queue for results of finished jobs.
  Queue<Result> results_;
};

} // namespace detail
} // namespace data
} // namespace torch
