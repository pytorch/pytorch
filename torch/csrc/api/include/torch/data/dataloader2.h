#pragma once

#include <torch/data/dataloader_options.h>
#include <torch/data/detail/data_shuttle.h>
#include <torch/data/detail/sequencers.h>
#include <torch/data/iterator.h>
#include <torch/data/worker_exception.h>
#include <torch/types.h>

#include <torch/csrc/utils/memory.h>
#include <torch/csrc/utils/variadic.h>

#include <c10/util/Exception.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace data {
template <typename Dataset>
class DataLoader2 {
 public:
  using Batch = typename Dataset::BatchType;
  using UnwrappedBatch = typename Batch::value_type;
  using BatchRequest = size_t;

  /// Constructs a new `DataLoader2` from a `dataset` to sample from and
  /// `options` to configure the `DataLoader2` with.
  DataLoader2(Dataset dataset, DataLoaderOptions options)
      : options_(std::move(options)),
        dataset_(std::move(dataset)),
        sequencer_(new_sequencer()) {
    for (size_t w = 0; w < options_.workers; ++w) {
      workers_.emplace_back([this] { this->worker_thread(); });
    }
  }

  virtual ~DataLoader2() {
    join();
  }

  /// Returns an iterator into the `DataLoader2`. The lifetime of the iterator
  /// is bound to the `DataLoader2`. In C++ standards language, the category of
  /// the iterator is `OutputIterator`. See
  /// https://en.cppreference.com/w/cpp/named_req/OutputIterator for what this
  /// means. In short: you may increment the iterator and dereference it, but
  /// cannot go back, or step forward more than one position at a time. When the
  /// `DataLoader2` is exhausted, it will compare equal with the special
  /// "sentinel" iterator returned by `DataLoader2::end()`. Most of the time,
  /// you should only use range-for loops to loop over the `DataLoader2`, but
  /// standard algorithms like `std::copy(dataloader.begin(), dataloader.end(),
  /// output_iterator)`  are supported too.
  Iterator<UnwrappedBatch> begin() {
    AT_CHECK(
        shuttle_.in_flight_jobs() == 0,
        "Attempted to get a new DataLoader2 iterator "
        "while another iterator is not yet exhausted");
    reset();
    return Iterator<UnwrappedBatch>(
        torch::make_unique<detail::ValidIterator<UnwrappedBatch>>(
            [this] { return this->next(); }));
  }

  /// Returns a special "sentinel" iterator that compares equal with a
  /// non-sentinel iterator once the `DataLoader2` is exhausted.
  Iterator<UnwrappedBatch> end() {
    return Iterator<UnwrappedBatch>(
        torch::make_unique<detail::SentinelIterator<UnwrappedBatch>>());
  }

  /// Joins the `DataLoader2`'s worker threads and drains internal queues.
  /// This function may only be invoked from the main thread (in which the
  /// `DataLoader2` lives).
  void join() {
    if (joined_) {
      return;
    }
    shuttle_.drain();
    // Send one 'quit' message per worker. Since a worker dies (exits its
    // thread) after receiving this message, each `QuitWorker()` message will be
    // read by exactly one worker.
    for (size_t w = 0; w < options_.workers; ++w) {
      push_job(QuitWorker());
    }
    for (auto& worker : workers_) {
      worker.join();
    }
    joined_ = true;
  }

  /// Returns the options with which the `DataLoader2` was configured.
  const FullDataLoaderOptions& options() const noexcept {
    return options_;
  }

 private:
  /// Simple mix-in to give something a sequence number.
  struct Sequenced {
    Sequenced() = default;
    Sequenced(size_t sqn) : sequence_number(sqn) {}
    size_t sequence_number;
  };

  struct QuitWorker {};
  struct StopEpoch {};

  /// A `Job` is either a `BatchRequest` (new indices to fetch data at) or a
  /// `QuitWorker` object, to indicate the worker should shut down.
  struct Job : Sequenced {
    Job() = default;
    Job(QuitWorker q, size_t sqn) : Sequenced(sqn), quit(q) {}
    Job(BatchRequest&& i, size_t sqn)
        : Sequenced(sqn), batch_request(std::move(i)) {}
    optional<QuitWorker> quit;
    optional<BatchRequest> batch_request;
  };

  /// The finished result of a job.
  struct Result : Sequenced {
    Result() = default;
    Result(optional<UnwrappedBatch>&& b, size_t sqn)
        : Sequenced(sqn), batch(std::move(b)) {}
    Result(StopEpoch s, size_t sqn) : Sequenced(sqn), stop_epoch(s) {}
    Result(std::exception_ptr exception, size_t sqn)
        : Sequenced(sqn), exception(std::move(exception)) {}
    optional<UnwrappedBatch> batch;
    optional<StopEpoch> stop_epoch;
    std::exception_ptr exception;
  };

  /// Resets the internal state of the `DataLoader2`, optionally pre-fetching
  /// new jobs.
  void reset(bool prefetch = true) {
    shuttle_.drain();
    sequence_number_ = 0;
    sequencer_ = new_sequencer();
    dataset_.reset();
    if (prefetch && options_.workers > 0) {
      this->prefetch();
    }
  }

  /// Schedules `requested_jobs` many new batches to be fetched. The actual
  /// number of jobs scheduled may be less if the `DataLoader2` exhausts.
  void prefetch(size_t requested_jobs) {
    while (requested_jobs-- > 0) {
      push_job(options_.batch_size);
    }
  }

  /// Schedules the maximum number of jobs (based on the `max_jobs` option).
  void prefetch() {
    prefetch(options_.max_jobs);
  }

  /// Returns the next batch of data, or an empty `optional` if the
  /// `DataLoader2` is exhausted. This operation will block until a batch is
  /// available.

  optional<UnwrappedBatch> next() {
    if (options_.workers > 0) {
      while (auto result = pop_result()) {
        if (result->exception) {
          throw WorkerException(result->exception);
        } else if (result->batch) {
          prefetch(1);
          return std::move(result->batch);
        }
      }
    } else {
      return dataset_.get_batch(options_.batch_size);
    }
    return at::nullopt;
  }

  /// The function that worker threads run.
  void worker_thread() {
    while (true) {
      auto job = shuttle_.pop_job();
      if (job.quit) {
        break;
      }
      try {
        auto batch = dataset_.get_batch(std::move(*job.batch_request));
        shuttle_.push_result({std::move(batch), job.sequence_number});
      } catch (...) {
        shuttle_.push_result({std::current_exception(), job.sequence_number});
      }
    }
  }

  template <typename T>
  void push_job(T value) {
    shuttle_.push_job({std::move(value), sequence_number_++});
  }

  optional<Result> pop_result() {
    return sequencer_->next(
        [this] { return this->shuttle_.pop_result(this->options_.timeout); });
  }

  std::unique_ptr<detail::sequencers::Sequencer<Result>> new_sequencer() {
    if (options_.enforce_ordering) {
      return torch::make_unique<detail::sequencers::OrderedSequencer<Result>>(
          options_.max_jobs);
    }
    return torch::make_unique<detail::sequencers::NoSequencer<Result>>();
  }

  /// The options the `DataLoader2` was configured with.
  const FullDataLoaderOptions options_;

  Dataset dataset_;

  /// The sequence number for the *next* batch to be retrieved from the
  /// dataset.
  size_t sequence_number_ = 0;

  /// The worker threads, running the `worker_thread()` method.
  std::vector<std::thread> workers_;

  /// The `DataShuttle` which takes care of the life cycle of a job.
  detail::DataShuttle<Job, Result> shuttle_;

  /// The `Sequencer`, which handles optional ordering of batches.
  std::unique_ptr<detail::sequencers::Sequencer<Result>> sequencer_;

  /// True if the `DataLoader2` has joined its worker threads.
  bool joined_ = false;
}; // namespace data

/// Creates a new `DataLoader2`, inferring the necessary template types from
/// the given arguments.
template <typename Dataset>
std::unique_ptr<DataLoader2<Dataset>> make_data_loader2(
    Dataset dataset,
    DataLoaderOptions options) {
  return torch::make_unique<DataLoader2<Dataset>>(
      std::move(dataset), std::move(options));
}
} // namespace data
} // namespace torch
