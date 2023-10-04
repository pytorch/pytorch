#pragma once

#include <torch/data/dataloader_options.h>
#include <torch/data/detail/data_shuttle.h>
#include <torch/data/detail/sequencers.h>
#include <torch/data/iterator.h>
#include <torch/data/samplers/random.h>
#include <torch/data/worker_exception.h>
#include <torch/types.h>

#include <torch/csrc/utils/variadic.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace data {
template <typename Dataset, typename Batch, typename BatchRequest>
class DataLoaderBase {
 public:
  using BatchType = Batch;
  using BatchRequestType = BatchRequest;

  /// Constructs a new DataLoader from a `dataset` to sample from, `options`
  /// to configure the DataLoader with, and a `sampler` that specifies the
  /// sampling strategy.
  DataLoaderBase(
      DataLoaderOptions options,
      std::unique_ptr<Dataset> main_thread_dataset = nullptr)
      : options_(std::move(options)),
        main_thread_dataset_(std::move(main_thread_dataset)),
        sequencer_(new_sequencer()) {}

  // NOLINTNEXTLINE(bugprone-exception-escape)
  virtual ~DataLoaderBase() {
    join();
  }

  /// Returns an iterator into the DataLoader. The lifetime of the iterator is
  /// bound to the DataLoader. In C++ standards language, the category of the
  /// iterator is `OutputIterator`. See
  /// https://en.cppreference.com/w/cpp/named_req/OutputIterator for what this
  /// means. In short: you may increment the iterator and dereference it, but
  /// cannot go back, or step forward more than one position at a time. When the
  /// DataLoader is exhausted, it will compare equal with the special
  /// "sentinel" iterator returned by `DataLoader::end()`. Most of the time, you
  /// should only use range-for loops to loop over the DataLoader, but
  /// standard algorithms like `std::copy(dataloader.begin(), dataloader.end(),
  /// output_iterator)`  are supported too.
  Iterator<Batch> begin() {
    TORCH_CHECK(
        shuttle_.in_flight_jobs() == 0,
        "Attempted to get a new DataLoader iterator "
        "while another iterator is not yet exhausted");
    reset();
    return Iterator<Batch>(std::make_unique<detail::ValidIterator<Batch>>(
        [this] { return this->next(); }));
  }

  /// Returns a special "sentinel" iterator that compares equal with a
  /// non-sentinel iterator once the DataLoader is exhausted.
  Iterator<Batch> end() {
    return Iterator<Batch>(std::make_unique<detail::SentinelIterator<Batch>>());
  }

  /// Joins the DataLoader's worker threads and drains internal queues.
  /// This function may only be invoked from the main thread (in which the
  /// DataLoader lives).
  void join() {
    if (joined_) {
      return;
    }
    shuttle_.drain();
    // Send one 'quit' message per worker. Since a worker dies (exits its
    // thread) after receiving this message, each `QuitWorker()` message will be
    // read by exactly one worker.
    for (const auto w : c10::irange(options_.workers)) {
      (void)w; // Suppress unused variable warning
      push_job(QuitWorker());
    }
    for (auto& worker : workers_) {
      worker.join();
    }
    joined_ = true;
  }

  /// Returns the options with which the DataLoader was configured.
  const FullDataLoaderOptions& options() const noexcept {
    return options_;
  }

 protected:
  /// Simple mix-in to give something a sequence number.
  struct Sequenced {
    Sequenced() = default;
    Sequenced(size_t sqn) : sequence_number(sqn) {}
    size_t sequence_number;
  };

  struct QuitWorker {};

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
    Result(optional<Batch>&& b, size_t sqn)
        : Sequenced(sqn), batch(std::move(b)) {}
    Result(std::exception_ptr exception, size_t sqn)
        : Sequenced(sqn), exception(std::move(exception)) {}
    optional<Batch> batch;
    std::exception_ptr exception;
  };

  /// Subclass hook for getting the next batch request. The stateless case will
  /// ask the sampler for a new batch request (e.g. a vector of indices), while
  /// the stateful one will simply return the batch size.
  virtual optional<BatchRequestType> get_batch_request() = 0;

  /// Resets the internal state of the DataLoader, optionally pre-fetching
  /// new jobs.
  virtual void reset() {
    shuttle_.drain();
    sequence_number_ = 0;
    sequencer_ = new_sequencer();
    prefetch();
  }

  /// Schedules `requested_jobs` many new batches to be fetched. The actual
  /// number of jobs scheduled may be less if the DataLoader exhausts.
  void prefetch(size_t requested_jobs) {
    for (const auto r : c10::irange(requested_jobs)) {
      (void)r; // Suppress unused variable
      if (auto batch_request = get_batch_request()) {
        this->push_job(std::move(*batch_request));
      } else {
        break;
      }
    }
  }

  /// Schedules the maximum number of jobs (based on the `max_jobs` option).
  void prefetch() {
    prefetch(options_.max_jobs);
  }

  /// Returns the next batch of data, or an empty `optional` if the DataLoader
  /// is exhausted. This operation will block until a batch is available if one
  /// is still expected.
  optional<BatchType> next() {
    if (options_.workers > 0) {
      while (optional<Result> result = this->pop_result()) {
        if (result->exception) {
          throw WorkerException(result->exception);
        } else if (result->batch) {
          prefetch(1);
          return std::move(result->batch);
        }
      }
    } else if (auto batch_request = get_batch_request()) {
      return this->main_thread_dataset_->get_batch(std::move(*batch_request));
    }
    return nullopt;
  }

  /// The function that worker threads run.
  void worker_thread(Dataset& dataset) {
    while (true) {
      auto job = shuttle_.pop_job();
      if (job.quit) {
        break;
      }
      try {
        auto batch = dataset.get_batch(std::move(*job.batch_request));
        shuttle_.push_result({std::move(batch), job.sequence_number});
      } catch (...) {
        shuttle_.push_result({std::current_exception(), job.sequence_number});
      }
    }
  }

  /// Convenience method that calls `shuttle_.push_job()` with the next sequence
  /// number.
  template <typename T>
  void push_job(T value) {
    shuttle_.push_job({std::move(value), sequence_number_++});
  }

  /// Convenience method that gets the next result from the sequencer.
  optional<Result> pop_result() {
    return sequencer_->next(
        [this] { return this->shuttle_.pop_result(this->options_.timeout); });
  }

  /// Convenience method that creates a new sequencer based on the
  /// `enforce_ordering` option.
  std::unique_ptr<detail::sequencers::Sequencer<Result>> new_sequencer() {
    if (options_.enforce_ordering) {
      return std::make_unique<detail::sequencers::OrderedSequencer<Result>>(
          options_.max_jobs);
    }
    return std::make_unique<detail::sequencers::NoSequencer<Result>>();
  }

  /// The options the DataLoader was configured with.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const FullDataLoaderOptions options_;

  /// The dataset for the main thread, only has a value if the number of
  /// worker threads was configured as zero, meaning the main thread has to do
  /// all the work (synchronously). NOTE: Really want this to be on the heap
  /// when empty, therefore `unique_ptr` and not `optional`.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<Dataset> main_thread_dataset_;

  /// The sequence number for the *next* batch to be retrieved from the
  /// dataset.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  size_t sequence_number_ = 0;

  /// The worker threads, running the `worker_thread()` method.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::thread> workers_;

  /// The `DataShuttle` which takes care of the life cycle of a job.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  detail::DataShuttle<Job, Result> shuttle_;

  /// The `Sequencer`, which handles optional ordering of batches.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<detail::sequencers::Sequencer<Result>> sequencer_;

  /// True if the DataLoader has joined its worker threads.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool joined_ = false;
};
} // namespace data
} // namespace torch
