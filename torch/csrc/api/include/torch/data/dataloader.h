#pragma once

#include <torch/data/dataloader_options.h>
#include <torch/data/detail/data_shuttle.h>
#include <torch/data/detail/sequencers.h>
#include <torch/data/iterator.h>
#include <torch/data/samplers/random.h>
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
template <typename Dataset, typename Sampler>
class DataLoader {
 public:
  using Batch = typename Dataset::BatchType;
  using BatchRequest = typename Sampler::BatchRequestType;

  /// Constructs a new `DataLoader` from a `dataset` to sample from, `options`
  /// to configure the `DataLoader` with, and a `sampler` that specifies the
  /// sampling strategy.
  DataLoader(Dataset dataset, DataLoaderOptions options, Sampler sampler)
      : options_(std::move(options)),
        sampler_(std::move(sampler)),
        sequencer_(new_sequencer()) {
    for (size_t w = 0; w < options_.workers; ++w) {
      // Here we copy the dataset into the worker thread closure. Each worker
      // has its own copy of the dataset. This means the dataset must be
      // trivially copiable, or else we don't expect more than one worker to
      // be in use.
      workers_.emplace_back(
          [this, dataset] { this->worker_thread(std::move(dataset)); });
    }
    if (options_.workers == 0) {
      main_thread_dataset_ = torch::make_unique<Dataset>(std::move(dataset));
    }
  }

  virtual ~DataLoader() {
    join();
  }

  /// Returns an iterator into the `DataLoader`. The lifetime of the iterator is
  /// bound to the `DataLoader`. In C++ standards language, the category of the
  /// iterator is `OutputIterator`. See
  /// https://en.cppreference.com/w/cpp/named_req/OutputIterator for what this
  /// means. In short: you may increment the iterator and dereference it, but
  /// cannot go back, or step forward more than one position at a time. When the
  /// `DataLoader` is exhausted, it will compare equal with the special
  /// "sentinel" iterator returned by `DataLoader::end()`. Most of the time, you
  /// should only use range-for loops to loop over the `DataLoader`, but
  /// standard algorithms like `std::copy(dataloader.begin(), dataloader.end(),
  /// output_iterator)`  are supported too.
  Iterator<Batch> begin() {
    AT_CHECK(
        shuttle_.in_flight_jobs() == 0,
        "Attempted to get a new DataLoader iterator "
        "while another iterator is not yet exhausted");
    reset();
    return Iterator<Batch>(torch::make_unique<detail::ValidIterator<Batch>>(
        [this] { return this->next(); }));
  }

  /// Returns a special "sentinel" iterator that compares equal with a
  /// non-sentinel iterator once the `DataLoader` is exhausted.
  Iterator<Batch> end() {
    return Iterator<Batch>(
        torch::make_unique<detail::SentinelIterator<Batch>>());
  }

  /// Joins the `DataLoader`'s worker threads and drains internal queues.
  /// This function may only be invoked from the main thread (in which the
  /// `DataLoader` lives).
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

  /// Returns the options with which the `DataLoader` was configured.
  const FullDataLoaderOptions& options() const noexcept {
    return options_;
  }

  /// Returns the sampler currently used by the `DataLoader`.
  const Sampler& sampler() const noexcept {
    return sampler_;
  }

  /// Returns the sampler currently used by the `DataLoader`.
  Sampler& sampler() noexcept {
    return sampler_;
  }

 private:
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
    Result(Batch&& b, size_t sqn) : Sequenced(sqn), batch(std::move(b)) {}
    Result(std::exception_ptr exception, size_t sqn)
        : Sequenced(sqn), exception(std::move(exception)) {}
    optional<Batch> batch;
    std::exception_ptr exception;
  };

  /// Resets the internal state of the `DataLoader`, optionally pre-fetching
  /// new jobs.
  void reset(bool prefetch = true) {
    shuttle_.drain();
    sampler_.reset();
    sequence_number_ = 0;
    sequencer_ = new_sequencer();
    if (prefetch) {
      this->prefetch();
    }
  }

  /// Schedules `requested_jobs` many new batches to be fetched. The actual
  /// number of jobs scheduled may be less if the `DataLoader` exhausts.
  void prefetch(size_t requested_jobs) {
    while (requested_jobs-- > 0) {
      if (auto batch_request = get_batch_request()) {
        push_job(std::move(*batch_request));
      } else {
        break;
      }
    }
  }

  /// Schedules the maximum number of jobs (based on the `max_jobs` option).
  void prefetch() {
    prefetch(options_.max_jobs);
  }

  /// Returns the next batch of data, or an empty `optional` if the `DataLoader`
  /// is exhausted. This operation will block until a batch is available.
  optional<Batch> next() {
    optional<Batch> batch;
    if (options_.workers > 0) {
      optional<Result> result = sequencer_->next(
          [this] { return this->shuttle_.pop_result(this->options_.timeout); });
      if (result) {
        if (result->exception) {
          throw WorkerException(result->exception);
        } else {
          AT_ASSERT(result->batch.has_value());
          batch = std::move(result->batch);
          prefetch(1);
        }
      }
    } else if (auto batch_request = get_batch_request()) {
      AT_ASSERT(main_thread_dataset_ != nullptr);
      batch = main_thread_dataset_->get_batch(std::move(*batch_request));
    }
    return batch;
  }

  /// The function that worker threads run.
  void worker_thread(Dataset dataset) {
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

  optional<BatchRequest> get_batch_request() {
    auto indices = sampler_.next(options_.batch_size);
    if (!indices ||
        (indices->size() < options_.batch_size && options_.drop_last)) {
      return nullopt;
    }
    AT_ASSERT(indices->size() > 0);
    return indices;
  }

  template <typename T>
  void push_job(T value) {
    shuttle_.push_job({std::move(value), sequence_number_++});
  }

  std::unique_ptr<detail::sequencers::Sequencer<Result>> new_sequencer() {
    if (options_.enforce_ordering) {
      return torch::make_unique<detail::sequencers::OrderedSequencer<Result>>(
          options_.max_jobs);
    }
    return torch::make_unique<detail::sequencers::NoSequencer<Result>>();
  }

  /// The options the `DataLoader` was configured with.
  const FullDataLoaderOptions options_;

  /// The dataset for the main thread, only has a value if the number of
  /// worker threads was configured as zero, meaning the main thread has to do
  /// all the work (synchronously). NOTE: Really want this to be on the heap
  /// when empty, therefore `unique_ptr` and not `optional`.
  std::unique_ptr<Dataset> main_thread_dataset_;

  /// The sampler with which new batch requests are created.
  Sampler sampler_;

  /// The sequence number for the *next* batch to be retrieved from the
  /// dataset.
  size_t sequence_number_ = 0;

  /// The worker threads, running the `worker_thread()` method.
  std::vector<std::thread> workers_;

  /// The `DataShuttle` which takes care of the life cycle of a job.
  detail::DataShuttle<Job, Result> shuttle_;

  /// The `Sequencer`, which handles optional ordering of batches.
  std::unique_ptr<detail::sequencers::Sequencer<Result>> sequencer_;

  /// True if the `DataLoader` has joined its worker threads.
  bool joined_ = false;
}; // namespace data

/// Creates a new `DataLoader`, inferring the necessary template types from
/// the given arguments.
template <typename Dataset, typename Sampler>
std::unique_ptr<DataLoader<Dataset, Sampler>> make_data_loader(
    Dataset dataset,
    DataLoaderOptions options,
    Sampler sampler) {
  return torch::make_unique<DataLoader<Dataset, Sampler>>(
      std::move(dataset), std::move(options), std::move(sampler));
}

/// Creates a new `DataLoader`, inferring the necessary template types from
/// the given arguments.
template <
    typename Dataset,
    typename Sampler = samplers::RandomSampler,
    typename =
        torch::enable_if_t<std::is_constructible<Sampler, size_t>::value>>
std::unique_ptr<DataLoader<Dataset, Sampler>> make_data_loader(
    Dataset dataset,
    DataLoaderOptions options = DataLoaderOptions()) {
  const optional<size_t> size = dataset.size();
  AT_CHECK(
      size.has_value(),
      "Expected the dataset to be sized in "
      "order to construct the Sampler");
  return make_data_loader(
      std::move(dataset), std::move(options), Sampler(*size));
}

} // namespace data
} // namespace torch
