#pragma once

#include <c10/util/irange.h>
#include <torch/arg.h>
#include <torch/csrc/utils/memory.h>
#include <torch/data/datasets/stateful.h>
#include <torch/data/samplers.h>
#include <queue>
#include <thread>

#include <torch/serialize.h>

namespace torch {
namespace data {
namespace datasets {

/// Interface for chunk reader, which performs data chunking and reading of
/// entire chunks.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text-file split based on seek
/// positions.
template <
    typename ExampleType_,
    typename ChunkType_ = std::vector<ExampleType_>>
class ChunkDataReader {
 public:
  virtual ~ChunkDataReader() = default;

  using ChunkType = ChunkType_;
  using ExampleType = ExampleType_;

  /// Read an entire chunk.
  virtual ChunkType read_chunk(size_t chunk_index) = 0;

  /// Returns the number of chunks available in this reader.
  virtual size_t chunk_count() = 0;

  /// This will clear any internal state associate with this reader.
  virtual void reset() = 0;
};

namespace detail {
/// BatchDataBuffer manages a queue of UnwrappedBatchData. After a new chunk is
/// loaded, BatchDataBuffer splits it into small batches and push them into the
/// queue. When get_batch is called from data loader, it pops cached batches and
/// return. If the cache is empty, it either waits to load more chunks or return
/// null if all chunks are loaded.
template <
    typename UnwrappedBatch,
    typename ExampleSampler = samplers::RandomSampler>
class BatchDataBuffer {
 public:
  using UnwrappedBatchType = UnwrappedBatch;
  using BatchType = torch::optional<UnwrappedBatchType>;
  using BatchRequestType = typename ExampleSampler::BatchRequestType;

  BatchDataBuffer(
      size_t batch_size,
      ExampleSampler& example_sampler,
      size_t queue_capacity)
      : batch_size_(batch_size),
        example_sampler_(example_sampler),
        queue_capacity_(queue_capacity) {}

  /// Return batch data from the queue. Called from the ChunkDataset main
  /// thread.
  BatchType get_batch() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_read_.wait(lock, [this] {
      // wait till there is available data in the queue or if all chunks are
      // loaded (i.e. the dataset is exhausted for this epoch)
      return (
          this->total_example_count_in_queue_ >= batch_size_ || this->stop_);
    });
    if (batch_queue_.empty()) {
      AT_ASSERT(stop_);
      // All batches have been retrieved. Return an empty batch.
      return nullopt;
    }

    UnwrappedBatchData batch = std::move(batch_queue_.front());
    batch_queue_.pop();
    if (batch.exception) {
      throw WorkerException(batch.exception);
    }

    total_example_count_in_queue_ -= batch.batch_data.size();
    lock.unlock();
    cv_write_.notify_all();

    return batch.batch_data;
  }

  /// Push preloaded chunks to batch queue. Called from the ChunkDataset worker
  /// threads.
  void add_chunk_data(UnwrappedBatchType data) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_write_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < this->queue_capacity_ ||
          this->stop_;
    });
    if (stop_) {
      // When stop_ is true, it means no further chunk loading is necessary.
      // Return without any further processing.
      return;
    }

    auto data_size = data.size();
    auto remaining_size = data_size;
    example_sampler_.reset(data_size);

    auto fill_batch = [&](size_t example_count, UnwrappedBatchType& batch) {
      auto batch_example_indices = this->example_sampler_.next(example_count);
      AT_ASSERT(
          batch_example_indices &&
          batch_example_indices.value().size() == example_count);
      BatchRequestType& indices = batch_example_indices.value();
      for (size_t i : indices) {
        TORCH_CHECK(i < data_size, "Index out of range");
        batch.emplace_back(std::move(data[i]));
      }
      remaining_size -= example_count;
    };

    if (!batch_queue_.empty()) {
      // if the queue has existing data, and the last batch doesn't have enough
      // examples to fill a batch_size batch, add more example to this batch
      // first.
      auto& batch = batch_queue_.back();
      size_t current_count = batch.batch_data.size();
      if (current_count < batch_size_) {
        auto example_count =
            std::min(remaining_size, batch_size_ - current_count);
        fill_batch(example_count, batch.batch_data);
      }
    }

    // If we still have data remaining after filling the last pushed batch, add
    // them to the queue too.
    // NOLINTNEXTLINE(bugprone-infinite-loop)
    while (remaining_size > 0) {
      UnwrappedBatchType current_batch;

      // Allocate the batch memory ahead of time.
      current_batch.reserve(batch_size_);

      auto example_count = std::min(remaining_size, batch_size_);
      fill_batch(example_count, current_batch);
      batch_queue_.emplace(std::move(current_batch));
    }
    total_example_count_in_queue_ += data_size;
    lock.unlock();
    cv_read_.notify_all();
  }

  /// Push exceptions thrown during preloading into batch queue. Called from
  /// the ChunkDataset worker threads.
  void add_chunk_data(std::exception_ptr e_ptr) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_write_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return (
          this->total_example_count_in_queue_ < this->queue_capacity_ ||
          this->stop_);
    });
    if (stop_) {
      // When stop_ is true, it means this current thread needs to be tore down,
      // the batch buffer will be discarded, so no need to enqueue any new
      // exceptions.
      return;
    }

    batch_queue_.emplace(e_ptr);
    lock.unlock();
    cv_read_.notify_all();
  }

  void stop() {
    {
      // Hold the lock before changing stop_ to prevent a race condition which
      // can cause a deadlock. To be more specific, conditional variable
      // cv_write_ waits on predicate stop_ in add_chunk_data(). The wait
      // happens in two steps: 1) while still holding the lock, check if
      // predicate is true; 2) if it is true, proceeds, otherwise, release the
      // lock and wait until notified. Without holding a lock, cv_write_'s
      // notification can happen in between step 1) and 2). In that case, as
      // cv_write_ is not in waiting status yet, so the notification is lost and
      // cv_write_ will sleep forever. By taking a lock before changing
      // predicate stop_, it is ensured updating and evaluating stop_ always
      // happen in a synchronized way
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }

    // notify all writers, wake them from wait to exit current method.
    cv_write_.notify_all();
    // notify all readers too.
    cv_read_.notify_all();
  }
  /// The batch size is needed to create batches from the chunk data. Similar to
  /// regular dataloader where the batches are created with prefetches,
  /// BatchDataBuffer perform the batch creation using the provided batch size.
  size_t batch_size_ = 0;

  /// count of total example stored in the queue
  size_t total_example_count_in_queue_ = 0;

  /// struct that contains a raw unwrapped batch unit. An unwrapped batch unit
  /// is the raw data without 'optional' wrapper. It can be a collection of
  /// images, utterances, e.t.c.
  struct UnwrappedBatchData {
    explicit UnwrappedBatchData(UnwrappedBatchType data)
        : batch_data(std::move(data)) {}

    // NOLINTNEXTLINE(modernize-pass-by-value)
    explicit UnwrappedBatchData(std::exception_ptr e) : exception(e) {}

    /// batch data to return
    UnwrappedBatchType batch_data;

    /// exception pointer which captures any abnormal exceptions while creating
    /// the batch.
    std::exception_ptr exception;
  };

  /// local cache to store example batches from loaded chunk
  std::queue<UnwrappedBatchData> batch_queue_;

  // sync batch_queue_ update.
  std::mutex queue_mutex_;

  std::condition_variable cv_read_;
  std::condition_variable cv_write_;

  ExampleSampler& example_sampler_;

  // configurable maximun number of elements the queue can hold at one time.
  size_t queue_capacity_;

  // When set to true, it wakes the writer threads from the wait and exit
  // current function call. This is needed when ChunkDataSet.Reset is called
  // while the previous epoch is not exhausted yet. When ChunkDataset is waiting
  // its preloader to finish previous work before tearing down the thread, the
  // preloader could be still waiting for the conditional variable, thus cause
  // the program to hang. This boolean is used to break this waiting condition.
  bool stop_ = false;
};
} // namespace detail

/// Options to configure a `ChunkDataset`.
struct ChunkDatasetOptions {
  ChunkDatasetOptions() = delete;
  ChunkDatasetOptions(
      size_t preloader_count,
      size_t batch_size,
      size_t cache_size = 2048,
      size_t cross_chunk_shuffle_count = 1)
      : preloader_count_(preloader_count),
        batch_size_(batch_size),
        cache_size_(cache_size),
        cross_chunk_shuffle_count_(cross_chunk_shuffle_count) {
    TORCH_CHECK(
        preloader_count_ > 0,
        "Preloader count is 0. At least one preloader needs to be specified.");
    TORCH_CHECK(
        batch_size_ > 0,
        "Batch size is 0. A positive batch size needs to be specified.");
    TORCH_CHECK(
        cache_size_ > 0,
        "Cache size is 0. A positive cache size needs to be specified.");
    TORCH_CHECK(
        cache_size_ >= batch_size_,
        "Cache size is less than batch size. Cache needs to be large enough to "
        "hold at least one batch.");
    TORCH_CHECK(
        cross_chunk_shuffle_count_ > 0,
        "cross_chunk_shuffle_count needs to be greater than 0.");
  }

  /// The number of worker thread to preload chunk data.
  TORCH_ARG(size_t, preloader_count);

  /// The size of each batch.
  TORCH_ARG(size_t, batch_size);

  /// The capacity of the queue for batch caching.
  TORCH_ARG(size_t, cache_size) = 2048;

  // The number of chunks to perfrom cross-chunk shuffling. Default to 1 meaning
  // no cross-chunk shuffling. When it is equal to n (n > 1), n random
  // chunks will be loaded at once and example shuffling will be performed
  // across all those n chunks.
  // Note: Usually the default config (1 chunk shuffle + example shuffle) is
  // good enough to generate random distributed data. Use this parameter only if
  // you know cross-shuffle is needed in your case. Also there is a performance
  // penalty when this value is greater than 1, as we need to do extra merge
  // between multiple chunks before performing example sampling.
  TORCH_ARG(size_t, cross_chunk_shuffle_count) = 1;
};

/// A stateful dataset that support hierarchical sampling and prefetching of
/// entre chunks.
///
/// Unlike regular dataset, chunk dataset require two samplers to operate and
/// keeps an internal state. `ChunkSampler` selects, which chunk to load next,
/// while the `ExampleSampler` determins the order of Examples that are returned
/// in each `get_batch` call. The hierarchical sampling approach used here is
/// inspired by this paper http://martin.zinkevich.org/publications/nips2010.pdf
template <
    typename ChunkReader,
    typename ChunkSampler = samplers::RandomSampler,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataset final
    : public StatefulDataset<
          ChunkDataset<ChunkReader, ChunkSampler, ExampleSampler>,
          typename ChunkReader::BatchType,
          size_t> {
 public:
  using BatchType = torch::optional<typename ChunkReader::BatchType>;
  using UnwrappedBatchType = typename ChunkReader::BatchType;
  using BatchRequestType = size_t;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  ChunkDataset(
      ChunkReader chunk_reader,
      ChunkSampler chunk_sampler,
      ExampleSampler example_sampler,
      ChunkDatasetOptions options,
      // NOLINTNEXTLINE(modernize-pass-by-value)
      std::function<void(UnwrappedBatchType&)> preprocessing_policy =
          std::function<void(UnwrappedBatchType&)>())
      : chunk_reader_(std::move(chunk_reader)),
        chunk_sampler_(std::move(chunk_sampler)),
        example_sampler_(std::move(example_sampler)),
        options_(std::move(options)),
        preprocessing_policy_(preprocessing_policy),
        quit_worker_(false),
        running_preloaders_(0),
        load_checkpoint_(false) {}

  ~ChunkDataset() override {
    // stop batch buffer first.
    if (batch_buffer_) {
      batch_buffer_->stop();
    }
    free_workers();
  }

  /// Default get_batch method of BatchDataset. This method returns
  /// Example batches created from the preloaded chunks. The implemenation
  /// is dataset agnostic and does not need overriding in different chunk
  /// datasets.
  BatchType get_batch(size_t batch_size) override {
    TORCH_CHECK(
        batch_buffer_ != nullptr,
        "Dataset needs to call reset() before calling get_batch().");

    TORCH_CHECK(
        batch_size == options_.batch_size(),
        "The requested batch size does not match with the initialized batch size.\n"
        " The requested batch size is ",
        batch_size,
        ", while the dataset is created with batch size equal to ",
        options_.batch_size());
    return batch_buffer_->get_batch();
  }

  /// Helper method around get_batch as `batch_size` is not strictly necessary
  BatchType get_batch() {
    return get_batch(options_.batch_size());
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset.
  void reset() override {
    // We need this to support partial data reads via dataloader iterator.
    if (batch_buffer_) {
      batch_buffer_->stop();
    }
    // free workers from previous reset if there is any.
    free_workers();
    preload_threads_.clear();

    if (!load_checkpoint_) {
      chunk_reader_.reset();
      chunk_sampler_.reset(chunk_reader_.chunk_count());
      load_checkpoint_ = false;
    }

    // Throw out any existing cached batch in the buffer and re-creates a new
    // chunk buffer.
    batch_buffer_ = torch::make_unique<
        detail::BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>(
        options_.batch_size(), example_sampler_, options_.cache_size());

    // create new workers for this new epoch.
    quit_worker_ = false;

    AT_ASSERT(running_preloaders_ == 0);
    running_preloaders_ = options_.preloader_count();
    for (const auto i : c10::irange(options_.preloader_count())) {
      preload_threads_.emplace_back([this, i]() { this->preloader(i); });
    }
  }

  /// size is not used for chunk dataset.
  optional<size_t> size() const override {
    return torch::nullopt;
  }

  // provide a references to chunk sampler. Used mainly in distributed data
  // loading to set the epoch number for the sampler.
  ChunkSamplerType& chunk_sampler() {
    return chunk_sampler_;
  }

  void save(serialize::OutputArchive& archive) const override {
    std::lock_guard<std::mutex> lock(chunk_index_guard_);
    chunk_sampler_.save(archive);
  }

  void load(serialize::InputArchive& archive) override {
    std::lock_guard<std::mutex> lock(chunk_index_guard_);
    chunk_sampler_.load(archive);
    load_checkpoint_ = true;
  }

 private:
  /// running on worker thread to preload chunk data.
  void preloader(size_t id) {
    while (!quit_worker_.load()) {
      try {
        std::vector<size_t> chunk_idx;
        {
          std::lock_guard<std::mutex> lock(chunk_index_guard_);
          if (auto chunk_sampler_result = chunk_sampler_.next(
                  this->options_.cross_chunk_shuffle_count())) {
            chunk_idx = chunk_sampler_result.value();
          } else {
            break;
          }
        }
        UnwrappedBatchType data = chunk_reader_.read_chunk(chunk_idx[0]);
        for (const auto i : c10::irange(1, chunk_idx.size())) {
          auto chunk_data = chunk_reader_.read_chunk(chunk_idx[i]);
          std::move(
              chunk_data.begin(), chunk_data.end(), std::back_inserter(data));
        }
        if (preprocessing_policy_) {
          preprocessing_policy_(data);
        }
        if (!data.empty()) { // skip empty chunks.
          batch_buffer_->add_chunk_data(std::move(data));
        }
      } catch (...) {
        batch_buffer_->add_chunk_data(std::current_exception());
      }
    }
    AT_ASSERT(running_preloaders_.load() > 0);
    --running_preloaders_;
    if (running_preloaders_.load() == 0) {
      // all preloaders are completed, so we can notify the batch_buffer.
      batch_buffer_->stop();
    }
  }

  /// Block the current thread until the workers finish execution and exit.
  void free_workers() {
    if (!quit_worker_.load()) {
      quit_worker_ = true;
      for (auto& worker_thread : preload_threads_) {
        worker_thread.join();
      }
    }
  }

 private:
  // Templated class that defines what is a chunk and how to read chunk data.
  // When a chunk is returned by chunk_reader_, ChunkDataset split it into
  // batches and caches them in batch_buffer_.
  ChunkReader chunk_reader_;

  // chunk sampler to shuffle different chunks
  ChunkSamplerType chunk_sampler_;

  // example sampler to shuffle examples in a specific chunk
  ExampleSamplerType example_sampler_;

  // batch data buffer which holds chunk data from preloading thread.
  std::shared_ptr<
      detail::BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>
      batch_buffer_;

  // worker thread pool
  std::vector<std::thread> preload_threads_;

  /// The options the Dataset was configured with.
  const ChunkDatasetOptions options_;

  // function pointer wrapper to apply custom processing over chunk data. This
  // is considered an advanced parameter for developers who want to apply a
  // pre-process to the chunk data before sampling into minibatch.
  // Different than the collate function, this policy is applied on the chunk
  // level, instead of minibatch level. When a chunk of data is loaded (multiple
  // chunks if cross_chunk_shuffle_count_ is greater than 1), this policy is
  // applied to the full loaded data. It is useful if developers want to
  // perform pre-processing (like bucketing) to the chunk data before
  // example sampler samples the data. By default it's an empty pointer and no
  // action will be taken.
  std::function<void(UnwrappedBatchType&)> preprocessing_policy_;

  // indicate whether the worker thread can be teared down
  std::atomic<bool> quit_worker_;

  // keep track of running preloaders to notify batch buffer. A value 0
  // indicates that the chunk loading is completed.
  std::atomic<size_t> running_preloaders_;

  // mutex to synchronize chunk sampler next() call.
  mutable std::mutex chunk_index_guard_;

  // boolean value to indicate whether we need to load the checkpoint for
  // chunk_sampler_.
  bool load_checkpoint_;
};
} // namespace datasets
} // namespace data
} // namespace torch
