#pragma once

#include <torch/data/datasets/stateful.h>

namespace torch {
namespace data {
namespace datasets {

/// Interface for chunk reader, which performs data chunking and reading of
/// entire chunks.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text-file split based on seek
/// positions.
template <typename Chunk = std::vector<Example<>>>
class ChunkDataReader {
 public:
  using ChunkType = Chunk;

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
    typename UnwrappedBatch = std::vector<Example<>>,
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
        queue_capacity_(queue_capacity),
        stop_(false) {}

  /// Return batch data from the queue. Called from the ChunkDataset main
  /// thread.
  BatchType get_batch() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_read_.wait(lock, [this] {
      // wait till there is available data in the queue or if all chunks are
      // loaded (i.e. the dataset is exhausted for this epoch)
      return (
          this->total_example_count_in_queue_ >= batch_size_ ||
          this->stop_.load());
    });
    if (batch_queue_.empty()) {
      AT_ASSERT(this->stop_.load());
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
          stop_.load();
    });

    if (stop_.load()){
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
          batch_example_indices.value().size() == example_count)
      BatchRequestType& indices = batch_example_indices.value();
      for (size_t i : indices) {
        AT_CHECK(i < data_size, "Index out of range");
        batch.emplace_back(std::move(data[i]));
      }
      remaining_size -= example_count;
    };

    if (!batch_queue_.empty()) {
      // if the queue has existing data, and the last batch doesn't have enough
      // examples to fill a batch_size batch, add more example to this batch first.
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
      return this->total_example_count_in_queue_ < this->queue_capacity_ || stop_.load();
    });

    if (stop_.load()){
      // When stop_ is true, it means this current thread needs to be tore down,
      // the batch buffer will be discarded, so no need to enqueue any new
      // exceptions.
      return;
    }

    batch_queue_.emplace(e_ptr);
    lock.unlock();
    cv_read_.notify_all();
  }

  void stop(){
    stop_ = true;

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

  /// struct that contains a raw unwrapped batch unit. An unwrapped batch unit is
  /// the raw data without 'optional' wrapper. It can be a collection of images,
  /// utterances, e.t.c.
  struct UnwrappedBatchData {
    explicit UnwrappedBatchData(UnwrappedBatchType data) : batch_data(std::move(data)) {}

    explicit UnwrappedBatchData(std::exception_ptr e) : exception(e) {}

    /// batch data to return
    UnwrappedBatchType batch_data;

    /// exception pointer which captures any abnormal exceptions while creating the
    /// batch.
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

  // When set to true, it wakes the writer threads from the wait and exit current
  // function call. This is needed when ChunkDataSet.Reset is called while the
  // previous epoch is not exhausted yet. When ChunkDataset is waiting its
  // preloader to finish previous work before tearing down the thread, the
  // preloader could be still waiting for the conditional variable, thus cause
  // the program to hang. This boolean is used to break this waiting condition.
  std::atomic<bool> stop_;
};
} // namespace detail

/// Options to configure a `ChunkDataset`.
struct ChunkDatasetOptions {
  ChunkDatasetOptions() = delete;
  ChunkDatasetOptions(
      size_t preloader_count,
      size_t batch_size,
      size_t cache_size = 2048)
      : preloader_count_(preloader_count),
        batch_size_(batch_size),
        cache_size_(cache_size) {
    AT_CHECK(
        preloader_count_ > 0,
        "Preloader count is 0. At least one preloader needs to be specified.");
    AT_CHECK(
        batch_size_ > 0,
        "Batch size is 0. A positive batch size needs to be specified.");
    AT_CHECK(
        cache_size_ > 0,
        "Cache size is 0. A positive cache size needs to be specified.");
    AT_CHECK(
        cache_size_ >= batch_size_,
        "Cache size is less than batch size. Cache needs to be large enough to "
        "hold at least one batch.");
  }

  /// The number of worker thread to preload chunk data.
  TORCH_ARG(size_t, preloader_count);

  /// The size of each batch.
  TORCH_ARG(size_t, batch_size);

  // the capacity of the queue for batch caching.
  TORCH_ARG(size_t, cache_size) = 2048;
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
      ChunkDatasetOptions options)
      : chunk_reader_(std::move(chunk_reader)),
        chunk_sampler_(std::move(chunk_sampler)),
        example_sampler_(std::move(example_sampler)),
        options_(std::move(options)),
        quit_worker_(false),
        running_preloaders_(0) {}

  virtual ~ChunkDataset() {
    free_workers();
  }

  /// Default get_batch method of BatchDataset. This method returns
  /// Example batches created from the preloaded chunks. The implemenation
  /// is dataset agnostic and does not need overriding in different chunk
  /// datasets.
  BatchType get_batch(size_t batch_size) override {
    AT_CHECK(
      batch_buffer_ != nullptr,
      "Dataset needs to call reset() before calling get_batch().");

    AT_CHECK(
      batch_size == options_.batch_size_,
      "The requested batch size does not match with the initialized batch size.\n"
      " The requested batch size is ", batch_size,
      ", while the dataset is created with batch size equal to ", options_.batch_size_);

    return batch_buffer_->get_batch();
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset.
  void reset() override {
    // free workers from previous reset if there is any.
    free_workers();
    preload_threads_.clear();

    chunk_reader_.reset();

    chunk_sampler_.reset(chunk_reader_.chunk_count());

    // Throw out any existing cached batch in the buffer and re-creates a new
    // chunk buffer.
    batch_buffer_ = torch::make_unique<
        detail::BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>(
        options_.batch_size_,
        example_sampler_,
        options_.cache_size_);

    // create new workers for this new epoch.
    quit_worker_ = false;

    AT_ASSERT(running_preloaders_ == 0);
    for (size_t i = 0; i < options_.preloader_count_; ++i) {
      preload_threads_.emplace_back([this, i]() { this->preloader(i); });
      ++running_preloaders_;
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

 private:
  /// running on worker thread to preload chunk data.
  void preloader(size_t id) {
    while (!quit_worker_.load()) {
      try {
        size_t chunk_id = 0;
        {
          std::lock_guard<std::mutex> lock(chunk_index_guard_);
          if (auto chunk_sampler_result = chunk_sampler_.next(1)) {
            chunk_id = chunk_sampler_result.value()[0];
          } else {
            break;
          }
        }
        UnwrappedBatchType data = chunk_reader_.read_chunk(chunk_id);
        if (!data.empty()) { // skip empty chunks.
          batch_buffer_->add_chunk_data(std::move(data));
        }
      } catch (...) {
        batch_buffer_->add_chunk_data(std::current_exception());
      }
    }
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
  std::shared_ptr<detail::BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>
      batch_buffer_;

  // worker thread pool
  std::vector<std::thread> preload_threads_;

  /// The options the Dataset was configured with.
  const ChunkDatasetOptions options_;

  // indicate whether the worker thread can be teared down
  std::atomic<bool> quit_worker_;

  // keep track of running preloaders to notify batch buffer. A value 0
  // indicates that the chunk loading is completed.
  std::atomic<size_t> running_preloaders_;

  // mutex to synchronize chunk sampler next() call.
  std::mutex chunk_index_guard_;
};
} // namespace datasets
} // namespace data
} // namespace torch
