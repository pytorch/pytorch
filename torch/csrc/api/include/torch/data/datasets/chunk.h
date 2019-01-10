#pragma once

#include <torch/csrc/utils/memory.h>
#include <torch/data/datasets/stateful.h>
#include <torch/data/example.h>
#include <torch/data/samplers.h>

namespace torch {
namespace data {
namespace datasets {

/// Interface for chunk reader, which performs data chunking and reading of
/// entire chunks.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text-file split based on seek
/// positions.
template <typename Self, typename Batch = std::vector<Example<>>>
class ChunkDataReader {
 public:
  using SelfType = Self;
  using BatchType = Batch;

  /// Read an entire chunk.
  virtual BatchType read_chunk(size_t chunk_index) = 0;

  /// Returns the number of chunks available in this reader.
  virtual size_t get_chunk_count() = 0;

  /// This will clear any internal state associate with this reader.
  virtual void reset() = 0;
};

/// A class that contains a raw unwrapped batch unit. An unwrapped batch unit is
/// the raw data without 'optional' wrapper. It can be a collection of images,
/// utterances, e.t.c.
template <typename UnwrappedBatch = std::vector<Example<>>>
struct UnwrappedBatchData {
 public:
  using UnwrappedBatchType = UnwrappedBatch;

  UnwrappedBatchData(UnwrappedBatchType data) : batch_data(std::move(data)) {}

  UnwrappedBatchData(std::exception_ptr e) : exception(e) {}

  /// batch data to return
  UnwrappedBatchType batch_data;

  /// exception pointer which captures any abnormal exceptions while creating the
  /// batch.
  std::exception_ptr exception;
};

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
      size_t num_chunks,
      size_t batch_size,
      ExampleSampler example_sampler,
      bool ignore_empty_chunk,
      size_t cache_size)
      : remaining_chunk_count_(num_chunks),
        batch_size_(batch_size),
        example_sampler_(std::move(example_sampler)),
        ignore_empty_chunk_(ignore_empty_chunk),
        queue_depth_(cache_size) {}

  /// Return batch data from the queue. Called from the ChunkDataSet main
  /// thread.
  BatchType get_batch(size_t batch_size) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cvr_.wait(lock, [this] {
      // wait till there is available data in the queue or if all chunks are
      // loaded (i.e. the data set is exhausted for this epoch)
      return (
          this->total_example_count_in_queue_ >= batch_size_ ||
          remaining_chunk_count_ == 0);
    });
    if (batch_queue_.empty()) {
      lock.unlock();
      AT_ASSERT(remaining_chunk_count_ == 0);

      // All batches have been retrieved. Return an empty batch.
      return nullopt;
    }

    auto batch_data = batch_queue_.front();
    batch_queue_.pop();
    if (batch_data.exception) {
      throw WorkerException(batch_data.exception);
    }

    this->total_example_count_in_queue_ -= batch_data.batch_data.size();
    lock.unlock();
    cvw_.notify_all(); // notify all writers.

    return batch_data.batch_data;
  }

  // skip one chunk
  void skip_chunk() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    AT_ASSERT(remaining_chunk_count_ > 0);
    remaining_chunk_count_--;
    lock.unlock();
    cvr_.notify_all();
  }

  /// Push preloaded chunks to batch queue. Called from the ChunkDataSet worker
  /// threads.
  void add_chunk_data(size_t index, UnwrappedBatchType data) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cvw_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < queue_depth_;
    });

    auto data_size = data.size();
    auto remaining_size = data_size;
    example_sampler_.reset(data_size);

    if (!batch_queue_.empty()) {
      // if the queue has exiting data, and the last batch doesn't have enough
      // examples to fill a batch_size batch, add more example to this batch first.
      auto& batch_data = batch_queue_.back();
      size_t current_count = batch_data.batch_data.size();
      if (current_count < batch_size_) {
        auto example_count =
            std::min(remaining_size, batch_size_ - current_count);
        auto batch_example_indices = example_sampler_.next(example_count);
        AT_ASSERT(
            batch_example_indices &&
            batch_example_indices.value().size() == example_count)
        BatchRequestType indices = batch_example_indices.value();
        for (size_t i : indices) {
          batch_data.batch_data.emplace_back(std::move(data[i]));
        }
        remaining_size -= example_count;
      }
    }

    // If we still have data remaining after filling the last pushed batch, add
    // them to the queue too.
    while (remaining_size > 0) {
      UnwrappedBatchType current_batch;

      // Allocate the batch memory ahead of time.
      current_batch.reserve(batch_size_);

      auto example_count = std::min(remaining_size, batch_size_);
      auto batch_example_indices = example_sampler_.next(example_count);
      AT_ASSERT(
          batch_example_indices &&
          batch_example_indices.value().size() == example_count)
      BatchRequestType indices = batch_example_indices.value();
      for (size_t i : indices) {
        current_batch.emplace_back(std::move(data[i]));
      }
      remaining_size -= example_count;
      UnwrappedBatchData<UnwrappedBatchType> batch_data(
          std::move(current_batch));
      batch_queue_.push(std::move(batch_data));
    }
    this->total_example_count_in_queue_ += data_size;

    AT_ASSERT(remaining_chunk_count_ > 0);
    this->remaining_chunk_count_--;

    lock.unlock();
    cvr_.notify_all();
  }

  /// Push exceptions throwed during preloading into batch queue. Called from
  /// the ChunkDataSet worker threads.
  void add_chunk_data(size_t index, std::exception_ptr e_ptr) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cvw_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < queue_depth_;
    });
    UnwrappedBatchData<UnwrappedBatchType> batch_data(e_ptr);
    batch_queue_.push(std::move(batch_data));

    AT_ASSERT(remaining_chunk_count_ > 0);
    remaining_chunk_count_--;
    lock.unlock();
    cvr_.notify_all(); // notify all readers.
  }

  /// count of remaining chunk to be loaded. It is initialized with the total
  /// chunk count and it decreases when a chunk data is retrieved. When this reaches
  /// to 0, no more chunk needs to be loaded.
  size_t remaining_chunk_count_ = 0;

  /// The batch size is needed to create batches from the chunk data. Similar to
  /// regular dataloader where the batches are created with prefetches,
  /// BatchDataBuffer perform the batch creation using the provided batch size.
  size_t batch_size_ = 0;

  /// count of total example stored in the queue
  size_t total_example_count_in_queue_ = 0;

  /// local cache to store example batches from loaded chunk
  std::queue<UnwrappedBatchData<UnwrappedBatchType>> batch_queue_;

  // sync batch_queue_ update.
  std::mutex queue_mutex_;

  std::condition_variable cvr_;
  std::condition_variable cvw_;

  ExampleSampler example_sampler_;

  // indicator for whether an empty chunk should be ignored. When it is true, an
  // example will throw, otherwise, this empty chunk is skipped.
  bool ignore_empty_chunk_ = false;

  // configurable queue depth for batch_queue_
  size_t queue_depth_;
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
class ChunkDataSet final
    : public StatefulDataset<
          ChunkDataSet<ChunkReader, ChunkSampler, ExampleSampler>,
          typename ChunkReader::BatchType,
          size_t> {
 public:
  using BatchType = torch::optional<typename ChunkReader::BatchType>;
  using UnwrappedBatchType = typename ChunkReader::BatchType;
  using BatchRequestType = size_t;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  ChunkDataSet(
      ChunkReader chunk_reader,
      ChunkSampler chunk_sampler,
      ExampleSampler example_sampler,
      size_t preloader_count,
      size_t batch_size,
      bool ignore_empty_chunk = false,
      size_t cache_size = 500)
      : chunk_reader_(std::move(chunk_reader)),
        chunk_sampler_(std::move(chunk_sampler)),
        example_sampler_(std::move(example_sampler)),
        preloader_count_(preloader_count),
        batch_size_(batch_size),
        ignore_empty_chunk_(ignore_empty_chunk),
        cache_size_(cache_size) {
    if (preloader_count_ == 0) {
      throw std::runtime_error(
          "Preloader count is 0. At least one preloader needs to be specified.");
    }

    if (batch_size == 0) {
      throw std::runtime_error(
          "Batch size is 0. A positive batch size needs to be specified.");
    }

    if (cache_size_ == 0) {
      throw std::runtime_error(
          "Cache size is 0. A positive cache size needs to be specified.");
    }

    if (cache_size < batch_size){
      throw std::runtime_error(
          "Cache size is less than batch size. Cache needs to be large enough to hold at least one batch.");
    }
    // chunk_sampler_ =
    //     std::make_unique<samplers::ThreadSafeSampler<ChunkSamplerType>>(
    //         std::move(chunk_sampler));
  }

  virtual ~ChunkDataSet() {
    free_workers();
  }

  /// Default get_batch method of BatchDataSet. This method returns
  /// Example batches created from the preloaded chunks. The implemenation
  /// is dataset agnostic and does not need overriding in different chunk
  /// data sets.
  BatchType get_batch(size_t batch_size) override {
    if (chunk_buffer_ == nullptr) {
      throw std::runtime_error(
          "Dataset needs to call reset() before calling get_batch().");
    }
    if (batch_size != batch_size_) {
      std::string error =
          "The requested batch size does not match with the initialized batch size.\n"
          " The requested batch size is " + std::to_string(batch_size) +
          ", while the data set is created with batch size equal to " +
          std::to_string(batch_size_);

      throw std::runtime_error(error);
    }
    return chunk_buffer_->get_batch(batch_size);
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset.
  virtual void reset() {
    // free workers from previous reset if there is any.
    free_workers();
    preload_threads_.clear();

    chunk_reader_.reset();

    size_t chunks_to_load = chunk_reader_.get_chunk_count();
    chunk_sampler_.reset(chunks_to_load);

    // Creates a new chunk buffer each time we reset the dataset.
    chunk_buffer_ = torch::make_unique<
        BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>(
        chunks_to_load,
        batch_size_,
        example_sampler_,
        ignore_empty_chunk_,
        cache_size_);

    // create new workers for this new epoch.
    quit_worker_ = false;

    for (size_t i = 0; i < preloader_count_; ++i) {
      preload_threads_.emplace_back(
          [this, i]() { this->preloader(i); });
    }
  }

  /// size is not used for chunk dataset.
  optional<size_t> size() const override {
    return torch::nullopt;
  }

 private:
  /// running on worker thread to preload chunk data.
  void preloader(size_t id) {
    while (!quit_worker_) {
      size_t chunk_id;
      try {
        auto chunk_sampler_result = chunk_sampler_.next(1);
        if (chunk_sampler_result.has_value()) {
          chunk_id = chunk_sampler_result.value()[0];
        } else {
          break;
        }
        UnwrappedBatchType data = chunk_reader_.read_chunk(chunk_id);
        if (data.empty()) {
          if (!ignore_empty_chunk_) {
            std::string error =
                "Chunk with index " + std::to_string(chunk_id) + " is empty";
            throw std::runtime_error(error);
          } else {
            // skip adding the current chunk data and move to the next.
            chunk_buffer_->skip_chunk();
          }
        }
        else {
          chunk_buffer_->add_chunk_data(chunk_id, std::move(data));
        }
      } catch (...) {
        chunk_buffer_->add_chunk_data(chunk_id, std::current_exception());
      }
    }
  }

  /// Block the current thread until the workers finish execution and exit.
  void free_workers() {
    if (!quit_worker_) {
      quit_worker_ = true;
      for (auto& worker_thread : preload_threads_) {
        worker_thread.join();
      }
    }
  }

 private:
  // chunk reader is responsible for reading chunk data
  ChunkReader chunk_reader_;

  // chunk sampler to shuffle different chunks
  samplers::ThreadSafeSampler<ChunkSamplerType> chunk_sampler_;

  // example sampler to shuffle examples in a specific chunk
  ExampleSamplerType example_sampler_;

  // chunk data buffer which holds chunk data from preloading thread.
  std::shared_ptr<BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>
      chunk_buffer_;

  // worker thread pool
  std::vector<std::thread> preload_threads_;

  // worker thread count
  size_t preloader_count_ = 0;

  size_t batch_size_ = 0;

  // if it is set to true, the dataset will quietly move to the next chunk when
  // the current one is empty. Otherwise, an exception is thrown on the empty
  // batch.
  bool ignore_empty_chunk_ = false;

  bool quit_worker_;

  size_t cache_size_;
};
} // namespace datasets
} // namespace data
} // namespace torch
