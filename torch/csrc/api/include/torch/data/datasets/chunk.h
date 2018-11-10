#pragma once

#include <c10/util/Exception.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/data/samplers/base.h>
#include <torch/data/samplers/sequential.h>

#include <algorithm>
#include <mutex>
#include <optional>
#include <thread>

/// A dataset that supports loading an entire chunk of data.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text file split based on seek
/// positions. ChunkDataSet extends the DataSet functionality to read an
/// antire chunk at once.

namespace torch {
namespace data {
namespace datasets {

/// A dataset that supports loading an entire chunk of data.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text file split based on seek
/// positions. ChunkDataSet extends the DataSet functionality to read an
/// antire chunk at once.
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename ChunkSampler = samplers::SequentialSampler,
    typename ExampleSampler = samplers::SequentialSampler>
class ChunkDataSet : public BatchDataset<Self, Batch> {
 public:
  /// Read an entire chunk. A derived class needs to override this method.
  virtual Batch read_chunk(size_t chunk_index) = 0;

  /// Return the chunk count. A derived class needs to override this method.
  virtual size_t get_chunk_count() = 0;

  void initialize() {
    /// After the data set is created, we need to call this method before
    /// getting batch data. This method will use some infomation that is not yet
    /// available in the constructor to configure chunk sampler and start
    /// working thread.
    chunk_count_ = get_chunk_count();
    chunk_sampler_.set_size(chunk_count_);

    for (size_t i = 0; i < prefetch_count_; ++i) {
      workers_.emplace_back([&]() { this->chunk_loading_worker(); });
    }
  }

  ChunkDataSet(
      size_t prefetch_count,
      ChunkSampler chunkSampler = samplers::SequentialSampler(0),
      ExampleSampler exampleSampler = samplers::SequentialSampler(0))
      : prefetch_count_(prefetch_count),
        chunk_sampler_(chunkSampler),
        example_sampler_(exampleSampler) {
    AT_ASSERT(prefetch_count >= 1);

    chunkData_.resize(prefetch_count_);
    example_indices_.resize(prefetch_count_);
    chunk_remaining_data_.resize(prefetch_count_);
    std::fill(chunk_remaining_data_.begin(), chunk_remaining_data_.end(), 0);
  }

  ~ChunkDataSet() {
    quit_loading_ = true;
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  /// Default get_batch method of BatchDataSet. This method will handle the
  /// chunk loading and creating of Example batches. The implemenation is
  /// dataset agnostic and does not need overriding in different chunk data
  /// sets.
  Batch get_batch(ArrayRef<size_t> indices) override {
    AT_ASSERT(indices.size() == 1);
    size_t batch_size = indices[0];
    Batch result;
    result.resize(batch_size);

    copy_remaining_data_if_needed(batch_size, result);

    /// To achieve an better performance, the get_batch process performs two
    /// steps:
    ///   1. Take a lock and get the randomized example indices for this batch.
    ///   The lock is needed
    ///      as multiple thread could call get_batch at the same time.
    ///   2. Release the lock, use the indices to retrieve the example data to
    ///   return.

    // Get example indices.
    std::vector<RandExampleIndexRange> example_indices =
        get_example_indices(batch_size);

    size_t count = 0;
    for (auto& index_info : example_indices) {
      for (int i = index_info.startRandIndex; i < index_info.endRandIndex;
           ++i) {
        auto example_id = example_indices_[index_info.chunkIndex][i];
        result[count++] = chunkData_[index_info.chunkIndex][example_id];
      }
    }

    {
      // Take the lock and update each chunk's remaining element. When the count
      // is reaching 0, the working thread will load a new chunk and re-calcuate
      // the remaining element count.
      std::unique_lock<std::mutex> lock(mutex_);
      for (auto& index_info : example_indices) {
        AT_ASSERT(
            chunk_remaining_data_[index_info.chunkIndex] >=
            (index_info.endRandIndex - index_info.startRandIndex));
        chunk_remaining_data_[index_info.chunkIndex] -=
            (index_info.endRandIndex - index_info.startRandIndex);
      }
    }

    return result;
  }

 private:
  void copy_remaining_data_if_needed(size_t batch_size, Batch& result) {
    // TODO: Now we are assuming there are enough available elements remaining
    // for this mini batch. It is possible that, all current available chunks
    // don't contain enough elements, and we need to stop and wait for new
    // batches to be loaded in the middle of the get_batch.
    // For example, if the mini batch size is 10, but we only have 5 elements
    // in total from all chunks, then after we get the 5 elements, we need to
    // wait for new chunks to be loaded before return.
    // To do that, we need to:
    //    1. take a lock
    //    2. copy the data (not just the indices)
    //    3. calculate remaining element count to return
    //    4. release a lock and take a cv waiting for new chunk to be loaded
    //    5. check again for all available elements. If it is still less than
    //    we need, repeat 1; otherwise return the indices.
  }

  optional<int> get_chunkId_to_load() {
    std::unique_lock<std::mutex> lock(mutex_);

    for (int i = 0; i < chunk_remaining_data_.size(); ++i) {
      if (chunk_remaining_data_[i] == 0) {
        auto it = std::find(
            in_flight_loading_chunk_.begin(), in_flight_loading_chunk_.end(), i);
        if (it == std::end(in_flight_loading_chunk_)) {
          in_flight_loading_chunk_.push_back(i);

          // We cannot set chunk_remaining_data_[i] to true yet as this chunk
          // hasn't been loaded yet.
          return i;
        }
      }
    }
    return {};
  }

  void chunk_loading_worker() {
    while (!quit_loading_) {
      if (!all_chunk_loaded_) {
        // Check if the ith chunk needs to be re-load.
        // If it is the case, lock the corresponding chunk remaining element
        // count, load the chunk, get the randomized index and swap it to
        // chunkData_[i]
        optional<int> chunkId = get_chunkId_to_load();

        if (chunkId) {
          auto next_chunk = chunk_sampler_.next(1);
          if (!next_chunk) {
            // We finished loading all chunks.
            // Waiting to reset for the next epoch.
            all_chunk_loaded_ = true;
            continue;
          }

          auto new_batch = read_chunk(next_chunk.value()[0]);

          example_sampler_.set_size(new_batch.size());
          auto new_element_indices = example_sampler_.next(new_batch.size());

          AT_ASSERT(new_element_indices); // this should not be nullopt.

          auto chunkId_value = chunkId.value();

          // Now update the data.
          {
            std::unique_lock<std::mutex> lock(mutex_);

            // it should be safe to override the data as currently no get_batch
            // should be using data from this memory.
            AT_ASSERT(chunk_remaining_data_[chunkId_value] == 0);

            chunk_remaining_data_[chunkId_value] = new_batch.size();

            chunkData_[chunkId_value] = std::move(new_batch);
            example_indices_[chunkId_value] =
                std::move(new_element_indices.value());

            auto it = std::find(
                in_flight_loading_chunk_.begin(),
                in_flight_loading_chunk_.end(),
                chunkId_value);
            AT_ASSERT(it != std::end(in_flight_loading_chunk_));
            in_flight_loading_chunk_.erase(it);
          }
          cv_.notify_all();
        }
      }
    }
  }

  struct RandExampleIndexRange {
    size_t chunkIndex;
    size_t startRandIndex;
    size_t endRandIndex;
  };

  std::vector<RandExampleIndexRange> get_example_indices(size_t batch_size) {
    std::vector<RandExampleIndexRange> batch_example_indices;
    size_t count = 0;

    while (count < batch_size) {
      std::unique_lock<std::mutex> lock(mutex_);

      // wait until the current chunk we are reading is available.
      cv_.wait(lock, [&] {
        return this->chunk_remaining_data_[current_reading_chunk] > 0;
      });

      size_t totalExampleInChunk =
          example_indices_[current_reading_chunk].size();

      if ((count < batch_size) && (example_position_ < totalExampleInChunk)) {
        size_t copiableItemCount = std::min(
            batch_size - count, totalExampleInChunk - example_position_);
        batch_example_indices.push_back(
            {current_reading_chunk,
             example_position_,
             example_position_ + copiableItemCount});

        example_position_ += copiableItemCount;
        count += copiableItemCount;
      }

      if (count < batch_size) {
        // move to the next chunk to read.
        current_reading_chunk = (current_reading_chunk + 1) % prefetch_count_;
        example_position_ = 0;
      } else {
        break;
      }
    }
    return batch_example_indices;
  }

 private:
  size_t prefetch_count_;
  ChunkSampler chunk_sampler_;
  ExampleSampler example_sampler_;

  // Store prefetched chunk data, each chunk is a batch in the vector.
  std::vector<Batch> chunkData_;

  // Store the randomized example index. It is one to one mapped to
  // data in chunkData_.
  std::vector<std::vector<size_t>> example_indices_;

  // Store the remaining data count in each chunk. When it reaches to
  // zero, it means the coreponding chunk is exhausted and a new chunk
  // can be loaded to this position.
  std::vector<size_t> chunk_remaining_data_;

  // Store chunk id that has started loading but the loading is not yet completed.
  std::vector<size_t> in_flight_loading_chunk_;

  // total chunk size.
  size_t chunk_count_;

  // pointer to the current reading chunk's position
  size_t current_reading_chunk = 0;
  // pointer to the current reading example's position
  size_t example_position_ = 0;

  bool all_chunk_loaded_ = false;
  bool quit_loading_ = false;

  // worker threads to do prefetching job
  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable cv_;
};

} // namespace datasets
} // namespace data
} // namespace torch
