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
  virtual Batch read_chunk(size_t chunk_index) = 0;

  /// Returns the number of chunks available in this reader.
  virtual size_t get_chunk_count() = 0;

  /// This will clear any internal state associate with this reader.
  virtual void reset() = 0;
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
  using BatchRequestType = size_t;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  ChunkDataSet(
      ChunkReader chunk_reader,
      ChunkSampler chunk_sampler,
      ExampleSampler example_sampler,
      size_t batch_size)
      : chunk_reader_(std::move(chunk_reader)),
        chunk_sampler_(std::move(chunk_sampler)),
        example_sampler_(std::move(example_sampler)),
        batch_size_(batch_size) {}

  /// Default get_batch method of BatchDataSet. This method returns
  /// Example batches created from the preloaded chunks. The implemenation
  /// is dataset agnostic and does not need overriding in different chunk
  /// data sets.
  BatchType get_batch(size_t batch_size) override {
    // Temporary: for API only testing.
    int index = chunk_index_.fetch_add(1);
    if (index < chunk_reader_.get_chunk_count()) {
      return chunk_reader_.read_chunk(index);
    }
    return torch::nullopt;
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset.
  virtual void reset() {
    chunk_reader_.reset();
  }

  /// size is not used for chunk dataset.
  optional<size_t> size() const override {
    return torch::nullopt;
  }

 private:
  ChunkReader chunk_reader_;
  ChunkSampler chunk_sampler_;
  ExampleSampler example_sampler_;
  size_t batch_size_;
  // Temporary: for API only testing.
  std::atomic<int> chunk_index_{0};
};
} // namespace datasets
} // namespace data
} // namespace torch
