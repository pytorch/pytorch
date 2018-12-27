#pragma once

#include <torch/csrc/utils/memory.h>
#include <torch/data/datasets/stateful.h>
#include <torch/data/example.h>
#include <torch/data/samplers.h>

namespace torch {
namespace data {
namespace datasets {

/// A stateful dataset that support hierarchical sampling and prefetching of
/// entre chunks.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text-file split based on seek
/// positions.
///
/// Unlike regular dataset, chunk dataset require two samplers to operate and
/// keeps an internal state. `ChunkSampler` selects, which chunk to load next,
/// while the `ExampleSampler` determins the order of Examples that are returned
/// in each `get_batch` call. The hierarchical sampling approach used here is
/// inspired by this paper http://martin.zinkevich.org/publications/nips2010.pdf
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename ChunkSampler = samplers::RandomSampler,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataSet : public StatefulDataset<Self, Batch, size_t> {
 public:
  using SelfType = Self;
  using BatchType = Batch;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  /// Read an entire chunk. A derived class needs to override this method.
  virtual Batch read_chunk(size_t chunk_index) = 0;

  /// Returns the chunk sampler for this dataset.
  virtual ChunkSampler get_chunk_sampler() = 0;

  /// Returns the example sampler for this dataset.
  virtual ExampleSampler get_example_sampler() = 0;

  /// returns the number of chunks available in this dataset.
  virtual size_t get_chunk_count() = 0;

  /// Default get_batch method of BatchDataSet. This method returns Example
  /// batches created from the preloaded chunks. The implemenation is dataset
  /// agnostic and does not need overriding in different chunk data sets.
  optional<Batch> get_batch(size_t batch_size) override {
    // Temporary: tests will have a simple implemenation.
    return torch::nullopt;
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset.
  virtual void reset() {}

  /// size is not used for chunk dataset.
  optional<size_t> size() const override {
    return torch::nullopt;
  }
};
} // namespace datasets
} // namespace data
} // namespace torch
