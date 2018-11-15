#pragma once

#include <c10/util/Exception.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/data/samplers.h>

namespace torch {
namespace data {
namespace datasets {
/// A stateful dataset that support hierarchical sampling and prefetching of
/// entre chunks. dataset that supports loading an entire chunk of data.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text file split based on seek
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
    typename BatchRequest = ArrayRef<size_t>,
    typename ChunkSampler = samplers::RandomSampler,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataSet : public BatchDataset<Self, Batch, BatchRequest> {
 public:
  /// Read an entire chunk. A derived class needs to override this method.
  /// This is the only API, other than the constructor that 
  virtual Batch read_chunk(size_t chunk_index) = 0;

  /// Default get_batch method of BatchDataSet. This method will handle the
  /// chunk loading and creating of Example batches. The implemenation is
  /// dataset agnostic and does not need overriding in different chunk data
  /// sets.
  Batch get_batch(BatchRequest indices) override {
    AT_ASSERT(indices.size() == 1);
    return Batch();
  }

  /// This will clear any internal state and starts= the internal prefetching
  /// mechanism for the chunk dataset.
  void reset() override {}
};
} // namespace datasets
} // namespace data
} // namespace torch