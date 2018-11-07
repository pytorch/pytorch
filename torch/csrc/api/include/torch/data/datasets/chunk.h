#pragma once

#include <c10/util/Exception.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>

namespace torch {
namespace data {
namespace datasets {
/// A dataset that supports loading an entire chunk of data.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text file split based on seek
/// positions. ChunkDataSet extends the DataSet functionality to read an
/// entire chunk at once.
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = ArrayRef<size_t>>
class ChunkDataSet : public BatchDataset<Self, Batch, BatchRequest> {
 public:
  /// Read an entire chunk. A derived class needs to override this method.
  virtual Batch read_chunk(size_t chunk_index) = 0;

  /// Default get_batch method of BatchDataSet. This method will handle the
  /// chunk loading and creating of Example batches. The implemenation is
  /// dataset agnostic and does not need overriding in different chunk data
  /// sets.
  Batch get_batch(BatchRequest indices) override {
    AT_ASSERT(indices.size() == 1);
    return Batch();
  }
};
} // namespace datasets
} // namespace data
} // namespace torch