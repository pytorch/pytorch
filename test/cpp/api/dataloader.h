#pragma once

#include <torch/data.h>

using namespace torch::data; // NOLINT

/* DummyChunkDataReader implementation with 3 chunks and 35 examples in total
 * Each chunk contains 10, 5, 20 examples respectively
 * For this class, BatchType is std::vector<int>, but custom types are supported
 */
class DummyChunkDataReader : public datasets::ChunkDataReader<int> {
 public:
  using BatchType = datasets::ChunkDataReader<int>::ChunkType;
  using DataType = datasets::ChunkDataReader<int>::ExampleType;

  /// Read an entire chunk.
  BatchType read_chunk(size_t chunk_index) override {
    BatchType batch_data;
    int start_index = chunk_index == 0
        ? 0
        : std::accumulate(chunk_sizes, chunk_sizes + chunk_index, 0);

    batch_data.resize(chunk_sizes[chunk_index]);

    std::iota(batch_data.begin(), batch_data.end(), start_index);
    return batch_data;
  }

  size_t chunk_count() override {
    return chunk_count_;
  };

  void reset() override{};

  const static size_t chunk_count_ = 3;
  size_t chunk_sizes[chunk_count_] = {10, 5, 20};
};
