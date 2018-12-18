#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/datasets/map.h>
#include <torch/data/example.h>

#include <torch/types.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace data {
namespace datasets {
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = ArrayRef<size_t>>
class StatefulBatchDataset
    : public BatchDataset<Self, optional<Batch>, BatchRequest> {
 public:
  virtual void reset() = 0;
};

template <typename Self, typename SingleExample = Example<>>
class StatefulDataset
    : public StatefulBatchDataset<Self, std::vector<SingleExample>> {
 public:
  using ExampleType = optional<SingleExample>;

  virtual optional<SingleExample> get(size_t index) = 0;

  optional<std::vector<SingleExample>> get_batch(
      ArrayRef<size_t> indices) override {
    optional<std::vector<SingleExample>> batch;
    for (const auto i : indices) {
      if (ExampleType example = get(i)) {
        if (!batch) {
          batch = std::vector<SingleExample>();
          batch->reserve(indices.size());
        }
        batch->push_back(*example);
      } else {
        break;
      }
    }
    return batch;
  }
};
} // namespace datasets
} // namespace data
} // namespace torch
