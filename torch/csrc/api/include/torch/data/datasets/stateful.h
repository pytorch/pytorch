#pragma once

#include <torch/data/datasets/base.h>
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
    typename BatchRequest = size_t>
struct StatefulDataset
    : public BatchDataset<Self, optional<Batch>, BatchRequest> {
  virtual void reset() = 0;
};
} // namespace datasets
} // namespace data
} // namespace torch
