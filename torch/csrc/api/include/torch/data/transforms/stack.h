#pragma once

#include <torch/data/example.h>
#include <torch/data/transforms/collate.h>
#include <torch/types.h>

#include <utility>
#include <vector>

namespace torch::data::transforms {

template <typename T = Example<>>
struct Stack;

/// A `Collation` for `Example<Tensor, Tensor>` types that stacks all data
/// tensors into one tensor, and all target (label) tensors into one tensor.
template <>
struct Stack<Example<>> : public Collation<Example<>> {
  Example<> apply_batch(std::vector<Example<>> examples) override {
    std::vector<torch::Tensor> data, targets;
    data.reserve(examples.size());
    targets.reserve(examples.size());
    for (auto& example : examples) {
      data.push_back(std::move(example.data));
      targets.push_back(std::move(example.target));
    }
    return {torch::stack(data), torch::stack(targets)};
  }
};

/// A `Collation` for `Example<Tensor, NoTarget>` types that stacks all data
/// tensors into one tensor.
template <>
struct Stack<TensorExample>
    : public Collation<Example<Tensor, example::NoTarget>> {
  TensorExample apply_batch(std::vector<TensorExample> examples) override {
    std::vector<torch::Tensor> data;
    data.reserve(examples.size());
    for (auto& example : examples) {
      data.push_back(std::move(example.data));
    }
    return torch::stack(data);
  }
};
} // namespace torch::data::transforms
