#pragma once

#include <torch/types.h>

#include <utility>
#include <vector>

namespace torch::data::transforms {

/// A transformation of a batch to a new batch.
template <typename InputBatch, typename OutputBatch>
class BatchTransform {
 public:
  using InputBatchType = InputBatch;
  using OutputBatchType = OutputBatch;

  virtual ~BatchTransform() = default;

  /// Applies the transformation to the given `input_batch`.
  virtual OutputBatch apply_batch(InputBatch input_batch) = 0;
};

/// A transformation of individual input examples to individual output examples.
///
/// Just like a `Dataset` is a `BatchDataset`, a `Transform` is a
/// `BatchTransform` that can operate on the level of individual examples rather
/// than entire batches. The batch-level transform is implemented (by default)
/// in terms of the example-level transform, though this can be customized.
template <typename Input, typename Output>
class Transform
    : public BatchTransform<std::vector<Input>, std::vector<Output>> {
 public:
  using InputType = Input;
  using OutputType = Output;

  /// Applies the transformation to the given `input`.
  virtual OutputType apply(InputType input) = 0;

  /// Applies the `transformation` over the entire `input_batch`.
  std::vector<Output> apply_batch(std::vector<Input> input_batch) override {
    std::vector<Output> output_batch;
    output_batch.reserve(input_batch.size());
    for (auto&& input : input_batch) {
      output_batch.push_back(apply(std::move(input)));
    }
    return output_batch;
  }
};
} // namespace torch::data::transforms
