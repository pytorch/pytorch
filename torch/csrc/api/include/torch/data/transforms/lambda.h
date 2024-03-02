#pragma once

#include <torch/data/transforms/base.h>

#include <functional>
#include <utility>
#include <vector>

namespace torch {
namespace data {
namespace transforms {

/// A `BatchTransform` that applies a user-provided functor to a batch.
template <typename Input, typename Output = Input>
class BatchLambda : public BatchTransform<Input, Output> {
 public:
  using typename BatchTransform<Input, Output>::InputBatchType;
  using typename BatchTransform<Input, Output>::OutputBatchType;
  using FunctionType = std::function<OutputBatchType(InputBatchType)>;

  /// Constructs the `BatchLambda` from the given `function` object.
  explicit BatchLambda(FunctionType function)
      : function_(std::move(function)) {}

  /// Applies the user-provided function object to the `input_batch`.
  OutputBatchType apply_batch(InputBatchType input_batch) override {
    return function_(std::move(input_batch));
  }

 private:
  FunctionType function_;
};

// A `Transform` that applies a user-provided functor to individual examples.
template <typename Input, typename Output = Input>
class Lambda : public Transform<Input, Output> {
 public:
  using typename Transform<Input, Output>::InputType;
  using typename Transform<Input, Output>::OutputType;
  using FunctionType = std::function<Output(Input)>;

  /// Constructs the `Lambda` from the given `function` object.
  explicit Lambda(FunctionType function) : function_(std::move(function)) {}

  /// Applies the user-provided function object to the `input`.
  OutputType apply(InputType input) override {
    return function_(std::move(input));
  }

 private:
  FunctionType function_;
};

} // namespace transforms
} // namespace data
} // namespace torch
