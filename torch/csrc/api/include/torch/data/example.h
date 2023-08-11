#pragma once

#include <torch/types.h>

namespace torch {
namespace data {

/// An `Example` from a dataset.
///
/// A dataset consists of data and an associated target (label).
template <typename Data = at::Tensor, typename Target = at::Tensor>
struct Example {
  using DataType = Data;
  using TargetType = Target;

  Example() = default;
  Example(Data data, Target target)
      : data(std::move(data)), target(std::move(target)) {}

  Data data;
  Target target;
};

namespace example {
using NoTarget = void;
} // namespace example

/// A specialization for `Example` that does not have a target.
///
/// This class exists so that code can be written for a templated `Example`
/// type, and work both for labeled and unlabeled datasets.
template <typename Data>
struct Example<Data, example::NoTarget> {
  using DataType = Data;
  using TargetType = example::NoTarget;

  Example() = default;
  /* implicit */ Example(Data data) : data(std::move(data)) {}

  // When a DataLoader returns an Example like this, that example should be
  // implicitly convertible to the underlying data type.

  operator Data&() {
    return data;
  }
  operator const Data&() const {
    return data;
  }

  Data data;
};

using TensorExample = Example<at::Tensor, example::NoTarget>;
} // namespace data
} // namespace torch
