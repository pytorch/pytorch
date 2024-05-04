#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <torch/csrc/Export.h>

#include <cstddef>
#include <string>

namespace torch {
namespace data {
namespace datasets {
/// The MNIST dataset.
class TORCH_API MNIST : public Dataset<MNIST> {
 public:
  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  /// Loads the MNIST dataset from the `root` path.
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
  explicit MNIST(const std::string& root, Mode mode = Mode::kTrain);

  /// Returns the `Example` at the given `index`.
  Example<> get(size_t index) override;

  /// Returns the size of the dataset.
  optional<size_t> size() const override;

  /// Returns true if this is the training subset of MNIST.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  bool is_train() const noexcept;

  /// Returns all images stacked into a single tensor.
  const Tensor& images() const;

  /// Returns all targets stacked into a single tensor.
  const Tensor& targets() const;

 private:
  Tensor images_, targets_;
};
} // namespace datasets
} // namespace data
} // namespace torch
