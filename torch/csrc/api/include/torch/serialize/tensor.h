#pragma once

namespace at {
class Tensor;
} // namespace at

namespace torch {
using at::Tensor;
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch

namespace torch {
/// Serializes a `Tensor` into an `OutputArchive`.
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Tensor& tensor);

/// Deserializes a `Tensor` from an `InputArchive`.
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Tensor& tensor);
} // namespace torch
