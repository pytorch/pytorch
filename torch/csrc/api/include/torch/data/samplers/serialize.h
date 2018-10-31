#pragma once

#include <torch/data/samplers/base.h>
#include <torch/serialize/archive.h>

namespace torch {
namespace data {
namespace samplers {
/// Serializes a `Sampler` into an `OutputArchive`.
template <typename BatchIndex>
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Sampler<BatchIndex>& sampler) {
  sampler.save(archive);
  return archive;
}

/// Deserializes a `Sampler` from an `InputArchive`.
template <typename BatchIndex>
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Sampler<BatchIndex>& sampler) {
  sampler.load(archive);
  return archive;
}
} // namespace samplers
} // namespace data
} // namespace torch
