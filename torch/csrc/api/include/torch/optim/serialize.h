#pragma once

#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace torch {
namespace optim {

// Note: These functions are all called `serialize()` so they can be called
// inside a template where the archive type is a template type and can thus be
// passed such that the appropriate overload is selected.

/// Utility function to save a value of `int64_t` type.
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const int64_t& value);

/// Utility function to load a value of `int64_t` type.
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    int64_t& value);

/// Utility function to save a vector of step buffers.
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const std::vector<int64_t>& steps);

/// Utility function to load a vector of step buffers.
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    std::vector<int64_t>& steps);

/// Utility function to save a vector of buffers.
template <typename BufferContainer>
void serialize(
    serialize::OutputArchive& archive,
    const std::string& key,
    const BufferContainer& buffers) {
  archive.write(
      key + "/size", torch::tensor(static_cast<int64_t>(buffers.size())));
  for (size_t index = 0; index < buffers.size(); ++index) {
    archive.write(
        key + "/" + std::to_string(index), buffers[index], /*is_buffer=*/true);
  }
}

/// Utility function to load a vector of buffers.
template <typename BufferContainer>
void serialize(
    serialize::InputArchive& archive,
    const std::string& key,
    BufferContainer& buffers) {
  buffers.clear();
  torch::Tensor size_tensor;
  archive.read(key + "/size", size_tensor);
  const size_t size = size_tensor.item<int64_t>();
  for (size_t index = 0; index < size; ++index) {
    buffers.emplace_back();
    archive.read(
        key + "/" + std::to_string(index), buffers.back(), /*is_buffer=*/true);
  }
}

#define _TORCH_OPTIM_SERIALIZE(name) \
  torch::optim::serialize(archive, #name, self.name)

} // namespace optim
} // namespace torch
