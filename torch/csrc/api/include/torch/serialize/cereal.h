#pragma once

#include <torch/optim.h>
#include <torch/serialize/base.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace torch {
namespace serialize {
class CerealWriter : public Writer {
 public:
  template <typename... Args>
  explicit CerealWriter(Args&&... args)
      : archive_(std::forward<Args>(args)...) {}

  void write(
      const std::string& key,
      const Tensor& tensor,
      bool is_buffer = false) override;

 private:
  cereal::BinaryOutputArchive archive_;
};

class CerealReader : public Reader {
 public:
  template <typename... Args>
  explicit CerealReader(Args&&... args)
      : archive_(std::forward<Args>(args)...) {}

  void read(const std::string& key, Tensor& tensor, bool is_buffer = false)
      override;

 private:
  cereal::BinaryInputArchive archive_;
};
} // namespace serialize

namespace detail {
int32_t scalarTypeId(torch::Dtype type);
torch::Dtype scalarTypeFromId(int32_t id);
int32_t backendId(at::Backend backend);
at::Backend backendFromId(int32_t id);
} // namespace detail
} // namespace torch

namespace cereal {
namespace agimpl {

template <class Archive>
void saveBinary(Archive& archive, void const* data, size_t size) {
  // In general, there's no direct `saveBinary`-like method on archives
  std::vector<char> v(
      static_cast<char const*>(data), static_cast<char const*>(data) + size);
  archive(v);
}
template <>
inline void saveBinary(
    BinaryOutputArchive& archive,
    void const* data,
    size_t size) {
  // Writes to output stream without extra copy
  archive.saveBinary(data, size);
}

template <class Archive>
void loadBinary(Archive& archive, void* data, size_t size) {
  // In general, there's no direct `loadBinary`-like method on archives
  std::vector<char> v(size);
  archive(v);
  std::memcpy(data, v.data(), size);
}
template <>
inline void loadBinary(BinaryInputArchive& archive, void* data, size_t size) {
  // Read from input stream without extra copy
  archive.loadBinary(data, size);
}

} // namespace agimpl

// Gradients will not be saved for variables
template <class Archive>
void save(Archive& archive, const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    int32_t typeId = ::torch::detail::scalarTypeId(torch::Dtype::Undefined);
    archive(CEREAL_NVP(typeId));
    return;
  } else {
    int32_t typeId = ::torch::detail::scalarTypeId(tensor.dtype());
    archive(CEREAL_NVP(typeId));
  }
  auto sizes = std::vector<int64_t>();
  auto buf = std::vector<uint8_t>();
  for (auto s : tensor.sizes()) {
    sizes.push_back(s);
  }
  auto contig = tensor.cpu().contiguous();
  int32_t backend = ::torch::detail::backendId(tensor.type().backend());

  archive(CEREAL_NVP(backend), CEREAL_NVP(sizes));
  agimpl::saveBinary(
      archive,
      contig.data_ptr(),
      tensor.numel() * tensor.type().elementSizeInBytes());
}

/**
 * We follow these rules for loading:
 * 1. If tensor is defined, and the same ScalarType as the saved tensor,
 *    then we simply copy the data into the tensor, with resizing.
 * 2. Otherwise, overwrite the provided tensor with the right type and backend
 **/
template <class Archive>
void load(Archive& archive, torch::Tensor& tensor) {
  torch::NoGradGuard guard;
  torch::Dtype type;
  int32_t typeId;
  archive(CEREAL_NVP(typeId));
  type = ::torch::detail::scalarTypeFromId(typeId);
  if (type == torch::Dtype::Undefined) {
    tensor = torch::Tensor();
    return;
  }

  int32_t backendId;
  auto sizes = std::vector<int64_t>();
  auto buf = std::vector<uint8_t>();
  archive(CEREAL_NVP(backendId), CEREAL_NVP(sizes));

  at::Backend backend = ::torch::detail::backendFromId(backendId);
  if (!tensor.defined() || tensor.dtype() != type) {
    tensor = torch::empty({}, at::TensorOptions(backend).dtype(type));
  }
  const auto required_grad = tensor.requires_grad();
  tensor.set_requires_grad(false);
  tensor.resize_(sizes);
  tensor.set_requires_grad(required_grad);

  if (tensor.type().is_cuda()) {
    // should actually use cudamemcpy probably
    auto cputensor = torch::empty(sizes, tensor.dtype());
    agimpl::loadBinary(
        archive,
        cputensor.data_ptr(),
        cputensor.numel() * cputensor.type().elementSizeInBytes());
    tensor.copy_(cputensor);
  } else {
    agimpl::loadBinary(
        archive,
        tensor.data_ptr(),
        tensor.numel() * tensor.type().elementSizeInBytes());
  }
}
} // namespace cereal
