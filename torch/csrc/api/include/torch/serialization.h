#pragma once

#include <fstream>

#include "cereal/archives/binary.hpp"
#include "cereal/types/polymorphic.hpp"

#include "cereal/types/string.hpp"
#include "cereal/types/unordered_map.hpp"
#include "cereal/types/vector.hpp"

namespace torch {

// Some convenience functions for saving and loading
template <typename T>
void save(std::ostream& stream, T const& obj) {
  cereal::BinaryOutputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void load(std::istream& stream, T& obj) {
  cereal::BinaryInputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void save(std::ostream& stream, T const* obj) {
  cereal::BinaryOutputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void load(std::istream& stream, T* obj) {
  cereal::BinaryInputArchive archive(stream);
  archive(*obj);
}
template <typename T>
void save(std::string const& path, T const& obj) {
  std::ofstream os(path, std::ios::binary);
  torch::save(os, obj);
}
template <typename T>
void load(std::string const& path, T& obj) {
  std::ifstream is(path, std::ios::binary);
  torch::load(is, obj);
}

namespace detail {

// We use our own hard-coded type<->id mapping so that serialization is robust
// wrt changes in ATen; see e.g. https://git.io/vxd6R
// The mapping is consistent with the ScalarType enum as of pytorch version
// v0.1.11-7675-ge94c67e.
inline int32_t scalarTypeId(at::ScalarType type) {
  switch (type) {
    case at::ScalarType::Byte: return 0;
    case at::ScalarType::Char: return 1;
    case at::ScalarType::Short: return 2;
    case at::ScalarType::Int: return 3;
    case at::ScalarType::Long: return 4;
    case at::ScalarType::Half: return 5;
    case at::ScalarType::Float: return 6;
    case at::ScalarType::Double: return 7;
    case at::ScalarType::Undefined: return 8;
    default:
      throw std::runtime_error(
          "Unknown scalar type: " + std::to_string(static_cast<int>(type)));
  }
}

inline at::ScalarType scalarTypeFromId(int32_t id) {
  switch (id) {
    case 0: return at::ScalarType::Byte;
    case 1: return at::ScalarType::Char;
    case 2: return at::ScalarType::Short;
    case 3: return at::ScalarType::Int;
    case 4: return at::ScalarType::Long;
    case 5: return at::ScalarType::Half;
    case 6: return at::ScalarType::Float;
    case 7: return at::ScalarType::Double;
    case 8: return at::ScalarType::Undefined;
    default:
      throw std::runtime_error("Unknown scalar type id: " + std::to_string(id));
  }
}

inline int32_t backendId(at::Backend backend) {
  switch (backend) {
    case at::Backend::CPU: return 0;
    case at::Backend::CUDA: return 1;
    case at::Backend::SparseCPU: return 2;
    case at::Backend::SparseCUDA: return 3;
    case at::Backend::Undefined: return 4;
    default:
      throw std::runtime_error(
          "Unknown backend: " + std::to_string(static_cast<int>(backend)));
  }
}

inline at::Backend backendFromId(int32_t id) {
  switch (id) {
    case 0: return at::Backend::CPU;
    case 1: return at::Backend::CUDA;
    case 2: return at::Backend::SparseCPU;
    case 3: return at::Backend::SparseCUDA;
    case 4: return at::Backend::Undefined;
    default:
      throw std::runtime_error("Unknown backend id: " + std::to_string(id));
  }
}

} // namespace detail
} // namespace torch

// This is super ugly and I don't know how to simplify it
CEREAL_REGISTER_TYPE(torch::SGD);
CEREAL_REGISTER_POLYMORPHIC_RELATION(torch::OptimizerImpl, torch::SGD);
CEREAL_REGISTER_TYPE(torch::Adagrad);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    torch::OptimizerImpl,
    torch::Adagrad);
CEREAL_REGISTER_TYPE(torch::RMSprop);
CEREAL_REGISTER_POLYMORPHIC_RELATION(
    torch::OptimizerImpl,
    torch::RMSprop);
CEREAL_REGISTER_TYPE(torch::Adam);
CEREAL_REGISTER_POLYMORPHIC_RELATION(torch::OptimizerImpl, torch::Adam);

namespace cereal {

namespace agimpl {

template <class Archive>
void saveBinary(Archive& archive, void const* data, std::size_t size) {
  // In general, there's no direct `saveBinary`-like method on archives
  std::vector<char> v(
      static_cast<char const*>(data), static_cast<char const*>(data) + size);
  archive(v);
}
template <>
inline void
saveBinary(BinaryOutputArchive& archive, void const* data, std::size_t size) {
  // Writes to output stream without extra copy
  archive.saveBinary(data, size);
}

template <class Archive>
void loadBinary(Archive& archive, void* data, std::size_t size) {
  // In general, there's no direct `loadBinary`-like method on archives
  std::vector<char> v(size);
  archive(v);
  std::memcpy(data, v.data(), size);
}
template <>
inline void
loadBinary(BinaryInputArchive& archive, void* data, std::size_t size) {
  // Read from input stream without extra copy
  archive.loadBinary(data, size);
}

} // namespace agimpl

// Gradients will not be saved for variables
template <class Archive>
void save(Archive& archive, at::Tensor const& tensor) {
  if (!tensor.defined()) {
    int32_t typeId = ::torch::detail::scalarTypeId(at::ScalarType::Undefined);
    archive(CEREAL_NVP(typeId));
    return;
  } else {
    int32_t typeId = ::torch::detail::scalarTypeId(tensor.type().scalarType());
    archive(CEREAL_NVP(typeId));
  }
  auto sizes = std::vector<int64_t>();
  auto buf = std::vector<uint8_t>();
  for (auto s : tensor.sizes()) {
    sizes.push_back(s);
  }
  auto contig = tensor.toBackend(at::kCPU).contiguous();
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
void load(Archive& archive, at::Tensor& tensor) {
  at::ScalarType type;
  int32_t typeId;
  archive(CEREAL_NVP(typeId));
  type = ::torch::detail::scalarTypeFromId(typeId);
  if (type == at::ScalarType::Undefined) {
    tensor = at::Tensor();
    return;
  }

  int32_t backendId;
  auto sizes = std::vector<int64_t>();
  auto buf = std::vector<uint8_t>();
  archive(CEREAL_NVP(backendId), CEREAL_NVP(sizes));

  at::Backend backend = ::torch::detail::backendFromId(backendId);
  if (!tensor.defined() || tensor.type().scalarType() != type) {
    tensor = at::getType(backend, type).tensor();
  }
  tensor.resize_(sizes);

  if (tensor.type().is_cuda()) {
    // should actually use cudamemcpy probably
    auto cputensor = at::CPU(tensor.type().scalarType()).tensor(sizes);
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

template <class Archive>
void load(Archive& archive, tag::Variable& var) {
  load(archive, var.data());
}

} // namespace cereal
