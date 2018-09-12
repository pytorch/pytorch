#include <torch/serialize/cereal.h>

#include <torch/optim.h>
#include <torch/serialize/base.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#include <cereal/cereal.hpp>

#include <exception>
#include <string>

namespace torch {
namespace serialize {

void CerealWriter::write(
    const std::string& key,
    const Tensor& tensor,
    bool is_buffer) {
  (void)is_buffer; // Not used for cereal
  archive_(cereal::make_nvp(key, tensor));
}

void CerealReader::read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  (void)is_buffer; // Not used for cereal
  archive_(cereal::make_nvp(key, tensor));
}
} // namespace serialize

namespace detail {
// We use our own hard-coded type<->id mapping so that serialization is robust
// wrt changes in ATen; see e.g. https://git.io/vxd6R
// The mapping is consistent with the ScalarType enum as of pytorch version
// v0.1.11-7675-ge94c67e.
int32_t scalarTypeId(torch::Dtype type) {
  switch (type) {
    case torch::Dtype::Byte:
      return 0;
    case torch::Dtype::Char:
      return 1;
    case torch::Dtype::Short:
      return 2;
    case torch::Dtype::Int:
      return 3;
    case torch::Dtype::Long:
      return 4;
    case torch::Dtype::Half:
      return 5;
    case torch::Dtype::Float:
      return 6;
    case torch::Dtype::Double:
      return 7;
    case torch::Dtype::Undefined:
      return 8;
    default:
      throw std::runtime_error(
          "Unknown scalar type: " + std::to_string(static_cast<int>(type)));
  }
}

torch::Dtype scalarTypeFromId(int32_t id) {
  switch (id) {
    case 0:
      return torch::Dtype::Byte;
    case 1:
      return torch::Dtype::Char;
    case 2:
      return torch::Dtype::Short;
    case 3:
      return torch::Dtype::Int;
    case 4:
      return torch::Dtype::Long;
    case 5:
      return torch::Dtype::Half;
    case 6:
      return torch::Dtype::Float;
    case 7:
      return torch::Dtype::Double;
    case 8:
      return torch::Dtype::Undefined;
    default:
      throw std::runtime_error("Unknown scalar type id: " + std::to_string(id));
  }
}

int32_t backendId(at::Backend backend) {
  switch (backend) {
    case at::Backend::CPU:
      return 0;
    case at::Backend::CUDA:
      return 1;
    case at::Backend::SparseCPU:
      return 2;
    case at::Backend::SparseCUDA:
      return 3;
    case at::Backend::Undefined:
      return 4;
    default:
      throw std::runtime_error(
          "Unknown backend: " + std::to_string(static_cast<int>(backend)));
  }
}

at::Backend backendFromId(int32_t id) {
  switch (id) {
    case 0:
      return at::Backend::CPU;
    case 1:
      return at::Backend::CUDA;
    case 2:
      return at::Backend::SparseCPU;
    case 3:
      return at::Backend::SparseCUDA;
    case 4:
      return at::Backend::Undefined;
    default:
      throw std::runtime_error("Unknown backend id: " + std::to_string(id));
  }
}
} // namespace detail
} // namespace torch
