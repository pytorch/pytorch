#pragma once

#include <ATen/core/functional.h>
#include <torch/csrc/Export.h>
#include <ATen/ATen.h>
#include <utility>
#include <c10/core/TensorOptions.h>

namespace torch { namespace utils {

/// Generate an ID for a combination of tensor backend + scalar type to be used
/// when ordering tensors ('like' tensors are grouped by pulling out their
/// backend + scalar type, so this function combines that into a single number)
inline size_t type_id(const at::Tensor& tensor) {
  return static_cast<size_t>(tensor.options().backend()) *
      static_cast<size_t>(at::ScalarType::NumOptions) +
      static_cast<size_t>(tensor.scalar_type());
}

inline at::Tensor flatten_dense_tensors(at::TensorList tensors) {
  return at::flatten_dense_tensors(tensors);
}

inline std::vector<at::Tensor> unflatten_dense_tensors(const at::Tensor& flat, at::TensorList tensors) {
  return at::unflatten_dense_tensors(flat, tensors);
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct TensorGroup {
  std::vector<at::Tensor> tensors;
  size_t size = 0;

  size_t type_id() {
    AT_ASSERT(!tensors.empty());
    return ::torch::utils::type_id(tensors[0]);
  }

  const at::TensorOptions options() {
    AT_ASSERT(!tensors.empty());
    return tensors[0].options();
  }
};

// Helper function that takes a list of tensors and splits them into tensor
// groups by the size limit and outputs these tensor groups. If the input
// tensors are of different tensor types, they will be split into different
// groups as well.
//
// Two options of splitting provided to the user,
//
// Imagine the size_limit is 256 and the list of input tensors are:
// tensor_a(fp16 - 128 bytes),
// tensor_b(fp32 - 256 bytes),
// tensor_c(fp16 - 128 bytes),
//
// when fine_grained == false:
// The function will read the list of tensors sequentially and accumulate
// enough tensors for each data type until the size_limit, therefore:
// it will output: {{tensor_a, tensor_c}, {tensor_b}}
//
// when fine_grained == true:
// The function will read the list of tensors sequentially and  accumulate
// enough tensors for all data types until the size_limit, and then split
// the accumulated tensors into different groups by data types, therefore:
// it will output: {{tensor_a}, {tensor_b}, {tensor_c}}
TORCH_API std::vector<TensorGroup> take_tensors(
    at::TensorList tensors,
    size_t size_limit,
    bool fine_grained = false);

TORCH_API void reorder_tensors_like(std::vector<at::Tensor>& tensors, at::TensorList order);

TORCH_API std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(at::TensorList tensors);

TORCH_API std::vector<at::Tensor> unflatten_sparse_tensors(
    const at::Tensor& flat_indices,
    const at::Tensor& flat_values,
    at::TensorList tensors);

}}
