#include "torch/csrc/utils/tensor_flatten.h"

#include <unordered_map>

namespace torch { namespace utils {

using namespace at;

std::vector<TensorGroup> take_tensors(TensorList tensors, std::size_t size_limit) {
  std::vector<TensorGroup> results;
  results.reserve(tensors.size()); // an overapproximation, but at least we won't have to copy stuff around
  std::unordered_map<at::Type*, TensorGroup> groups;
  for (const auto & tensor : tensors) {
    auto & type = tensor.type();
    std::size_t tensor_size;
    if (type.is_sparse()) {
      const auto& indices = tensor._indices();
      const auto& values = tensor._values();
      tensor_size = indices.numel() * indices.type().elementSizeInBytes() +
                    values.numel() * indices.type().elementSizeInBytes();
    } else {
      tensor_size = tensor.numel() * type.elementSizeInBytes();
    }
    auto & type_group = groups[&type];
    type_group.tensors.push_back(tensor);
    type_group.size += tensor_size;
    if (type_group.size + tensor_size >= size_limit) {
      results.emplace_back();
      std::swap(results.back(), type_group);
    }
  }
  // End case. Look for any remaining groups and return them.
  for (auto & entry : groups) {
    auto & group = entry.second;
    if (group.size > 0) {
      results.emplace_back(std::move(group));
    }
  }
  return results;
}

void reorder_tensors_like(std::vector<Tensor>& tensors, TensorList order) {
  TORCH_ASSERT(tensors.size() == order.size());
  std::unordered_map<at::Type*, std::vector<std::size_t>> type_indices;
  for (std::size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i)
    type_indices[&tensors[i].type()].push_back(i);

  std::unordered_map<at::Type*, std::size_t> type_used;
  std::vector<Tensor> ordered_tensors;
  ordered_tensors.reserve(tensors.size());
  for (auto & tmpl_tensor : order) {
    auto * type = &tmpl_tensor.type();
    auto & indices = type_indices[type];
    auto & used = type_used[type];
    ordered_tensors.push_back(tensors[indices[used++]]);
  }
  std::swap(tensors, ordered_tensors);
}

namespace {

at::Tensor get_indices(const at::Tensor& t) {
  return t._indices();
}

at::Tensor get_values(const at::Tensor& t) {
  return t._values();
}

}

std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(at::TensorList tensors) {
  auto flat_indices = flatten_dense_tensors(fmap(tensors, &get_indices));
  auto flat_values = flatten_dense_tensors(fmap(tensors, &get_values));
  return std::make_pair(flat_indices, flat_values);
}

std::vector<at::Tensor> unflatten_sparse_tensors(
        const at::Tensor& flat_indices, const at::Tensor& flat_values,
        at::TensorList tensors) {
  if (tensors.size() == 0) return {};

  auto indices = unflatten_dense_tensors(flat_indices, fmap(tensors, &get_indices));
  auto values = unflatten_dense_tensors(flat_values, fmap(tensors, &get_values));

  std::vector<at::Tensor> outputs;
  outputs.reserve(tensors.size());
  auto & type = tensors[0].type();
  for (std::size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i)
    outputs.emplace_back(type._sparse_coo_tensor_unsafe(indices[i], values[i], tensors[i].sizes()));
  return outputs;
}


}}
