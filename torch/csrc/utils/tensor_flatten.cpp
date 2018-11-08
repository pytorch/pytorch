#include "torch/csrc/utils/tensor_flatten.h"

#include <map>
#include <unordered_map>

namespace torch { namespace utils {

using namespace at;

std::vector<TensorGroup> take_tensors(
    TensorList tensors,
    size_t size_limit,
    bool fine_grained) {
  std::vector<TensorGroup> results;
  // an overapproximation, but at least we won't have to copy stuff around
  results.reserve(tensors.size());
  std::map<TypeID, TensorGroup> groups;
  size_t cur_group_size = 0;

  for (const auto & tensor : tensors) {
    auto& type = tensor.type();
    size_t tensor_size;
    if (type.is_sparse()) {
      const auto& indices = tensor._indices();
      const auto& values = tensor._values();
      tensor_size = indices.numel() * indices.type().elementSizeInBytes() +
                    values.numel() * indices.type().elementSizeInBytes();
    } else {
      tensor_size = tensor.numel() * type.elementSizeInBytes();
    }

    auto& type_group = groups[type.ID()];
    type_group.tensors.push_back(tensor);

    if (fine_grained) {
      cur_group_size += tensor_size;
      // Regardless the type, the current total size exceeds the limit
      if (cur_group_size >= size_limit) {
        // Spill all types to separate groups in results
        for (auto& entry : groups) {
          auto& group = entry.second;
          results.emplace_back(std::move(group));
        }
        cur_group_size = 0;
        groups.clear();
      }
    } else {
      type_group.size += tensor_size;
      if (type_group.size >= size_limit) {
        results.emplace_back();
        std::swap(results.back(), type_group);
      }
    }
  }
  // End case. Look for any remaining groups and return them.
  for (auto& entry : groups) {
    auto& group = entry.second;
    if (!fine_grained && group.size == 0) {
      continue;
    }
    results.emplace_back(std::move(group));
  }
  return results;
}

void reorder_tensors_like(std::vector<Tensor>& tensors, TensorList order) {
  AT_ASSERT(tensors.size() == order.size());
  std::unordered_map<at::Type*, std::vector<size_t>> type_indices;
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i)
    type_indices[&tensors[i].type()].push_back(i);

  std::unordered_map<at::Type*, size_t> type_used;
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
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i) {
    auto &ref_t = tensors[i];
    auto t = at::_sparse_coo_tensor_unsafe(indices[i], values[i], ref_t.sizes());
    outputs.emplace_back(t._coalesced_(ref_t.is_coalesced()));
  }
  return outputs;
}


}}
