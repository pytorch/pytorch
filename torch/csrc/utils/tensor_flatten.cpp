#include <torch/csrc/utils/tensor_flatten.h>

#include <map>
#include <unordered_map>

namespace torch::utils {

using namespace at;

std::vector<TensorGroup> take_tensors(
    TensorList tensors,
    size_t size_limit,
    bool fine_grained) {
  std::vector<TensorGroup> results;
  // an overapproximation, but at least we won't have to copy stuff around
  results.reserve(tensors.size());
  std::map<int64_t, TensorGroup> groups;
  size_t cur_group_size = 0;

  for (const auto& tensor : tensors) {
    size_t tensor_size = 0;
    if (tensor.is_sparse()) {
      const auto& indices = tensor._indices();
      const auto& values = tensor._values();
      tensor_size = indices.numel() * indices.element_size() +
          values.numel() * indices.element_size();
    } else {
      tensor_size = tensor.numel() * tensor.element_size();
    }

    auto& type_group = groups[static_cast<int64_t>(type_id(tensor))];
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
    if (group.tensors.empty()) {
      continue;
    }
    results.emplace_back(std::move(group));
  }
  return results;
}

void reorder_tensors_like(std::vector<Tensor>& tensors, TensorList order) {
  AT_ASSERT(tensors.size() == order.size());
  std::unordered_map<size_t, std::vector<size_t>> type_id_to_indices;
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i)
    type_id_to_indices[type_id(tensors[i])].push_back(i);

  std::unordered_map<size_t, size_t> type_id_to_type_used;
  std::vector<Tensor> ordered_tensors;
  ordered_tensors.reserve(tensors.size());
  for (auto& tmpl_tensor : order) {
    size_t tmpl_type_id = type_id(tmpl_tensor);
    auto& indices = type_id_to_indices[tmpl_type_id];
    auto& used = type_id_to_type_used[tmpl_type_id];
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

} // namespace

std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(
    at::TensorList tensors) {
  auto flat_indices = utils::flatten_dense_tensors(fmap(tensors, &get_indices));
  auto flat_values = utils::flatten_dense_tensors(fmap(tensors, &get_values));
  return std::make_pair(flat_indices, flat_values);
}

std::vector<at::Tensor> unflatten_sparse_tensors(
    const at::Tensor& flat_indices,
    const at::Tensor& flat_values,
    at::TensorList tensors) {
  if (tensors.empty())
    return {};

  auto indices =
      utils::unflatten_dense_tensors(flat_indices, fmap(tensors, &get_indices));
  auto values =
      utils::unflatten_dense_tensors(flat_values, fmap(tensors, &get_values));

  std::vector<at::Tensor> outputs;
  outputs.reserve(tensors.size());
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i) {
    auto& ref_t = tensors[i];
    auto t =
        at::_sparse_coo_tensor_unsafe(indices[i], values[i], ref_t.sizes());
    outputs.emplace_back(t._coalesced_(ref_t.is_coalesced()));
  }
  return outputs;
}

} // namespace torch::utils
