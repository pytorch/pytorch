#pragma once

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_distributed {

//////////////////////////////////////////////////////////////////////////////
// Distributed operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
std::pair<LazyTensor, torch::lazy::Value> all_reduce(
    const LazyTensor& input, const torch::lazy::Value& token, AllReduceType reduce_type,
    double scale, std::vector<std::vector<lazy_tensors::int64>> groups);

torch::lazy::Value all_reduce_(LazyTensor& input, const torch::lazy::Value& token,
                      AllReduceType reduce_type, double scale,
                      std::vector<std::vector<lazy_tensors::int64>> groups);

torch::lazy::Value all_reduce(std::vector<LazyTensor>* inputs, const torch::lazy::Value& token,
                     AllReduceType reduce_type, double scale,
                     std::vector<std::vector<lazy_tensors::int64>> groups);

std::pair<LazyTensor, torch::lazy::Value> all_to_all(
    const LazyTensor& input, const torch::lazy::Value& token,
    lazy_tensors::int64 split_dimension, lazy_tensors::int64 concat_dimension,
    lazy_tensors::int64 split_count,
    std::vector<std::vector<lazy_tensors::int64>> groups);

std::pair<LazyTensor, torch::lazy::Value> collective_permute(
    const LazyTensor& input, const torch::lazy::Value& token,
    std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
        source_target_pairs);

LazyTensor get_dimensions_size(const LazyTensor& input,
                               std::vector<lazy_tensors::int64> dimensions);

}  // namespace lazy_tensor_distributed
}  // namespace torch_lazy_tensors
