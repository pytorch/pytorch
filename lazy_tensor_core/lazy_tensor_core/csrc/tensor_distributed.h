#pragma once

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include <torch/csrc/lazy/core/tensor.h>

namespace torch_lazy_tensors {
namespace lazy_tensor_distributed {

//////////////////////////////////////////////////////////////////////////////
// Distributed operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
// std::pair<torch::lazy::LazyTensorPtr, torch::lazy::Value> all_reduce(
//     const torch::lazy::LazyTensorPtr& input, const torch::lazy::Value& token,
//     AllReduceType reduce_type, double scale,
//     std::vector<std::vector<int64_t>> groups);

// torch::lazy::Value all_reduce_(torch::lazy::LazyTensorPtr& input,
//                                const torch::lazy::Value& token,
//                                AllReduceType reduce_type, double scale,
//                                std::vector<std::vector<int64_t>> groups);

// torch::lazy::Value all_reduce(std::vector<torch::lazy::LazyTensorPtr>* inputs,
//                               const torch::lazy::Value& token,
//                               AllReduceType reduce_type, double scale,
//                               std::vector<std::vector<int64_t>> groups);

// std::pair<torch::lazy::LazyTensorPtr, torch::lazy::Value> all_to_all(
//     const torch::lazy::LazyTensorPtr& input, const torch::lazy::Value& token,
//     int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
//     std::vector<std::vector<int64_t>> groups);

// std::pair<torch::lazy::LazyTensorPtr, torch::lazy::Value> collective_permute(
//     const torch::lazy::LazyTensorPtr& input, const torch::lazy::Value& token,
//     std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

// torch::lazy::LazyTensorPtr get_dimensions_size(const torch::lazy::LazyTensorPtr& input,
//                                std::vector<int64_t> dimensions);

}  // namespace lazy_tensor_distributed
}  // namespace torch_lazy_tensors
