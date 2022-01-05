#pragma once

#include "lazy_tensor_core/csrc/cross_replica_reduces.h"
#include <torch/csrc/lazy/core/tensor.h>

namespace torch_lazy_tensors {
namespace lazy_tensor_distributed {

//////////////////////////////////////////////////////////////////////////////
// Distributed operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
std::pair<torch::lazy::LazyTensor, torch::lazy::Value> all_reduce(
    const torch::lazy::LazyTensor& input, const torch::lazy::Value& token,
    AllReduceType reduce_type, double scale,
    std::vector<std::vector<int64_t>> groups);

torch::lazy::Value all_reduce_(torch::lazy::LazyTensor& input,
                               const torch::lazy::Value& token,
                               AllReduceType reduce_type, double scale,
                               std::vector<std::vector<int64_t>> groups);

torch::lazy::Value all_reduce(std::vector<torch::lazy::LazyTensor>* inputs,
                              const torch::lazy::Value& token,
                              AllReduceType reduce_type, double scale,
                              std::vector<std::vector<int64_t>> groups);

std::pair<torch::lazy::LazyTensor, torch::lazy::Value> all_to_all(
    const torch::lazy::LazyTensor& input, const torch::lazy::Value& token,
    int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
    std::vector<std::vector<int64_t>> groups);

std::pair<torch::lazy::LazyTensor, torch::lazy::Value> collective_permute(
    const torch::lazy::LazyTensor& input, const torch::lazy::Value& token,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

torch::lazy::LazyTensor get_dimensions_size(const torch::lazy::LazyTensor& input,
                               std::vector<int64_t> dimensions);

}  // namespace lazy_tensor_distributed
}  // namespace torch_lazy_tensors
