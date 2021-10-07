#pragma once

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_distributed {

//////////////////////////////////////////////////////////////////////////////
// Distributed operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
std::pair<LazyTensor, ir::Value> all_reduce(
    const LazyTensor& input, const ir::Value& token, AllReduceType reduce_type,
    double scale, std::vector<std::vector<lazy_tensors::int64>> groups);

ir::Value all_reduce_(LazyTensor& input, const ir::Value& token,
                      AllReduceType reduce_type, double scale,
                      std::vector<std::vector<lazy_tensors::int64>> groups);

ir::Value all_reduce(std::vector<LazyTensor>* inputs, const ir::Value& token,
                     AllReduceType reduce_type, double scale,
                     std::vector<std::vector<lazy_tensors::int64>> groups);

std::pair<LazyTensor, ir::Value> all_to_all(
    const LazyTensor& input, const ir::Value& token,
    lazy_tensors::int64 split_dimension, lazy_tensors::int64 concat_dimension,
    lazy_tensors::int64 split_count,
    std::vector<std::vector<lazy_tensors::int64>> groups);

std::pair<LazyTensor, ir::Value> collective_permute(
    const LazyTensor& input, const ir::Value& token,
    std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
        source_target_pairs);

LazyTensor get_dimensions_size(const LazyTensor& input,
                               std::vector<lazy_tensors::int64> dimensions);

}  // namespace lazy_tensor_distributed
}  // namespace torch_lazy_tensors
