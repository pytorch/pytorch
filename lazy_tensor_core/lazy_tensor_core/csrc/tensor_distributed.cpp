#include "lazy_tensor_core/csrc/tensor_distributed.h"

#include "lazy_tensor_core/csrc/ops/all_reduce.h"
#include "lazy_tensor_core/csrc/ops/all_to_all.h"
#include "lazy_tensor_core/csrc/ops/collective_permute.h"
#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_distributed {

std::pair<LazyTensor, ir::Value> all_reduce(
    const LazyTensor& input, const ir::Value& token, AllReduceType reduce_type,
    double scale, std::vector<std::vector<lazy_tensors::int64>> groups) {
  std::vector<ir::Value> input_values({input.GetIrValue()});
  ir::NodePtr node = ir::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  return {input.CreateFrom(ir::Value(node, 0)), ir::Value(node, 1)};
}

ir::Value all_reduce_(LazyTensor& input, const ir::Value& token,
                      AllReduceType reduce_type, double scale,
                      std::vector<std::vector<lazy_tensors::int64>> groups) {
  std::vector<ir::Value> input_values({input.GetIrValue()});
  ir::NodePtr node = ir::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  input.SetInPlaceIrValue(ir::Value(node, 0));
  return ir::Value(node, 1);
}

ir::Value all_reduce(std::vector<LazyTensor>* inputs, const ir::Value& token,
                     AllReduceType reduce_type, double scale,
                     std::vector<std::vector<lazy_tensors::int64>> groups) {
  std::vector<ir::Value> input_values;
  input_values.reserve(inputs->size());
  for (auto& input : *inputs) {
    input_values.push_back(input.GetIrValue());
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  for (size_t i = 0; i < inputs->size(); ++i) {
    (*inputs)[i].SetInPlaceIrValue(ir::Value(node, i));
  }
  return ir::Value(node, inputs->size());
}

std::pair<LazyTensor, ir::Value> all_to_all(
    const LazyTensor& input, const ir::Value& token,
    lazy_tensors::int64 split_dimension, lazy_tensors::int64 concat_dimension,
    lazy_tensors::int64 split_count,
    std::vector<std::vector<lazy_tensors::int64>> groups) {
  ir::NodePtr node = ir::MakeNode<ir::ops::AllToAll>(
      input.GetIrValue(), token, split_dimension, concat_dimension, split_count,
      std::move(groups));
  return {input.CreateFrom(ir::Value(node, 0)), ir::Value(node, 1)};
}

LazyTensor get_dimensions_size(const LazyTensor& input,
                               std::vector<lazy_tensors::int64> dimensions) {
  return input.CreateFrom(ir::MakeNode<ir::ops::GetDimensionsSize>(
                              input.GetIrValue(), std::move(dimensions)),
                          at::ScalarType::Int);
}

std::pair<LazyTensor, ir::Value> collective_permute(
    const LazyTensor& input, const ir::Value& token,
    std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
        source_target_pairs) {
  ir::NodePtr node = ir::MakeNode<ir::ops::CollectivePermute>(
      input.GetIrValue(), token, std::move(source_target_pairs));
  return {input.CreateFrom(ir::Value(node, 0)), ir::Value(node, 1)};
}

}  // namespace lazy_tensor_distributed
}  // namespace torch_lazy_tensors
