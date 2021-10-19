#include "lazy_tensor_core/csrc/tensor_distributed.h"

#include "lazy_tensor_core/csrc/ops/all_reduce.h"
#include "lazy_tensor_core/csrc/ops/all_to_all.h"
#include "lazy_tensor_core/csrc/ops/collective_permute.h"
#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_distributed {

std::pair<LazyTensor, torch::lazy::Value> all_reduce(
    const LazyTensor& input, const torch::lazy::Value& token, AllReduceType reduce_type,
    double scale, std::vector<std::vector<lazy_tensors::int64>> groups) {
  std::vector<torch::lazy::Value> input_values({input.GetIrValue()});
  NodePtr node = torch::lazy::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  return {input.CreateFrom(torch::lazy::Value(node, 0)), torch::lazy::Value(node, 1)};
}

torch::lazy::Value all_reduce_(LazyTensor& input, const torch::lazy::Value& token,
                      AllReduceType reduce_type, double scale,
                      std::vector<std::vector<lazy_tensors::int64>> groups) {
  std::vector<torch::lazy::Value> input_values({input.GetIrValue()});
  NodePtr node = torch::lazy::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  input.SetInPlaceIrValue(torch::lazy::Value(node, 0));
  return torch::lazy::Value(node, 1);
}

torch::lazy::Value all_reduce(std::vector<LazyTensor>* inputs, const torch::lazy::Value& token,
                     AllReduceType reduce_type, double scale,
                     std::vector<std::vector<lazy_tensors::int64>> groups) {
  std::vector<torch::lazy::Value> input_values;
  input_values.reserve(inputs->size());
  for (auto& input : *inputs) {
    input_values.push_back(input.GetIrValue());
  }
  NodePtr node = torch::lazy::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  for (size_t i = 0; i < inputs->size(); ++i) {
    (*inputs)[i].SetInPlaceIrValue(torch::lazy::Value(node, i));
  }
  return torch::lazy::Value(node, inputs->size());
}

std::pair<LazyTensor, torch::lazy::Value> all_to_all(
    const LazyTensor& input, const torch::lazy::Value& token,
    lazy_tensors::int64 split_dimension, lazy_tensors::int64 concat_dimension,
    lazy_tensors::int64 split_count,
    std::vector<std::vector<lazy_tensors::int64>> groups) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::AllToAll>(
      input.GetIrValue(), token, split_dimension, concat_dimension, split_count,
      std::move(groups));
  return {input.CreateFrom(torch::lazy::Value(node, 0)), torch::lazy::Value(node, 1)};
}

LazyTensor get_dimensions_size(const LazyTensor& input,
                               std::vector<lazy_tensors::int64> dimensions) {
  return input.CreateFrom(torch::lazy::MakeNode<ir::ops::GetDimensionsSize>(
                              input.GetIrValue(), std::move(dimensions)),
                          at::ScalarType::Int);
}

std::pair<LazyTensor, torch::lazy::Value> collective_permute(
    const LazyTensor& input, const torch::lazy::Value& token,
    std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
        source_target_pairs) {
  NodePtr node = torch::lazy::MakeNode<ir::ops::CollectivePermute>(
      input.GetIrValue(), token, std::move(source_target_pairs));
  return {input.CreateFrom(torch::lazy::Value(node, 0)), torch::lazy::Value(node, 1)};
}

}  // namespace lazy_tensor_distributed
}  // namespace torch_lazy_tensors
