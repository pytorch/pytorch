#include "lazy_tensor_core/csrc/tensor_distributed.h"

#include "lazy_tensor_core/csrc/ops/all_reduce.h"
#include "lazy_tensor_core/csrc/ops/all_to_all.h"
#include "lazy_tensor_core/csrc/ops/collective_permute.h"
#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"

namespace torch_lazy_tensors {
namespace lazy_tensor_distributed {

std::pair<torch::lazy::LazyTensor, torch::lazy::Value> all_reduce(
    const torch::lazy::LazyTensor& input, const torch::lazy::Value& token,
    AllReduceType reduce_type, double scale,
    std::vector<std::vector<int64_t>> groups) {
  std::vector<torch::lazy::Value> input_values({input.GetIrValue()});
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  return {torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 0), input.GetDevice()), torch::lazy::Value(node, 1)};
}

torch::lazy::Value all_reduce_(torch::lazy::LazyTensor& input,
                               const torch::lazy::Value& token,
                               AllReduceType reduce_type, double scale,
                               std::vector<std::vector<int64_t>> groups) {
  std::vector<torch::lazy::Value> input_values({input.GetIrValue()});
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  input.SetInPlaceIrValue(torch::lazy::Value(node, 0));
  return torch::lazy::Value(node, 1);
}

torch::lazy::Value all_reduce(std::vector<torch::lazy::LazyTensor>* inputs,
                              const torch::lazy::Value& token,
                              AllReduceType reduce_type, double scale,
                              std::vector<std::vector<int64_t>> groups) {
  std::vector<torch::lazy::Value> input_values;
  input_values.reserve(inputs->size());
  for (auto& input : *inputs) {
    input_values.push_back(input.GetIrValue());
  }
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::AllReduce>(
      reduce_type, input_values, token, scale, std::move(groups));
  for (size_t i = 0; i < inputs->size(); ++i) {
    (*inputs)[i].SetInPlaceIrValue(torch::lazy::Value(node, i));
  }
  return torch::lazy::Value(node, inputs->size());
}

std::pair<torch::lazy::LazyTensor, torch::lazy::Value> all_to_all(
    const torch::lazy::LazyTensor& input, const torch::lazy::Value& token,
    int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
    std::vector<std::vector<int64_t>> groups) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::AllToAll>(
      input.GetIrValue(), token, split_dimension, concat_dimension, split_count,
      std::move(groups));
  return {torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 0), input.GetDevice()), torch::lazy::Value(node, 1)};
}

torch::lazy::LazyTensor get_dimensions_size(const torch::lazy::LazyTensor& input,
                               std::vector<int64_t> dimensions) {
  return torch::lazy::LazyTensor::Create(torch::lazy::MakeNode<ir::ops::GetDimensionsSize>(
      input.GetIrValue(), std::move(dimensions)), input.GetDevice());
}

std::pair<torch::lazy::LazyTensor, torch::lazy::Value> collective_permute(
    const torch::lazy::LazyTensor& input, const torch::lazy::Value& token,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs) {
  torch::lazy::NodePtr node = torch::lazy::MakeNode<ir::ops::CollectivePermute>(
      input.GetIrValue(), token, std::move(source_target_pairs));
  return {torch::lazy::LazyTensor::Create(torch::lazy::Value(node, 0), input.GetDevice()), torch::lazy::Value(node, 1)};
}

}  // namespace lazy_tensor_distributed
}  // namespace torch_lazy_tensors
