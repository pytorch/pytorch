#include <ATen/Config.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

// Check whether accumu is used by add's post ops.
// if accumu is used by the add's post ops, we can't write it.
auto accumu_use_check = [](const torch::jit::Node* add_node,
                           const torch::jit::Value* accumu_value) {
  bool accumu_same_used = false;
  auto accumu_uses = accumu_value->uses();
  std::for_each(
      accumu_uses.begin(), accumu_uses.end(), [&](torch::jit::Use& u) {
        // if one user is the after nodes of add. we can't write accumu.
        if (u.user != add_node && !u.user->isBefore(add_node)) {
          accumu_same_used = true;
        }
      });
  return accumu_same_used;
};

// Check add operator's alpha is one.
bool alpha_is_one_value(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  bool alpha_is_one = false;
  if (vmap.find("alpha") != vmap.end()) {
    auto alpha = toIValue(match.values_map.at(vmap.at("alpha")));
    if (alpha.has_value() &&
        ((alpha.value().isDouble() && alpha.value().toDouble() == 1.0) ||
         (alpha.value().isInt() &&
          static_cast<int>(alpha.value().toInt()) == 1))) {
      alpha_is_one = true;
    }
  }
  return alpha_is_one;
}

template <bool accumu_is_right = true>
bool add_accumu_can_be_write(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  int accumu_index = accumu_is_right ? 1 : 0;
  auto add_node = match.values_map.at(vmap.at("res"))->node();
  auto conv_res = add_node->inputs().at(1 - accumu_index);
  auto accumu = add_node->inputs().at(accumu_index);
  auto graph = add_node->owningGraph();
  // If accumu is graph's input or ouput, we can't write the accumu.
  if (std::count(graph->inputs().begin(), graph->inputs().end(), accumu) > 0 ||
      std::count(graph->outputs().begin(), graph->outputs().end(), accumu) >
          0) {
    return false;
  }
  bool accumu_same_used = accumu_use_check(add_node, accumu);
  // If accumu is used by other ops after add_node, or is a constant,
  // or add_node's inputs have one none-tensor, we can't write the accumu.
  if (accumu_same_used ||
      accumu->node()->kind() == torch::jit::prim::Constant ||
      !accumu->type()->cast<torch::jit::TensorType>() ||
      !conv_res->type()->cast<TensorType>()) {
    return false;
  }

  auto conv_res_size_option =
      conv_res->type()->cast<TensorType>()->sizes().concrete_sizes();

  auto accumu_size_option =
      accumu->type()->cast<TensorType>()->sizes().concrete_sizes();
  if (!conv_res_size_option.has_value() || !accumu_size_option.has_value()) {
    return false;
  }

  auto conv_res_size_value = conv_res_size_option.value();
  auto accumu_size_value = accumu_size_option.value();

  auto conv_res_stride_option =
      conv_res->type()->cast<TensorType>()->strides().concrete_sizes();

  auto accumu_stride_option =
      accumu->type()->cast<TensorType>()->strides().concrete_sizes();
  if (!conv_res_stride_option.has_value() ||
      !accumu_stride_option.has_value()) {
    return false;
  }

  auto conv_res_stride_value = conv_res_stride_option.value();
  auto accumu_stride_value = accumu_stride_option.value();

  auto conv_res_dtype_option =
      conv_res->type()->cast<TensorType>()->scalarType();
  auto accumu_dtype_option = accumu->type()->cast<TensorType>()->scalarType();
  if (!conv_res_dtype_option || !accumu_dtype_option) {
    return false;
  }
  auto conv_res_device_option = conv_res->type()->cast<TensorType>()->device();
  auto accumu_device_option = accumu->type()->cast<TensorType>()->device();
  if (!conv_res_device_option || !accumu_device_option) {
    return false;
  }
  // Make sue add_node' input has same size, stride, dtype and device.
  if (conv_res_size_value.empty() || accumu_size_value.empty() ||
      conv_res_size_value != accumu_size_value ||
      conv_res_stride_value.empty() || accumu_stride_value.empty() ||
      conv_res_stride_value != accumu_stride_value ||
      conv_res_dtype_option.value() != accumu_dtype_option.value() ||
      conv_res_device_option.value() != accumu_device_option.value()) {
    return false;
  }
  // Fro accumu = accumu + alpha*conv_output, alpha should be one.
  if (!accumu_is_right && !alpha_is_one_value(match, vmap)) {
    return false;
  }
  return true;
}

bool add_accumu_on_right(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  return add_accumu_can_be_write<true>(match, vmap);
}

bool add_accumu_on_left(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  return add_accumu_can_be_write<false>(match, vmap);
}

} // namespace jit
} // namespace torch
