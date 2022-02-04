#include "torch/csrc/lazy/ts_backend/ops/optim.h"

#include <tuple>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/util.h>

#include "ATen/core/ivalue.h"
#include "ATen/core/jit_type.h"
#include "MethodOperators.h"
#include "c10/util/irange.h"
#include "core/TensorBody.h"
#include "lazy/core/ir.h"
#include "lazy/core/tensor.h"
#include <ATen/core/stack.h>

namespace torch {
namespace lazy {

namespace {

std::tuple<std::vector<at::Tensor>, std::vector<int64_t>> flatten_with_positions(
    std::vector<std::vector<at::Tensor>> tensor_lists) {

  std::vector<int64_t> list_sizes;
  list_sizes.reserve(tensor_lists.size());
  size_t i = 0;
  std::vector<at::Tensor> flattened_lists;
  
  for (auto i : c10::irange(tensor_lists.size())) {
    const std::vector<at::Tensor>& list = tensor_lists[i];
    // TODO: split into another loop with `emplace_back` if this is too inefficient
    flattened_lists.insert(flattened_lists.end(), list.begin(), list.end());
    list_sizes[i] = list.size();
  }

  return std::make_tuple(flattened_lists, list_sizes);
}

// TODO: move to CUDA extensions
RegisterOperators reg({
    torch::jit::Operator(
    "LazyCUDA::L2NormMpCuda(Tensor[] list, int[] sizes, int chunk_size, bool? per_tensor_python) -> Tensor, Tensor",
    [](torch::jit::Stack& stack) {
      RECORD_FUNCTION("_grad_sum_to_size", std::vector<c10::IValue>());
      at::TensorList tl; 
      std::vector<int64_t> sizes;
      int chunk_size;
      c10::IValue ptp;
      torch::jit::pop(stack, tl, sizes, chunk_size, ptp);
      // TODO: run the cuda code
      torch::jit::push(stack, at::Tensor{}, at::Tensor{});
    },
    c10::AliasAnalysisKind::FROM_SCHEMA),
});

}

std::tuple<at::Tensor, at::Tensor> buildL2NormMpCuda(int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python) {
  auto lists_w_pos = flatten_with_positions(tensor_lists);
  auto vals = c10::fmap(std::get<0>(lists_w_pos), [](const at::Tensor& t) { return torch::lazy::GetLtcTensor(t).GetIrValue(); });
  auto flag_lt = torch::lazy::GetLtcTensor(noop_flag);
  vals.push_back(flag_lt.GetIrValue());

  auto l2node = std::make_shared<L2NormMpCuda>(vals, std::get<1>(lists_w_pos), chunk_size, per_tensor_python);
  // TODO: these need better names, but I don't know what they mean.
  auto node1 = torch::lazy::CreateAtenFromLtcTensor(LazyTensor::Create(torch::lazy::Value(l2node, 0), flag_lt.GetDevice()));
  auto node2 = torch::lazy::CreateAtenFromLtcTensor(LazyTensor::Create(torch::lazy::Value(l2node, 1), flag_lt.GetDevice()));
  return std::make_tuple(node1, node2);
}

L2NormMpCuda::L2NormMpCuda(OpList tensor_args, const std::vector<int64_t>& list_sizes, int chunk_size, c10::optional<bool> per_tensor_python)
    : torch::lazy::TsNode(
          torch::lazy::OpKind(
              c10::Symbol::fromQualString("LazyCUDA::L2NormMpCuda")),
          tensor_args,
          // TODO: would be nice to have shape inference for this
          {Shape{}},
          2,
          torch::lazy::MHash(
              chunk_size,
              list_sizes,
              per_tensor_python)),
      list_sizes_(list_sizes),
      chunk_size_(chunk_size),
      per_tensor_python_(per_tensor_python) {}

torch::lazy::TSOpVector L2NormMpCuda::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;

  std::vector<torch::jit::Value*> ten_vals;
  ten_vals.reserve(operands().size() - 1);
  for (auto i : c10::irange(operands().size() - 1)) {
    ten_vals.emplace_back(loctx->GetOutputOp(operand(i)));
  }
  auto noop_flag = loctx->GetOutputOp(operands().back());

  auto tensor_list = loctx->graph()->insertNode(loctx->graph()->createList(at::TensorType::get(), ten_vals))->output();
  auto chunk_size = loctx->graph()->insertConstant(chunk_size_);
  auto pos_list = loctx->graph()->insertConstant(list_sizes_);

  auto per_tensor_python = (per_tensor_python_) ? 
    loctx->graph()->insertConstant(*per_tensor_python_) :
    loctx->graph()->insertConstant(c10::IValue{});
  per_tensor_python->setType(c10::OptionalType::create(c10::BoolType::get()));
  auto optim_node = loctx->graph()->create(op().op, {tensor_list, pos_list, chunk_size, per_tensor_python}, 2);
  loctx->graph()->insertNode(optim_node);

  return {optim_node->output(0), optim_node->output(1)};
}

std::string L2NormMpCuda::ToString() const {
  std::stringstream ss;
  ss << torch::lazy::TsNode::ToString();
  return ss.str();
}

} // namespace
} // namespace lazy
