#include <torch/csrc/lazy/ts_backend/ts_node_lowering.h>

#include <ATen/Functions.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/ts_backend/ir_builder.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch {
namespace lazy {

TSOpVector LowerBuiltin(
    const torch::lazy::Node* node,
    std::shared_ptr<torch::jit::GraphFunction> function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  return LowerTSBuiltin(function, node->op().op, arguments, kwarguments);
}
TSOpVector LowerBuiltin(
    c10::Symbol sym,
    std::shared_ptr<torch::jit::GraphFunction> function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  return LowerTSBuiltin(function, sym, arguments, kwarguments);
}

TSOpVector LowerTSBuiltin(
    std::shared_ptr<torch::jit::GraphFunction> function,
    c10::Symbol sym,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments) {
  auto builtin =
      std::make_shared<torch::jit::BuiltinFunction>(sym, at::nullopt);
  auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
  auto ret = magic_method->call({}, *function, arguments, kwarguments, 0);
  auto& sv = dynamic_cast<torch::jit::SimpleValue&>(*ret);
  if (sv.getValue()->type()->kind() == c10::TypeKind::TupleType) {
    const auto tuple_call_result = sv.asTuple({}, *function);
    TSOpVector tuple_result;
    for (const auto& tuple_component : tuple_call_result) {
      auto tuple_component_sv =
          dynamic_cast<torch::jit::SimpleValue*>(tuple_component.get());
      tuple_result.push_back(tuple_component_sv->getValue());
    }
    return tuple_result;
  }
  return {sv.getValue()};
}

torch::jit::Value* GenerateClone(
    torch::jit::Value* val,
    std::shared_ptr<torch::jit::GraphFunction> function) {
  std::vector<torch::jit::NamedValue> clone_arguments;
  clone_arguments.emplace_back(val);
  TSOpVector cloned = LowerBuiltin(at::aten::clone, function, clone_arguments);
  TORCH_CHECK_EQ(cloned.size(), 1);
  return cloned.front();
}

void GenerateCopy(
    torch::jit::Value* destination,
    torch::jit::Value* source,
    std::shared_ptr<torch::jit::GraphFunction> function) {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(destination);
  arguments.emplace_back(source);
  LowerBuiltin(at::aten::copy_, function, arguments);
}

torch::jit::Value* GenerateSlice(
    torch::jit::Value* base,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step,
    std::shared_ptr<torch::jit::GraphFunction> function) {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(base);
  arguments.emplace_back(dim);
  arguments.emplace_back(start);
  arguments.emplace_back(end);
  arguments.emplace_back(step);
  TSOpVector selected = LowerBuiltin(at::aten::slice, function, arguments);
  TORCH_CHECK_EQ(selected.size(), 1);
  return selected.front();
}

// Node Lowerings

// Default node lowering
TSOpVector TsNode::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  for (const torch::lazy::Output& output : operands()) {
    arguments.emplace_back(loctx->GetOutputOp(output));
  }
  return LowerBuiltin(this, function, arguments);
}

// Non-native ops
torch::lazy::TSOpVector Cast::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(dtype);
  return LowerBuiltin(at::aten::to, function, arguments);
}

torch::lazy::TSOpVector DeviceData::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  auto infoptr = data_->info();
  auto deviceDataInfoPtr =
      (torch::lazy::LazyGraphExecutor::DeviceDataInfo*)infoptr;
  if (GRAPH_DUMP_ENABLED) {
    LOG(ERROR) << "Lowering device data node, tensor id "
               << deviceDataInfoPtr->tensor_id << std::endl;
  }
  return {loctx->GetParameter(data_)};
}

torch::lazy::TSOpVector Expand::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(size);
  auto expand_out = LowerBuiltin(this, function, arguments);
  if (is_scalar_expand) {
    // The aten::expand operations sets all strides to 0 when the original is
    // of rank 0. This leads to false positives when checking for internal
    // memory overlap, because at::has_internal_overlap returns
    // MemOverlap::YES when a stride is set to 0.
    TORCH_CHECK_EQ(expand_out.size(), 1);
    return {GenerateClone(expand_out.front(), function)};
  }
  return expand_out;
}

torch::lazy::TSOpVector Scalar::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  auto options =
      at::TensorOptions()
          .device(torch::lazy::getBackend()->EagerFallbackDeviceType())
          .dtype(shape().scalar_type());
  return {loctx->graph()->insertConstant(at::scalar_tensor(value, options))};
}

} // namespace lazy
} // namespace torch
