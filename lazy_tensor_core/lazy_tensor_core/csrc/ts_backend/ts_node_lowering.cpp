#include <torch/csrc/lazy/ts_backend/ts_node_lowering.h>

#include <ATen/Functions.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/ts_backend/ops/cast.h>
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>
#include <torch/csrc/lazy/ts_backend/ops/expand.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/ts_backend/ops/scalar.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/core/view_ops/as_strided.h>
#include <torch/csrc/lazy/core/view_ops/as_strided_view_update.h>
#include <torch/csrc/lazy/core/view_ops/narrow.h>
#include <torch/csrc/lazy/core/view_ops/narrow_view_update.h>
#include <torch/csrc/lazy/core/view_ops/permute.h>
#include <torch/csrc/lazy/core/view_ops/select.h>
#include <torch/csrc/lazy/core/view_ops/select_view_update.h>
#include <torch/csrc/lazy/core/view_ops/view.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>

#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"

namespace torch {
namespace lazy {

class TSNodeLowering : public TSNodeLoweringInterface {
 public:
  TSNodeLowering(const std::string& name, torch::lazy::TSLoweringContext* loctx)
      : loctx_(loctx),
        function_(loctx ? std::make_shared<torch::jit::GraphFunction>(
                              name, loctx->graph(), nullptr)
                        : nullptr) {}

  torch::lazy::TSLoweringContext* loctx() { return loctx_; }

  bool Lower(const torch::lazy::Node* node) override {
    if (auto* tsnode = dynamic_cast<const torch::lazy::TsNode*>(node)) {
      // First, we call the node lowering function, which exists for newly
      // codegenned or refactored nodes
      TSOpVector ops = tsnode->Lower(function_, loctx());
      if (ops.empty()) {
        // Then fall back to legacy lowering code, which should be gradually
        // removed
        ops = LowerNonCodegenOps(node);
      }
      if (ops.empty()) {
        return false;
      }
      CHECK_EQ(node->num_outputs(), ops.size());
      for (size_t i = 0; i < ops.size(); ++i) {
        loctx()->AssignOutputOp(torch::lazy::Output(node, i), ops[i]);
      }
      return true;
    }
    throw std::runtime_error(
        "Expected torch::lazy::TsNode but could not dynamic cast");
  }

  // TODO(whc) this is for legacy/non-codegen Ops, and after moving most ops
  // to codegen we should delete this and put all the lowering logic into Node
  // classes
  TSOpVector LowerNonCodegenOps(const torch::lazy::Node* node) {
    if (node->op().op == at::aten::as_strided) {
      return LowerAsStrided(torch::lazy::NodeCast<torch::lazy::AsStrided>(
          node, torch::lazy::OpKind(at::aten::as_strided)));
    }
    if (node->op() == *torch::lazy::ltc_as_strided_view_update) {
      return LowerAsStridedViewUpdate(
          torch::lazy::NodeCast<torch::lazy::AsStridedViewUpdate>(
              node, *torch::lazy::ltc_as_strided_view_update));
    }
    if (node->op() == *torch::lazy::ltc_cast) {
      return LowerCast(torch::lazy::NodeCast<torch::lazy::Cast>(
          node, *torch::lazy::ltc_cast));
    }
    if (node->op() == *torch::lazy::ltc_select_view_update) {
      return LowerSelectViewUpdate(
          torch::lazy::NodeCast<torch::lazy::SelectViewUpdate>(
              node, *torch::lazy::ltc_select_view_update));
    }
    if (node->op() == *torch::lazy::ltc_narrow_view_update) {
      return LowerNarrowViewUpdate(
          torch::lazy::NodeCast<torch::lazy::NarrowViewUpdate>(
              node, *torch::lazy::ltc_narrow_view_update));
    }
    if (node->op().op == at::prim::Constant) {
      return LowerScalar(torch::lazy::NodeCast<torch::lazy::Scalar>(
          node, torch::lazy::OpKind(at::prim::Constant)));
    }
    if (node->op().op == at::aten::bernoulli) {
      std::vector<torch::jit::NamedValue> arguments;
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
      return LowerBuiltin(node, arguments);
    }
    if (node->op().op == at::aten::native_batch_norm) {
      return LowerBatchNorm(
          torch::lazy::NodeCast<
              torch_lazy_tensors::ir::ops::TSNativeBatchNormForward>(
              node, torch::lazy::OpKind(at::aten::native_batch_norm)));
    }
    if (node->op().op == at::aten::native_batch_norm_backward) {
      return LowerBatchNormBackward(
          torch::lazy::NodeCast<
              torch_lazy_tensors::ir::ops::TSNativeBatchNormBackward>(
              node, torch::lazy::OpKind(at::aten::native_batch_norm_backward)));
    }
    if (node->op().op == at::aten::expand) {
      return LowerExpand(
          torch::lazy::NodeCast<torch::lazy::Expand>(
              node, torch::lazy::OpKind(at::aten::expand)));
    }
    if (node->op().op == at::aten::narrow) {
      return LowerNarrow(torch::lazy::NodeCast<torch::lazy::Narrow>(
          node, torch::lazy::OpKind(at::aten::narrow)));
    }
    if (node->op().op == at::aten::permute) {
      return LowerPermute(torch::lazy::NodeCast<torch::lazy::Permute>(
          node, torch::lazy::OpKind(at::aten::permute)));
    }
    if (node->op().op == at::aten::select) {
      return LowerSelect(torch::lazy::NodeCast<torch::lazy::Select>(
          node, torch::lazy::OpKind(at::aten::select)));
    }
    if (node->op().op == at::aten::squeeze) {
      return LowerSqueeze(
          torch::lazy::NodeCast<torch_lazy_tensors::ir::ops::Squeeze>(
              node, torch::lazy::OpKind(at::aten::squeeze)));
    }
    if (node->op().op == at::aten::stack) {
      return LowerStack(
          torch::lazy::NodeCast<torch_lazy_tensors::ir::ops::Stack>(
              node, torch::lazy::OpKind(at::aten::stack)));
    }
    if (node->op().op == at::aten::unsqueeze) {
      return LowerUnsqueeze(
          torch::lazy::NodeCast<torch_lazy_tensors::ir::ops::Unsqueeze>(
              node, torch::lazy::OpKind(at::aten::unsqueeze)));
    }
    if (node->op().op == at::aten::view) {
      return LowerView(torch::lazy::NodeCast<torch::lazy::View>(
          node, torch::lazy::OpKind(at::aten::view)));
    }
    if (node->op() == *torch::lazy::ltc_device_data) {
      const torch::lazy::DeviceData* device_data_node =
          torch::lazy::NodeCast<torch::lazy::DeviceData>(
              node, *torch::lazy::ltc_device_data);
      auto infoptr = device_data_node->data()->info();
      auto deviceDataInfoPtr = (torch::lazy::LazyGraphExecutor::DeviceDataInfo*) infoptr;
      if (GRAPH_DUMP_ENABLED) {
        LOG(ERROR) << "Lowering device data node, tensor id " << deviceDataInfoPtr->tensor_id << std::endl;
      }
      return {loctx()->GetParameter(device_data_node->data())};
    }

    std::vector<torch::jit::NamedValue> arguments;
    for (const torch::lazy::Output& output : node->operands()) {
      arguments.emplace_back(loctx()->GetOutputOp(output));
    }
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerBuiltin(
      const torch::lazy::Node* node,
      const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
    return LowerTSBuiltin(function_, node->op().op, arguments, kwarguments);
  }
  TSOpVector LowerBuiltin(
      c10::Symbol sym, const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
    return LowerTSBuiltin(function_, sym, arguments, kwarguments);
  }

  TSOpVector LowerAsStrided(const torch::lazy::AsStrided* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->size());
    arguments.emplace_back(node->stride());
    arguments.emplace_back(node->storage_offset());
    TSOpVector as_strided_out = LowerBuiltin(node, arguments);
    CHECK_EQ(as_strided_out.size(), 1);
    return {GenerateClone(as_strided_out.front())};
  }

  TSOpVector LowerAsStridedViewUpdate(
      const torch::lazy::AsStridedViewUpdate* node) {
    torch::jit::Value* destination =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    const torch::lazy::Output& input_op = node->operand(1);
    const torch::lazy::Shape& input_shape =
        torch::lazy::GetShapeFromTsOutput(input_op);
    const auto input_dimensions = input_shape.sizes();
    std::vector<torch::jit::NamedValue> dest_arguments;
    dest_arguments.emplace_back(destination);
    dest_arguments.emplace_back(
        std::vector<int64_t>(input_dimensions.begin(), input_dimensions.end()));
    dest_arguments.emplace_back(node->stride());
    dest_arguments.emplace_back(node->storage_offset());
    TSOpVector as_strided_out =
        LowerBuiltin(at::aten::as_strided, dest_arguments);
    CHECK_EQ(as_strided_out.size(), 1);
    torch::jit::Value* as_strided = as_strided_out.front();
    GenerateCopy(as_strided, loctx()->GetOutputOp(input_op));
    return {destination};
  }

  TSOpVector LowerBatchNorm(
      const torch_lazy_tensors::ir::ops::TSNativeBatchNormForward* node) {
    std::vector<torch::jit::NamedValue> arguments;
    for (size_t i = 0; i < 5; ++i) {
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(i)));
    }
    arguments.emplace_back(node->training());
    arguments.emplace_back(node->momentum());
    arguments.emplace_back(node->eps());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerBatchNormBackward(
      const torch_lazy_tensors::ir::ops::TSNativeBatchNormBackward* node) {
    std::vector<torch::jit::NamedValue> arguments;
    for (size_t i = 0; i < 3; ++i) {
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(i)));
    }
    const auto& operands = node->operands();
    c10::optional<at::Tensor> null_arg;
    if (operands.size() == 5) {
      arguments.emplace_back(null_arg);
      arguments.emplace_back(null_arg);
    }
    for (size_t i = 3; i < operands.size(); ++i) {
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(i)));
    }
    arguments.emplace_back(node->training());
    arguments.emplace_back(node->eps());
    arguments.emplace_back(node->output_mask());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerCast(const torch::lazy::Cast* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->dtype());
    return LowerBuiltin(at::aten::to, arguments);
  }

  TSOpVector LowerExpand(const torch::lazy::Expand* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->size());
    auto expand_out = LowerBuiltin(node, arguments);
    if (node->is_scalar_expand()) {
      // The aten::expand operations sets all strides to 0 when the original is
      // of rank 0. This leads to false positives when checking for internal
      // memory overlap, because at::has_internal_overlap returns
      // MemOverlap::YES when a stride is set to 0.
      CHECK_EQ(expand_out.size(), 1);
      return {GenerateClone(expand_out.front())};
    }
    return expand_out;
  }

  TSOpVector LowerNarrow(const torch::lazy::Narrow* node) {
    const torch::lazy::Output& input = node->operand(0);
    torch::jit::Value* base = loctx()->GetOutputOp(input);
    const auto& base_indices = node->base_indices();
    const auto& sizes = node->sizes();
    const torch::lazy::Shape& input_shape =
        torch::lazy::GetShapeFromTsOutput(input);
    CHECK_EQ(sizes.size(), base_indices.size());
    CHECK_EQ(input_shape.dim(), base_indices.size());
    for (size_t dim = 0; dim < base_indices.size(); ++dim) {
      int64_t start = base_indices[dim];
      base = GenerateSlice(/*base=*/base, /*dim=*/dim, /*start=*/start,
                           /*end=*/start + sizes[dim], /*step=*/1);
    }
    return {base};
  }

  TSOpVector LowerPermute(const torch::lazy::Permute* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->dims());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerScalar(const torch::lazy::Scalar* node) {
    const at::Scalar& value = node->value();
    const torch::lazy::Shape& shape = node->shape();
    auto options =
        at::TensorOptions()
            .device(torch::lazy::getBackend()->EagerFallbackDeviceType())
            .dtype(shape.scalar_type());
    return {
        loctx()->graph()->insertConstant(at::scalar_tensor(value, options))};
  }

  TSOpVector LowerSelect(const torch::lazy::Select* node) {
    int64_t step = torch::lazy::Select::GetStride(node->start(), node->end(),
                                                  node->stride());
    torch::jit::Value* base = loctx()->GetOutputOp(node->operand(0));
    return {GenerateSlice(/*base=*/base, /*dim=*/node->dim(),
                          /*start=*/node->start(), /*end=*/node->end(),
                          /*step=*/step)};
  }

  TSOpVector LowerSqueeze(const torch_lazy_tensors::ir::ops::Squeeze* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    if (node->dim() != -1) {
      arguments.push_back(node->dim());
    }
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerStack(const torch_lazy_tensors::ir::ops::Stack* stack) {
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::Value*> tensor_list;
    const auto& operands = stack->operands();
    CHECK(operands.size() == 1);
    auto graph = function_->graph();
    arguments.emplace_back(loctx()->GetOutputOp(operands.back()));
    arguments.emplace_back(stack->dim());
    return LowerBuiltin(stack, arguments);
  }

  TSOpVector LowerSelectViewUpdate(const torch::lazy::SelectViewUpdate* node) {
    torch::jit::Value* dest =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    int64_t step = torch::lazy::Select::GetStride(node->start(), node->end(),
                                                  node->stride());
    torch::jit::Value* selected = GenerateSlice(
        /*base=*/dest, /*dim=*/node->dim(), /*start=*/node->start(),
        /*end=*/node->end(), /*step=*/step);
    GenerateCopy(selected, loctx()->GetOutputOp(node->operand(1)));
    return {dest};
  }

  TSOpVector LowerNarrowViewUpdate(const torch::lazy::NarrowViewUpdate* node) {
    torch::jit::Value* dest =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    const auto& base_indices = node->base_indices();
    const torch::lazy::Output& source_argument = node->operand(1);
    const torch::lazy::Shape& source_shape =
        torch::lazy::GetShapeFromTsOutput(source_argument);
    CHECK_EQ(source_shape.dim(), base_indices.size());
    torch::jit::Value* base = dest;
    for (size_t dim = 0; dim < base_indices.size(); ++dim) {
      int64_t start = base_indices[dim];
      base = GenerateSlice(/*base=*/base, /*dim=*/dim, /*start=*/start,
                           /*end=*/start + source_shape.size(dim),
                           /*step=*/1);
    }
    GenerateCopy(base, loctx()->GetOutputOp(source_argument));
    return {dest};
  }

  TSOpVector LowerUnsqueeze(
      const torch_lazy_tensors::ir::ops::Unsqueeze* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->dim());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerView(const torch::lazy::View* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->output_size());
    return LowerBuiltin(at::aten::reshape, arguments);
  }

  torch::jit::Value* GenerateClone(torch::jit::Value* val) {
    std::vector<torch::jit::NamedValue> clone_arguments;
    clone_arguments.emplace_back(val);
    TSOpVector cloned = LowerBuiltin(at::aten::clone, clone_arguments);
    CHECK_EQ(cloned.size(), 1);
    return cloned.front();
  }

  void GenerateCopy(torch::jit::Value* destination, torch::jit::Value* source) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(destination);
    arguments.emplace_back(source);
    LowerBuiltin(at::aten::copy_, arguments);
  }

  torch::jit::Value* GenerateSlice(torch::jit::Value* base, int64_t dim,
                                   int64_t start, int64_t end, int64_t step) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(base);
    arguments.emplace_back(dim);
    arguments.emplace_back(start);
    arguments.emplace_back(end);
    arguments.emplace_back(step);
    TSOpVector selected = LowerBuiltin(at::aten::slice, arguments);
    CHECK_EQ(selected.size(), 1);
    return selected.front();
  }
  torch::lazy::TSLoweringContext* loctx_;
  std::shared_ptr<torch::jit::GraphFunction> function_;
};

std::unique_ptr<TSNodeLoweringInterface> TSNodeLoweringInterface::Create(
    torch::lazy::LoweringContext* loctx) {
  return std::make_unique<TSNodeLowering>(
      "TSNodeLowering", static_cast<torch::lazy::TSLoweringContext*>(loctx));
}

TSOpVector LowerTSBuiltin(
    std::shared_ptr<torch::jit::GraphFunction> function, c10::Symbol sym,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments) {
  auto builtin =
      std::make_shared<torch::jit::BuiltinFunction>(sym, at::nullopt);
  auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
  auto ret = magic_method->call({}, *function, arguments, kwarguments, 0);
  auto sv = dynamic_cast<torch::jit::SimpleValue*>(ret.get());
  CHECK(sv);
  if (sv->getValue()->type()->kind() == c10::TypeKind::TupleType) {
    const auto tuple_call_result = sv->asTuple({}, *function);
    TSOpVector tuple_result;
    for (const auto& tuple_component : tuple_call_result) {
      auto tuple_component_sv =
          dynamic_cast<torch::jit::SimpleValue*>(tuple_component.get());
      tuple_result.push_back(tuple_component_sv->getValue());
    }
    return tuple_result;
  }
  return {sv->getValue()};
}

}  // namespace lazy
}  // namespace torch
