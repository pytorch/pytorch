#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

#include <ATen/core/Reduction.h>
#include <ATen/native/ConvUtils.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/jit.h>

#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/generic_slice.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/repeat.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/update_slice.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {

class TSNodeLowering : public TSNodeLoweringInterface {
 public:
  TSNodeLowering(const std::string& name, ts_backend::TSLoweringContext* loctx)
      : loctx_(loctx),
        function_(loctx ? std::make_shared<torch::jit::GraphFunction>(
                              name, loctx->graph(), nullptr)
                        : nullptr) {}

  ts_backend::TSLoweringContext* loctx() { return loctx_; }

  bool Lower(const torch::lazy::Node* node) override {
    if (auto* tsnode =
            dynamic_cast<const torch_lazy_tensors::ir::TsNode*>(node)) {
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
    throw std::runtime_error("Expected TsNode but could not dynamic cast");
  }

  // TODO(whc) this is for legacy/non-codegen Ops, and after moving most ops
  // to codegen we should delete this and put all the lowering logic into Node
  // classes
  TSOpVector LowerNonCodegenOps(const torch::lazy::Node* node) {
    if (node->op().op == at::aten::as_strided) {
      return LowerAsStrided(torch::lazy::NodeCast<ir::ops::AsStrided>(
          node, ir::OpKind(at::aten::as_strided)));
    }
    if (node->op() == *ir::ops::ltc_as_strided_view_update) {
      return LowerAsStridedViewUpdate(
          torch::lazy::NodeCast<ir::ops::AsStridedViewUpdate>(
              node, *ir::ops::ltc_as_strided_view_update));
    }
    if (node->op() == *ir::ops::ltc_cast) {
      return LowerCast(
          torch::lazy::NodeCast<ir::ops::Cast>(node, *ir::ops::ltc_cast));
    }
    if (node->op() == *ir::ops::ltc_generic_slice) {
      return LowerGenericSlice(torch::lazy::NodeCast<ir::ops::GenericSlice>(
          node, *ir::ops::ltc_generic_slice));
    }
    if (node->op() == *ir::ops::ltc_select) {
      return LowerSelect(
          torch::lazy::NodeCast<ir::ops::Select>(node, *ir::ops::ltc_select));
    }
    if (node->op() == *ir::ops::ltc_unselect) {
      return LowerUnselect(torch::lazy::NodeCast<ir::ops::Unselect>(
          node, *ir::ops::ltc_unselect));
    }
    if (node->op() == *ir::ops::ltc_update_slice) {
      return LowerUpdateSlice(torch::lazy::NodeCast<ir::ops::UpdateSlice>(
          node, *ir::ops::ltc_update_slice));
    }
    if (node->op().op == at::prim::Constant) {
      auto scalar_node = dynamic_cast<const ir::ops::Scalar*>(node);
      if (scalar_node) {
        return LowerScalar(scalar_node);
      }
      return LowerConstant(torch::lazy::NodeCast<ir::ops::Constant>(
          node, ir::OpKind(at::prim::Constant)));
    }
    if (node->op().op == at::aten::bernoulli) {
      std::vector<torch::jit::NamedValue> arguments;
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
      return LowerBuiltin(node, arguments);
    }
    if (node->op().op == at::aten::convolution_backward_overrideable) {
      return LowerConvolutionBackwardOverrideable(
          torch::lazy::NodeCast<ir::ops::ConvolutionBackwardOverrideable>(
              node, ir::OpKind(at::aten::convolution_backward_overrideable)));
    }
    if (node->op().op == at::aten::convolution_overrideable) {
      return LowerConvolutionOverrideable(
          torch::lazy::NodeCast<ir::ops::ConvolutionOverrideable>(
              node, ir::OpKind(at::aten::convolution_overrideable)));
    }
    if (node->op().op == at::aten::native_batch_norm) {
      return LowerBatchNorm(
          torch::lazy::NodeCast<ir::ops::TSNativeBatchNormForward>(
              node, ir::OpKind(at::aten::native_batch_norm)));
    }
    if (node->op().op == at::aten::native_batch_norm_backward) {
      return LowerBatchNormBackward(
          torch::lazy::NodeCast<ir::ops::TSNativeBatchNormBackward>(
              node, ir::OpKind(at::aten::native_batch_norm_backward)));
    }
    if (node->op().op == at::aten::constant_pad_nd) {
      return LowerConstantPad(torch::lazy::NodeCast<ir::ops::ConstantPadNd>(
          node, ir::OpKind(at::aten::constant_pad_nd)));
    }
    if (node->op().op == at::aten::expand) {
      return LowerExpand(torch::lazy::NodeCast<ir::ops::Expand>(
          node, ir::OpKind(at::aten::expand)));
    }
    if (node->op().op == at::aten::permute) {
      return LowerPermute(torch::lazy::NodeCast<ir::ops::Permute>(
          node, ir::OpKind(at::aten::permute)));
    }
    if (node->op().op == at::aten::repeat) {
      return LowerRepeat(torch::lazy::NodeCast<ir::ops::Repeat>(
          node, ir::OpKind(at::aten::repeat)));
    }
    if (node->op().op == at::aten::squeeze) {
      return LowerSqueeze(torch::lazy::NodeCast<ir::ops::Squeeze>(
          node, ir::OpKind(at::aten::squeeze)));
    }
    if (node->op().op == at::aten::stack) {
      return LowerStack(torch::lazy::NodeCast<ir::ops::Stack>(
          node, ir::OpKind(at::aten::stack)));
    }
    if (node->op().op == at::aten::unsqueeze) {
      return LowerUnsqueeze(torch::lazy::NodeCast<ir::ops::Unsqueeze>(
          node, ir::OpKind(at::aten::unsqueeze)));
    }
    if (node->op().op == at::aten::view) {
      return LowerView(torch::lazy::NodeCast<ir::ops::View>(
          node, ir::OpKind(at::aten::view)));
    }
    if (node->op() == *ir::ops::ltc_device_data) {
      const ir::ops::DeviceData* device_data_node =
          torch::lazy::NodeCast<ir::ops::DeviceData>(node,
                                                     *ir::ops::ltc_device_data);
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

  TSOpVector LowerAsStrided(const ir::ops::AsStrided* node) {
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
      const ir::ops::AsStridedViewUpdate* node) {
    torch::jit::Value* destination =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    const torch::lazy::Output& input_op = node->operand(1);
    const torch::lazy::Shape& input_shape = ir::GetShapeFromTsOutput(input_op);
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

  TSOpVector LowerBatchNorm(const ir::ops::TSNativeBatchNormForward* node) {
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
      const ir::ops::TSNativeBatchNormBackward* node) {
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

  TSOpVector LowerCast(const ir::ops::Cast* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->dtype());
    return LowerBuiltin(at::aten::to, arguments);
  }

  TSOpVector LowerConvolutionBackwardOverrideable(
      const ir::ops::ConvolutionBackwardOverrideable* conv) {
    const auto& operands = conv->operands();
    CHECK(!operands.empty());

    std::vector<torch::jit::NamedValue> arguments;

    // TODO: Clean up after convolution unification is done.
    auto& ctx = at::globalContext();
    CHECK(ctx.userEnabledCuDNN() &&
          compiler::getBackend()
                  ->HardwareDeviceType() == at::kCUDA);

    // See cudnn_convolution_backward/cudnn_convolution_transpose_backward in
    // native_functions.yaml
    arguments.emplace_back(loctx()->GetOutputOp(operands[1]));
    arguments.emplace_back(loctx()->GetOutputOp(operands[0]));
    arguments.emplace_back(loctx()->GetOutputOp(operands[2]));

    arguments.emplace_back(conv->padding());
    if (conv->transposed()) {
      arguments.emplace_back(conv->output_padding());
    }
    arguments.emplace_back(conv->stride());
    arguments.emplace_back(conv->dilation());
    arguments.emplace_back(conv->groups());
    arguments.emplace_back(ctx.benchmarkCuDNN());  // benchmark
    arguments.emplace_back(ctx.deterministicCuDNN() ||
                           ctx.deterministicAlgorithms());  // deterministic
    arguments.emplace_back(ctx.allowTF32CuDNN());           // allow_tf3
    std::array<bool, 2> output_mask = {conv->output_mask()[0],
                                       conv->output_mask()[1]};
    arguments.emplace_back(output_mask);

    auto result =
        conv->transposed()
            ? LowerBuiltin(at::aten::cudnn_convolution_transpose_backward,
                           arguments)
            : LowerBuiltin(at::aten::cudnn_convolution_backward, arguments);
    // cudnn_convolution_backward/cudnn_convolution_transpose_backward only
    // returns 2 tensors
    result.push_back(nullptr);
    return result;
  }

  TSOpVector LowerConvolutionOverrideable(
      const ir::ops::ConvolutionOverrideable* conv) {
    constexpr size_t kBiasOperandsOffset = 2;
    const auto& operands = conv->operands();
    CHECK(!operands.empty());

    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(operands[0]));
    arguments.emplace_back(loctx()->GetOutputOp(operands[1]));
    // bias is optional
    c10::optional<at::Tensor> nullArg;
    if (operands.size() <= kBiasOperandsOffset) {
      arguments.emplace_back(nullArg);
    } else {
      arguments.emplace_back(
          loctx()->GetOutputOp(operands[kBiasOperandsOffset]));
    }
    arguments.emplace_back(conv->stride());
    arguments.emplace_back(conv->padding());
    arguments.emplace_back(conv->dilation());
    arguments.emplace_back(conv->transposed());
    arguments.emplace_back(conv->output_padding());
    arguments.emplace_back(conv->groups());
    // TODO: backend information is exposed too early here.
    // Clean up after convolution unification is done.
    auto& ctx = at::globalContext();
    arguments.emplace_back(ctx.benchmarkCuDNN());  // benchmark
    arguments.emplace_back(ctx.deterministicCuDNN() ||
                           ctx.deterministicAlgorithms());  // deterministic
    arguments.emplace_back(ctx.userEnabledCuDNN());         // cudnn_enabled
    arguments.emplace_back(ctx.allowTF32CuDNN());           // allow_tf32

    // Invoke aten::_convolution instead of aten::convolution_overrideable
    return LowerBuiltin(at::aten::_convolution, arguments);
  }

  TSOpVector LowerConstant(const ir::ops::Constant* node) {
    at::Tensor value = node->value().value();
    if (compiler::getBackend()
            ->HardwareDeviceType() == at::kCUDA) {
      value = value.cuda();
    }
    return {loctx()->graph()->insertConstant(value)};
  }

  TSOpVector LowerConstantPad(const ir::ops::ConstantPadNd* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->pad());
    arguments.emplace_back(node->value());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerExpand(const ir::ops::Expand* node) {
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

  TSOpVector LowerGenericSlice(const ir::ops::GenericSlice* node) {
    const torch::lazy::Output& input = node->operand(0);
    torch::jit::Value* base = loctx()->GetOutputOp(input);
    const auto& base_indices = node->base_indices();
    const auto& sizes = node->sizes();
    const torch::lazy::Shape& input_shape = ir::GetShapeFromTsOutput(input);
    CHECK_EQ(sizes.size(), base_indices.size());
    CHECK_EQ(input_shape.dim(), base_indices.size());
    for (size_t dim = 0; dim < base_indices.size(); ++dim) {
      int64_t start = base_indices[dim];
      base = GenerateSlice(/*base=*/base, /*dim=*/dim, /*start=*/start,
                           /*end=*/start + sizes[dim], /*step=*/1);
    }
    return {base};
  }

  TSOpVector LowerPermute(const ir::ops::Permute* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->dims());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerRepeat(const ir::ops::Repeat* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->repeats());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerScalar(const ir::ops::Scalar* node) {
    const at::Scalar& value = node->value();
    const torch::lazy::Shape& shape = node->shape();
    auto options =
        at::TensorOptions()
            .device(compiler::getBackend()
                        ->HardwareDeviceType())
            .dtype(shape.scalar_type());
    return {
        loctx()->graph()->insertConstant(at::scalar_tensor(value, options))};
  }

  TSOpVector LowerSelect(const ir::ops::Select* node) {
    int64_t step =
        ir::ops::Select::GetStride(node->start(), node->end(), node->stride());
    torch::jit::Value* base = loctx()->GetOutputOp(node->operand(0));
    return {GenerateSlice(/*base=*/base, /*dim=*/node->dim(),
                          /*start=*/node->start(), /*end=*/node->end(),
                          /*step=*/step)};
  }

  TSOpVector LowerSqueeze(const ir::ops::Squeeze* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    if (node->dim() != -1) {
      arguments.push_back(node->dim());
    }
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerStack(const ir::ops::Stack* stack) {
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::Value*> tensor_list;
    const auto& operands = stack->operands();
    CHECK(!operands.empty());
    for (const torch::lazy::Output& operand : operands) {
      tensor_list.emplace_back(loctx()->GetOutputOp(operand));
    }
    auto graph = function_->graph();
    arguments.emplace_back(
        graph
            ->insertNode(graph->createList(tensor_list[0]->type(), tensor_list))
            ->output());
    arguments.emplace_back(stack->dim());
    return LowerBuiltin(stack, arguments);
  }

  TSOpVector LowerUnselect(const ir::ops::Unselect* node) {
    torch::jit::Value* dest =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    int64_t step =
        ir::ops::Select::GetStride(node->start(), node->end(), node->stride());
    torch::jit::Value* selected = GenerateSlice(
        /*base=*/dest, /*dim=*/node->dim(), /*start=*/node->start(),
        /*end=*/node->end(), /*step=*/step);
    GenerateCopy(selected, loctx()->GetOutputOp(node->operand(1)));
    return {dest};
  }

  TSOpVector LowerUpdateSlice(const ir::ops::UpdateSlice* node) {
    torch::jit::Value* dest =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    const auto& base_indices = node->base_indices();
    const torch::lazy::Output& source_argument = node->operand(1);
    const torch::lazy::Shape& source_shape =
        ir::GetShapeFromTsOutput(source_argument);
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

  TSOpVector LowerUnsqueeze(const ir::ops::Unsqueeze* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->dim());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerView(const ir::ops::View* node) {
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
  ts_backend::TSLoweringContext* loctx_;
  std::shared_ptr<torch::jit::GraphFunction> function_;
};

std::unique_ptr<TSNodeLoweringInterface> TSNodeLoweringInterface::Create(
    ir::LoweringContext* loctx) {
  return std::make_unique<TSNodeLowering>(
      "TSNodeLowering", static_cast<ts_backend::TSLoweringContext*>(loctx));
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

}  // namespace compiler
}  // namespace torch_lazy_tensors
