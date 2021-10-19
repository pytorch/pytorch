#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

#include <ATen/core/Reduction.h>
#include <ATen/native/ConvUtils.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/jit.h>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/cat.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/convolution_backward_overrideable.h"
#include "lazy_tensor_core/csrc/ops/convolution_overrideable.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/generic_slice.h"
#include "lazy_tensor_core/csrc/ops/index_select.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu_backward.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/nll_loss_backward.h"
#include "lazy_tensor_core/csrc/ops/nll_loss_forward.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/repeat.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/stack.h"
#include "lazy_tensor_core/csrc/ops/sum.h"
#include "lazy_tensor_core/csrc/ops/threshold.h"
#include "lazy_tensor_core/csrc/ops/threshold_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_embedding_dense_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_backward.h"
#include "lazy_tensor_core/csrc/ops/ts_native_batch_norm_forward.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/update_slice.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_computation_client.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_lowering_context.h"
#include "lazy_tensors/permutation_util.h"

namespace torch_lazy_tensors {
namespace compiler {


class TSNodeLowering : public torch_lazy_tensors::compiler::TSNodeLoweringInterface {
 public:
  TSNodeLowering(const std::string& name, ts_backend::TSLoweringContext* loctx)
      : TSNodeLoweringInterface(loctx),
        function_(loctx ? std::make_shared<torch::jit::GraphFunction>(
                              name, loctx->graph(), nullptr)
                        : nullptr) {}

  bool Lower(const torch::lazy::Node* node) override {
    if (auto* tsnode = dynamic_cast<const torch_lazy_tensors::ir::TsNode*>(node)) {
      TSOpVector ops = tsnode->Lower(*this, function_, loctx());
      if (ops.empty()) {
        return false;
      }
      LTC_CHECK_EQ(node->num_outputs(), ops.size());
      for (size_t i = 0; i < ops.size(); ++i) {
        loctx()->AssignOutputOp(torch::lazy::Output(node, i), ops[i]);
      }
      return true;
    }
    throw std::runtime_error("Expected TsNode but could not dynamic cast");
  }

  lazy_tensors::Shape Infer(const torch::lazy::Node* node) override {
    if (node->op() == *ir::ops::ltc_generic_slice) {
      auto generic_slice = torch::lazy::NodeCast<ir::ops::GenericSlice>(
          node, *ir::ops::ltc_generic_slice);
      const torch::lazy::Output& argument = node->operand(0);
      return lazy_tensors::Shape(ir::GetShapeFromTsOutput(argument).element_type(),
                                 generic_slice->sizes());
    }
    if (node->op() == *ir::ops::ltc_update_slice) {
      const torch::lazy::Output& argument = node->operand(0);
      return ir::GetShapeFromTsOutput(argument);
    }
    switch (node->op().op) {
      case at::aten::expand: {
        auto expand =
            torch::lazy::NodeCast<ir::ops::Expand>(node, ir::OpKind(at::aten::expand));
        const torch::lazy::Output& argument = node->operand(0);
        return lazy_tensors::Shape(ir::GetShapeFromTsOutput(argument).element_type(),
                                   expand->size());
      }
      case at::aten::embedding_dense_backward: {
        return InferEmbeddingDenseBackward(
            torch::lazy::NodeCast<ir::ops::TSEmbeddingDenseBackward>(
                node, ir::OpKind(at::aten::embedding_dense_backward)));
      }
      case at::aten::index_select: {
        return InferIndexSelect(torch::lazy::NodeCast<ir::ops::IndexSelect>(
            node, ir::OpKind(at::aten::index_select)));
      }
      case at::aten::matmul: {
        // Only used from bmm currently.
        return InferBmm(node);
      }
      case at::aten::cat: {
        return InferCat(
            torch::lazy::NodeCast<ir::ops::Cat>(node, ir::OpKind(at::aten::cat)));
      }
      case at::aten::convolution_backward_overrideable: {
        return InferConvolutionBackwardOverrideable(
            torch::lazy::NodeCast<ir::ops::ConvolutionBackwardOverrideable>(
                node, ir::OpKind(at::aten::convolution_backward_overrideable)));
      }
      case at::aten::convolution_overrideable: {
        return InferConvolutionOverrideable(
            torch::lazy::NodeCast<ir::ops::ConvolutionOverrideable>(
                node, ir::OpKind(at::aten::convolution_overrideable)));
      }
      case at::aten::addmm:
      case at::aten::mm: {
        return InferMm(node);
      }
      case at::aten::leaky_relu_backward:
      case at::aten::nll_loss_backward: {
        const torch::lazy::Output& input = node->operand(1);
        return ir::GetShapeFromTsOutput(input);
      }
      case at::aten::native_batch_norm: {
        return InferBatchNorm(node);
      }
      case at::aten::native_batch_norm_backward: {
        return InferBatchNormBackward(node);
      }
      case at::aten::nll_loss_forward: {
        return InferNllLossForward(torch::lazy::NodeCast<ir::ops::NllLossForward>(
              node, ir::OpKind(at::aten::nll_loss_forward)));
      }
      case at::aten::permute: {
        auto permute =
            torch::lazy::NodeCast<ir::ops::Permute>(node, ir::OpKind(at::aten::permute));
        const torch::lazy::Output& argument = node->operand(0);
        return ir::ops::Permute::MakePermuteShape(ir::GetShapeFromTsOutput(argument),
                                                  permute->dims());
      }
      // activation and unary op do not change shape
      case at::aten::leaky_relu:
      case at::aten::pow:
      case at::aten::relu:
      case at::aten::relu_:
      case at::aten::sqrt: {
        const torch::lazy::Output& argument = node->operand(0);
        return ir::GetShapeFromTsOutput(argument);
      }
      case at::aten::repeat: {
        return InferRepeat(
            torch::lazy::NodeCast<ir::ops::Repeat>(node, ir::OpKind(at::aten::repeat)));
      }
      case at::aten::squeeze: {
        return InferSqueeze(torch::lazy::NodeCast<ir::ops::Squeeze>(
            node, ir::OpKind(at::aten::squeeze)));
      }
      case at::aten::stack: {
        return InferStack(
            torch::lazy::NodeCast<ir::ops::Stack>(node, ir::OpKind(at::aten::stack)));
      }
      case at::aten::sum: {
        return InferSum(
            torch::lazy::NodeCast<ir::ops::Sum>(node, ir::OpKind(at::aten::sum)));
      }
      case at::aten::constant_pad_nd: {
        auto constant_pad_nd = torch::lazy::NodeCast<ir::ops::ConstantPadNd>(
            node, ir::OpKind(at::aten::constant_pad_nd));
        const torch::lazy::Output& argument = node->operand(0);
        const lazy_tensors::Shape& argument_shape = ir::GetShapeFromTsOutput(argument);
        const auto argument_dimensions = argument_shape.dimensions();
        const auto& pad = constant_pad_nd->pad();
        LTC_CHECK_EQ(argument_dimensions.size() * 2, pad.size());
        std::vector<lazy_tensors::int64> padded_dimensions(
            argument_dimensions.begin(), argument_dimensions.end());
        size_t i = 0;
        for (auto rit = pad.rbegin(); rit != pad.rend(); rit += 2, ++i) {
          padded_dimensions[i] += (*rit + *(rit + 1));
        }
        return lazy_tensors::Shape(argument_shape.element_type(),
                                   padded_dimensions);
      }
      case at::aten::eq:
      case at::aten::ge:
      case at::aten::gt:
      case at::aten::le:
      case at::aten::lt:
      case at::aten::ne: {
        return InferComparison(node);
      }
      default:
        LTC_LOG(FATAL) << *node << "Not implemented yet.";
    }
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
      return LowerCast(torch::lazy::NodeCast<ir::ops::Cast>(node, *ir::ops::ltc_cast));
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
      return LowerUnselect(
          torch::lazy::NodeCast<ir::ops::Unselect>(node, *ir::ops::ltc_unselect));
    }
    if (node->op() == *ir::ops::ltc_update_slice) {
      return LowerUpdateSlice(
          torch::lazy::NodeCast<ir::ops::UpdateSlice>(node, *ir::ops::ltc_update_slice));
    }
    if (node->op().op == at::prim::Constant) {
      auto scalar_node = dynamic_cast<const ir::ops::Scalar*>(node);
      if (scalar_node) {
        return LowerScalar(scalar_node);
      }
      return LowerConstant(torch::lazy::NodeCast<ir::ops::Constant>(
          node, ir::OpKind(at::prim::Constant)));
    }
    if (node->op().op == at::aten::addmm) {
      std::vector<torch::jit::NamedValue> arguments;
      // The addmm operator in PyTorch takes bias first.
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(2)));
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(1)));
      return LowerBuiltin(node, arguments);
    }
    if (node->op().op == at::aten::bernoulli) {
      std::vector<torch::jit::NamedValue> arguments;
      arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
      return LowerBuiltin(node, arguments);
    }
    if (node->op().op == at::aten::cat) {
      return LowerCat(
          torch::lazy::NodeCast<ir::ops::Cat>(node, ir::OpKind(at::aten::cat)));
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
      return LowerBatchNorm(torch::lazy::NodeCast<ir::ops::TSNativeBatchNormForward>(
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
    if (node->op().op == at::aten::embedding_dense_backward) {
      return LowerEmbeddingDenseBackward(
          torch::lazy::NodeCast<ir::ops::TSEmbeddingDenseBackward>(
              node, ir::OpKind(at::aten::embedding_dense_backward)));
    }
    if (node->op().op == at::aten::expand) {
      return LowerExpand(
          torch::lazy::NodeCast<ir::ops::Expand>(node, ir::OpKind(at::aten::expand)));
    }
    if (node->op().op == at::aten::index_select) {
      return LowerIndexSelect(torch::lazy::NodeCast<ir::ops::IndexSelect>(
          node, ir::OpKind(at::aten::index_select)));
    }
    if (node->op().op == at::aten::leaky_relu) {
      return LowerLeakyRelu(torch::lazy::NodeCast<ir::ops::LeakyRelu>(
          node, ir::OpKind(at::aten::leaky_relu)));
    }
    if (node->op().op == at::aten::leaky_relu_backward) {
      return LowerLeakyReluBackward(torch::lazy::NodeCast<ir::ops::LeakyReluBackward>(
          node, ir::OpKind(at::aten::leaky_relu_backward)));
    }
    if (node->op().op == at::aten::nll_loss_backward) {
      return LowerNllLossBackward(
          torch::lazy::NodeCast<ir::ops::NllLossBackward>(
              node, ir::OpKind(at::aten::nll_loss_backward)));
    }
    if (node->op().op == at::aten::nll_loss_forward) {
      return LowerNllLossForward(torch::lazy::NodeCast<ir::ops::NllLossForward>(node,
          ir::OpKind(at::aten::nll_loss_forward)));
    }
    if (node->op().op == at::aten::permute) {
      return LowerPermute(
          torch::lazy::NodeCast<ir::ops::Permute>(node, ir::OpKind(at::aten::permute)));
    }
    if (node->op().op == at::aten::repeat) {
      return LowerRepeat(
          torch::lazy::NodeCast<ir::ops::Repeat>(node, ir::OpKind(at::aten::repeat)));
    }
    if (node->op().op == at::aten::squeeze) {
      return LowerSqueeze(
          torch::lazy::NodeCast<ir::ops::Squeeze>(node, ir::OpKind(at::aten::squeeze)));
    }
    if (node->op().op == at::aten::stack) {
      return LowerStack(
          torch::lazy::NodeCast<ir::ops::Stack>(node, ir::OpKind(at::aten::stack)));
    }
    if (node->op().op == at::aten::sum) {
      return LowerSum(
          torch::lazy::NodeCast<ir::ops::Sum>(node, ir::OpKind(at::aten::sum)));
    }
    if (node->op().op == at::aten::threshold) {
      return LowerThreshold(
          torch::lazy::NodeCast<ir::ops::Threshold>(node, ir::OpKind(at::aten::threshold)));
    }
    if (node->op().op == at::aten::threshold_backward) {
      return LowerThresholdBackward(
          torch::lazy::NodeCast<ir::ops::ThresholdBackward>(node, ir::OpKind(at::aten::threshold_backward)));
    }
    if (node->op().op == at::aten::unsqueeze) {
      return LowerUnsqueeze(torch::lazy::NodeCast<ir::ops::Unsqueeze>(
          node, ir::OpKind(at::aten::unsqueeze)));
    }
    if (node->op().op == at::aten::view) {
      return LowerView(
          torch::lazy::NodeCast<ir::ops::View>(node, ir::OpKind(at::aten::view)));
    }
    if (node->op() == *ir::ops::ltc_device_data) {
      const ir::ops::DeviceData* device_data_node =
          torch::lazy::NodeCast<ir::ops::DeviceData>(node, *ir::ops::ltc_device_data);
      return {loctx()->GetParameter(device_data_node->data())};
    }
    std::vector<torch::jit::NamedValue> arguments;
    for (const torch::lazy::Output& output : node->operands()) {
      arguments.emplace_back(loctx()->GetOutputOp(output));
    }
    return LowerBuiltin(node, arguments);
  }

 private:
  static lazy_tensors::Shape InferComparison(const torch::lazy::Node* node) {
    const torch::lazy::Output& lhs = node->operand(0);
    const torch::lazy::Output& rhs = node->operand(1);
    return lazy_tensors::Shape(
        lazy_tensors::PrimitiveType::PRED,
        Helpers::GetPromotedShape(ir::GetShapeFromTsOutput(lhs).dimensions(),
                                  ir::GetShapeFromTsOutput(rhs).dimensions()));
  }

  static lazy_tensors::Shape InferBatchNorm(const torch::lazy::Node* node) {
    const torch::lazy::Output& input = node->operand(0);
    const torch::lazy::Output& running_mean = node->operand(3);
    const torch::lazy::Output& running_var = node->operand(4);
    return lazy_tensors::ShapeUtil::MakeTupleShape(
        {ir::GetShapeFromTsOutput(input), ir::GetShapeFromTsOutput(running_mean), ir::GetShapeFromTsOutput(running_var)});
  }

  static lazy_tensors::Shape InferBatchNormBackward(const torch::lazy::Node* node) {
    const torch::lazy::Output& input = node->operand(1);
    const torch::lazy::Output& weight = node->operand(2);
    return lazy_tensors::ShapeUtil::MakeTupleShape(
        {ir::GetShapeFromTsOutput(input), ir::GetShapeFromTsOutput(weight), ir::GetShapeFromTsOutput(weight)});
  }

  static lazy_tensors::Shape InferNllLossForward(
      const ir::ops::NllLossForward* node) {
    static constexpr size_t kDimension2D = 2;

    auto inputShape = ir::GetShapeFromTsOutput(node->operand(0));
    auto scalarShape = lazy_tensors::Shape(inputShape.element_type(), {});
    if (node->reduction() == ReductionMode::kNone &&
        inputShape.dimensions_size() == kDimension2D) {
      auto batchSize = inputShape.dimensions(0);
      return lazy_tensors::ShapeUtil::MakeTupleShape(
          {lazy_tensors::Shape(inputShape.element_type(), {batchSize})
              , std::move(scalarShape)});
    }

    return lazy_tensors::ShapeUtil::MakeTupleShape({
        scalarShape, std::move(scalarShape)});
  }

  static lazy_tensors::Shape InferBmm(const torch::lazy::Node* node) {
    const torch::lazy::Output& tensor1 = node->operand(0);
    const torch::lazy::Output& tensor2 = node->operand(1);
    const lazy_tensors::Shape& tensor1_shape = ir::GetShapeFromTsOutput(tensor1);
    const lazy_tensors::Shape& tensor2_shape = ir::GetShapeFromTsOutput(tensor2);
    LTC_CHECK_EQ(tensor1_shape.rank(), 3);
    LTC_CHECK_EQ(tensor2_shape.rank(), 3);
    lazy_tensors::int64 b = tensor1_shape.dimensions(0);
    lazy_tensors::int64 n = tensor1_shape.dimensions(1);
    lazy_tensors::int64 m1 = tensor1_shape.dimensions(2);
    LTC_CHECK_EQ(tensor2_shape.dimensions(0), b);
    LTC_CHECK_EQ(tensor2_shape.dimensions(1), m1);
    lazy_tensors::int64 p = tensor2_shape.dimensions(2);
    return lazy_tensors::Shape(tensor1_shape.element_type(), {b, n, p});
  }

  static lazy_tensors::Shape InferCat(const ir::ops::Cat* node) {
    const auto& operands = node->operands();
    LTC_CHECK(!operands.empty());
    lazy_tensors::Shape output_shape = ir::GetShapeFromTsOutput(operands[0]);
    size_t cat_dimension_size = 0;
    for (const torch::lazy::Output& operand : operands) {
      cat_dimension_size += ir::GetShapeFromTsOutput(operand).dimensions(node->dim());
    }
    output_shape.set_dimensions(node->dim(), cat_dimension_size);
    return output_shape;
  }

  static lazy_tensors::Shape InferConvolutionBackwardOverrideable(
      const ir::ops::ConvolutionBackwardOverrideable* conv_backward) {
    const torch::lazy::Output& self = conv_backward->operand(0);
    const torch::lazy::Output& input = conv_backward->operand(1);
    const torch::lazy::Output& weight = conv_backward->operand(2);
    return lazy_tensors::ShapeUtil::MakeTupleShape(
        {ir::GetShapeFromTsOutput(input), ir::GetShapeFromTsOutput(weight), ir::GetShapeFromTsOutput(self)});
  }

  static lazy_tensors::Shape InferConvolutionOverrideable(
      const ir::ops::ConvolutionOverrideable* conv) {
    const auto& operands = conv->operands();
    LTC_CHECK(!operands.empty());

    // TODO: Shape::dimensions() returns a Span and converting it to
    // a vector of int is awkard. Clean up this after we switch to a
    // PyTorch shape.
    const auto input_shape = ir::GetShapeFromTsOutput(operands[0]);
    const auto& input_size =
        std::vector<int64_t>(input_shape.dimensions().begin(),
                             input_shape.dimensions().end());
    const auto weight_shape = ir::GetShapeFromTsOutput(operands[1]);
    const auto& weight_size =
        std::vector<int64_t>(weight_shape.dimensions().begin(),
                             weight_shape.dimensions().end());
    const auto& dilation = conv->dilation();
    const auto& padding = conv->padding();
    const auto& stride = conv->stride();
    const auto& output_padding = conv->output_padding();
    const auto& groups = conv->groups();

    if (!conv->transposed()) {
      return lazy_tensors::Shape(
          input_shape.element_type(),
          at::native::conv_output_size(input_size, weight_size, padding, stride,
                                       dilation));
    } else {
      return lazy_tensors::Shape(input_shape.element_type(),
                                 at::native::conv_input_size(
                                     input_size, weight_size, padding,
                                     output_padding, stride, dilation, groups));
    }
  }

  static lazy_tensors::Shape InferEmbeddingDenseBackward(
      const ir::ops::TSEmbeddingDenseBackward* node) {
    const torch::lazy::Output& grad_output = node->operand(0);
    const lazy_tensors::Shape& grad_output_shape = ir::GetShapeFromTsOutput(grad_output);
    return lazy_tensors::Shape(
        grad_output_shape.element_type(),
        {node->num_weights(),
         grad_output_shape.dimensions(grad_output_shape.rank() - 1)});
  }

  static lazy_tensors::Shape InferIndexSelect(
      const ir::ops::IndexSelect* node) {
    const torch::lazy::Output& input = node->operand(0);
    const torch::lazy::Output& index = node->operand(1);
    const lazy_tensors::Shape& index_shape = ir::GetShapeFromTsOutput(index);
    LTC_CHECK_EQ(index_shape.rank(), 1);
    const lazy_tensors::Shape& input_shape = ir::GetShapeFromTsOutput(input);
    const auto input_dimensions = input_shape.dimensions();
    std::vector<lazy_tensors::int64> output_dimensions(input_dimensions.begin(),
                                                       input_dimensions.end());
    LTC_CHECK_GE(node->dim(), 0);
    LTC_CHECK_LT(node->dim(), input_shape.rank());
    output_dimensions[node->dim()] = index_shape.dimensions(0);
    return lazy_tensors::Shape(input_shape.element_type(), output_dimensions);
  }

  static lazy_tensors::Shape InferMm(const torch::lazy::Node* node) {
    const torch::lazy::Output& tensor1 = node->operand(0);
    const torch::lazy::Output& tensor2 = node->operand(1);
    const lazy_tensors::Shape& tensor1_shape = ir::GetShapeFromTsOutput(tensor1);
    const lazy_tensors::Shape& tensor2_shape = ir::GetShapeFromTsOutput(tensor2);
    LTC_CHECK_EQ(tensor1_shape.rank(), 2);
    LTC_CHECK_EQ(tensor2_shape.rank(), 2);
    lazy_tensors::int64 n = tensor1_shape.dimensions(0);
    lazy_tensors::int64 m = tensor1_shape.dimensions(1);
    LTC_CHECK_EQ(tensor2_shape.dimensions(0), m);
    lazy_tensors::int64 p = tensor2_shape.dimensions(1);
    return lazy_tensors::Shape(tensor1_shape.element_type(), {n, p});
  }

  static lazy_tensors::Shape InferRepeat(const ir::ops::Repeat* repeat) {
    const torch::lazy::Output& input = repeat->operand(0);
    const lazy_tensors::Shape& input_shape = ir::GetShapeFromTsOutput(input);
    const auto& repeats = repeat->repeats();
    LTC_CHECK_GE(repeats.size(), input_shape.rank());

    lazy_tensors::int64 num_new_dimensions =
        repeats.size() - input_shape.rank();
    std::vector<lazy_tensors::int64> padded_size(num_new_dimensions, 1);
    padded_size.insert(padded_size.end(), input_shape.dimensions().begin(),
                       input_shape.dimensions().end());
    std::vector<lazy_tensors::int64> target_size(repeats.size());
    for (const auto idx : c10::irange(repeats.size())) {
      target_size[idx] = padded_size[idx] * repeats[idx];
    }
    return lazy_tensors::Shape(input_shape.element_type(), target_size);
  }

  static lazy_tensors::Shape InferSqueeze(const ir::ops::Squeeze* squeeze) {
    const torch::lazy::Output& argument = squeeze->operand(0);
    const lazy_tensors::Shape& argument_shape = ir::GetShapeFromTsOutput(argument);
    const auto output_sizes =
        BuildSqueezedDimensions(argument_shape.dimensions(), squeeze->dim());
    return lazy_tensors::Shape(argument_shape.element_type(), output_sizes);
  }

  static lazy_tensors::Shape InferStack(const ir::ops::Stack* stack) {
    const auto& inputs = stack->operands();
    LTC_CHECK(!inputs.empty());
    const lazy_tensors::Shape& input_shape = ir::GetShapeFromTsOutput(inputs[0]);
    for (const torch::lazy::Output& input : inputs) {
      LTC_CHECK_EQ(ir::GetShapeFromTsOutput(input), input_shape);
    }
    const auto input_dimensions = input_shape.dimensions();
    std::vector<lazy_tensors::int64> output_dimensions(input_dimensions.begin(),
                                                       input_dimensions.end());
    LTC_CHECK_GE(stack->dim(), 0);
    LTC_CHECK_LE(stack->dim(), output_dimensions.size());
    output_dimensions.insert(output_dimensions.begin() + stack->dim(),
                             inputs.size());
    return lazy_tensors::Shape(input_shape.element_type(), output_dimensions);
  }

  static lazy_tensors::Shape InferSum(const ir::ops::Sum* sum) {
    const torch::lazy::Output& argument = sum->operand(0);
    const lazy_tensors::Shape& argument_shape = ir::GetShapeFromTsOutput(argument);
    const auto argument_dimensions = argument_shape.dimensions();
    std::vector<lazy_tensors::int64> output_dimensions;
    const auto& sum_dimensions = sum->dimensions();
    for (lazy_tensors::int64 i = 0; i < argument_shape.rank(); ++i) {
      auto it = std::find(sum_dimensions.begin(), sum_dimensions.end(), i);
      if (it == sum_dimensions.end()) {
        output_dimensions.push_back(argument_dimensions[i]);
      } else if (sum->keep_reduced_dimensions()) {
        output_dimensions.push_back(1);
      }
    }
    lazy_tensors::PrimitiveType element_type =
        sum->dtype() ? torch_lazy_tensors::TensorTypeToLtcType(*sum->dtype())
                     : argument_shape.element_type();
    return lazy_tensors::Shape(element_type, output_dimensions);
  }

  TSOpVector LowerBuiltin(
      const torch::lazy::Node* node,
      const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
    return LowerBuiltin(node->op().op, arguments, kwarguments);
  }

  TSOpVector LowerBuiltin(
      c10::Symbol sym, const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {}) override {
    auto builtin =
        std::make_shared<torch::jit::BuiltinFunction>(sym, at::nullopt);
    auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
    auto ret = magic_method->call({}, *function_, arguments, kwarguments, 0);
    auto sv = dynamic_cast<torch::jit::SimpleValue*>(ret.get());
    LTC_CHECK(sv);
    if (sv->getValue()->type()->kind() == c10::TypeKind::TupleType) {
      const auto tuple_call_result = sv->asTuple({}, *function_);
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

  TSOpVector LowerAsStrided(const ir::ops::AsStrided* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->size());
    arguments.emplace_back(node->stride());
    arguments.emplace_back(node->storage_offset());
    TSOpVector as_strided_out = LowerBuiltin(node, arguments);
    LTC_CHECK_EQ(as_strided_out.size(), 1);
    return {GenerateClone(as_strided_out.front())};
  }

  TSOpVector LowerAsStridedViewUpdate(
      const ir::ops::AsStridedViewUpdate* node) {
    torch::jit::Value* destination =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    const torch::lazy::Output& input_op = node->operand(1);
    const lazy_tensors::Shape& input_shape = ir::GetShapeFromTsOutput(input_op);
    const auto input_dimensions = input_shape.dimensions();
    std::vector<torch::jit::NamedValue> dest_arguments;
    dest_arguments.emplace_back(destination);
    dest_arguments.emplace_back(std::vector<lazy_tensors::int64>(
        input_dimensions.begin(), input_dimensions.end()));
    dest_arguments.emplace_back(node->stride());
    dest_arguments.emplace_back(node->storage_offset());
    TSOpVector as_strided_out =
        LowerBuiltin(at::aten::as_strided, dest_arguments);
    LTC_CHECK_EQ(as_strided_out.size(), 1);
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

  TSOpVector LowerCat(const ir::ops::Cat* cat) {
    std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::Value*> tensor_list;
    const auto& operands = cat->operands();
    LTC_CHECK(!operands.empty());
    for (const torch::lazy::Output& operand : operands) {
      tensor_list.emplace_back(loctx()->GetOutputOp(operand));
    }
    auto graph = function_->graph();
    arguments.emplace_back(
        graph
            ->insertNode(graph->createList(tensor_list[0]->type(), tensor_list))
            ->output());
    arguments.emplace_back(cat->dim());
    return LowerBuiltin(cat, arguments);
  }

  TSOpVector LowerConvolutionBackwardOverrideable(
      const ir::ops::ConvolutionBackwardOverrideable* conv) {
    const auto& operands = conv->operands();
    LTC_CHECK(!operands.empty());

    std::vector<torch::jit::NamedValue> arguments;

    // TODO: Clean up after convolution unification is done.
    auto& ctx = at::globalContext();
    LTC_CHECK(
        ctx.userEnabledCuDNN() &&
        lazy_tensors::compiler::TSComputationClient::HardwareDeviceType() ==
            at::kCUDA);

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
    LTC_CHECK(!operands.empty());

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
    if (lazy_tensors::compiler::TSComputationClient::HardwareDeviceType() ==
        at::kCUDA) {
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

  TSOpVector LowerEmbeddingDenseBackward(
      const ir::ops::TSEmbeddingDenseBackward* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(1)));
    arguments.emplace_back(node->num_weights());
    arguments.emplace_back(node->padding_idx());
    arguments.emplace_back(node->scale_grad_by_freq());
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
      LTC_CHECK_EQ(expand_out.size(), 1);
      return {GenerateClone(expand_out.front())};
    }
    return expand_out;
  }

  TSOpVector LowerGenericSlice(const ir::ops::GenericSlice* node) {
    const torch::lazy::Output& input = node->operand(0);
    torch::jit::Value* base = loctx()->GetOutputOp(input);
    const auto& base_indices = node->base_indices();
    const auto& sizes = node->sizes();
    const lazy_tensors::Shape& input_shape = ir::GetShapeFromTsOutput(input);
    LTC_CHECK_EQ(sizes.size(), base_indices.size());
    LTC_CHECK_EQ(input_shape.rank(), base_indices.size());
    for (size_t dim = 0; dim < base_indices.size(); ++dim) {
      lazy_tensors::int64 start = base_indices[dim];
      base = GenerateSlice(/*base=*/base, /*dim=*/dim, /*start=*/start,
                           /*end=*/start + sizes[dim], /*step=*/1);
    }
    return {base};
  }

  TSOpVector LowerIndexSelect(const ir::ops::IndexSelect* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(node->dim());
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(1)));
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerLeakyRelu(const ir::ops::LeakyRelu* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.push_back(node->negative_slope());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerLeakyReluBackward(const ir::ops::LeakyReluBackward* node) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(0)));
    arguments.emplace_back(loctx()->GetOutputOp(node->operand(1)));
    arguments.push_back(node->negative_slope());
    arguments.push_back(node->self_is_result());
    return LowerBuiltin(node, arguments);
  }

  lazy_tensors::int64 GetReduction(ReductionMode reductionMode) {
    switch (reductionMode) {
      case ReductionMode::kMean:
        return at::Reduction::Mean;
      case ReductionMode::kNone:
        return at::Reduction::None;
      case ReductionMode::kSum:
        return at::Reduction::Sum;
    }
    LTC_ERROR() << "Unknown reduction mode: " << static_cast<int64_t>(reductionMode);
  }

  TSOpVector LowerNllLossBackward(const ir::ops::NllLossBackward* node) {
    // For TS, only weight is optional.
    static constexpr size_t kWeightOperandsOffset = 3;
    static constexpr size_t kWeightedOperandSize = 5;

    auto& operands = node->operands();
    LTC_CHECK_GT(operands.size(), kWeightOperandsOffset);

    std::vector<torch::jit::NamedValue> arguments;
    for (size_t i = 0; i < kWeightOperandsOffset; ++i) {
      arguments.emplace_back(loctx()->GetOutputOp(operands[i]));
    }

    c10::optional<at::Tensor> nullArg;
    if (operands.size() < kWeightedOperandSize) {
      arguments.emplace_back(nullArg);
    } else {
      arguments.emplace_back(loctx()->GetOutputOp(operands[kWeightOperandsOffset]));
    }

    arguments.emplace_back(GetReduction(node->reduction()));
    arguments.emplace_back(node->ignore_index());
    arguments.emplace_back(loctx()->GetOutputOp(operands.back()));

    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerNllLossForward(const ir::ops::NllLossForward* node) {
    static constexpr size_t kWeightOperandsOffset = 2;

    auto& operands = node->operands();
    LTC_CHECK_GE(operands.size(), kWeightOperandsOffset);

    std::vector<torch::jit::NamedValue> arguments;
    for (size_t i = 0; i < kWeightOperandsOffset; ++i) {
      arguments.emplace_back(loctx()->GetOutputOp(operands[i]));
    }

    c10::optional<at::Tensor> nullArg;
    if (operands.size() <= kWeightOperandsOffset) {
      arguments.emplace_back(nullArg);
    } else {
      arguments.emplace_back(loctx()->GetOutputOp(operands[kWeightOperandsOffset]));
    }

    arguments.emplace_back(GetReduction(node->reduction()));
    arguments.emplace_back(node->ignore_index());

    return LowerBuiltin(node, arguments);
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
    const lazy_tensors::Shape& shape = node->shape();
    auto options =
        at::TensorOptions()
            .device(lazy_tensors::compiler::TSComputationClient::
                        HardwareDeviceType())
            .dtype(lazy_tensors::PrimitiveToScalarType(shape.element_type()));
    return {
        loctx()->graph()->insertConstant(at::scalar_tensor(value, options))};
  }

  TSOpVector LowerSelect(const ir::ops::Select* node) {
    lazy_tensors::int64 step =
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
    LTC_CHECK(!operands.empty());
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

  TSOpVector LowerSum(const ir::ops::Sum* sum) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(loctx()->GetOutputOp(sum->operand(0)));
    arguments.emplace_back(sum->dimensions());
    arguments.emplace_back(sum->keep_reduced_dimensions());
    std::vector<torch::jit::NamedValue> kwarguments;
    kwarguments.emplace_back("dtype", sum->dtype());
    return LowerBuiltin(sum, arguments, kwarguments);
  }

  TSOpVector LowerThreshold(const ir::ops::Threshold* node) {
    std::vector<torch::jit::NamedValue> arguments;
    for (auto& operand : node->operands()) {
      arguments.emplace_back(loctx()->GetOutputOp(operand));
    }
    arguments.emplace_back(node->threshold());
    arguments.emplace_back(node->value());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerThresholdBackward(const ir::ops::ThresholdBackward* node) {
    std::vector<torch::jit::NamedValue> arguments;
    for (auto& operand : node->operands()) {
      arguments.emplace_back(loctx()->GetOutputOp(operand));
    }
    arguments.emplace_back(node->threshold());
    return LowerBuiltin(node, arguments);
  }

  TSOpVector LowerUnselect(const ir::ops::Unselect* node) {
    torch::jit::Value* dest =
        GenerateClone(loctx()->GetOutputOp(node->operand(0)));
    lazy_tensors::int64 step =
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
    const lazy_tensors::Shape& source_shape = ir::GetShapeFromTsOutput(source_argument);
    LTC_CHECK_EQ(source_shape.rank(), base_indices.size());
    torch::jit::Value* base = dest;
    for (size_t dim = 0; dim < base_indices.size(); ++dim) {
      lazy_tensors::int64 start = base_indices[dim];
      base = GenerateSlice(/*base=*/base, /*dim=*/dim, /*start=*/start,
                           /*end=*/start + source_shape.dimensions(dim),
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
    LTC_CHECK_EQ(cloned.size(), 1);
    return cloned.front();
  }

  void GenerateCopy(torch::jit::Value* destination, torch::jit::Value* source) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(destination);
    arguments.emplace_back(source);
    LowerBuiltin(at::aten::copy_, arguments);
  }

  torch::jit::Value* GenerateSlice(torch::jit::Value* base,
                                   lazy_tensors::int64 dim,
                                   lazy_tensors::int64 start,
                                   lazy_tensors::int64 end,
                                   lazy_tensors::int64 step) {
    std::vector<torch::jit::NamedValue> arguments;
    arguments.emplace_back(base);
    arguments.emplace_back(dim);
    arguments.emplace_back(start);
    arguments.emplace_back(end);
    arguments.emplace_back(step);
    TSOpVector selected = LowerBuiltin(at::aten::slice, arguments);
    LTC_CHECK_EQ(selected.size(), 1);
    return selected.front();
  }

  ts_backend::TSLoweringContext* loctx() {
    return static_cast<ts_backend::TSLoweringContext*>(loctx_);
  }

  std::shared_ptr<torch::jit::GraphFunction> function_;
};

TSNodeLoweringInterface* GetTSNodeLowering() {
  static TSNodeLoweringInterface* ts_node_lowering =
      new TSNodeLowering("ltc-ts", nullptr);
  return ts_node_lowering;
}

std::unique_ptr<NodeLowering> CreateTSNodeLowering(ir::LoweringContext* loctx) {
  return std::make_unique<TSNodeLowering>(
      "ltc-ts", static_cast<ts_backend::TSLoweringContext*>(loctx));
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
