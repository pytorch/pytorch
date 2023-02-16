#include <type_inference.h>

#include <ATen/AccumulateType.h>
#include <c10/core/ScalarType.h>
#include <instrumentation.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/ExpandUtils.h>
#include <ATen/core/jit_type.h>
#include <ATen/native/TypeProperties.h>
#include <type_promotion.h>

namespace nvfuser {

namespace {

at::ScalarType toAccumulateType(const torch::jit::TensorTypePtr& op) {
  TORCH_INTERNAL_ASSERT(
      op->scalarType().has_value(), "Missing Type Information.");
  return at::toAccumulateType(op->scalarType().value(), true /* is_cuda */);
}

bool hasTypeAndDevice(const torch::jit::TensorTypePtr& op) {
  return op != nullptr && op->device().has_value() &&
      op->scalarType().has_value();
}

void copyScalarTypeAndDeviceToOutput(
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Device> device,
    torch::jit::Node* node,
    size_t index = 0) {
  auto out = node->output(index)->type()->cast<at::TensorType>();
  TORCH_INTERNAL_ASSERT(
      out != nullptr,
      "Expect target node's type pointer to be non-nullptr, but get nullptr");
  if (!hasTypeAndDevice(out)) {
    node->output(index)->setType(torch::jit::TensorType::create(
        dtype, device, c10::nullopt, c10::nullopt));
  }
}

void copyScalarTypeAndDeviceToOutput(
    torch::jit::TensorTypePtr from,
    torch::jit::Node* node,
    size_t index = 0) {
  copyScalarTypeAndDeviceToOutput(
      from->scalarType(), from->device(), node, index);
}

at::TensorTypePtr getInputTensorType(
    torch::jit::Node* node,
    size_t index,
    bool optional = false) {
  auto tensor_type = node->input(index)->type()->cast<at::TensorType>();
  if (optional && tensor_type == nullptr) {
    return tensor_type;
  }

  // (not optional) implies (tensor_type not equal nullptr)
  TORCH_CHECK(
      optional || tensor_type != nullptr,
      "Input ",
      index,
      " for operation ",
      node->kind().toDisplayString(),
      " needs to be a tensor.");

  TORCH_CHECK(
      hasTypeAndDevice(tensor_type),
      "Input ",
      index,
      " for operation ",
      node->kind().toDisplayString(),
      " is missing Type or Device Information.");
  return tensor_type;
}

/* NaiveTypePropagator
 *   Populate type/device tag on tensor, this is a transition module to
 *   cover the absence of type inference in codegen cuda fuser.
 *
 * We only cover operations supported in codegen. We focus on propagate concrete
 * types.
 * It does NOT handle aliases (not supported in codegen anyway); Type promotion
 * is not guaranteed to be consistent with PyTorch (we need to serve the need of
 * codegen instead).
 */
class NaiveTypePropagator {
 public:
  NaiveTypePropagator(std::shared_ptr<torch::jit::Graph> graph)
      : graph_(std::move(graph)) {}

  void PropagateOnBlock(torch::jit::Block* block) {
    for (torch::jit::Node* node : block->nodes()) {
      PropagateOnNode(node);
    }
  }

  void PropagateOnNode(torch::jit::Node* node) {
    switch (node->kind()) {
      // Constant:
      case at::prim::Constant: {
        if (node->output()->type()->isSubtypeOf(
                torch::jit::TensorType::get())) {
          node->output()->inferTypeFrom(node->t(at::attr::value));
        }
        break;
      }
      // unary operations
      case at::aten::threshold:
      case at::aten::clamp:
      case at::aten::abs:
      case at::aten::neg:
      case at::aten::ceil:
      case at::aten::floor:
      case at::aten::round:
      case at::aten::trunc:
      case at::aten::frac:
      case at::aten::leaky_relu:
      case at::aten::relu:
      case at::aten::silu:
      case at::aten::gelu:
      case at::aten::softplus:
      case at::aten::bitwise_not:
      // TODO: rand_like should support cast.
      case at::aten::rand_like: {
        unary_type(node);
        break;
      }
      // unary float operations
      case at::aten::log:
      case at::aten::log10:
      case at::aten::log1p:
      case at::aten::log2:
      case at::aten::lgamma:
      case at::aten::exp:
      case at::aten::exp2:
      case at::aten::expm1:
      case at::aten::erf:
      case at::aten::erfc:
      case at::aten::erfinv:
      case at::aten::cos:
      case at::aten::acos:
      case at::aten::acosh:
      case at::aten::cosh:
      case at::aten::sin:
      case at::aten::asin:
      case at::aten::asinh:
      case at::aten::sinh:
      case at::aten::tan:
      case at::aten::atan:
      case at::aten::atanh:
      case at::aten::sqrt:
      case at::aten::rsqrt:
      case at::aten::reciprocal:
      case at::aten::sigmoid:
      case at::aten::tanh: {
        unary_float_type(node);
        break;
      }
      // unary is
      case at::aten::isfinite:
      case at::aten::isinf:
      case at::aten::isnan:
      case at::aten::isneginf:
      case at::aten::isposinf:
      case at::aten::isreal: {
        copyScalarTypeAndDeviceToOutput(
            c10::ScalarType::Bool, c10::nullopt, node);
        break;
      }
      // binary float
      case at::aten::atan2: {
        binary_type(node, TypePromotion::float_op_config);
        break;
      }
      // binary operations that forward meta info and broadcast shape:
      case at::aten::gelu_backward:
      case at::aten::tanh_backward:
      case at::aten::mul:
      case at::aten::div:
      case at::aten::min:
      case at::aten::max:
      // TODO: first operand for pow can be Tensor / Scalar
      case at::aten::pow:
      case at::aten::remainder:
      case at::aten::threshold_backward:
      case at::aten::fmod:
      case at::aten::lerp:
      // add/sub could be ternary op and the third argument does not contribute
      // to neither type promotion nor shape.
      // TODO: Include alpha check for add/sub
      case at::aten::add:
      case at::aten::sub:
      case at::aten::rsub:
      case at::aten::bitwise_and:
      case at::aten::__and__:
      case at::aten::bitwise_or:
      case at::aten::__or__:
      case at::aten::bitwise_xor:
      case at::aten::__xor__:
      case at::aten::bitwise_left_shift:
      case at::aten::__lshift__:
      case at::aten::bitwise_right_shift:
      case at::aten::__rshift__: {
        binary_type(node);
        break;
      }
      // binary comparison
      case at::aten::lt:
      case at::aten::le:
      case at::aten::gt:
      case at::aten::ge:
      case at::aten::ne:
      case at::aten::eq: {
        binary_broadcast_type(
            node,
            getInputTensorType(node, 0, false),
            getInputTensorType(node, 1, true),
            at::ScalarType::Bool);
        break;
      }
      case at::aten::where: {
        binary_broadcast_type(
            node,
            getInputTensorType(node, 1, true),
            getInputTensorType(node, 2, true));
        break;
      }
      case at::aten::addcmul: {
        auto promoted_type = binary_broadcast_type(
            nullptr,
            getInputTensorType(node, 1, true),
            getInputTensorType(node, 2, true));
        binary_broadcast_type(
            node, promoted_type, getInputTensorType(node, 0, true));
        break;
      }
      case at::aten::native_dropout: {
        auto out_type = getInputTensorType(node, 0);
        copyScalarTypeAndDeviceToOutput(out_type, node, 0);
        copyScalarTypeAndDeviceToOutput(
            out_type->withScalarType(at::ScalarType::Bool), node, 1);
        break;
      }
      case at::aten::native_dropout_backward:
      case at::aten::dropout:
      case at::aten::instance_norm:
      case at::aten::batch_norm:
      case at::aten::layer_norm: {
        copyScalarTypeAndDeviceToOutput(getInputTensorType(node, 0), node);
        break;
      }
      case at::aten::_batch_norm_impl_index_backward:
      case at::aten::native_batch_norm_backward: {
        int mask_index = -1;
        if (node->kind() ==
            c10::Symbol::fromQualString(
                "aten::_batch_norm_impl_index_backward")) {
          mask_index = 10;
        } else if (
            node->kind() ==
            c10::Symbol::fromQualString("aten::native_batch_norm_backward")) {
          mask_index = 9;
        } else {
          TORCH_INTERNAL_ASSERT(
              false, "unidentified node kind", node->kind().toDisplayString());
        }
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
        auto out_mask_list =
            torch::jit::constant_as<c10::List<bool>>(node->input(mask_index));
        TORCH_INTERNAL_ASSERT(
            out_mask_list.has_value(),
            "Missing output mask for batch_norm_backward");
        std::vector<int> output_mask;
        for (const auto value : out_mask_list->vec()) {
          output_mask.emplace_back(static_cast<int>(value));
        }

        auto grad_input_type = getInputTensorType(node, 1);
        if (output_mask[0]) {
          copyScalarTypeAndDeviceToOutput(grad_input_type, node, 0);
        }

        if (output_mask[1]) {
          if (auto weight_type = getInputTensorType(node, 3, true)) {
            auto acc_weight_type =
                weight_type->withScalarType(toAccumulateType(weight_type));
            copyScalarTypeAndDeviceToOutput(acc_weight_type, node, 1);
          }
        }

        // TODO: Use shape information from weight tensor
        // OR get dtype information for bias tensor
        if (output_mask[2]) {
          auto bias_type = at::TensorType::create(
              toAccumulateType(grad_input_type),
              *grad_input_type->device(),
              c10::nullopt,
              c10::nullopt);
          copyScalarTypeAndDeviceToOutput(bias_type, node, 2);
        }
        break;
      }
      case at::aten::_batch_norm_impl_index: {
        auto out_type = getInputTensorType(node, 0);
        copyScalarTypeAndDeviceToOutput(out_type, node, 0);

        auto mean_invstd_type = at::TensorType::create(
            toAccumulateType(out_type),
            *out_type->device(),
            c10::nullopt,
            c10::nullopt);
        copyScalarTypeAndDeviceToOutput(mean_invstd_type, node, 1);
        copyScalarTypeAndDeviceToOutput(mean_invstd_type, node, 2);

        // TODO: not that it matters, but mark the right type here;
        auto reserve_type = at::TensorType::create(
            *out_type->scalarType(),
            *out_type->device(),
            c10::nullopt,
            c10::nullopt);
        copyScalarTypeAndDeviceToOutput(reserve_type, node, 3);
        node->output(4)->setType(at::IntType::get());
        break;
      }
      case at::aten::native_batch_norm:
      case at::aten::native_layer_norm: {
        auto out_type = getInputTensorType(node, 0);
        copyScalarTypeAndDeviceToOutput(out_type, node, 0);

        auto mean_invstd_type = at::TensorType::create(
            toAccumulateType(out_type),
            *out_type->device(),
            c10::nullopt,
            c10::nullopt);
        copyScalarTypeAndDeviceToOutput(mean_invstd_type, node, 1);
        copyScalarTypeAndDeviceToOutput(mean_invstd_type, node, 2);
        break;
      }
      case at::aten::native_layer_norm_backward: {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
        auto out_mask_list =
            torch::jit::constant_as<c10::List<bool>>(node->input(7));
        TORCH_INTERNAL_ASSERT(
            out_mask_list.has_value(), "output mask for layer_norm_backward");
        std::vector<int> output_mask;
        for (const auto value : out_mask_list->vec()) {
          output_mask.emplace_back(static_cast<int>(value));
        }

        if (output_mask[0]) {
          copyScalarTypeAndDeviceToOutput(getInputTensorType(node, 0), node, 0);
        }

        if (output_mask[1]) {
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
          if (auto weight_type = getInputTensorType(node, 5, true)) {
            copyScalarTypeAndDeviceToOutput(weight_type, node, 1);
          }
        }

        if (output_mask[2]) {
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
          if (auto bias_type = getInputTensorType(node, 6, true)) {
            copyScalarTypeAndDeviceToOutput(bias_type, node, 2);
          }
        }
        break;
      }
      case at::aten::log_softmax:
      case at::aten::softmax: {
        auto out_type = getInputTensorType(node, 0);

        // accept dtype input to `aten::softmax` node
        if (!node->input(2)->type()->isSubtypeOf(
                static_cast<c10::TypePtr>(at::NoneType::get()))) {
          if (auto opt_ivalue = toIValue(node->input(2))) {
            out_type = out_type->withScalarType(opt_ivalue->toScalarType());
          }
        }
        copyScalarTypeAndDeviceToOutput(out_type, node);
        break;
      }
      case at::aten::_softmax: {
        auto out_type = getInputTensorType(node, 0);

        const auto half_to_float =
            torch::jit::constant_as<bool>(node->input(2));
        TORCH_CHECK(
            half_to_float.has_value(),
            "half_to_float bool doesn't have a value.");
        if (half_to_float.value()) {
          out_type = out_type->withScalarType(at::ScalarType::Float);
        }

        copyScalarTypeAndDeviceToOutput(out_type, node);
        break;
      }
      case at::aten::_log_softmax_backward_data:
      case at::aten::_softmax_backward_data: {
        auto out_type = getInputTensorType(node, 0);
        if (auto opt_ivalue = toIValue(node->input(3))) {
          out_type = out_type->withScalarType(opt_ivalue->toScalarType());
        }
        copyScalarTypeAndDeviceToOutput(out_type, node);
        break;
      }
      case at::aten::amax:
      case at::aten::amin:
      case at::aten::mean:
      case at::aten::sum: {
        auto out_type = getInputTensorType(node, 0);

        // accept dtype input to `aten::sum` && `aten::mean`  node
        if (node->kind() == at::aten::mean || node->kind() == at::aten::sum) {
          if (!node->input(3)->type()->isSubtypeOf(
                  static_cast<c10::TypePtr>(at::NoneType::get()))) {
            if (auto opt_ivalue = toIValue(node->input(3))) {
              out_type = out_type->withScalarType(opt_ivalue->toScalarType());
            }
          }
        }
        const auto dims =
            torch::jit::constant_as<c10::List<int64_t>>(node->input(1));
        const auto keepdim = torch::jit::constant_as<bool>(node->input(2));
        TORCH_CHECK(
            dims.has_value() && keepdim.has_value(),
            "Shape inference cannot handle options.");
        unary_reduce_type(node, out_type, dims->vec(), keepdim.value());
        break;
      }
      case at::aten::std:
      case at::aten::var: {
        auto out_type = getInputTensorType(node, 0);
        const auto dims =
            torch::jit::constant_as<c10::List<int64_t>>(node->input(1));
        const auto keepdim = torch::jit::constant_as<bool>(node->input(3));
        TORCH_CHECK(
            dims.has_value() && keepdim.has_value(),
            "Shape inference cannot handle options.");
        unary_reduce_type(node, out_type, dims->vec(), keepdim.value());
        break;
      }
      case at::aten::sum_to_size:
      case at::aten::_grad_sum_to_size: {
        auto out_type = node->input(0)->type()->cast<at::TensorType>();
        copyScalarTypeAndDeviceToOutput(out_type->withDim(c10::nullopt), node);
        break;
      }
      case at::aten::expand_copy:
      case at::prim::expand_as_copy:
      case at::prim::flatten_copy:
      case at::aten::permute_copy:
      case at::aten::_reshape_copy:
      case at::aten::squeeze_copy:
      case at::aten::t_copy:
      case at::aten::transpose_copy:
      case at::aten::unsqueeze_copy:
      case at::aten::index_select:
      case at::aten::gather:
      case at::aten::view_copy: {
        auto out_type = node->input(0)->type()->cast<at::TensorType>();
        copyScalarTypeAndDeviceToOutput(out_type, node);
        break;
      }
      case at::aten::type_as: {
        const auto type0 = getInputTensorType(node, 0);
        const auto type1 = getInputTensorType(node, 1);
        copyScalarTypeAndDeviceToOutput(
            type0->withScalarType(type1->scalarType()), node);
        break;
      }
      case at::aten::to:
      case at::aten::_to_copy: {
        const auto type0 = getInputTensorType(node, 0);
        const auto out_dtype = toIValue(node->input(1));
        if (out_dtype.has_value() && out_dtype->isInt()) {
          copyScalarTypeAndDeviceToOutput(
              type0->withScalarType(out_dtype->toScalarType()), node);
        } else {
          TORCH_CHECK(
              !out_dtype.has_value() || out_dtype->isNone(),
              "dtype for cast unrecognized ",
              out_dtype->tagKind());
          copyScalarTypeAndDeviceToOutput(type0, node);
        }
        break;
      }
      case at::prim::add_optional: {
        const auto type0 = getInputTensorType(node, 0);
        // const auto type1 = getInputTensorType(node, 1, true);
        // note: add_optional is supposed to replace an inplace add on input0,
        // so we just directly forward dtype
        TORCH_CHECK(type0 != nullptr);
        copyScalarTypeAndDeviceToOutput(type0, node);
        break;
      }
      case at::aten::_autocast_to_reduced_precision: {
        const auto in_type = node->input(0)->type()->cast<at::TensorType>();
        TORCH_CHECK(
            hasTypeAndDevice(in_type),
            "Type and device propagation has failed, or was not provided enough information.");
        const auto in_device = in_type->device();
        const auto cuda_enabled = torch::jit::constant_as<bool>(node->input(1));
        const auto cpu_enabled = torch::jit::constant_as<bool>(node->input(2));
        const auto cuda_dtype =
            torch::jit::constant_as<c10::ScalarType>(node->input(3));
        const auto cpu_dtype =
            torch::jit::constant_as<c10::ScalarType>(node->input(4));
        TORCH_CHECK(
            cuda_enabled.has_value() && cpu_enabled.has_value() &&
                cuda_dtype.has_value() && cpu_dtype.has_value(),
            "_autocast_to_reduced_precision requires all scalar inputs to be constant.");
        if (in_type->scalarType() == at::ScalarType::Float) {
          if (in_device->is_cuda() && cuda_enabled.value()) {
            copyScalarTypeAndDeviceToOutput(
                in_type->withScalarType(cuda_dtype.value()), node);
            break;
          } else if (in_device->is_cpu() && cpu_enabled.value()) {
            copyScalarTypeAndDeviceToOutput(
                in_type->withScalarType(cpu_dtype.value()), node);
            break;
          }
        }
        copyScalarTypeAndDeviceToOutput(in_type, node);
        break;
      }
      case at::aten::_autocast_to_full_precision: {
        const auto in_type = node->input(0)->type()->cast<at::TensorType>();
        TORCH_CHECK(
            hasTypeAndDevice(in_type),
            "Type and device propagation has failed, or was not provided enough information.");
        const auto in_scalar_type = in_type->scalarType();
        const auto in_device = in_type->device();
        const auto cuda_enabled = torch::jit::constant_as<bool>(node->input(1));
        const auto cpu_enabled = torch::jit::constant_as<bool>(node->input(2));
        TORCH_CHECK(
            cuda_enabled.has_value() && cpu_enabled.has_value(),
            "_autocast_to_full_precision requires enable flag to be constant.");

        if ((in_scalar_type == at::ScalarType::Half ||
             in_scalar_type == at::ScalarType::BFloat16) &&
            ((in_device->is_cuda() && cuda_enabled.value()) ||
             (in_device->is_cpu() && cpu_enabled.value()))) {
          copyScalarTypeAndDeviceToOutput(
              in_type->withScalarType(at::ScalarType::Float), node);
        } else {
          copyScalarTypeAndDeviceToOutput(in_type, node);
        }
        break;
      }
      default:
        TORCH_CHECK(
            false,
            "type inference failed, unrecognized operation encountered:",
            node->kind().toDisplayString());
        // TODO: generate a proper error log, as this probably means something
        //       went unexpected.
        break;
    }
  }

  void run() {
    PropagateOnBlock(graph_->block());
  }

 protected:
  void unary_type(torch::jit::Node* node) {
    auto op = getInputTensorType(node, 0, false);
    copyScalarTypeAndDeviceToOutput(op, node);
  }

  void unary_float_type(torch::jit::Node* node) {
    auto op = getInputTensorType(node, 0, false);
    copyScalarTypeAndDeviceToOutput(
        computeTypes(TypePromotion::float_op_config, {op}),
        *op->device(),
        node);
  }

  void unary_reduce_type(
      torch::jit::Node* node,
      const torch::jit::TensorTypePtr& op,
      const std::vector<int64_t>& dims,
      bool keepdim) {
    TORCH_CHECK(
        hasTypeAndDevice(op),
        "Type and device propagation has failed, or was not provided enough information.");
    copyScalarTypeAndDeviceToOutput(op, node);
  }

  void binary_type(
      torch::jit::Node* node,
      TypePromotionConfig config = TypePromotion::default_op_config) {
    auto op0 = node->input(0)->type();
    auto op1 = node->input(1)->type();
    auto op0_tensor_type = op0->cast<at::TensorType>();
    auto op1_tensor_type = op1->cast<at::TensorType>();
    TORCH_CHECK(
        hasTypeAndDevice(op0_tensor_type) || hasTypeAndDevice(op1_tensor_type),
        "At least one operand must be a tensor.");
    auto ptr = (op0_tensor_type != nullptr) ? op0_tensor_type : op1_tensor_type;
    copyScalarTypeAndDeviceToOutput(
        computeTypes(config, {op0, op1}), *ptr->device(), node);
  }

  // TODO: we should comply to codegen type promotion.
  torch::jit::TensorTypePtr binary_broadcast_type(
      torch::jit::Node* node,
      torch::jit::TensorTypePtr const& op0,
      torch::jit::TensorTypePtr const& op1,
      c10::optional<at::ScalarType> scalar_type = c10::nullopt) {
    torch::jit::TensorTypePtr out;
    TORCH_CHECK(
        op0 != nullptr || op1 != nullptr,
        "Scalar operations on binary broadcast type, not supported yet.");

    c10::ScalarType promoted_scalar_type;
    c10::optional<c10::Device> device;
    if (op0 != nullptr && op1 != nullptr) {
      TORCH_CHECK(
          hasTypeAndDevice(op0) && hasTypeAndDevice(op1),
          "Type and device propagation has failed, or was not provided enough information.");
      promoted_scalar_type = scalar_type.has_value()
          ? *scalar_type
          : c10::promoteTypes(*op0->scalarType(), *op1->scalarType());
      device = *op0->device();
    } else {
      auto ptr = (op0 != nullptr) ? op0 : op1;
      TORCH_CHECK(
          hasTypeAndDevice(ptr),
          "Type and device propagation has failed, or was not provided enough information.");
      promoted_scalar_type =
          scalar_type.has_value() ? *scalar_type : *ptr->scalarType();
      device = *ptr->device();
    }
    if (node != nullptr) {
      copyScalarTypeAndDeviceToOutput(promoted_scalar_type, device, node);
    }

    return torch::jit::TensorType::create(
        promoted_scalar_type, device, c10::nullopt, c10::nullopt);
  }

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
};

} // namespace

void TypePropagate(std::shared_ptr<torch::jit::Graph>& graph) {
  FUSER_PERF_SCOPE("TypePropagate");
  GRAPH_DUMP("Before TypePropagate: ", graph);
  NaiveTypePropagator(graph).run();
  GRAPH_DUMP("After TypePropagate: ", graph);
}

} // namespace nvfuser
