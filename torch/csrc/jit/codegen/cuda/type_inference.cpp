#include <torch/csrc/jit/codegen/cuda/type_inference.h>

#include <aten/src/ATen/AccumulateType.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/ExpandUtils.h>
#include <ATen/core/jit_type.h>
#include <ATen/native/TypeProperties.h>
#include <torch/csrc/jit/codegen/cuda/type_promotion.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

at::ScalarType toAccumulateType(const TensorTypePtr& op) {
  TORCH_INTERNAL_ASSERT(
      op->scalarType().has_value(), "Missing Type Information.");
  return at::toAccumulateType(op->scalarType().value(), true /* is_cuda */);
}

bool hasTypeAndDevice(const TensorTypePtr& op) {
  return op != nullptr && op->device().has_value() &&
      op->scalarType().has_value();
}

TensorTypePtr getInputTensorType(
    Node* node,
    size_t index,
    bool optional = false) {
  auto tensor_type = node->input(index)->type()->cast<TensorType>();
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
  NaiveTypePropagator(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  void PropagateOnBlock(Block* block) {
    for (Node* node : block->nodes()) {
      PropagateOnNode(node);
    }
  }

  void PropagateOnNode(Node* node) {
    switch (node->kind()) {
      // Constant:
      case prim::Constant: {
        if (node->output()->type()->isSubtypeOf(TensorType::get())) {
          node->output()->inferTypeFrom(node->t(attr::value));
        }
        break;
      }
      // unary operations
      case aten::threshold:
      case aten::clamp:
      case aten::abs:
      case aten::neg:
      case aten::ceil:
      case aten::floor:
      case aten::round:
      case aten::trunc:
      case aten::frac:
      case aten::relu:
      case aten::silu:
      case aten::gelu:
      case aten::softplus:
      case aten::bitwise_not:
      // TODO: rand_like should support cast.
      case aten::rand_like: {
        node->output()->setType(unary_type(node));
        break;
      }
      // unary float operations
      case aten::log:
      case aten::log10:
      case aten::log1p:
      case aten::log2:
      case aten::lgamma:
      case aten::exp:
      case aten::expm1:
      case aten::erf:
      case aten::erfc:
      case aten::cos:
      case aten::acos:
      case aten::cosh:
      case aten::sin:
      case aten::asin:
      case aten::sinh:
      case aten::tan:
      case aten::atan:
      case aten::atanh:
      case aten::sqrt:
      case aten::rsqrt:
      case aten::reciprocal:
      case aten::sigmoid:
      case aten::tanh: {
        node->output()->setType(unary_float_type(node));
        break;
      }
      // binary float
      case aten::atan2: {
        node->output()->setType(binary_float_type(node));
        break;
      }
      // binary operations that forward meta info and broadcast shape:
      case aten::gelu_backward:
      case aten::mul:
      case aten::div:
      case aten::min:
      case aten::max:
      // TODO: first operand for pow can be Tensor / Scalar
      case aten::pow:
      case aten::remainder:
      case aten::threshold_backward:
      case aten::fmod:
      case aten::lerp:
      // add/sub could be ternary op and the third argument does not contribute
      // to neither type promotion nor shape.
      // TODO: Include alpha check for add/sub
      case aten::add:
      case aten::sub: {
        node->output()->setType(binary_type(node));
        break;
      }
      // Type can be int or bool for "and" and "or", if both are bool should be
      // bool, if both int should be int, otherwise would have errored
      case aten::__and__:
      case aten::__or__: {
        const auto promoted_type = binary_broadcast_type(
            getInputTensorType(node, 0, true),
            getInputTensorType(node, 1, true),
            node->input(0)->type()->cast<TensorType>()->scalarType() ==
                    at::ScalarType::Bool
                ? at::ScalarType::Bool
                : at::ScalarType::Int);
        break;
      }
      // Real int ops
      case aten::__xor__:
      case aten::__lshift__:
      case aten::__rshift__: {
        const auto promoted_type = binary_broadcast_type(
            getInputTensorType(node, 0, true),
            getInputTensorType(node, 1, true),
            at::ScalarType::Int);
        node->output()->setType(promoted_type);
        break;
      }
      // binary comparison
      case aten::lt:
      case aten::le:
      case aten::gt:
      case aten::ge:
      case aten::ne:
      case aten::eq: {
        const auto promoted_type = binary_broadcast_type(
            getInputTensorType(node, 0, false),
            getInputTensorType(node, 1, true),
            at::ScalarType::Bool);
        node->output()->setType(promoted_type);
        break;
      }
      case aten::where: {
        const auto promoted_type = binary_broadcast_type(
            getInputTensorType(node, 1, true),
            getInputTensorType(node, 2, true));
        node->output()->setType(promoted_type);
        break;
      }
      case aten::addcmul: {
        auto promoted_type = binary_broadcast_type(
            getInputTensorType(node, 1, true),
            getInputTensorType(node, 2, true));
        promoted_type = binary_broadcast_type(
            promoted_type, getInputTensorType(node, 0, true));
        node->output()->setType(promoted_type);
        break;
      }
      case aten::native_dropout_backward:
      case aten::dropout: {
        node->output()->setType(getInputTensorType(node, 0));
        break;
      }
      case aten::native_dropout: {
        auto out_type = getInputTensorType(node, 0);
        node->output(0)->setType(out_type);

        auto mask_type = TensorType::create(
            at::ScalarType::Bool, *out_type->device(), c10::nullopt, false);

        node->output(1)->setType(mask_type);
        break;
      }
      case aten::instance_norm:
      case aten::batch_norm: {
        node->output()->setType(getInputTensorType(node, 0));
        break;
      }
      case aten::_batch_norm_impl_index_backward: {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
        auto out_mask_list = constant_as<c10::List<bool>>(node->input(10));
        TORCH_INTERNAL_ASSERT(
            out_mask_list.has_value(),
            "Missing output mask for batch_norm_backward");
        std::vector<int> output_mask;
        for (const auto value : out_mask_list->vec()) {
          output_mask.emplace_back(static_cast<int>(value));
        }

        auto grad_input_type = getInputTensorType(node, 1);
        if (output_mask[0]) {
          node->output(0)->setType(grad_input_type);
        }

        if (output_mask[1]) {
          if (auto weight_type = getInputTensorType(node, 3, true)) {
            auto acc_weight_type =
                weight_type->withScalarType(toAccumulateType(weight_type));
            node->output(1)->setType(acc_weight_type);
          }
        }

        // TODO: Use shape information from weight tensor
        // OR get dtype information for bias tensor
        if (output_mask[2]) {
          auto bias_type = TensorType::create(
              toAccumulateType(grad_input_type),
              *grad_input_type->device(),
              c10::nullopt,
              c10::nullopt);
          node->output(2)->setType(bias_type);
        }
        break;
      }
      case aten::_batch_norm_impl_index: {
        auto out_type = getInputTensorType(node, 0);
        node->output(0)->setType(out_type);

        auto mean_invstd_type = TensorType::create(
            toAccumulateType(out_type),
            *out_type->device(),
            c10::nullopt,
            c10::nullopt);
        node->output(1)->setType(mean_invstd_type);
        node->output(2)->setType(mean_invstd_type);

        // TODO: not that it matters, but mark the right type here;
        auto reserve_type = TensorType::create(
            *out_type->scalarType(),
            *out_type->device(),
            c10::nullopt,
            c10::nullopt);
        node->output(3)->setType(reserve_type);
        node->output(4)->setType(IntType::get());
        break;
      }
      case aten::native_batch_norm: {
        auto out_type = getInputTensorType(node, 0);
        node->output(0)->setType(out_type);

        auto mean_invstd_type = TensorType::create(
            toAccumulateType(out_type),
            *out_type->device(),
            c10::nullopt,
            c10::nullopt);
        node->output(1)->setType(mean_invstd_type);
        node->output(2)->setType(mean_invstd_type);
        break;
      }
      case aten::layer_norm: {
        node->output(0)->setType(getInputTensorType(node, 0));
        break;
      }
      case aten::native_layer_norm: {
        auto out_type = getInputTensorType(node, 0);
        node->output(0)->setType(out_type);

        auto mean_invstd_type = TensorType::create(
            toAccumulateType(out_type),
            *out_type->device(),
            c10::nullopt,
            c10::nullopt);
        node->output(1)->setType(mean_invstd_type);
        node->output(2)->setType(mean_invstd_type);
        break;
      }
      case aten::native_layer_norm_backward: {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
        auto out_mask_list = constant_as<c10::List<bool>>(node->input(7));
        TORCH_INTERNAL_ASSERT(
            out_mask_list.has_value(), "output mask for layer_norm_backward");
        std::vector<int> output_mask;
        for (const auto value : out_mask_list->vec()) {
          output_mask.emplace_back(static_cast<int>(value));
        }

        if (output_mask[0]) {
          node->output(0)->setType(getInputTensorType(node, 0));
        }

        if (output_mask[1]) {
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
          if (auto weight_type = getInputTensorType(node, 5, true)) {
            node->output(1)->setType(weight_type);
          }
        }

        if (output_mask[2]) {
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
          if (auto bias_type = getInputTensorType(node, 6, true)) {
            node->output(2)->setType(bias_type);
          }
        }
        break;
      }
      case aten::softmax: {
        auto out_type = getInputTensorType(node, 0);

        // accept dtype input to `aten::softmax` node
        if (!node->input(2)->type()->isSubtypeOf(
                static_cast<c10::TypePtr>(NoneType::get()))) {
          if (auto opt_ivalue = toIValue(node->input(2))) {
            out_type = out_type->withScalarType(opt_ivalue->toScalarType());
          }
        }
        node->output()->setType(out_type);
        break;
      }
      case aten::_softmax: {
        auto out_type = getInputTensorType(node, 0);

        const auto half_to_float = constant_as<bool>(node->input(2));
        TORCH_CHECK(
            half_to_float.has_value(),
            "half_to_float bool doesn't have a value.");
        if (half_to_float.value()) {
          out_type = out_type->withScalarType(at::ScalarType::Float);
        }

        node->output()->setType(out_type);
        break;
      }
      case aten::_softmax_backward_data: {
        auto out_type = getInputTensorType(node, 0);
        if (auto opt_ivalue = toIValue(node->input(3))) {
          out_type = out_type->withScalarType(opt_ivalue->toScalarType());
        }
        node->output()->setType(out_type);
        break;
      }
      case aten::amax:
      case aten::mean:
      case aten::sum: {
        auto out_type = getInputTensorType(node, 0);

        // accept dtype input to `aten::sum` && `aten::mean`  node
        if (node->kind() == aten::mean || node->kind() == aten::sum) {
          if (!node->input(3)->type()->isSubtypeOf(
                  static_cast<c10::TypePtr>(NoneType::get()))) {
            if (auto opt_ivalue = toIValue(node->input(3))) {
              out_type = out_type->withScalarType(opt_ivalue->toScalarType());
            }
          }
        }
        const auto dims = constant_as<c10::List<int64_t>>(node->input(1));
        const auto keepdim = constant_as<bool>(node->input(2));
        TORCH_CHECK(
            dims.has_value() && keepdim.has_value(),
            "Shape inference cannot handle options.");
        node->output()->setType(
            unary_reduce_type(out_type, dims->vec(), keepdim.value()));
        break;
      }
      case aten::sum_to_size:
      case aten::_grad_sum_to_size: {
        auto out_type = node->input(0)->type()->cast<TensorType>();
        node->output()->setType(out_type->withDim(c10::nullopt));
        break;
      }
      /*
      // TODO: Enable view in parser by detecting non-alias view operation
      case aten::view:
      case aten::reshape: {
        auto out_type = node->input(0)->type()->cast<TensorType>();
        auto size_optional = constant_as<c10::List<int64_t>>(node->input(1));
        TORCH_INTERNAL_ASSERT(
            size_optional.has_value(), "The size parameter is required.");
        auto new_size = size_optional->vec();
        node->output()->setType(out_type->withSizes(new_size));
        break;
      }
      */
      case aten::type_as: {
        const auto type0 = getInputTensorType(node, 0);
        const auto type1 = getInputTensorType(node, 1);
        node->output()->setType(type0->withScalarType(type1->scalarType()));
        break;
      }
      case aten::to: {
        const auto type0 = getInputTensorType(node, 0);
        const auto out_dtype = toIValue(node->input(1));
        TORCH_CHECK(out_dtype, "No output type specified");
        node->output()->setType(
            type0->withScalarType(out_dtype->toScalarType()));
        break;
      }
      case prim::add_optional: {
        const auto type0 = getInputTensorType(node, 0);
        // const auto type1 = getInputTensorType(node, 1, true);
        // note: add_optional is supposed to replace an inplace add on input0,
        // so we just directly forward dtype
        TORCH_CHECK(type0 != nullptr);
        node->output()->setType(type0);
        break;
      }
      case aten::_autocast_to_reduced_precision: {
        const auto in_type = node->input(0)->type()->cast<TensorType>();
        TORCH_CHECK(
            hasTypeAndDevice(in_type),
            "Type and device propagation has failed, or was not provided enough information.");
        const auto in_scalar_type = in_type->scalarType();
        const auto in_device = in_type->device();
        const auto cuda_enabled = constant_as<bool>(node->input(1));
        const auto cpu_enabled = constant_as<bool>(node->input(2));
        const auto cuda_dtype = constant_as<c10::ScalarType>(node->input(3));
        const auto cpu_dtype = constant_as<c10::ScalarType>(node->input(4));
        TORCH_CHECK(
            cuda_enabled.has_value() && cpu_enabled.has_value() &&
                cuda_dtype.has_value() && cpu_dtype.has_value(),
            "_autocast_to_reduced_precision requires all scalar inputs to be constant.");
        if (in_type->scalarType() == at::ScalarType::Float) {
          if (in_device->is_cuda() && cuda_enabled.value()) {
            node->output()->setType(
                in_type->withScalarType(cuda_dtype.value()));
            break;
          } else if (in_device->is_cpu() && cpu_enabled.value()) {
            node->output()->setType(in_type->withScalarType(cpu_dtype.value()));
            break;
          }
        }
        node->output()->setType(in_type);
        break;
      }
      case aten::_autocast_to_full_precision: {
        const auto in_type = node->input(0)->type()->cast<TensorType>();
        TORCH_CHECK(
            hasTypeAndDevice(in_type),
            "Type and device propagation has failed, or was not provided enough information.");
        const auto in_scalar_type = in_type->scalarType();
        const auto in_device = in_type->device();
        const auto cuda_enabled = constant_as<bool>(node->input(1));
        const auto cpu_enabled = constant_as<bool>(node->input(2));
        TORCH_CHECK(
            cuda_enabled.has_value() && cpu_enabled.has_value(),
            "_autocast_to_full_precision requires enable flag to be constant.");

        if ((in_scalar_type == at::ScalarType::Half ||
             in_scalar_type == at::ScalarType::BFloat16) &&
            ((in_device->is_cuda() && cuda_enabled.value()) ||
             (in_device->is_cpu() && cpu_enabled.value()))) {
          node->output()->setType(
              in_type->withScalarType(at::ScalarType::Float));
        } else {
          node->output()->setType(in_type);
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
  TensorTypePtr unary_type(Node* node) {
    auto op = getInputTensorType(node, 0, false);
    return TensorType::create(
        *op->scalarType(), *op->device(), c10::nullopt, c10::nullopt);
  }

  TensorTypePtr unary_float_type(Node* node) {
    auto op = getInputTensorType(node, 0, false);
    return TensorType::create(
        computeTypes(TypePromotion::float_op_config, {op}),
        *op->device(),
        c10::nullopt,
        c10::nullopt);
  }

  TensorTypePtr unary_reduce_type(
      const TensorTypePtr& op,
      const std::vector<int64_t>& dims,
      bool keepdim) {
    TORCH_CHECK(
        hasTypeAndDevice(op),
        "Type and device propagation has failed, or was not provided enough information.");
    return TensorType::create(
        *op->scalarType(), *op->device(), c10::nullopt, c10::nullopt);
  }

  TensorTypePtr binary_type(Node* node) {
    auto op0 = node->input(0)->type();
    auto op1 = node->input(1)->type();
    auto op0_tensor_type = op0->cast<TensorType>();
    auto op1_tensor_type = op1->cast<TensorType>();
    TORCH_CHECK(
        hasTypeAndDevice(op0_tensor_type) || hasTypeAndDevice(op1_tensor_type),
        "At least one operand must be a tensor.");
    auto ptr = (op0_tensor_type != nullptr) ? op0_tensor_type : op1_tensor_type;
    return TensorType::create(
        computeTypes(TypePromotion::default_op_config, {op0, op1}),
        *ptr->device(),
        c10::nullopt,
        c10::nullopt);
  }

  TensorTypePtr binary_float_type(Node* node) {
    auto op0 = getInputTensorType(node, 0, false);
    auto op1 = node->input(1)->type();
    return TensorType::create(
        computeTypes(TypePromotion::float_op_config, {op0, op1}),
        *op0->device(),
        c10::nullopt,
        c10::nullopt);
  }

  // TODO: we should comply to codegen type promotion.
  TensorTypePtr binary_broadcast_type(
      TensorTypePtr const& op0,
      TensorTypePtr const& op1,
      c10::optional<at::ScalarType> scalar_type = c10::nullopt) {
    TORCH_CHECK(
        op0 != nullptr || op1 != nullptr,
        "Scalar operations on binary broadcast type, not supported yet.");

    if (op0 != nullptr && op1 != nullptr) {
      TORCH_CHECK(
          hasTypeAndDevice(op0) && hasTypeAndDevice(op1),
          "Type and device propagation has failed, or was not provided enough information.");
      auto promoted_scalar_type = scalar_type.has_value()
          ? *scalar_type
          : c10::promoteTypes(*op0->scalarType(), *op1->scalarType());

      return TensorType::create(
          promoted_scalar_type, *op0->device(), c10::nullopt, c10::nullopt);
    } else {
      auto ptr = (op0 != nullptr) ? op0 : op1;
      TORCH_CHECK(
          hasTypeAndDevice(ptr),
          "Type and device propagation has failed, or was not provided enough information.");
      return TensorType::create(
          scalar_type.has_value() ? *scalar_type : *ptr->scalarType(),
          *ptr->device(),
          c10::nullopt,
          c10::nullopt);
    }
  }

 private:
  std::shared_ptr<Graph> graph_;
};

} // namespace

void TypePropagate(std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("TypePropagate");
  NaiveTypePropagator(graph).run();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
