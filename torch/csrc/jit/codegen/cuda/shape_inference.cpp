#include <torch/csrc/jit/codegen/cuda/shape_inference.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/ExpandUtils.h>
#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

/* NaiveShapeTypePropagator
 *   Populate shape/type/device tag on tensor, this is a transition module to
 *   cover the absence of shape/type inference in codegen cuda fuser.
 *
 * We only cover operations supported in codegen. We focus on propagate concrete
 * shapes/types.
 * It does NOT handle aliases (not supported in codegen anyway); Type promotion
 * is not guaranteed to be consistent with PyTorch (we need to serve the need of
 * codegen instead).
 */
class NaiveShapeTypePropagator {
 public:
  NaiveShapeTypePropagator(std::shared_ptr<Graph> graph)
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
      // unary operations that forward meta info and shape:
      case aten::neg:
      case aten::abs:
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
      case aten::sqrt:
      case aten::rsqrt:
      case aten::ceil:
      case aten::floor:
      case aten::round:
      case aten::trunc:
      case aten::frac:
      case aten::reciprocal:
      case aten::relu:
      case aten::sigmoid:
      case aten::threshold:
      case aten::clamp:
      case aten::gelu:
      case aten::tanh: {
        TORCH_CHECK(
            node->input(0)->type()->cast<TensorType>()->isComplete(),
            "shape propagation failed");
        node->output()->setType(node->input(0)->type()->cast<TensorType>());
        break;
      }
      // TODO: rand_like should support cast.
      case aten::rand_like: {
        TORCH_CHECK(
            node->input(0)->type()->cast<TensorType>()->isComplete(),
            "shape propagation failed");
        node->output()->setType(node->input(0)->type()->cast<TensorType>());
        break;
      }
      // binary operations that forward meta info and broadcast shape:
      case aten::mul:
      case aten::div:
      case aten::atan2:
      // TODO: double check type casting logic for min/max/pow
      case aten::min:
      case aten::max:
      case aten::pow:
      case aten::remainder:
      case aten::fmod:
      case aten::lerp:
      // add/sub could be ternary op and the third argument does not contribute
      // to neither type promoteion nor shape.
      case aten::add:
      case aten::sub: {
        auto promoted_type = binary_broadcast_type(
            node->input(0)->type()->cast<TensorType>(),
            node->input(1)->type()->cast<TensorType>());
        node->output()->setType(promoted_type);
        break;
      }
      // TODO: double check type casting logic for operations commented out.
      case aten::lt:
      case aten::le:
      case aten::gt:
      case aten::ge:
      case aten::ne:
      case aten::eq: {
        auto promoted_type = binary_broadcast_type(
            node->input(0)->type()->cast<TensorType>(),
            node->input(1)->type()->cast<TensorType>(),
            at::ScalarType::Bool);
        node->output()->setType(promoted_type);
        break;
      }
      case aten::where: {
        auto promoted_type = binary_broadcast_type(
            node->input(1)->type()->cast<TensorType>(),
            node->input(2)->type()->cast<TensorType>());
        node->output()->setType(promoted_type);
        break;
      }
      case aten::addcmul: {
        auto promoted_type = binary_broadcast_type(
            node->input(1)->type()->cast<TensorType>(),
            node->input(2)->type()->cast<TensorType>());
        promoted_type = binary_broadcast_type(
            promoted_type, node->input(0)->type()->cast<TensorType>());
        node->output()->setType(promoted_type);
        break;
      }
      default:
        TORCH_CHECK(false, "shape/type inference failed.");
        // TODO: generate a proper error log, as this probably means something
        //       went unexpected.
        break;
    }
  }

  void run() {
    PropagateOnBlock(graph_->block());
  }

 protected:
  // TODO: we should comply to codegen type promotion.
  TensorTypePtr binary_broadcast_type(
      TensorTypePtr const& op0,
      TensorTypePtr const& op1,
      c10::optional<at::ScalarType> scalar_type = c10::nullopt) {
    TORCH_CHECK(
        op0 != nullptr || op1 != nullptr, "no scalar operation supported yet.");

    if (op0 != nullptr && op1 != nullptr) {
      auto expanded_size = at::infer_size(
          *op0->sizes().concrete_sizes(), *op1->sizes().concrete_sizes());
      auto promoted_scalar_type = scalar_type.has_value()
          ? scalar_type.value()
          : c10::promoteTypes(*op0->scalarType(), *op1->scalarType());
      // TODO: maybe contiguous is not what we want in case when layout
      //       propagation could be beneficial.
      return TensorType::createContiguous(
          promoted_scalar_type, *op0->device(), expanded_size);
    } else {
      auto ptr = (op0 != nullptr) ? op0 : op1;
      TORCH_CHECK(ptr->isComplete(), "shape propagation failed");
      return TensorType::createContiguous(
          scalar_type.has_value() ? scalar_type.value() : *ptr->scalarType(),
          *ptr->device(),
          *ptr->sizes().concrete_sizes());
    }
  }

 private:
  std::shared_ptr<Graph> graph_;
};

} // namespace

TORCH_CUDA_API void ShapeTypePropagate(std::shared_ptr<Graph>& graph) {
  NaiveShapeTypePropagator(graph).run();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
