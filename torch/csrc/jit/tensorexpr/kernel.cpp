#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <c10/util/string_utils.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {
namespace tensorexpr {

static int te_cuda_pointwise_loop_levels = -1;
static int te_cuda_pointwise_block_count = -1;
static int te_cuda_pointwise_block_size = -1;
static bool fallback_allowed = false;
static bool te_generate_block_code = false;

bool setFallbackAllowed(bool value) {
  bool old_value = fallback_allowed;
  fallback_allowed = value;
  return old_value;
}

bool fallbackAllowed() {
  static const char* enable_c_str = std::getenv("PYTORCH_TENSOREXPR_FALLBACK");
  if (!enable_c_str) {
    return fallback_allowed;
  }
  if (std::string(enable_c_str) == "0") {
    return false;
  }
  return true;
}

bool fallbackEnforced() {
  static const char* enable_c_str = std::getenv("PYTORCH_TENSOREXPR_FALLBACK");
  if (!enable_c_str) {
    return fallback_allowed;
  }
  if (std::string(enable_c_str) == "2") {
    return true;
  }
  return false;
}

int& getTECudaPointwiseLoopLevels() {
  return te_cuda_pointwise_loop_levels;
}

int& getTECudaPointwiseBlockCount() {
  return te_cuda_pointwise_block_count;
}

int& getTECudaPointwiseBlockSize() {
  return te_cuda_pointwise_block_size;
}

// TODO: Remove this global var
// Ideally Block code gen should be decided
// based on device type in tensor.
bool& getTEGenerateBlockCode() {
  return te_generate_block_code;
}

c10::optional<at::Device> pickDeviceType(
    const at::ArrayRef<torch::jit::Value*>& inputs) {
  c10::optional<at::Device> device = c10::nullopt;
  for (auto const& input : inputs) {
    auto tt = input->type()->cast<TensorType>();
    if (tt && tt->device()) {
      if (device && *device != *tt->device()) {
        return c10::nullopt;
      }
      device = *tt->device();
    }
  }
  return device;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

static at::ScalarType tensorType(Tensor* t) {
  return static_cast<at::ScalarType>(t->body()->dtype().scalar_type());
}

static std::vector<ExprHandle> computeIndicesToBroadcast(
    const std::vector<ExprHandle>& outputAxes,
    const std::vector<ExprHandle>& inputSizes) {
  if (outputAxes.size() < inputSizes.size()) {
    throw malformed_input("Cannot broadcast to a lower rank tensor");
  }
  std::vector<ExprHandle> bcast;
  auto axisIt = outputAxes.rbegin();
  auto sizeIt = inputSizes.rbegin();
  while (sizeIt != inputSizes.rend()) {
    auto const& size = sizeIt->AsNode<IntImm>();
    if (size && size->value() == 1) {
      bcast.emplace_back(0);
    } else {
      bcast.emplace_back(*axisIt);
    }
    ++axisIt;
    ++sizeIt;
  }
  std::reverse(bcast.begin(), bcast.end());
  return bcast;
}

ExprHandle TensorExprKernel::broadcast(
    Tensor* t,
    const std::vector<ExprHandle>& axes) {
  return t->call(computeIndicesToBroadcast(
      axes, ExprVectorToExprHandleVector(t->buf()->dims())));
}

ExprHandle TensorExprKernel::chunk(
    Tensor* t,
    size_t chunkIdx,
    size_t dim,
    size_t chunks,
    const std::vector<ExprHandle>& axes) {
  auto sizes = bufferSizes(t);
  size_t step = sizes[dim] / chunks;

  std::vector<ExprHandle> indices;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i == dim) {
      indices.push_back(axes[i] + IntImm::make((int)chunkIdx * (int)step));
    } else {
      indices.push_back(axes[i]);
    }
  }

  return t->call(indices);
}

ExprHandle TensorExprKernel::tensorOrConstant(
    const torch::jit::Value* v,
    const std::vector<ExprHandle>& axes) {
  auto ti = tensors_.find(v->unique());
  if (ti != tensors_.end()) {
    return broadcast(ti->second, axes);
  }
  return constant(v);
}

std::vector<ExprHandle> TensorExprKernel::sizesFromVaryingShape(
    const c10::VaryingShape<int64_t>& shape) {
  std::vector<ExprHandle> dims;
  for (size_t i = 0; i < *shape.size(); i++) {
    dims.push_back(IntImm::make(*shape[i]));
  }
  return dims;
}

std::vector<DimArg> TensorExprKernel::dimsFromSizes(
    const std::vector<ExprHandle>& sizes) {
  std::vector<DimArg> dimArgs;
  for (size_t idx = 0; idx < sizes.size(); idx++) {
    dimArgs.emplace_back(DimArg(sizes[idx], "i" + c10::to_string(idx)));
  }
  return dimArgs;
}

std::vector<ExprHandle> TensorExprKernel::sizesForValue(
    const torch::jit::Value* v) {
  if (known_sizes_.count(v)) {
    return known_sizes_.at(v);
  }

  // If the shape is present in the type info, just extract it from here. No
  // need to infer it.
  if (v->type()->kind() == TypeKind::TensorType) {
    auto tt = v->type()->cast<TensorType>();
    if (tt->isComplete()) {
      return sizesFromVaryingShape(tt->sizes());
    }
  }

  if (v->type()->isSubtypeOf(FloatType::get()) ||
      v->type()->isSubtypeOf(IntType::get())) {
    return {1};
  }
  if (v->type()->isSubtypeOf(NoneType::get())) {
    return {};
  }

  known_sizes_[v] = inferSizesForValue(v);
  return known_sizes_.at(v);
}

std::vector<ExprHandle> TensorExprKernel::inferSizesForValue(
    const torch::jit::Value* v) {
  switch (v->node()->kind()) {
    case aten::_cast_Float:
    case aten::sigmoid:
    case aten::reciprocal:
    case aten::neg:
    case aten::relu:
    case aten::log:
    case aten::log10:
    case aten::log2:
    case aten::exp:
    case aten::expm1:
    case aten::erf:
    case aten::erfc:
    case aten::cos:
    case aten::sin:
    case aten::tan:
    case aten::rand_like:
    case aten::acos:
    case aten::asin:
    case aten::cosh:
    case aten::sinh:
    case aten::atan:
    case aten::tanh:
    case aten::sqrt:
    case aten::rsqrt:
    case aten::abs:
    case aten::ceil:
    case aten::floor:
    case aten::round:
    case aten::trunc:
    case aten::frac:
    case aten::lgamma:
      return sizesForValue(v->node()->input());

    case aten::sub:
    case aten::add:
    case aten::mul:
    case aten::div:
    case aten::__and__:
    case aten::__or__:
    case aten::__xor__:
    case aten::__lshift__:
    case aten::__rshift__:
    case aten::eq:
    case aten::ne:
    case aten::ge:
    case aten::gt:
    case aten::le:
    case aten::lt:
    case aten::min:
    case aten::max:
    case aten::type_as:
    case aten::pow:
    case aten::fmod:
    case aten::remainder:
    case aten::atan2: {
      std::vector<std::vector<ExprHandle>> shapes;
      for (size_t idx = 0; idx < 2; idx++) {
        torch::jit::Value* inp = v->node()->input(idx);
        shapes.push_back(sizesForValue(inp));
      }
      return broadcastShapes(shapes);
    }

    case aten::lerp:
    case aten::clamp:
    case aten::threshold:
    case aten::where: {
      std::vector<std::vector<ExprHandle>> shapes;
      for (size_t idx = 0; idx < 3; idx++) {
        torch::jit::Value* inp = v->node()->input(idx);
        shapes.push_back(sizesForValue(inp));
      }
      return broadcastShapes(shapes);
    }

    case aten::addcmul: {
      std::vector<std::vector<ExprHandle>> shapes;
      for (size_t idx = 0; idx < 4; idx++) {
        torch::jit::Value* inp = v->node()->input(idx);
        shapes.push_back(sizesForValue(inp));
      }
      return broadcastShapes(shapes);
    }

    case prim::ConstantChunk: {
      auto shape = sizesForValue(v->node()->input());
      int dim = v->node()->i(attr::dim);
      int chunks = v->node()->i(attr::chunks);
      shape[dim] = IRSimplifier::simplify(shape[dim] / chunks);
      return shape;
    }

    case aten::unsqueeze: {
      auto const& n = v->node();
      auto shape = sizesForValue(n->input(0));

      int64_t dim = constant(n->input(1)).AsNode<IntImm>()->value();
      // From the documentation
      // (https://pytorch.org/docs/master/generated/torch.unsqueeze.html):
      //
      // A dim value within the range [-input.dim() - 1, input.dim() + 1) can be
      // used. Negative dim will correspond to unsqueeze() applied at dim = dim
      // + input.dim() + 1.
      if (dim < 0) {
        dim = dim + shape.size() + 1;
      }
      if (dim < 0 || dim > shape.size()) {
        throw std::runtime_error("Invalid 'dim' input in aten::unsqueeze");
      }

      shape.insert(shape.begin() + dim, ExprHandle(1));
      return shape;
    }

    case aten::cat: {
      // In JIT IR, aten::cat usually appears with the following nodes around
      // it:
      //   %dim : int = prim::Constant[value=0]()
      //   %inputs : Tensor[] = prim::ListConstruct(%a, %b, ...)
      //   %cat_output : Tensor = aten::cat(%inputs, %dim)
      // Shapes of the input tensors could only differ at the dimension %dim.
      // The sizes of the output tensor on that dimension is a sum of the
      // corresponding sizes of the input tensors, the other dimension have the
      // same sizes.
      auto const& n = v->node();
      auto inputs = n->input(0)->node()->inputs();
      TORCH_INTERNAL_ASSERT(n->input(1)->node()->kind() == prim::Constant);
      int64_t dim = n->input(1)->node()->i(attr::value);

      ExprHandle concat_size = IntImm::make(0);
      for (auto input : inputs) {
        concat_size = concat_size + sizesForValue(input)[dim];
      }
      concat_size = IRSimplifier::simplify(concat_size);
      auto shape = sizesForValue(inputs[0]);
      shape[dim] = concat_size;
      return shape;
    }
    case aten::slice:
      throw std::runtime_error(
          "Shape info is not implemented for this kind of node");

    default: {
      GRAPH_DEBUG("Can't infer sizes for the node: ", *v->node());
      GRAPH_DEBUG("Full fusion group graph:\n", *v->node()->owningGraph());
      throw std::runtime_error("Unhandled node kind");
    }
  }
}

ExprHandle TensorExprKernel::constant(const torch::jit::Value* v) {
  if (v->node()->kind() == prim::Constant) {
    const auto val = toIValue(v).value();
    if (val.isDouble()) {
      return FloatImm::make(static_cast<float>(val.toDouble()));
    } else if (val.isInt()) {
      return IntImm::make(val.toInt());
    } else if (val.isBool()) {
      return BoolImm::make(val.toBool());
    } else if (val.isNone()) {
      // This is just a placeholder so we don't throw.  None-handling
      // is operator-specific and should be handled properly in
      // the operator-specific lowering code.
      return IntImm::make(0);
    } else {
      throw unsupported_dtype();
    }
  }

  if (!scalars_.count(v->unique())) {
    throw malformed_input("no scalar in Constant");
  }

  return scalars_.at(v->unique());
}

void TensorExprKernel::promoteInputs(std::vector<ExprHandle>& inputs) {
  if (inputs.empty()) {
    return;
  }

  // Find the highest type among the inputs.
  ScalarType highType = inputs[0].dtype().scalar_type();
  for (const auto input : inputs) {
    highType = promoteTypes(highType, input.dtype().scalar_type());
  }

  for (ExprHandle& e : inputs) {
    if (e.dtype().scalar_type() == highType) {
      continue;
    }

    switch (highType) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    e = cast<Type>(e);        \
    break;
      AT_FORALL_SCALAR_TYPES_AND2(Half, Bool, TYPE_CASE);
#undef TYPE_CASE
      default:
        throw unsupported_dtype();
    }
  }
}

ExprHandle TensorExprKernel::demoteOutput(
    const ExprHandle& e,
    const torch::jit::Value* v) {
  if (v->type()->kind() != TypeKind::TensorType) {
    return e;
  }

  if (!v->isCompleteTensor()) {
    return e;
  }

  auto tt = *v->type()->cast<TensorType>()->scalarType();

  if (tt == static_cast<at::ScalarType>(e.dtype().scalar_type())) {
    return e;
  }

  switch (tt) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name) \
  case at::ScalarType::Name:  \
    return cast<Type>(e);
    AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
    case at::ScalarType::Bool:
      return cast<bool>(e);
    default:
      throw unsupported_dtype();
  }

  return e;
}

static bool isOne(ExprHandle e) {
  auto const& n = e.AsNode<IntImm>();
  if (!n) {
    return false;
  }
  return n->value() == 1;
}

std::vector<ExprHandle> TensorExprKernel::broadcastShapes(
    std::vector<std::vector<ExprHandle>> shapes) {
  size_t n = shapes.size();
  if (n == 1) {
    return shapes[0];
  }
  auto res1 = broadcastShapes(shapes[n - 2], shapes[n - 1]);
  shapes[n - 2] = res1;
  shapes.pop_back();
  auto res2 = broadcastShapes(shapes);
  return res2;
}
std::vector<ExprHandle> TensorExprKernel::broadcastShapes(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  auto at = a.rbegin();
  auto bt = b.rbegin();
  std::vector<ExprHandle> ret;
  while (at != a.rend() || bt != b.rend()) {
    if (at == a.rend()) {
      hasBroadcast_ = true;
      ret.push_back(*bt++);
      continue;
    }
    if (bt == b.rend()) {
      hasBroadcast_ = true;
      ret.push_back(*at++);
      continue;
    }
    // TODO: if neither *at nor *bt is 1, ensure they are identical
    // expressions.  Nb: `==` doesn't work since that simply produces a new
    // ExprHandle.
    ExprHandle dim = *at;
    if (isOne(*at)) {
      if (!isOne(*bt)) {
        dim = *bt;
        hasBroadcast_ = true;
      }
    }
    ret.push_back(dim);
    at++;
    bt++;
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

std::vector<ExprHandle> TensorExprKernel::valueShape(
    const torch::jit::Value* v) {
  auto it = tensors_.find(v->unique());
  if (it == tensors_.end()) {
    return {};
  }
  return ExprVectorToExprHandleVector(it->second->dims());
}

Tensor* TensorExprKernel::computeOneOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(const ExprHandle&)>& innerExpr) {
  auto const& n = v->node();
  auto const& shape = valueShape(n->inputs()[0]);
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, innerExpr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], indices)};

        promoteInputs(inputs);
        ExprHandle compute = innerExpr(inputs[0]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::computeTwoOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  auto const& n = v->node();
  auto const& shape =
      broadcastShapes(valueShape(n->inputs()[0]), valueShape(n->inputs()[1]));
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, innerExpr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], indices),
            tensorOrConstant(n->inputs()[1], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute = innerExpr(inputs[0], inputs[1]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::computeTwoOperandWithAlpha(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  auto const& n = v->node();
  auto const& shape =
      broadcastShapes(valueShape(n->inputs()[0]), valueShape(n->inputs()[1]));
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, innerExpr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], indices),
            tensorOrConstant(n->inputs()[1], indices),
            tensorOrConstant(n->inputs()[2], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute = innerExpr(inputs[0], inputs[2] * inputs[1]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::computeConditionWithTwoOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  auto const& n = v->node();
  std::vector<std::vector<ExprHandle>> shapes;
  for (size_t idx = 0; idx < 2; idx++) {
    torch::jit::Value* inp = n->input(idx);
    shapes.push_back(sizesForValue(inp));
  }
  auto const& shape = broadcastShapes(shapes);
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, innerExpr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[1], indices),
            tensorOrConstant(n->inputs()[2], indices),
        };

        promoteInputs(inputs);
        // First expr is the condition, which we don't promote
        inputs.emplace(
            inputs.begin(), tensorOrConstant(n->inputs()[0], indices));
        ExprHandle compute = innerExpr(inputs[0], inputs[1], inputs[2]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::computeThreeOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  auto const& n = v->node();
  std::vector<std::vector<ExprHandle>> shapes;
  for (size_t idx = 0; idx < 3; idx++) {
    torch::jit::Value* inp = n->input(idx);
    shapes.push_back(sizesForValue(inp));
  }
  auto const& shape = broadcastShapes(shapes);
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, innerExpr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], indices),
            tensorOrConstant(n->inputs()[1], indices),
            tensorOrConstant(n->inputs()[2], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute = innerExpr(inputs[0], inputs[1], inputs[2]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::computeFourOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&)>& innerExpr) {
  auto const& n = v->node();
  std::vector<std::vector<ExprHandle>> shapes;
  for (size_t idx = 0; idx < 4; idx++) {
    torch::jit::Value* inp = n->input(idx);
    shapes.push_back(sizesForValue(inp));
  }
  auto const& shape = broadcastShapes(shapes);
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, innerExpr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], indices),
            tensorOrConstant(n->inputs()[1], indices),
            tensorOrConstant(n->inputs()[2], indices),
            tensorOrConstant(n->inputs()[3], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute =
            innerExpr(inputs[0], inputs[1], inputs[2], inputs[3]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::computeValue(const torch::jit::Value* v) {
  switch (v->node()->kind()) {
    case aten::add: {
      auto add_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        return lhs + rhs;
      };
      TORCH_INTERNAL_ASSERT(
          v->node()->inputs().size() == 2 || v->node()->inputs().size() == 3);
      return (v->node()->inputs().size() > 2)
          ? computeTwoOperandWithAlpha("aten_add", v, add_lambda)
          : computeTwoOperand("aten_add", v, add_lambda);
    } break;

    case aten::_cast_Float: {
      return computeOneOperand("aten_cast_float", v, [](const ExprHandle& a) {
        return cast<float>(a);
      });
    } break;

    case aten::sub: {
      auto sub_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        return lhs - rhs;
      };
      TORCH_INTERNAL_ASSERT(
          v->node()->inputs().size() == 2 || v->node()->inputs().size() == 3);
      return (v->node()->inputs().size() > 2)
          ? computeTwoOperandWithAlpha("aten_sub", v, sub_lambda)
          : computeTwoOperand("aten_sub", v, sub_lambda);
    } break;

    case aten::mul: {
      return computeTwoOperand(
          "aten_mul", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs * rhs;
          });
    } break;

    case aten::div: {
      return computeTwoOperand(
          "aten_div", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs / rhs;
          });
    } break;

    case aten::__and__: {
      return computeTwoOperand(
          "aten_and", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs & rhs;
          });
    } break;

    case aten::__or__: {
      return computeTwoOperand(
          "aten_or", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs | rhs;
          });
    } break;

    case aten::__xor__: {
      return computeTwoOperand(
          "aten_xor", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs ^ rhs;
          });
    } break;

    case aten::__lshift__: {
      return computeTwoOperand(
          "aten_lshift", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs << rhs;
          });
    } break;

    case aten::__rshift__: {
      return computeTwoOperand(
          "aten_rshift", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs >> rhs;
          });
    } break;

    case aten::addcmul: {
      return computeFourOperand(
          "aten_addcmul",
          v,
          [](const ExprHandle& a0,
             const ExprHandle& a1,
             const ExprHandle& a2,
             const ExprHandle& a3) { return a0 + a3 * a1 * a2; });
    } break;

    case aten::eq: {
      return computeTwoOperand(
          "aten_eq", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs == rhs);
          });
    } break;

    case aten::ne: {
      return computeTwoOperand(
          "aten_ne", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs != rhs);
          });
    } break;
    case aten::ge: {
      return computeTwoOperand(
          "aten_ge", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs >= rhs);
          });
    } break;

    case aten::gt: {
      return computeTwoOperand(
          "aten_gt", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs > rhs);
          });
    } break;

    case aten::le: {
      return computeTwoOperand(
          "aten_le", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs <= rhs);
          });
    } break;

    case aten::lt: {
      return computeTwoOperand(
          "aten_lt", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs < rhs);
          });
    } break;

    case aten::min: {
      return computeTwoOperand(
          "aten_min", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Min::make(lhs, rhs, false);
          });
    } break;

    case aten::max: {
      return computeTwoOperand(
          "aten_max", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Max::make(lhs, rhs, false);
          });
    } break;

    case aten::clamp: {
      bool noMin = false;
      bool noMax = false;
      if (v->node()->input(1)->node()->kind() == prim::Constant) {
        const auto val = toIValue(v->node()->input(1)).value();
        if (val.isNone()) {
          noMin = true;
        }
      }

      if (v->node()->input(2)->node()->kind() == prim::Constant) {
        const auto val = toIValue(v->node()->input(2)).value();
        if (val.isNone()) {
          noMax = true;
        }
      }

      return computeThreeOperand(
          "aten_clamp",
          v,
          [noMin, noMax](
              const ExprHandle& in,
              const ExprHandle& min,
              const ExprHandle& max) {
            if (noMin && noMax) {
              return in;
            } else if (noMin) {
              return CompareSelect::make(in, max, max, in, kGT);
            } else if (noMax) {
              return CompareSelect::make(in, min, min, in, kLT);
            } else {
              return CompareSelect::make(
                  in,
                  min,
                  min,
                  CompareSelect::make(in, max, max, in, kGT),
                  kLT);
            }
          });
    } break;

    case aten::sigmoid: {
      return computeOneOperand(
          "aten_sigmoid", v, [](const ExprHandle& a) { return sigmoid(a); });
    } break;

    case aten::reciprocal: {
      return computeOneOperand("aten_reciprocal", v, [](const ExprHandle& a) {
        return ExprHandle(1.0f) / a;
      });
    } break;

    case aten::neg: {
      return computeOneOperand("aten_neg", v, [](const ExprHandle& a) {
        return ExprHandle(-0) - a;
      });
    } break;

    case aten::relu: {
      return computeOneOperand("aten_relu", v, [](const ExprHandle& a) {
        auto zero = Cast::make(a.dtype(), 0);
        return ifThenElse(CompareSelect::make(a, zero, kLT), zero, a);
      });
    } break;

    case aten::log: {
      return computeOneOperand(
          "aten_log", v, [](const ExprHandle& a) { return log(a); });
    } break;

    case aten::log10: {
      return computeOneOperand(
          "aten_log10", v, [](const ExprHandle& a) { return log10(a); });
    } break;

    case aten::log2: {
      return computeOneOperand(
          "aten_log2", v, [](const ExprHandle& a) { return log2(a); });
    } break;

    case aten::exp: {
      return computeOneOperand(
          "aten_exp", v, [](const ExprHandle& a) { return exp(a); });
    } break;

    case aten::expm1: {
      return computeOneOperand(
          "aten_expm1", v, [](const ExprHandle& a) { return expm1(a); });
    } break;

    case aten::erf: {
      return computeOneOperand(
          "aten_erf", v, [](const ExprHandle& a) { return erf(a); });
    } break;

    case aten::erfc: {
      return computeOneOperand(
          "aten_erfc", v, [](const ExprHandle& a) { return erfc(a); });
    } break;

    case aten::cos: {
      return computeOneOperand(
          "aten_cos", v, [](const ExprHandle& a) { return cos(a); });
    } break;

    case aten::sin: {
      return computeOneOperand(
          "aten_sin", v, [](const ExprHandle& a) { return sin(a); });
    } break;

    case aten::tan: {
      return computeOneOperand(
          "aten_tan", v, [](const ExprHandle& a) { return tan(a); });
    } break;

    case aten::type_as: {
      return computeTwoOperand(
          "aten_type_as", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Cast::make(rhs.dtype(), lhs);
          });
    } break;

    case aten::rand_like: {
      hasRandom_ = true;
      return computeOneOperand("aten_rand_like", v, [](const ExprHandle& a) {
        return Intrinsics::make(IntrinsicsOp::kRand, a.dtype());
      });
    } break;

    case aten::pow: {
      return computeTwoOperand(
          "aten_pow", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            const FloatImm* floatImm = rhs.AsNode<FloatImm>();
            if (floatImm) {
              float imm = floatImm->value();
              if (imm == 1.0f) {
                return lhs;
              } else if (imm == 2.0f) { // NOLINT
                return lhs * lhs;
              } else if (imm == 3.0f) { // NOLINT
                return (lhs * lhs) * lhs;
              } else if (imm == 4.0f) { // NOLINT
                ExprHandle tmp = lhs * lhs;
                return tmp * tmp;
              } else if (imm == 0.5f) { // NOLINT
                return sqrt(lhs);
              } else if (imm == 0.0f) {
                return ExprHandle(1.0f);
              } else if (imm == -0.5f) { // NOLINT
                return rsqrt(lhs);
              } else if (imm == -1.0f) {
                return ExprHandle(1.0f) / lhs;
              } else if (imm == -2.0f) { // NOLINT
                return ExprHandle(1.0f) / (lhs * lhs);
              }
            }

            const Cast* floatCast = rhs.AsNode<Cast>();
            if (floatCast) {
              const IntImm* intImm =
                  dynamic_cast<const IntImm*>(floatCast->src_value());
              if (intImm) {
                float imm = static_cast<float>(intImm->value());
                if (imm == 1) {
                  return lhs;
                } else if (imm == 2) {
                  return lhs * lhs;
                } else if (imm == 3) {
                  return (lhs * lhs) * lhs;
                } else if (imm == 4) {
                  ExprHandle tmp = lhs * lhs;
                  return tmp * tmp;
                } else if (imm == 0) {
                  return ExprHandle(1.0f);
                } else if (imm == -1) {
                  return ExprHandle(1.0f) / lhs;
                } else if (imm == -2) {
                  return ExprHandle(1.0f) / (lhs * lhs);
                }
              }
            }
            return pow(lhs, rhs);
          });
    } break;

    case aten::fmod: {
      return computeTwoOperand(
          "aten_fmod", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return fmod(lhs, rhs);
          });
    } break;

    case aten::lerp: {
      return computeThreeOperand(
          "aten_lerp",
          v,
          [](const ExprHandle& a,
             const ExprHandle& end,
             const ExprHandle& weight) { return a + weight * (end - a); });
    } break;
    case aten::remainder: {
      return computeTwoOperand(
          "aten_remainder",
          v,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return fmod((rhs + fmod(lhs, rhs)), rhs);
          });

    } break;

    case aten::acos: {
      return computeOneOperand(
          "aten_acos", v, [](const ExprHandle& a) { return acos(a); });
    } break;

    case aten::asin: {
      return computeOneOperand(
          "aten_asin", v, [](const ExprHandle& a) { return asin(a); });
    } break;

    case aten::cosh: {
      return computeOneOperand(
          "aten_cosh", v, [](const ExprHandle& a) { return cosh(a); });
    } break;

    case aten::sinh: {
      return computeOneOperand(
          "aten_sinh", v, [](const ExprHandle& a) { return sinh(a); });
    } break;

    case aten::atan: {
      return computeOneOperand(
          "aten_atan", v, [](const ExprHandle& a) { return atan(a); });
    } break;

    case aten::atan2: {
      return computeTwoOperand(
          "aten_atan2", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return atan2(lhs, rhs);
          });
    } break;

    case aten::tanh: {
      return computeOneOperand(
          "aten_tanh", v, [](const ExprHandle& a) { return tanh(a); });
    } break;

    case aten::sqrt: {
      return computeOneOperand(
          "aten_sqrt", v, [](const ExprHandle& a) { return sqrt(a); });
    } break;

    case aten::rsqrt: {
      return computeOneOperand(
          "aten_rsqrt", v, [](const ExprHandle& a) { return rsqrt(a); });
    } break;

    case aten::abs: {
      return computeOneOperand(
          "aten_abs", v, [](const ExprHandle& a) { return fabs(a); });
    } break;

    case aten::ceil: {
      return computeOneOperand(
          "aten_ceil", v, [](const ExprHandle& a) { return ceil(a); });
    } break;

    case aten::floor: {
      return computeOneOperand(
          "aten_floor", v, [](const ExprHandle& a) { return floor(a); });
    } break;

    case aten::round: {
      return computeOneOperand(
          "aten_round", v, [](const ExprHandle& a) { return round(a); });
    } break;

    case aten::trunc: {
      return computeOneOperand(
          "aten_trunc", v, [](const ExprHandle& a) { return trunc(a); });
    } break;

    case aten::threshold: {
      return computeThreeOperand(
          "aten_threshold",
          v,
          [](const ExprHandle& a,
             const ExprHandle& threshold,
             const ExprHandle& value) {
            return ifThenElse(CompareSelect::make(a, threshold, kLE), value, a);
          });
    } break;

    case aten::where: {
      return computeConditionWithTwoOperand(
          "aten_where",
          v,
          [](const ExprHandle& a0, const ExprHandle& a1, const ExprHandle& a2) {
            return ifThenElse(a0, a1, a2);
          });
    } break;

    case aten::frac: {
      return computeOneOperand(
          "aten_frac", v, [](const ExprHandle& a) { return a - floor(a); });
    } break;

    case aten::lgamma: {
      return computeOneOperand(
          "aten_lgamma", v, [](const ExprHandle& a) { return lgamma(a); });
    } break;

    case prim::ConstantChunk: {
      return Compute(
          "prim_constantchunk",
          dimsFromSizes(sizesForValue(v)),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            int64_t dim = n->i(attr::dim);
            int64_t chunks = n->i(attr::chunks);
            std::vector<ExprHandle> indices(axes.begin(), axes.end());
            return chunk(
                tensors_.at(n->inputs()[0]->unique()),
                v->offset(),
                dim,
                chunks,
                indices);
          });
    }

    case aten::cat: {
      return Compute(
          "aten_cat",
          dimsFromSizes(sizesForValue(v)),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            auto inputs = n->inputs()[0]->node()->inputs();
            size_t dim = n->inputs()[1]->node()->i(attr::value);

            std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
            ExprHandle load = tensorOrConstant(inputs[0], newAxes);
            size_t offset = bufferSizes(tensors_.at(inputs[0]->unique()))[dim];
            newAxes[dim] = newAxes[dim] - IntImm::make(offset);

            for (size_t ii = 1; ii < inputs.size(); ++ii) {
              load = ifThenElse(
                  CompareSelect::make(axes[dim], IntImm::make(offset), kLT),
                  load,
                  tensorOrConstant(inputs[ii], newAxes));
              offset += bufferSizes(tensors_.at(inputs[ii]->unique()))[dim];
              newAxes[dim] = axes[dim] - IntImm::make(offset);
            }

            return load;
          });
    }
    case aten::slice: {
      return Compute(
          "aten_slice",
          dimsFromSizes(sizesForValue(v)),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            int dim = constant(n->inputs()[1]).AsNode<IntImm>()->value();
            ExprHandle start = constant(n->inputs()[2]);
            ExprHandle stride = constant(n->inputs()[4]);

            std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
            newAxes[dim] = stride * newAxes[dim] + start;
            return tensorOrConstant(n->inputs()[0], newAxes);
          });
    }

    case aten::unsqueeze: {
      return Compute(
          "aten_unsqueeze",
          dimsFromSizes(sizesForValue(v)),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            int64_t dim = constant(n->inputs()[1]).AsNode<IntImm>()->value();
            if (dim < 0) {
              if (axes.size() == 0) {
                throw malformed_input("axes are zero handling unsqueeze");
              }
              dim += axes.size();
            }
            // To construct an expression for an 'unsqueezed' tensor we need to
            // drop the DIM-th axis, i.e.
            //    unsqueezed_v[i,j,k,l] = v[i,j,l] # dim = 2 - drop index 'k'
            //                 0 1 2 3
            std::vector<ExprHandle> indices;
            int64_t i = 0;
            for (auto a : axes) {
              if (i++ != dim) {
                indices.emplace_back(ExprHandle(a.node()));
              }
            }

            return tensorOrConstant(n->inputs()[0], indices);
          });
    }

    case aten::sum: {
      return computeSum(v);
    }

    default: {
      throw std::runtime_error("Unhandled node kind");
    }
  }
}

void TensorExprKernel::flattenTensors(BackendType backendType) {
  if (backendType != BackendType::kCudaCodeGen &&
      backendType != BackendType::kBlockCodeGen) {
    // We only need to flatten for GPU, for other backends just use the same
    // tensors.
    flatTensorOutputs_ = tensorOutputs_;
    return;
  }

  flatTensorOutputs_.resize(tensorOutputs_.size());
  for (size_t tensorIdx = 0; tensorIdx < tensorOutputs_.size(); tensorIdx++) {
    Tensor* tensor = tensorOutputs_[tensorIdx];
    ExprHandle totalCount = ExprHandle(tensor->dim(0));
    for (int i = 1; i < tensor->ndim(); i++) {
      const IntImm* totalCountImm = totalCount.AsNode<IntImm>();
      const IntImm* tensorDimImm = dynamic_cast<const IntImm*>(tensor->dim(i));
      if (totalCountImm && tensorDimImm) {
        // TODO: switch to real constant folding when it is available.
        totalCount = ExprHandle(totalCountImm->value() * tensorDimImm->value());
      } else {
        totalCount = totalCount * ExprHandle(tensor->dim(i));
      }
    }
    // Flatten the index for GPU kernels.
    // TODO: move this to fusing axis when it is ready.
    Tensor* newOut = Compute(
        tensor->func_var()->name_hint() + "_flat",
        {totalCount},
        [tensor](const VarHandle& index) -> ExprHandle {
          std::vector<ExprHandle> dims;
          ExprHandle value = index;
          for (int i = tensor->ndim() - 1; i >= 0; i--) {
            ExprHandle idx = value;
            if (i > 0) {
              idx = Mod::make(value, ExprHandle(tensor->dim(i)));
            }
            dims.push_back(idx);
            value = value / ExprHandle(tensor->dim(i));
          }
          std::reverse(dims.begin(), dims.end());
          return tensor->call(dims);
        });
    flatTensorOutputs_[tensorIdx] = newOut;
  }
}

Stmt* TensorExprKernel::generateStmt(BackendType backendType) {
  flattenTensors(backendType);

  torch::jit::tensorexpr::LoopNest l(flatTensorOutputs_);
  GRAPH_DEBUG("Original Stmt:\n", std::to_string(l.root_stmt()), "\n");

  bool hasReduction = NodeFinder<ReduceOp>::find(l.root_stmt()).size() != 0;

  // Compute non-output tensors_ inline
  for (auto& p : tensors_) {
    if (!l.hasLoopBodyFor(p.second) || hasReduction) {
      continue;
    }
    Stmt* loop = l.getLoopBodyFor(p.second);
    if (torch::jit::tensorexpr::HasRand(loop).has_rand()) {
      l.computeInlineWithRandom(loop);
    } else {
      l.computeInline(loop);
    }
  }
  if (backendType == kCudaCodeGen) {
    for (size_t i = 0; i < flatTensorOutputs_.size(); i++) {
      Tensor* tensor = flatTensorOutputs_[i];

      // For every output tensor we've created a flattened 1D tensor - let's
      // mark the original output tensor with computeInline
      l.computeInline(l.getLoopBodyFor(tensorOutputs_[i]));

      int loopLevels = getTECudaPointwiseLoopLevels();
      const int kDefaultLoopLevels = 2;
      loopLevels = (loopLevels > 0) ? loopLevels : kDefaultLoopLevels;
      int blockCount = getTECudaPointwiseBlockCount();
      int blockSize = getTECudaPointwiseBlockSize();

      if (loopLevels == 2) {
        For* outer;
        For* inner;
        const int kDefaultBlockSize = 512;
        if (blockSize < 0) {
          blockSize = kDefaultBlockSize;
        }
        std::vector<For*> loops = l.getLoopStmtsFor(tensor);
        l.splitWithMask(loops[0], blockSize, &outer, &inner);
        l.setGPUBlockIndex(outer, 0);
        l.setGPUThreadIndex(inner, 0);
      } else if (loopLevels == 3) {
        For* outer;
        For* inner;
        For* inner1;
        For* inner2;
        // TODO: change the number of microprocessors
        const int kDefaultBlockCount = 1280;
        const int kDefaultBlockSize = 256;
        blockCount = (blockCount > 0) ? blockCount : kDefaultBlockCount;
        blockSize = (blockSize > 0) ? blockSize : kDefaultBlockSize;
        std::vector<For*> loops = l.getLoopStmtsFor(tensor);
        l.splitWithMask(loops[0], blockCount * blockSize, &outer, &inner);
        l.splitWithMask(inner, blockSize, &inner1, &inner2);
        l.setGPUBlockIndex(inner1, 0);
        l.setGPUThreadIndex(inner2, 0);
      } else {
        throw std::runtime_error(
            "Invalid loop-level: " + c10::to_string(loopLevels));
      }
    }
  }

  if (backendType == kBlockCodeGen) {
    auto block_analysis = std::make_unique<CreateBufferMap>();
    for (size_t i = 0; i < flatTensorOutputs_.size(); i++) {
      const int default_fp16_blocksize = 16;
      const int default_uint8_blocksize = 32;
      int blockSize = default_fp16_blocksize;
      // We only handle looplevels == 2 for now
      Tensor* tensor = flatTensorOutputs_[i];
      // Run Block analysis to get multi dim buffer info
      auto root_stmt = l.root_stmt();
      root_stmt->accept(block_analysis.get());

      if (tensor->buf()->dtype().scalar_type() == ScalarType::Byte) {
        blockSize = default_uint8_blocksize;
      }
      l.computeInline(l.getLoopBodyFor(tensorOutputs_[i]));
      For* outer;
      For* inner;
      std::vector<For*> loops = l.getLoopStmtsFor(tensor);
      TORCH_INTERNAL_ASSERT(loops.size() > 0, "loops should not be empty");
      l.splitWithMask(loops[0], blockSize, &outer, &inner);
      l.setGPUBlockIndex(outer, 0);
      l.setGPUThreadIndex(inner, 0);
      l.setBufferMap(outer, block_analysis->getBufferMap());
    }
  }

  l.prepareForCodegen();

  if (backendType == kLLVMCodeGen && !hasReduction) {
    std::vector<For*> innerLoops;
    std::vector<For*> worklist;

    // Find outer-most For loops
    if (For* rootF = dynamic_cast<For*>(l.root_stmt())) {
      worklist.push_back(rootF);
    } else if (Block* body = dynamic_cast<Block*>(l.root_stmt())) {
      std::vector<Block*> blocks = {body};
      while (blocks.size()) {
        Block* b = blocks.back();
        blocks.pop_back();

        for (Stmt* s : *b) {
          if (For* f = dynamic_cast<For*>(s)) {
            worklist.push_back(f);
          } else if (Block* b2 = dynamic_cast<Block*>(s)) {
            blocks.push_back(b2);
          }
        }
      }
    }

    // Traverse the For loop nest find inner-most loops, which are
    // vectorization candidates.
    while (worklist.size()) {
      For* f = worklist.back();
      worklist.pop_back();

      bool containsSubLoops = false;
      if (Block* body = dynamic_cast<Block*>(f->body())) {
        for (Stmt* s2 : *body) {
          if (For* f2 = dynamic_cast<For*>(s2)) {
            containsSubLoops = true;
            worklist.push_back(f2);
          }
        }
      }

      if (!containsSubLoops) {
        innerLoops.push_back(f);
      }
    }

    // vectorize inner loops.
    for (For* loop : innerLoops) {
      For* outer1;
      For* split1;
      For* tail1;

      static const int kBodyVectorWidth = 8;
      l.splitWithTail(loop, kBodyVectorWidth, &outer1, &split1, &tail1);
      l.vectorize(split1);

      if (tail1) {
        For* outer2;
        For* split2;
        For* tail2;
        static const int kTailVectorWidth = 4;
        l.splitWithTail(tail1, kTailVectorWidth, &outer2, &split2, &tail2);
        l.vectorize(split2);
      }
    }
  }

  Stmt* stmt = l.root_stmt();
  // Arithmetic Simplification.
  stmt = IRSimplifier::simplify(stmt);
  GRAPH_DEBUG("Final Stmt:\n", std::to_string(stmt), "\n");
  return stmt;
}

std::string TensorExprKernel::getCodeGenName(BackendType backendType) {
  switch (backendType) {
    case kCudaCodeGen:
      return "cuda_codegen";
    case kLLVMCodeGen:
      return "llvm_codegen";
    case kSimpleIREval:
      return "simple_ir_eval";
    case kBlockCodeGen:
      return "block_codegen";
    default:
      throw std::runtime_error(
          "invalid backend type: " +
          c10::to_string(static_cast<int>(backendType)));
  }
}

std::vector<CodeGen::BufferArg> TensorExprKernel::prepareBufferArgs() {
  std::vector<CodeGen::BufferArg> params;
  for (auto const& arg : kernelArgs_) {
    params.push_back(arg.buffer());
    for (auto const& size : arg.sizes()) {
      params.emplace_back(size.var);
    }
    for (auto const& stride : arg.strides()) {
      params.emplace_back(stride.var);
    }
  }
  for (auto& o : flatTensorOutputs_) {
    params.emplace_back(o);
  }
  return params;
}

template <typename T>
static bool isValidPrimProperty(const c10::optional<T>& a, T b) {
  return !a.has_value() || *a == b;
}

TensorExprKernel::BackendType TensorExprKernel::inferBackendTypeFromDevice(
    at::Device device) {
  BackendType backendType = BackendType::kUninitialized;
  if (device.type() == at::kCUDA) {
    backendType = kCudaCodeGen;
  } else if (device.type() == at::kCPU && getTEGenerateBlockCode()) {
    backendType = kBlockCodeGen;
  } else if (device.type() == at::kCPU) {
#ifdef TORCH_ENABLE_LLVM
    backendType = kLLVMCodeGen;
#else
    backendType = kSimpleIREval;
#endif
  } else {
    throw std::runtime_error("Invalid device type");
  }
  return backendType;
}

void TensorExprKernel::bindInput(const torch::jit::Value* input) {
  auto const& t = input->type();
  switch (t->kind()) {
    case TypeKind::TensorType: {
      auto tt = input->type()->cast<TensorType>();
      Buffer inBuffer(
          "t" + input->debugName(),
          ToDtype(static_cast<ScalarType>(*tt->scalarType())),
          {0});
      std::vector<DimArg> inputTensorDims;
      for (size_t i = 0; i < *tt->sizes().size(); i++) {
        auto const size = *tt->sizes()[i];
        inputTensorDims.emplace_back(
            DimArg(IntImm::make(size), "i" + c10::to_string(i)));
      }
      auto const strides = tt->strides();
      tensors_.emplace(
          input->unique(),
          Compute(
              "input" + c10::to_string(tensors_.size() + 1),
              inputTensorDims,
              [&](const std::vector<VarHandle>& axes) {
                ExprHandle idx = 0;
                for (size_t i = 0; i < axes.size(); i++) {
                  idx = idx + axes[i] * IntImm::make(*strides[i]);
                }
                return inBuffer(idx);
              }));
      kernelArgs_.emplace_back(
          inBuffer, std::vector<ShapeArg>(), std::vector<ShapeArg>());
      break;
    }
    case TypeKind::FloatType: {
      VarHandle v("v" + input->debugName(), kFloat);
      kernelArgs_.emplace_back(v);
      scalars_.emplace(input->unique(), v);
      break;
    }
    case TypeKind::BoolType: {
      VarHandle v("v" + input->debugName(), kBool);
      kernelArgs_.emplace_back(v);
      scalars_.emplace(input->unique(), v);
      break;
    }
    case TypeKind::IntType: {
      VarHandle v("v" + input->debugName(), kInt);
      kernelArgs_.emplace_back(v);
      scalars_.emplace(input->unique(), v);
      break;
    }
    default: {
      throw unsupported_dtype();
      break;
    }
  }
}

namespace {

// Remove all indices from axes positions.
std::vector<VarHandle> squeezeIndices(
    const ParameterList& indices,
    const std::vector<size_t>& axes) {
  std::vector<VarHandle> indices_squeezed;
  for (size_t dim = 0; dim < indices.size(); ++dim) {
    if (!std::count(axes.begin(), axes.end(), dim)) {
      indices_squeezed.push_back(indices[dim]);
    }
  }
  return indices_squeezed;
}

} // namespace

Tensor* TensorExprKernel::computeSum(const torch::jit::Value* v) {
  auto reduction_info = getReductionInfo(v->node());
  return Reduce(
      "sum",
      reduction_info.outputDims,
      Sum(),
      [&](ParameterList& indices) {
        const auto& axes = reduction_info.axes;
        // "Squeeze" out indices inserted when keepdim is set.
        auto indices_squeezed =
            reduction_info.keepdim ? squeezeIndices(indices, axes) : indices;
        TORCH_INTERNAL_ASSERT(axes.size() <= indices_squeezed.size());
        // Move innermost indices into axes positions:
        //   1. Fill the outermost indices first.
        //   2. Insert the innermost indices into the correct axis position,
        //   displacing the outermost indices as needed.
        std::vector<ExprHandle> indices_exprs;
        size_t i = 0;
        for (; i < indices_squeezed.size() - axes.size(); ++i) {
          indices_exprs.push_back(indices_squeezed[i]);
        }
        for (auto axis : axes) {
          indices_exprs.insert(
              indices_exprs.begin() + axis, indices_squeezed[i]);
          ++i;
        }
        auto indexed = tensorOrConstant(v->node()->input(0), indices_exprs);
        if (reduction_info.dtype) {
          return Cast::make(*reduction_info.dtype, indexed);
        } else {
          return indexed;
        }
      },
      reduction_info.reductionDims);
}

TensorExprKernel::ReductionInfo TensorExprKernel::getReductionInfo(
    const torch::jit::Node* node) {
  std::vector<size_t> axes;
  bool keepdim = false;
  // aten::sum takes the input tensor named self.
  auto sizes = sizesForValue(node->namedInput(attr::self));
  const auto inputs = node->inputs();
  int rank = sizes.size();
  if (inputs.size() > 2) {
    auto nodeAxes = getReductionAxes(node);
    // Canonicalize axes: wrap around, sort and make unique.
    for (auto axis : nodeAxes) {
      axes.push_back(at::maybe_wrap_dim(axis, rank));
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    keepdim = node->get(attr::keepdim)->toBool();
  } else {
    axes.resize(sizes.size());
    std::iota(axes.begin(), axes.end(), 0);
  }
  // Axes go into reduction dimensions.
  std::vector<DimArg> reductionDims;
  reductionDims.reserve(sizes.size());
  for (size_t axis : axes) {
    reductionDims.emplace_back(sizes[axis]);
  }
  auto allDims = dimsFromSizes(sizes);
  std::vector<DimArg> outputDims;
  // Output dimensions are the complement of axes. When keepdim is set, a
  // one-sized dimension is inserted for each axis.
  for (size_t dim = 0; dim < allDims.size(); ++dim) {
    if (!std::count(axes.begin(), axes.end(), dim)) {
      outputDims.emplace_back(sizes[dim]);
    } else if (keepdim) {
      outputDims.emplace_back(1);
    }
  }
  c10::optional<Dtype> dtype;
  auto dtypeValue = node->get(attr::dtype);
  if (!dtypeValue->isNone()) {
    auto scalarType = static_cast<ScalarType>(dtypeValue->toInt());
    dtype = ToDtype(scalarType);
  }
  return {reductionDims, outputDims, axes, keepdim, dtype};
}

std::vector<int64_t> TensorExprKernel::getReductionAxes(
    const torch::jit::Node* node) {
  std::vector<int64_t> axes;
  auto axesNode = node->namedInput(attr::dim)->node();
  // There are two possible representations for reduction axes:
  //   1. A prim::ListConstruct of integer constants.
  //   2. A prim::Constant list of integer ival's.
  // We need to handle both of them.
  if (axesNode->kind() == prim::ListConstruct) {
    for (auto axisNode : axesNode->inputs()) {
      axes.push_back(constant(axisNode).AsNode<IntImm>()->value());
    }
    return axes;
  }
  TORCH_INTERNAL_ASSERT(axesNode->kind() == prim::Constant);
  TORCH_INTERNAL_ASSERT(axesNode->kindOf(attr::value) == AttributeKind::ival);
  const auto& genericList = axesNode->ival(attr::value).toList();
  for (const IValue axisNode : genericList) {
    axes.push_back(axisNode.toInt());
  }
  return axes;
}

void TensorExprKernel::compile() {
  KernelScope kernelScope(&kernelArena_);
  // Bind inputs to buffers.
  nInputs_ = graph_->inputs().size();
  for (auto const& input : graph_->inputs()) {
    bindInput(input);
    inputTypes_.push_back(input->type());
  }

  // Bind nodes to tensor compute expressions.
  for (auto const& n : graph_->nodes()) {
    if (n->kind() == prim::Constant || n->kind() == prim::ListConstruct) {
      continue;
    } else {
      for (auto const& output : n->outputs()) {
        if (output->hasUses()) {
          tensors_.emplace(output->unique(), computeValue(output));
        }
      }
    }
    if (hasRandom_ && hasBroadcast_) {
      throw std::runtime_error(
          "Cannot support broadcast and random within one kernel");
    }
  }

  // Move output operands from `tensors_` to `tensorOutputs_`
  for (const auto& output : graph_->outputs()) {
    if (!tensors_.count(output->unique())) {
      throw malformed_input("cannot find output Tensor");
    }
    tensorOutputs_.emplace_back(tensors_.at(output->unique()));
    tensors_.erase(output->unique());
  }

  device_ = *pickDeviceType(graph_->inputs());
  BackendType backendType = inferBackendTypeFromDevice(device_);
  Stmt* stmt = generateStmt(backendType);
  // Set up formal params (inputs, then outputs) for kernel.
  std::vector<CodeGen::BufferArg> params = prepareBufferArgs();

  // Generate code.
  codegen_ = CreateCodeGen(getCodeGenName(backendType), stmt, params, device_);
}

TensorExprKernel::TensorExprKernel(const std::shared_ptr<Graph>& subgraph)
    : graph_(subgraph), code_(subgraph, "") {
  if (!fallbackAllowed()) {
    compile();
    return;
  }

  try {
    compile();
  } catch (...) {
    fallback_ = true;
  }
}

void TensorExprKernel::run(Stack& stack) {
  if (fallbackEnforced()) {
    fallback(stack);
    return;
  }
  if (!fallbackAllowed()) {
    runKernel(stack);
    return;
  }

  if (fallback_) {
    fallback(stack);
    return;
  }
  try {
    runKernel(stack);
  } catch (...) {
    fallback_ = true;
    fallback(stack);
  }
}

std::vector<CodeGen::CallArg> TensorExprKernel::prepareRunArgs(
    const at::ArrayRef<IValue>& inputs,
    std::vector<at::Tensor>& outputs) {
  std::map<const Expr*, int32_t> varToSize;

  std::vector<CodeGen::CallArg> runArgs;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto const& input = inputs[i];
    if (input.isInt()) {
      runArgs.emplace_back((int32_t)input.toInt());
    } else if (input.isDouble()) {
      runArgs.emplace_back((float)input.toDouble());
    } else if (input.isTensor()) {
      auto const& tensor = input.toTensor();
      runArgs.emplace_back(tensor.data_ptr());
      for (auto const& size : kernelArgs_[i].sizes()) {
        int32_t s = tensor.sizes()[size.idx];
        runArgs.emplace_back(s);
        varToSize[size.var.node()] = s;
      }
      for (auto const& stride : kernelArgs_[i].strides()) {
        int32_t s = tensor.strides()[stride.idx];
        runArgs.emplace_back(s);
      }
    }
  }

  for (auto& o : tensorOutputs_) {
    std::vector<int64_t> tensorSize;
    for (const Expr* dim : o->dims()) {
      auto it = varToSize.find(dim);
      if (it != varToSize.end()) {
        tensorSize.push_back(it->second);
      } else {
        const IntImm* s = dynamic_cast<const IntImm*>(dim);
        if (!s) {
          throw malformed_input("output expected Int", dim);
        }
        tensorSize.push_back(s->value());
      }
    }

    outputs.push_back(at::empty(
        tensorSize, c10::TensorOptions(tensorType(o)).device(device_)));
    runArgs.emplace_back(outputs.back().data_ptr());
  }
  return runArgs;
}

Stmt* TensorExprKernel::getCodeGenStmt() {
  return codegen_->stmt();
}

void TensorExprKernel::runKernel(Stack& stack) {
  KernelScope kernelScope(&kernelArena_);

  // Set up arguments (inputs, then outputs) for kernel call.
  auto inputs = last(stack, nInputs_);
  std::vector<at::Tensor> outputs;

  std::vector<CodeGen::CallArg> runArgs = prepareRunArgs(inputs, outputs);

  // Call the kernel.
  codegen_->call(runArgs);

  // Update the stack.
  drop(stack, nInputs_);
  for (auto& o : outputs) {
    push_one(stack, std::move(o));
  }
}
