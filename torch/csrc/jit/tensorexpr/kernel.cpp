#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <ATen/ExpandUtils.h>
#include <ATen/TensorGeometry.h>
#include <c10/util/string_utils.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
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
static bool te_must_use_llvm_on_cpu = true;
static bool cat_wo_conditionals = false; // NOLINT

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
  if (tensorexpr::getTEGenerateBlockCode()) {
    return false;
  }
  if (!enable_c_str) {
    return fallback_allowed;
  }
  if (std::string(enable_c_str) == "2") {
    return true;
  }
  return false;
}

bool dontUseLLVMFlag() {
  static const char* enable_c_str =
      std::getenv("PYTORCH_TENSOREXPR_DONT_USE_LLVM");
  if (!enable_c_str) {
    return false;
  }
  return std::string(enable_c_str) == "1";
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

bool& getTEMustUseLLVMOnCPU() {
  return te_must_use_llvm_on_cpu;
}

bool& getCatWoConditionals() {
  return cat_wo_conditionals;
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

size_t normalizeAndCheckIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }

  if (idx < 0 || idx >= list_size) {
    AT_ERROR("Invalid index ", idx, " for list_size", list_size);
  }
  return static_cast<size_t>(idx);
}

static at::ScalarType tensorType(Tensor* t) {
  return static_cast<at::ScalarType>(t->buf()->dtype().scalar_type());
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
    int64_t dim,
    int64_t chunks,
    const std::vector<ExprHandle>& axes) {
  auto norm_dim = normalizeAndCheckIndex(dim, axes.size());
  auto sizes = bufferSizes(t);
  size_t step = sizes[norm_dim] / chunks;

  std::vector<ExprHandle> indices;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i == norm_dim) {
      indices.push_back(axes[i] + IntImm::make((int)chunkIdx * (int)step));
    } else {
      indices.push_back(axes[i]);
    }
  }

  return t->call(indices);
}

ExprHandle promoteToDtype(ExprHandle e, ScalarType dt) {
  if (e.dtype().scalar_type() == dt) {
    return e;
  }

  switch (dt) {
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
  return e;
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
    case aten::to:
    case aten::sigmoid:
    case aten::reciprocal:
    case aten::neg:
    case aten::relu:
    case aten::batch_norm:
    case aten::isnan:
    case aten::log:
    case aten::log10:
    case aten::log1p:
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
    case aten::hardtanh:
    case aten::sqrt:
    case aten::rsqrt:
    case aten::abs:
    case aten::ceil:
    case aten::floor:
    case aten::round:
    case aten::trunc:
    case aten::frac:
    case aten::lgamma:
    case aten::type_as:
    case aten::masked_fill:
      return sizesForValue(v->node()->input(0));

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
      // Negative dim will correspond to dim = dim + input.dim().
      auto const& n = v->node();
      auto inputs = n->input(0)->node()->inputs();
      if (inputs.size() == 0) {
        throw std::runtime_error("Empty input list is passed to aten::cat");
      }

      TORCH_INTERNAL_ASSERT(n->input(1)->node()->kind() == prim::Constant);
      int64_t dim = n->input(1)->node()->i(attr::value);
      auto shape = sizesForValue(inputs[0]);
      size_t norm_dim = normalizeAndCheckIndex(dim, shape.size());
      ExprHandle concat_dim_size = 0;
      for (auto input : inputs) {
        concat_dim_size = concat_dim_size + sizesForValue(input)[norm_dim];
      }
      concat_dim_size = IRSimplifier::simplify(concat_dim_size);
      shape[norm_dim] = concat_dim_size;
      return shape;
    }

    case aten::softmax:
    case aten::log_softmax:
      // Output of softmax / log_softmax has the same shape as input 0.
      return sizesForValue(v->node()->input(0));

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

ExprHandle promoteIntegerToDefaultType(const ExprHandle& e) {
  auto scalarType = static_cast<c10::ScalarType>(e.dtype().scalar_type());
  if (!c10::isIntegralType(scalarType, /*includeBool*/ true)) {
    return e;
  }

  auto defaultType = c10::typeMetaToScalarType(c10::get_default_dtype());

  // We intend to promote Integers to floating-point types
  TORCH_INTERNAL_ASSERT(
      !c10::isIntegralType(defaultType, /*includeBool*/ true));

  return Cast::make(
      Dtype(
          static_cast<tensorexpr::ScalarType>(defaultType), e.dtype().lanes()),
      e);
}

ExprHandle promoteHalfToFloat(const ExprHandle& e) {
  auto scalarType = static_cast<c10::ScalarType>(e.dtype().scalar_type());
  auto floatType = static_cast<c10::ScalarType>(tensorexpr::ScalarType::Float);
  if (c10::isFloatingType(scalarType) &&
      (c10::elementSize(scalarType) < c10::elementSize(floatType))) {
    return Cast::make(
        Dtype(tensorexpr::ScalarType::Float, e.dtype().lanes()), e);
  } else {
    return e;
  }
}

bool TensorExprKernel::checkTypes(
    const ScalarType highType,
    const int typeConstraints) {
  if (typeConstraints == kAllTypes) {
    return true;
  }

  if (is_integral(highType)) {
    return (typeConstraints & kIntegralTypes) != 0;
  } else if (is_floating_point(highType)) {
    return (typeConstraints & kFloatingPointTypes) != 0;
  } else if (highType == ScalarType::Bool) {
    return (typeConstraints & kBoolType) != 0;
  }

  // assume JIT not supporting complex and qint yet
  TORCH_INTERNAL_ASSERT((typeConstraints & (kQintTypes | kComplexTypes)) == 0);
  return false;
}

void TensorExprKernel::promoteInputs(
    std::vector<ExprHandle>& inputs,
    const int typeConstraints) {
  if (inputs.empty()) {
    return;
  }

  // Find the highest type among the inputs.
  ScalarType highType = inputs[0].dtype().scalar_type();
  for (const auto input : inputs) {
    highType = promoteTypes(highType, input.dtype().scalar_type());
  }

  if (!checkTypes(highType, typeConstraints)) {
    throw unsupported_dtype();
  }

  for (ExprHandle& e : inputs) {
    e = promoteToDtype(e, highType);
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

  auto tt = *v->type()->castRaw<TensorType>()->scalarType();

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
  return ExprVectorToExprHandleVector(it->second->buf()->dims());
}

Tensor* TensorExprKernel::computeOneOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(const ExprHandle&)>& innerExpr,
    const int checkParamTypes) {
  auto const& n = v->node();
  auto const& shape = valueShape(n->inputs()[0]);
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, innerExpr, checkParamTypes](
          const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], indices)};
        promoteInputs(inputs, checkParamTypes);
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
        innerExpr,
    bool promote_inputs) {
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
      [this, v, innerExpr, promote_inputs](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], indices),
            tensorOrConstant(n->inputs()[1], indices),
            tensorOrConstant(n->inputs()[2], indices),
        };

        if (promote_inputs) {
          promoteInputs(inputs);
        }
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

namespace {

// Convert boolean to integer, if needed.
ExprHandle boolToInteger(const ExprHandle& x) {
  return x.dtype().scalar_type() == ScalarType::Bool ? cast<int>(x) : x;
}

} // namespace

c10::optional<ScalarType> findDtypeForValue(const torch::jit::Value* v) {
  if (v->type()->kind() == TypeKind::TensorType) {
    auto tt = v->type()->cast<TensorType>();
    if (tt->scalarType()) {
      return static_cast<ScalarType>(*tt->scalarType());
    }
  }
  return c10::nullopt;
}

Tensor* TensorExprKernel::computeValue(const torch::jit::Value* v) {
  switch (v->node()->kind()) {
    case aten::add: {
      auto add_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        return boolToInteger(lhs) + boolToInteger(rhs);
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

    case aten::to: {
      // see handling of aten::to in tensorexpr_fuser.cpp for why we only
      // need to handle the first input
      auto node = v->node();
      return computeOneOperand("aten_to", v, [node](const ExprHandle& a) {
        auto output_dtype = findDtypeForValue(node->output());
        TORCH_INTERNAL_ASSERT(output_dtype);
        return Cast::make(ToDtype(*output_dtype), a);
      });
    } break;

    case aten::sub: {
      auto sub_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        // NB: sub isn't supported on boolean, no need to promote to integer.
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
            return boolToInteger(lhs) * boolToInteger(rhs);
          });
    } break;

    case aten::div: {
      return computeTwoOperand(
          "aten_div", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return promoteIntegerToDefaultType(lhs) /
                promoteIntegerToDefaultType(rhs);
          });
    } break;

    case aten::__and__: {
      return computeTwoOperand(
          "aten_and", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) & boolToInteger(rhs);
          });
    } break;

    case aten::__or__: {
      return computeTwoOperand(
          "aten_or", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) | boolToInteger(rhs);
          });
    } break;

    case aten::__xor__: {
      return computeTwoOperand(
          "aten_xor", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) ^ boolToInteger(rhs);
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
            return Min::make(boolToInteger(lhs), boolToInteger(rhs), false);
          });
    } break;

    case aten::max: {
      return computeTwoOperand(
          "aten_max", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Max::make(boolToInteger(lhs), boolToInteger(rhs), false);
          });
    } break;

    case aten::masked_fill: {
      return computeThreeOperand(
          "aten_masked_fill",
          v,
          [](const ExprHandle& input,
             const ExprHandle& mask,
             const ExprHandle& value) {
            // value needs to promote to input, not vice versa
            auto val = promoteToDtype(value, input.dtype().scalar_type());
            return ifThenElse(mask, val, input);
          },
          /*promote_inputs*/ false);
    }

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
            auto cast = [&](const ExprHandle& e) {
              return Cast::make(in.dtype(), e);
            };

            if (noMin && noMax) {
              return in;
            } else if (noMin) {
              auto cmax = cast(max);
              return CompareSelect::make(in, cmax, cmax, in, kGT);
            } else if (noMax) {
              auto cmin = cast(min);
              return CompareSelect::make(in, cmin, cmin, in, kLT);
            } else {
              auto cmax = cast(max);
              auto cmin = cast(min);
              auto mm = CompareSelect::make(in, cmin, cmin, in, kLT);
              return CompareSelect::make(mm, cmax, cmax, mm, kGT);
            }
          },
          false /* promote_inputs */);
    } break;

    case aten::sigmoid: {
      return computeOneOperand("aten_sigmoid", v, [](const ExprHandle& a) {
        return sigmoid(promoteIntegerToDefaultType(a));
      });
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

    case aten::isnan: {
      return computeOneOperand("aten_isnan", v, [](const ExprHandle& a) {
        if (!a.dtype().is_floating_point()) {
          return IntImm::make(0);
        }
        return isnan(a);
      });
    } break;

    case aten::relu: {
      return computeOneOperand("aten_relu", v, [](const ExprHandle& a) {
        auto zero = Cast::make(a.dtype(), 0);
        return CompareSelect::make(a, zero, zero, a, kLT);
      });
    } break;

    case aten::batch_norm: {
      bool hasWeight = true;
      bool hasBias = true;

      if (v->node()->input(1)->node()->kind() == prim::Constant) {
        const auto val = toIValue(v->node()->input(1)).value();
        if (val.isNone()) {
          hasWeight = false;
        }
      }

      if (v->node()->input(2)->node()->kind() == prim::Constant) {
        const auto val = toIValue(v->node()->input(2)).value();
        if (val.isNone()) {
          hasBias = false;
        }
      }

      auto const& shape = valueShape(v->node()->inputs()[0]);
      return Compute(
          "aten_batch_norm",
          c10::fmap<DimArg>(shape),
          [this, v, hasWeight, hasBias](const std::vector<VarHandle>& axes) {
            TORCH_INTERNAL_ASSERT(axes.size() >= 2);
            auto const& n = v->node();
            // axes: N, C, H, W
            std::vector<ExprHandle> indices(axes.begin(), axes.end());
            ExprHandle c = indices[1];

            // Parameter list:
            // input, weight, bias, mean, var, training, momentum, eps,
            // cudnn_enabled
            std::vector<ExprHandle> inputs = {
                tensorOrConstant(n->input(0), indices), // input
                tensorOrConstant(n->input(3), {c}), // mean
                tensorOrConstant(n->input(4), {c}), // var
                constant(n->input(7)) // eps
            };
            if (hasWeight) {
              inputs.push_back(tensorOrConstant(n->input(1), {c}));
            }
            if (hasBias) {
              inputs.push_back(tensorOrConstant(n->input(2), {c}));
            }
            promoteInputs(inputs);

            ExprHandle input = inputs[0];
            ExprHandle mean = inputs[1];
            ExprHandle var = inputs[2];
            ExprHandle eps = inputs[3];
            ExprHandle weight = FloatImm::make(1);
            ExprHandle bias = FloatImm::make(0);

            if (hasWeight) {
              weight = inputs[4];
            }
            if (hasBias) {
              bias = inputs[5];
            }

            auto inv_var = rsqrt(var + eps);
            auto alpha = inv_var * weight;
            auto beta = bias - mean * alpha;
            auto output = input * alpha + beta;
            return demoteOutput(output, n->output());
          });
    } break;

    case aten::log: {
      return computeOneOperand("aten_log", v, [](const ExprHandle& a) {
        return log(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::log10: {
      return computeOneOperand("aten_log10", v, [](const ExprHandle& a) {
        return log10(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::log1p: {
      return computeOneOperand("aten_log1p", v, [](const ExprHandle& a) {
        return log1p(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::log2: {
      return computeOneOperand("aten_log2", v, [](const ExprHandle& a) {
        return log2(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::exp: {
      return computeOneOperand("aten_exp", v, [](const ExprHandle& a) {
        return exp(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::expm1: {
      return computeOneOperand("aten_expm1", v, [](const ExprHandle& a) {
        return expm1(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::erf: {
      return computeOneOperand("aten_erf", v, [](const ExprHandle& a) {
        return erf(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::erfc: {
      return computeOneOperand("aten_erfc", v, [](const ExprHandle& a) {
        return erfc(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::cos: {
      return computeOneOperand("aten_cos", v, [](const ExprHandle& a) {
        return cos(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::sin: {
      return computeOneOperand("aten_sin", v, [](const ExprHandle& a) {
        return sin(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::tan: {
      return computeOneOperand("aten_tan", v, [](const ExprHandle& a) {
        return tan(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::type_as: {
      auto const& n = v->node();
      Tensor* rhs = tensors_.at(n->inputs()[1]->unique());
      auto dtype = rhs->buf()->dtype();
      return computeOneOperand(
          "aten_type_as", v, [dtype](const ExprHandle& lhs) {
            return Cast::make(dtype, lhs);
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
            if (!rhs.node()->isConstant()) {
              return pow(lhs, rhs);
            }
            double val =
                immediateAs<double>(IRSimplifier::simplify(rhs.node()));

            if (val == 1.0f) {
              return lhs;
            } else if (val == 2.0f) { // NOLINT
              return lhs * lhs;
            } else if (val == 3.0f) { // NOLINT
              return (lhs * lhs) * lhs;
            } else if (val == 4.0f) { // NOLINT
              ExprHandle tmp = lhs * lhs;
              return tmp * tmp;
            } else if (val == 0.5f) { // NOLINT
              return sqrt(lhs);
            } else if (val == 0.0f) {
              return ExprHandle(1.0f);
            } else if (val == -0.5f) { // NOLINT
              return rsqrt(lhs);
            } else if (val == -1.0f) {
              return ExprHandle(1.0f) / lhs;
            } else if (val == -2.0f) { // NOLINT
              return ExprHandle(1.0f) / (lhs * lhs);
            }
            return pow(lhs, rhs);
          });
    } break;

    case aten::fmod: {
      return computeTwoOperand(
          "aten_fmod", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return fmod(promoteHalfToFloat(lhs), promoteHalfToFloat(rhs));
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
      auto imodImpl = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        return Mod::make(lhs, rhs);
      };
      auto fmodImpl = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        auto lhs_t = promoteHalfToFloat(lhs);
        auto rhs_t = promoteHalfToFloat(rhs);
        return fmod((rhs_t + fmod(lhs_t, rhs_t)), rhs_t);
      };
      {
        auto const& n = v->node();
        auto const& shape = broadcastShapes(
            valueShape(n->inputs()[0]), valueShape(n->inputs()[1]));
        return Compute(
            "aten_remainder",
            c10::fmap<DimArg>(shape),
            [&](const std::vector<VarHandle>& axes) {
              auto const& n = v->node();
              std::vector<ExprHandle> indices(axes.begin(), axes.end());
              std::vector<ExprHandle> inputs = {
                  tensorOrConstant(n->inputs()[0], indices),
                  tensorOrConstant(n->inputs()[1], indices),
              };

              promoteInputs(inputs);
              bool allInt = true;
              for (auto& e : inputs) {
                if (e.dtype().is_floating_point()) {
                  allInt = false;
                  break;
                }
              }
              if (allInt) {
                return demoteOutput(
                    imodImpl(inputs[0], inputs[1]), n->output());
              } else {
                return demoteOutput(
                    fmodImpl(inputs[0], inputs[1]), n->output());
              }
            });
      }

    } break;

    case aten::acos: {
      return computeOneOperand("aten_acos", v, [](const ExprHandle& a) {
        return acos(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::asin: {
      return computeOneOperand("aten_asin", v, [](const ExprHandle& a) {
        return asin(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::cosh: {
      return computeOneOperand("aten_cosh", v, [](const ExprHandle& a) {
        return cosh(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::sinh: {
      return computeOneOperand("aten_sinh", v, [](const ExprHandle& a) {
        return sinh(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::atan: {
      return computeOneOperand("aten_atan", v, [](const ExprHandle& a) {
        return atan(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::atan2: {
      return computeTwoOperand(
          "aten_atan2", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return atan2(
                promoteIntegerToDefaultType(lhs),
                promoteIntegerToDefaultType(rhs));
          });
    } break;

    case aten::tanh: {
      return computeOneOperand("aten_tanh", v, [](const ExprHandle& a) {
        return tanh(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::hardtanh: {
      return computeThreeOperand(
          "aten_hardtanh",
          v,
          [](const ExprHandle& a,
             const ExprHandle& min_val,
             const ExprHandle& max_val) {
            auto mm = CompareSelect::make(a, min_val, min_val, a, kLT);
            return CompareSelect::make(mm, max_val, max_val, mm, kGT);
          });
    } break;

    case aten::sqrt: {
      return computeOneOperand("aten_sqrt", v, [](const ExprHandle& a) {
        return tensorexpr::sqrt(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::rsqrt: {
      return computeOneOperand("aten_rsqrt", v, [](const ExprHandle& a) {
        return rsqrt(promoteIntegerToDefaultType(a));
      });
    } break;

    case aten::abs: {
      return computeOneOperand(
          "aten_abs",
          v,
          [](const ExprHandle& a) {
            return tensorexpr::abs(promoteHalfToFloat(a));
          },
          kIntegralTypes | kFloatingPointTypes | kBoolType);
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
          "aten_frac",
          v,
          [](const ExprHandle& a) {
            auto aa = promoteHalfToFloat(a);
            return aa - floor(aa);
          },
          kFloatingPointTypes);
    } break;

    case aten::lgamma: {
      return computeOneOperand("aten_lgamma", v, [](const ExprHandle& a) {
        return lgamma(promoteIntegerToDefaultType(a));
      });
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
      if (getCatWoConditionals()) {
        return computeCatWoConditionals(v);
      }
      return Compute(
          "aten_cat",
          dimsFromSizes(sizesForValue(v)),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            auto inputs = n->inputs()[0]->node()->inputs();
            if (inputs.size() == 0) {
              throw std::runtime_error(
                  "Empty input list is passed to aten::cat");
            }

            // Some of the inputs can be empty tensors, we need to skip them
            // when we construct the expression, but we need to take them into
            // account in dtype promotion.
            std::vector<const torch::jit::Value*> nonempty_inputs;
            for (auto input : inputs) {
              if (input->type()->kind() == TypeKind::TensorType) {
                auto tt = input->type()->cast<TensorType>();
                if (tt->isComplete() && tt->sizes().size() && tt->sizes()[0] &&
                    *tt->sizes()[0]) {
                  nonempty_inputs.push_back(input);
                }
              }
            }

            // When all inputs are empty tensors, the tensor we create for this
            // computation would contain no elements, so it doesn't really
            // matter what we return here, so just return 0.
            if (!nonempty_inputs.size()) {
              return ExprHandle(0);
            }

            int64_t dim_ = n->inputs()[1]->node()->i(attr::value);
            size_t dim = normalizeAndCheckIndex(dim_, axes.size());
            // Promote input types.
            // Note that we need to consider all inputs, including empty - they
            // also affect the resultant dtype.
            auto maybe_dtype = findDtypeForValue(inputs[0]);
            TORCH_INTERNAL_ASSERT(
                maybe_dtype, "Cannot find dtype for one of aten::cat inputs");
            ScalarType highType = *maybe_dtype;
            for (const auto input : inputs) {
              auto maybe_dtype = findDtypeForValue(input);
              TORCH_INTERNAL_ASSERT(
                  maybe_dtype, "Cannot find dtype for one of aten::cat inputs");
              highType = promoteTypes(highType, *maybe_dtype);
            }

            // Now we know the final dtype, we know what inputs are non-empty,
            // and we know that there is at least one such an input. With all
            // that we construct a tensor expression performing the
            // concatenation.
            // The expression we build here is a cascading if-then-else that
            // essentially represents:
            //
            //              inp1[i, j, k]         if 0   < i < l1,
            // out[i,j,k] = inp2[i, j-l1, k]      if l1 =< i < l1 + l2,
            //              ...
            //              inpN[i, j-l_N_1, k]   if l1+l2+...l_N_1  < i
            // where l_i is the corresponding size of the i-th input.
            std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
            ExprHandle load = promoteToDtype(
                tensorOrConstant(nonempty_inputs[0], newAxes), highType);
            size_t offset =
                bufferSizes(tensors_.at(nonempty_inputs[0]->unique()))[dim];
            newAxes[dim] = newAxes[dim] - IntImm::make(offset);

            for (size_t ii = 1; ii < nonempty_inputs.size(); ++ii) {
              auto input = nonempty_inputs[ii];
              load = ifThenElse(
                  CompareSelect::make(axes[dim], IntImm::make(offset), kLT),
                  load,
                  promoteToDtype(tensorOrConstant(input, newAxes), highType));

              offset += bufferSizes(tensors_.at(input->unique()))[dim];
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

    case aten::softmax: {
      return computeSoftmax(v, false);
    }

    case aten::log_softmax: {
      return computeSoftmax(v, true);
    }

    default: {
      throw std::runtime_error("Unhandled node kind");
    }
  }
}

Stmt* TensorExprKernel::transformLoops(BackendType backendType, Stmt* st) {
  std::unordered_set<const Buf*> output_bufs;
  for (auto t : tensorOutputs_) {
    output_bufs.insert(t->buf());
  }
  torch::jit::tensorexpr::LoopNest l(st, output_bufs);
  GRAPH_DEBUG("Original Stmt:\n", std::to_string(l.root_stmt()), "\n");

  bool hasReduction = NodeFinder<ReduceOp>::find(l.root_stmt()).size() != 0;

  // For Block codegen we create a map of tensor dims before
  // inlining. Like GPU codegen we need to inline. But the order
  // where this analysis is run matters.
  auto block_analysis = std::make_unique<CreateBufferMap>();
  if (backendType == kBlockCodeGen) {
    // Run Block analysis to get multi dim buffer info
    auto root_stmt = l.root_stmt();
    root_stmt->accept(block_analysis.get());
  }

  // inlining output & intermediate buffers can duplicate computation.
  // it slows down cpu code generation but is enabled on gpu because it avoids
  // difficult synchronization logic across blocks.
  bool allow_duplicated_work =
      (backendType == kCudaCodeGen || backendType == kBlockCodeGen);
  l.inlineIntermediateBufs(allow_duplicated_work);

  if (backendType == kCudaCodeGen) {
    for (auto tensor : tensorOutputs_) {
      std::vector<For*> loops = l.getLoopStmtsFor(tensor);
      TORCH_INTERNAL_ASSERT(!loops.empty(), "loops should not be empty");
      For* flattened = nullptr;
      LoopNest::flatten(loops, &flattened);
      assert(flattened);

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
        l.splitWithMask(flattened, blockSize, &outer, &inner);
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
        l.splitWithMask(flattened, blockCount * blockSize, &outer, &inner);
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
    for (auto tensor : tensorOutputs_) {
      const int default_fp16_blocksize = 16;
      const int default_uint8_blocksize = 32;
      int blockSize = default_fp16_blocksize;
      // We only handle looplevels == 2 for now
      if (tensor->buf()->dtype().scalar_type() == ScalarType::Byte) {
        blockSize = default_uint8_blocksize;
      }
      std::vector<For*> loops = l.getLoopStmtsFor(tensor);
      TORCH_INTERNAL_ASSERT(!loops.empty(), "loops should not be empty");
      For* flattened = nullptr;
      LoopNest::flatten(loops, &flattened);
      assert(flattened);

      For* outer = nullptr;
      For* inner = nullptr;
      l.splitWithMask(flattened, blockSize, &outer, &inner);
      l.setGPUBlockIndex(outer, 0);
      l.setGPUThreadIndex(inner, 0);
      l.setBufferMap(outer, block_analysis->getBufferMap());
    }
  }

  l.prepareForCodegen();

  if (backendType == kLLVMCodeGen && !hasReduction) {
    l.vectorizeInnerLoops();
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
    backendType = dontUseLLVMFlag() ? kSimpleIREval : kLLVMCodeGen;
#else
    backendType = kSimpleIREval;
#endif
    if (getTEMustUseLLVMOnCPU() && backendType == kSimpleIREval) {
      throw std::runtime_error("LLVM Backend not found");
    }
  } else {
    throw std::runtime_error("Invalid device type");
  }
  return backendType;
}

static bool isValidIdentifierChar(char c, size_t pos) {
  return islower(c) || isupper(c) || c == '_' || (pos > 0 && isdigit(c));
}

// replaces all invalid characters with underscore
std::string sanitizeName(const std::string& input_name) {
  std::stringstream sanitized_name;
  for (size_t i = 0; i < input_name.size(); ++i) {
    if (isValidIdentifierChar(input_name[i], i)) {
      sanitized_name << input_name[i];
    } else {
      sanitized_name << "_";
    }
  }
  return sanitized_name.str();
}

// we use the debug names in printing cuda code, they need to be removed
// of characters that can't be used in a variable identifier
void TensorExprKernel::genInputDebugNames() {
  std::unordered_map<std::string, const torch::jit::Value*> name_to_value;
  std::unordered_set<std::string> name_set;
  std::unordered_map<const torch::jit::Value*, std::string> value_to_name;
  for (const torch::jit::Value* input : graph_->inputs()) {
    std::string sanitized_name = sanitizeName(input->debugName());
    // we could get fancier here, but name conflict is extremely unlikely
    while (name_set.count(sanitized_name)) {
      sanitized_name = sanitized_name + "_";
    }
    value_to_name[input] = sanitized_name;
    name_set.insert(sanitized_name);
  }
  input_name_map_ = std::move(value_to_name);
}

void TensorExprKernel::bindInput(const torch::jit::Value* input) {
  auto const& t = input->type();
  switch (t->kind()) {
    case TypeKind::TensorType: {
      auto tt = input->type()->cast<TensorType>();
      Placeholder inBuffer(
          "t" + input_name_map_[input],
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
                return inBuffer.load(idx);
              }));
      bufferArgs_.emplace_back(inBuffer);
      break;
    }
    case TypeKind::FloatType: {
      VarHandle v("v" + input_name_map_[input], kDouble);
      bufferArgs_.emplace_back(v);
      scalars_.emplace(input->unique(), v);
      break;
    }
    case TypeKind::BoolType: {
      VarHandle v("v" + input_name_map_[input], kBool);
      bufferArgs_.emplace_back(v);
      scalars_.emplace(input->unique(), v);
      break;
    }
    case TypeKind::IntType: {
      VarHandle v("v" + input_name_map_[input], kLong);
      bufferArgs_.emplace_back(v);
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

Tensor* TensorExprKernel::computeSoftmax(
    const torch::jit::Value* v,
    bool log_softmax) {
  // Softmax is computed as follows:
  //    softmax(vi) = exp(vi) / sum(exp(vi))
  //
  // In order to avoid overflow issues due to exp of a large number, we
  // subtract the max of that dim before computing exp.
  //    softmax(vi) = exp(vi - max(vi)) / sum(exp(vi - max(vi)))
  //
  // This is implemented as 4 loopnests:
  //   - First loop computes the max over the softmax dim.
  //   - Second loop computes exp for every element in v after subtracting
  //     the max of the softmax dim it belongs to.
  //   - Third loop computes the sum over the softmax dim.
  //   - Final loop computes softmax for every element in v.

  // LogSoftmax is computed as follows:
  //    log_softmax(vi) = log(softmax(vi))
  //                    = vi - log(sum(exp(vi)))
  //
  // Using the same max trick as above:
  //    log_softmax(vi) = vi - max(vi) - log(sum(exp(vi - max(vi))))
  //
  // This is implemented as 5 loopnests:
  //   - First loop computes the max over the softmax dim.
  //   - Second loop computes exp for every element in v after subtracting
  //     the max of the softmax dim it belongs to.
  //   - Third loop computes the sum over the softmax dim.
  //   - Fourth loop computes log for every element in the sum.
  //   - Final loop computes the log_softmax for every element in v.

  TORCH_INTERNAL_ASSERT(v->node()->inputs().size() == 3);
  auto output_dims = dimsFromSizes(sizesForValue(v));

  // We do not handle None for dims (input 1) because that is supposed to
  // be deprecated.
  TORCH_INTERNAL_ASSERT(v->node()->input(1)->node()->kind() == prim::Constant);
  int64_t rank =
      *v->node()->input(0)->type()->castRaw<TensorType>()->sizes().size();
  size_t softmax_dim =
      normalizeAndCheckIndex(v->node()->input(1)->node()->i(attr::value), rank);
  std::vector<DimArg> non_softmax_dims;
  for (size_t i = 0; i < output_dims.size(); ++i) {
    if (i != softmax_dim) {
      non_softmax_dims.push_back(output_dims[i]);
    }
  }

  // Softmax implementation includes two reductions, one to find the max and
  // the other to calculate the sum along the softmax dim. These reductions
  // will have the softmax dimension as the inner most loop. So, the innermost
  // index in the indices will refer to the softmax dimension.

  // Update the indices by moving the softmax dimension index to the
  // appropriate position.
  auto move_softmax_dim_index_to_pos = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices;
    for (auto ind : indices) {
      new_indices.push_back(ind);
    }
    for (size_t i = softmax_dim; i < indices.size() - 1; ++i) {
      new_indices[i + 1] = indices[i];
    }
    new_indices[softmax_dim] = indices[indices.size() - 1];
    return new_indices;
  };

  // Remove the index corresponding to the softmax dimension.
  auto remove_softmax_dim_index = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (i != softmax_dim) {
        new_indices.push_back(indices[i]);
      }
    }
    return new_indices;
  };

  auto convert_indices_to_expr_handle = [&](const ParameterList& indices) {
    std::vector<ExprHandle> new_indices(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      new_indices[i] = indices[i];
    }
    return new_indices;
  };

  c10::optional<Dtype> dtype = ToDtype(ScalarType::None);
  auto maybe_dtype = v->node()->get(attr::dtype);
  if (maybe_dtype && !maybe_dtype->isNone()) {
    dtype = ToDtype(static_cast<ScalarType>(maybe_dtype->toInt()));
  }

  auto max = Reduce(
      "aten_softmax_max",
      non_softmax_dims,
      Maximum(dtype.value()),
      [&](ParameterList& indices) {
        return tensorOrConstant(
            v->node()->inputs()[0], move_softmax_dim_index_to_pos(indices));
      },
      {output_dims[softmax_dim]});
  auto e =
      Compute("aten_softmax_exp", output_dims, [&](ParameterList& indices) {
        auto inp = tensorOrConstant(
            v->node()->inputs()[0], convert_indices_to_expr_handle(indices));
        return exp(inp - max->call(remove_softmax_dim_index(indices)));
      });
  auto sum = Reduce(
      "aten_softmax_sum",
      non_softmax_dims,
      Sum(),
      [&](ParameterList& indices) {
        return e->call(move_softmax_dim_index_to_pos(indices));
      },
      {output_dims[softmax_dim]});
  if (!log_softmax) {
    auto result =
        Compute("aten_softmax", output_dims, [&](ParameterList& indices) {
          return e->call(indices) /
              sum->call(remove_softmax_dim_index(indices));
        });
    return new Tensor(
        result->buf(),
        new Block({max->stmt(), e->stmt(), sum->stmt(), result->stmt()}));
  }

  auto log_sum = Compute(
      "aten_softmax_log_sum", non_softmax_dims, [&](ParameterList& indices) {
        return log(sum->call(indices));
      });
  auto result =
      Compute("aten_log_softmax", output_dims, [&](ParameterList& indices) {
        auto inp = tensorOrConstant(
            v->node()->inputs()[0], convert_indices_to_expr_handle(indices));
        auto non_softmax_indices = remove_softmax_dim_index(indices);
        return inp - max->call(non_softmax_indices) -
            log_sum->call(non_softmax_indices);
      });
  return new Tensor(
      result->buf(),
      new Block(
          {max->stmt(),
           e->stmt(),
           sum->stmt(),
           log_sum->stmt(),
           result->stmt()}));
}

Tensor* TensorExprKernel::computeCatWoConditionals(const torch::jit::Value* v) {
  auto const& n = v->node();
  auto inputs = n->inputs()[0]->node()->inputs();
  if (inputs.size() == 0) {
    throw std::runtime_error("Empty input list is passed to aten::cat");
  }

  // Some of the inputs can be empty tensors, we need to skip them
  // when we construct the expression, but we need to take them into
  // account in dtype promotion.
  std::vector<const torch::jit::Value*> nonempty_inputs;
  for (auto input : inputs) {
    if (input->type()->kind() == TypeKind::TensorType) {
      auto tt = input->type()->cast<TensorType>();
      if (tt->isComplete() && tt->sizes().size() && tt->sizes()[0] &&
          *tt->sizes()[0]) {
        nonempty_inputs.push_back(input);
      }
    }
  }

  // Promote input types.
  // Note that we need to consider all inputs, including empty - they
  // also affect the resultant dtype.
  auto maybe_dtype = findDtypeForValue(inputs[0]);
  TORCH_INTERNAL_ASSERT(
      maybe_dtype, "Cannot find dtype for one of aten::cat inputs");
  ScalarType highType = *maybe_dtype;
  for (const auto input : inputs) {
    auto maybe_dtype = findDtypeForValue(input);
    TORCH_INTERNAL_ASSERT(
        maybe_dtype, "Cannot find dtype for one of aten::cat inputs");
    highType = promoteTypes(highType, *maybe_dtype);
  }

  // Now we build one loop per input:
  //
  // for i
  //   for j
  //     for k
  //       output[i,j,k] = inp1[i,j,k]
  // for i
  //   for j
  //     for k
  //       output[i,j+l1,k] = inp2[i,j,k]
  // for i
  //   for j
  //     for k
  //       output[i,j+l2,k] = inp3[i,j,k]

  auto output_sizes = inferSizesForValue(v);
  auto output_sizes_expr = ExprHandleVectorToExprVector(output_sizes);
  auto output_buf = new Buf("aten_cat", output_sizes_expr, ToDtype(highType));

  int64_t concat_dim = n->input(1)->node()->i(attr::value);
  auto shape = sizesForValue(inputs[0]);
  size_t norm_concat_dim = normalizeAndCheckIndex(concat_dim, shape.size());

  auto gen_code_for_input = [&](const torch::jit::Value* inp,
                                size_t inp_pos,
                                const Expr* concat_dim_size,
                                const std::vector<ExprHandle>& dims) {
    std::vector<Var*> for_vars(dims.size());
    std::vector<const Expr*> load_indices(dims.size());
    std::vector<const Expr*> store_indices(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      for_vars[i] = new Var(
          "i" + c10::to_string(inp_pos) + "_" + c10::to_string(i), kInt);
      load_indices[i] = for_vars[i];
      if (i == norm_concat_dim) {
        store_indices[i] = new Add(for_vars[i], concat_dim_size);
      } else {
        store_indices[i] = for_vars[i];
      }
    }
    auto inp_buf = tensors_.at(inp->unique())->buf();
    auto load_expr = new Load(inp_buf, load_indices, new IntImm(1));
    auto load_promoted = promoteToDtype(ExprHandle(load_expr), highType);
    Stmt* st = new Store(
        output_buf, store_indices, load_promoted.node(), new IntImm(1));
    for (size_t i = dims.size(); i > 0; --i) {
      st = new For(for_vars[i - 1], new IntImm(0), dims[i - 1].node(), st);
    }
    return st;
  };

  Expr* concat_dim_size = nullptr;
  auto block = new Block({});
  for (size_t i = 0; i < nonempty_inputs.size(); ++i) {
    auto input_dims = sizesForValue(nonempty_inputs[i]);
    if (concat_dim_size == nullptr) {
      concat_dim_size = new IntImm(0);
    }
    block->append_stmt(
        gen_code_for_input(nonempty_inputs[i], i, concat_dim_size, input_dims));
    concat_dim_size =
        new Add(concat_dim_size, input_dims[norm_concat_dim].node());
  }
  return new Tensor(output_buf, IRSimplifier::simplify(block));
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

template <typename T>
std::vector<size_t> reverse_sort_indices(const std::vector<T>& v) {
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
    return v[i1] > v[i2];
  });
  return idx;
}

bool denseAndNonOverlapping(
    at::ArrayRef<int64_t> sizes,
    at::ArrayRef<int64_t> strides) {
  return (strides == at::infer_dense_strides(sizes, strides));
}

Tensor* TensorExprKernel::convertOutputToCorrectStrides(torch::jit::Value* v) {
  const TensorTypePtr& tt = v->type()->expect<TensorType>();
  TORCH_INTERNAL_ASSERT(tensors_.count(v->unique()));
  Tensor* tensor = tensors_[v->unique()];

  TORCH_INTERNAL_ASSERT(tt->sizes().concrete_sizes());
  const auto sizes = *tt->sizes().concrete_sizes();
  std::vector<int64_t> default_strides = TensorType::contiguousStridesOf(sizes);
  TORCH_INTERNAL_ASSERT(tt->strides().concrete_sizes());
  const std::vector<int64_t> strides = *tt->strides().concrete_sizes();
  // All Tensors in NNC are layed out in default, contiguous layout.
  // If the output is also default contiguous we don't need to do anything
  if (strides == default_strides) {
    return tensor;
  }
  // If the tensor is not dense or overlaps, we have
  // no way of matching the profiled striding
  if (!denseAndNonOverlapping(sizes, strides)) {
    return tensor;
  }

  auto dims = dimsFromSizes(sizesForValue(v));
  // We need to convert the output tensor so that its values are layed
  // so that whene viewed from the output strides the values are correct.
  // A contiguous Tensor of size(2, 3) with values 0-5 is layed out as:
  // [0] [1] [2] [3] [4] [5]
  // The same valued tensor with strides (2, 1) would be layed out like
  // [0] [3] [1] [4] [2] [5]
  // When we are doing the re-ordering of values into the output tensor,
  // we are iterating per-element of the input, ad we are fixed
  // in indexing in to the output tensor at [i, j] = val
  // `val` we want here is equal to the indices for the output
  // tensor that would have given the same position as the output
  // The position is equal to the sum of stride[i] * index[i],
  // and we can can calculate the equivalent indices in the
  // output tensor strides by iteratively computing the index of
  // the biggest stride:
  // absolute = ...
  // for stride in strides_from_largest_to_smallest:
  //     cur_idx = absolute // stride
  //     absolute = absolute % stride

  return Compute(
      "output_1", dims, [&](const std::vector<VarHandle>& axes_input) {
        std::vector<ExprHandle> axes(axes_input.begin(), axes_input.end());
        auto absolute_position = IntImm::make(0);
        for (size_t i = 0; i < axes.size(); ++i) {
          absolute_position =
              absolute_position + (IntImm::make(default_strides[i]) * axes[i]);
        }
        std::vector<size_t> sorted_stride_indices =
            reverse_sort_indices(strides);
        std::vector<ExprHandle> new_axes(sorted_stride_indices.size());
        for (size_t stride_index : sorted_stride_indices) {
          auto stride = strides[stride_index];
          auto index = Div::make(absolute_position, IntImm::make(stride));
          absolute_position =
              Mod::make(absolute_position, IntImm::make(stride));
          new_axes[stride_index] = index;
        }
        return tensor->call(new_axes);
      });
}

void TensorExprKernel::compile() {
  KernelScope kernelScope(&kernelArena_);
  GRAPH_DUMP("TensorExprKernel graph:", graph_);

  // Vector to collect the Stmts corresponding to all tensors.
  std::vector<Stmt*> tensor_stmts;

  // Bind inputs to buffers.
  nInputs_ = graph_->inputs().size();
  genInputDebugNames();
  for (auto const& input : graph_->inputs()) {
    bindInput(input);
    inputTypes_.push_back(input->type());
    if (input->type()->kind() == TypeKind::TensorType) {
      tensor_stmts.push_back(Stmt::clone(tensors_.at(input->unique())->stmt()));
    }
  }

  // Bind nodes to tensor compute expressions.
  for (auto const& n : graph_->nodes()) {
    if (n->kind() == prim::Constant || n->kind() == prim::ListConstruct) {
      continue;
    } else {
      for (auto const& output : n->outputs()) {
        if (output->hasUses()) {
          tensors_.emplace(output->unique(), computeValue(output));
          tensor_stmts.push_back(
              Stmt::clone(tensors_.at(output->unique())->stmt()));
        }
      }
    }
    if (hasRandom_ && hasBroadcast_) {
      throw std::runtime_error(
          "Cannot support broadcast and random within one kernel");
    }
  }

  device_ = *pickDeviceType(graph_->inputs());

  // Move output operands from `tensors_` to `tensorOutputs_`
  for (const auto& output : graph_->outputs()) {
    if (!tensors_.count(output->unique())) {
      throw malformed_input("cannot find output Tensor");
    }
    // The "strided" tensor will be incorrect if used in NNC,
    // since NNC views it as contiguous. Only convert it to the right
    // strides at the end of the kernel (if already contiguous it's a no-op)
    Tensor* properly_strided_output = convertOutputToCorrectStrides(output);
    if (tensors_.at(output->unique()) != properly_strided_output) {
      tensor_stmts.push_back(Stmt::clone(properly_strided_output->stmt()));
    }
    tensors_[output->unique()] = properly_strided_output;
    const auto& tt = output->type()->expect<TensorType>();
    auto sizes = *tt->sizes().concrete_sizes();
    tensorOutputSizes_.push_back(sizes);
    auto strides = *tt->strides().concrete_sizes();

    // If the tensor is not dense or overlaps, we have
    // no way of matching the profiled striding
    if (denseAndNonOverlapping(sizes, strides)) {
      tensorOutputStrides_.push_back(*tt->strides().concrete_sizes());
    } else {
      tensorOutputStrides_.push_back(TensorType::contiguousStridesOf(sizes));
    }

    tensorOutputs_.emplace_back(tensors_.at(output->unique()));
    bufferArgs_.emplace_back(tensors_.at(output->unique()));
    tensorOutputTensorOptions_.emplace_back(
        c10::TensorOptions(tensorType(tensors_[output->unique()]))
            .device(device_));
    tensors_.erase(output->unique());
  }

  BackendType backendType = inferBackendTypeFromDevice(device_);
  Stmt* stmt = transformLoops(backendType, new Block(tensor_stmts));

  // Generate code.
  codegen_ = CreateCodeGen(
      getCodeGenName(backendType),
      stmt,
      bufferArgs_,
      device_,
      SubgraphUtils::generateNameForGraph(graph_));
}

TensorExprKernel::TensorExprKernel(const std::shared_ptr<Graph>& subgraph)
    : graph_(subgraph), code_(subgraph, "") {
  allow_fallback_ = fallbackAllowed();
  if (!allow_fallback_) {
    compile();
    return;
  }

  use_fallback_ = fallbackEnforced();
  if (use_fallback_) {
    return;
  }

  try {
    compile();
  } catch (...) {
    use_fallback_ = true;
  }
}

void TensorExprKernel::run(Stack& stack) {
  if (!use_fallback_ && !allow_fallback_) {
    runKernel(stack);
  } else if (!use_fallback_ && allow_fallback_) {
    try {
      runKernel(stack);
    } catch (...) {
      fallback(stack);
    }
  } else {
    fallback(stack);
  }
}

std::vector<CodeGen::CallArg> TensorExprKernel::prepareRunArgs(
    const at::ArrayRef<IValue>& inputs,
    std::vector<at::Tensor>& outputs) {
  std::vector<CodeGen::CallArg> runArgs;
  runArgs.reserve(inputs.size() + tensorOutputs_.size());

  for (const auto& input : inputs) {
    if (input.isInt()) {
      runArgs.emplace_back(input.toInt());
    } else if (input.isDouble()) {
      runArgs.emplace_back(input.toDouble());
    } else if (input.isTensor()) {
      runArgs.emplace_back(input.toTensor().data_ptr());
    }
  }

  for (size_t i = 0, e = tensorOutputs_.size(); i < e; ++i) {
    auto const& opts = tensorOutputTensorOptions_[i];
    outputs.emplace_back(codegen_->empty_strided(
        tensorOutputSizes_[i],
        tensorOutputStrides_[i],
        opts.dtype,
        opts.layout,
        opts.device,
        opts.pinned_memory));
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
