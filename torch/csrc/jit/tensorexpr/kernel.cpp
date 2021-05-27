#include <c10/util/variant.h>
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
#include <torch/csrc/jit/tensorexpr/operators/conv2d.h>
#include <iostream>

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {
namespace tensorexpr {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static int te_cuda_pointwise_loop_levels = -1;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static int te_cuda_pointwise_block_count = -1;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static int te_cuda_pointwise_block_size = -1;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool fallback_allowed = false;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool te_generate_block_code = false;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool te_must_use_llvm_on_cpu = true;
static bool cat_wo_conditionals = true; // NOLINT

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

// If v is a Tensor with concretely-known sizes and dtype, return them, else
// nullopt.
c10::optional<TensorInfo> getTensorInfoJit(torch::jit::Value* v) {
  auto const& it = v->type()->cast<TensorType>();
  if (!it) {
    return c10::nullopt;
  }
  if (!it->isComplete()) {
    return c10::nullopt;
  }
  if (!it->scalarType()) {
    return c10::nullopt;
  }
  auto concrete_sizes = it->sizes().concrete_sizes();
  if (!concrete_sizes) {
    return c10::nullopt;
  }
  return TensorInfo{*concrete_sizes, *it->scalarType()};
}
c10::optional<TensorInfo> getTensorInfo(BufHandle b) {
  std::vector<int64_t> dims;
  for (auto dim : b.dims()) {
    auto val = dynamic_cast<const IntImm*>(dim.node());
    if (!val) {
      return c10::nullopt;
    }
    dims.push_back(val->value());
  }
  return TensorInfo{dims, static_cast<at::ScalarType>(b.dtype().scalar_type())};
}

std::vector<int64_t> _pair_int(ArgValue v) {
  if (auto t = c10::get_if<IntList>(&v)) {
    return {(*t)[0], (*t)[1]};
  }
  auto i = c10::get<int64_t>(v);
  return {i, i};
}
std::vector<int64_t> _pair_int(IValue v) {
  if (v.isIntList()) {
    return v.toIntVector();
  } else {
    return {v.toInt(), v.toInt()};
  }
}

bool conv2dIsSupported(
    const TensorInfo& input,
    const TensorInfo& weight,
    const TensorInfo& bias,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& pad,
    const std::vector<int64_t>& dilation,
    int64_t groups) {
  if (input.dtype != c10::ScalarType::Float ||
      weight.dtype != c10::ScalarType::Float ||
      bias.dtype != c10::ScalarType::Float) {
    GRAPH_DEBUG("only float32 allowed");
    return false;
  }
  if (input.dims.size() != 4 || weight.dims.size() != 4 ||
      bias.dims.size() != 1) {
    GRAPH_DEBUG("inputs are the wrong size");
    return false;
  }
  auto Cin = input.dims[1];
  auto Cout = weight.dims[0];
  auto CperG = weight.dims[1];
  if (Cin != Cout || Cin != groups || CperG != 1) {
    GRAPH_DEBUG("not depthwise");
    return false;
  }
  auto KH = weight.dims[2];
  auto KW = weight.dims[3];
  if (KH != 3 || KW != 3) {
    GRAPH_DEBUG("not 3x3");
    return false;
  }
  if (stride.size() != 2 || stride[0] != stride[1]) {
    GRAPH_DEBUG("unsupported stride");
    return false;
  }
  if (pad.size() != 2 || pad[0] != pad[1]) {
    GRAPH_DEBUG("unsupported pad");
    return false;
  }
  if (dilation.size() != 2 || dilation[0] != 1 || dilation[1] != 1) {
    GRAPH_DEBUG("unsupported dilation");
    return false;
  }
  return true;
}
// The fuser only supports conv2d with very specific properties:
// - Static shapes: 4-d input and filter, 1-d bias.
// - Constant strides/padding/dilation/groups
// - Equal padding and strides, dilation == 1.
// - Depthwise (groups == in_channels == out_channels)
// - 3x3 kernel
bool conv2dIsSupportedJit(const torch::jit::Node* node) {
  auto const& input = getTensorInfoJit(node->input(0));
  auto const& weight = getTensorInfoJit(node->input(1));
  auto const& bias = getTensorInfoJit(node->input(2));
  auto const& stride = toIValue(node->input(3));
  auto const& pad = toIValue(node->input(4));
  auto const& dilation = toIValue(node->input(5));
  auto const& groups = toIValue(node->input(6));

  // Everything should be statically known.
  if (!input || !weight || !bias || !stride || !pad || !dilation || !groups) {
    GRAPH_DEBUG("some params aren't static");
    return false;
  }
  return conv2dIsSupported(
      *input,
      *weight,
      *bias,
      _pair_int(*stride),
      _pair_int(*pad),
      _pair_int(*dilation),
      groups->toInt());
}

// The fuser currently only supports matmul of 2D x 2D matrices
bool matmulIsSupported(const torch::jit::Node* node) {
  auto const& input0 = getTensorInfoJit(node->input(0));
  auto const& input1 = getTensorInfoJit(node->input(1));

  // Everything should be statically known.
  if (!input0 || !input1) {
    GRAPH_DEBUG("matmulIsSupported: Input shapes aren't static");
    return false;
  }

  // Proper ndim for tensor inputs.
  if (input0->dims.size() != 2 || input1->dims.size() != 2) {
    GRAPH_DEBUG("matmulIsSupported: Unsupported input sizes");
    return false;
  }

  return true;
}

void annotateInputShapes(
    const std::shared_ptr<Graph>& graph,
    const std::vector<c10::optional<at::Tensor>>& example_inputs) {
  TORCH_INTERNAL_ASSERT(graph->inputs().size() == example_inputs.size());
  for (size_t idx = 0; idx < example_inputs.size(); idx++) {
    if (auto t = example_inputs[idx]) {
      auto concrete_tensor_type = tensorTypeInCurrentExecutionContext(*t);
      graph->inputs().at(idx)->setType(concrete_tensor_type);
    }
  }
}

std::shared_ptr<Graph> removeUnusedSelfArgument(
    const std::shared_ptr<Graph>& graph) {
  if (graph->inputs().size() == 0) {
    return graph;
  }
  jit::Value* self_argument = graph->inputs().at(0);
  if (self_argument->uses().size() != 0 ||
      !self_argument->type()->is_module()) {
    return graph;
  }
  std::shared_ptr<Graph> res = graph->copy();
  res->eraseInput(0);
  return res;
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

static at::ScalarType tensorType(const Buf* b) {
  return static_cast<at::ScalarType>(b->dtype().scalar_type());
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

std::vector<int64_t> bufferSizes(const Buf* b) {
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < b->ndim(); i++) {
    sizes.push_back(dynamic_cast<const IntImm*>(b->dim(i))->value());
  }
  return sizes;
}

ExprHandle TensorExprKernel::chunk(
    const Buf* b,
    size_t chunkIdx,
    int64_t dim,
    int64_t chunks,
    const std::vector<ExprHandle>& axes) {
  auto norm_dim = normalizeAndCheckIndex(dim, axes.size());
  auto sizes = bufferSizes(b);
  size_t step = sizes[norm_dim] / chunks;

  std::vector<ExprHandle> indices;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i == norm_dim) {
      indices.push_back(axes[i] + IntImm::make((int)chunkIdx * (int)step));
    } else {
      indices.push_back(axes[i]);
    }
  }

  return BufHandle(b).load(indices);
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

ExprHandle broadcast(BufHandle b, const std::vector<ExprHandle>& axes) {
  return b.load(computeIndicesToBroadcast(axes, b.dims()));
}

ExprHandle constant(const ArgValue& v) {
  if (auto s = c10::get_if<tensorexpr::VarHandle>(&v)) {
    return *s;
  } else if (auto d = c10::get_if<double>(&v)) {
    return DoubleImm::make(*d);
  } else if (auto i = c10::get_if<int64_t>(&v)) {
    return LongImm::make(*i);
  } else if (auto b = c10::get_if<bool>(&v)) {
    return BoolImm::make(*b);
  } else if (c10::get_if<ArgNone>(&v)) {
    // This is just a placeholder so we don't throw.  None-handling
    // is operator-specific and should be handled properly in
    // the operator-specific lowering code.
    return IntImm::make(0);
  } else {
    throw unsupported_dtype("Trying to convert unsupported dtype to constant");
  }
}

ExprHandle tensorOrConstant(
    const ArgValue& v,
    const std::vector<ExprHandle>& axes) {
  if (auto b = c10::get_if<BufHandle>(&v)) {
    return broadcast(*b, axes);
  }
  return constant(v);
}
ExprHandle TensorExprKernel::constant(const torch::jit::Value* v) {
  if (v->node()->kind() == prim::Constant) {
    const auto val = toIValue(v).value();
    if (val.isDouble()) {
      return DoubleImm::make(val.toDouble());
    } else if (val.isInt()) {
      return LongImm::make(val.toInt());
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

  if (!scalars_.count(v)) {
    throw malformed_input("no scalar in Constant");
  }

  return scalars_.at(v);
}

ExprHandle TensorExprKernel::tensorOrConstant(
    const torch::jit::Value* v,
    const std::vector<ExprHandle>& axes) {
  auto ti = bufs_.find(v);
  if (ti != bufs_.end()) {
    return broadcast(BufHandle(ti->second), axes);
  }
  return constant(v);
}

// Convert boolean to integer, if needed.
ExprHandle boolToInteger(const ExprHandle& x) {
  return x.dtype().scalar_type() == ScalarType::Bool ? cast<int>(x) : x;
}

ArgValue TensorExprKernel::toArg(const torch::jit::Value* v) const {
  auto ti = bufs_.find(v);
  if (ti != bufs_.end()) {
    return BufHandle(ti->second);
  }
  if (v->node()->kind() == prim::ListConstruct) {
    std::vector<ArgValue> vec;
    for (auto el : v->node()->inputs()) {
      vec.push_back(toArg(el));
    }
    if (vec.size() == 0) {
      return BufList(); // Return arbitrarily typed vector
    } else if (c10::get_if<BufHandle>(&vec[0])) {
      return convertVecArgValue<BufHandle>(vec);
    } else if (c10::get_if<int64_t>(&vec[0])) {
      return convertVecArgValue<int64_t>(vec);
    }
    throw unsupported_dtype();
  }
  if (v->node()->kind() == prim::Constant) {
    const auto val = toIValue(v).value();
    if (val.isDouble()) {
      return val.toDouble();
    } else if (val.isInt()) {
      return val.toInt();
    } else if (val.isBool()) {
      return val.toBool();
    } else if (val.isNone()) {
      // This is just a placeholder so we don't throw.  None-handling
      // is operator-specific and should be handled properly in
      // the operator-specific lowering code.
      return ArgNone();
    } else if (val.isIntList()) {
      return val.toIntVector();
    } else {
      throw unsupported_dtype(val.type()->str());
    }
  }

  if (!scalars_.count(v)) {
    throw malformed_input("no scalar in Constant");
  }
  return scalars_.at(v);
}

std::vector<ExprHandle> TensorExprKernel::sizesFromVaryingShape(
    const c10::VaryingShape<int64_t>& shape) {
  std::vector<ExprHandle> dims;
  for (size_t i = 0; i < *shape.size(); i++) {
    dims.push_back(IntImm::make(*shape[i]));
  }
  return dims;
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
    case aten::gelu:
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
    case aten::hardswish:
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
      return broadcastShapesMut(shapes);
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
      return broadcastShapesMut(shapes);
    }

    case aten::addcmul: {
      std::vector<std::vector<ExprHandle>> shapes;
      for (size_t idx = 0; idx < 4; idx++) {
        torch::jit::Value* inp = v->node()->input(idx);
        shapes.push_back(sizesForValue(inp));
      }
      return broadcastShapesMut(shapes);
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

      int64_t dim = toIValue(n->input(1))->toInt();
      // From the documentation
      // (https://pytorch.org/docs/master/generated/torch.unsqueeze.html):
      //
      // A dim value within the range [-input.dim() - 1, input.dim() + 1) can be
      // used. Negative dim will correspond to unsqueeze() applied at dim = dim
      // + input.dim() + 1.
      if (dim < 0) {
        dim = dim + shape.size() + 1;
      }
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
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

ExprHandle clamp(
    const ExprHandle& cmin,
    const ExprHandle& cmax,
    const ExprHandle& input) {
  auto mm = CompareSelect::make(input, cmin, cmin, input, kLT);
  return CompareSelect::make(mm, cmax, cmax, mm, kGT);
}

bool checkTypes(const ScalarType highType, const int typeConstraints) {
  if (typeConstraints == kAllTypes) {
    return true;
  }

  if (c10::isIntegralType(highType, false)) {
    return (typeConstraints & kIntegralTypes) != 0;
  } else if (c10::isFloatingType(highType)) {
    return (typeConstraints & kFloatingPointTypes) != 0;
  } else if (highType == ScalarType::Bool) {
    return (typeConstraints & kBoolType) != 0;
  }

  // assume JIT not supporting complex and qint yet
  TORCH_INTERNAL_ASSERT((typeConstraints & (kQintTypes | kComplexTypes)) == 0);
  return false;
}

void promoteInputs(
    std::vector<ExprHandle>& inputs,
    const int typeConstraints = kAllTypes) {
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

ExprHandle demoteOutput(
    const ExprHandle& e,
    const c10::optional<ScalarType> type) {
  if (!type.has_value()) {
    return e;
  }
  if (*type == e.dtype().scalar_type()) {
    return e;
  }

  switch (*type) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    return cast<Type>(e);
    AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::Bool:
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

std::pair<std::vector<ExprHandle>, bool> broadcastShapesImpl(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  auto at = a.rbegin();
  auto bt = b.rbegin();
  std::vector<ExprHandle> ret;
  bool hasBroadcast = false;
  while (at != a.rend() || bt != b.rend()) {
    if (at == a.rend()) {
      hasBroadcast = true;
      ret.push_back(*bt++);
      continue;
    }
    if (bt == b.rend()) {
      hasBroadcast = true;
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
        hasBroadcast = true;
      }
    }
    ret.push_back(dim);
    at++;
    bt++;
  }
  std::reverse(ret.begin(), ret.end());
  return {ret, hasBroadcast};
}

std::pair<std::vector<ExprHandle>, bool> broadcastShapesImpl(
    std::vector<std::vector<ExprHandle>> shapes) {
  size_t n = shapes.size();
  if (n == 1) {
    return {shapes[0], false};
  }
  auto res1 = broadcastShapesImpl(shapes[n - 2], shapes[n - 1]);
  shapes[n - 2] = res1.first;
  shapes.pop_back();
  auto res2 = broadcastShapesImpl(shapes);
  return {res2.first, (res1.second || res2.second)};
}

std::vector<ExprHandle> broadcastShapes(
    std::vector<std::vector<ExprHandle>> shapes) {
  return broadcastShapesImpl(shapes).first;
}

std::vector<ExprHandle> broadcastShapes(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  return broadcastShapesImpl(a, b).first;
}

std::vector<ExprHandle> TensorExprKernel::broadcastShapesMut(
    std::vector<std::vector<ExprHandle>> shapes) {
  auto res = broadcastShapesImpl(shapes);
  if (res.second) {
    hasBroadcast_ = true;
  }
  return res.first;
}

std::vector<ExprHandle> TensorExprKernel::broadcastShapesMut(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  auto res = broadcastShapesImpl(a, b);
  if (res.second) {
    hasBroadcast_ = true;
  }
  return res.first;
}

std::vector<ExprHandle> valueShape(const ArgValue& v) {
  if (auto b = c10::get_if<tensorexpr::BufHandle>(&v)) {
    return b->dims();
  }
  return {};
}

Tensor* computeOneOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&)>& innerExpr,
    const int checkParamTypes = kAllTypes) {
  return Compute(
      name,
      c10::fmap<DimArg>(outputShape),
      [inputValues, outputType, innerExpr, checkParamTypes](
          const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices)};
        promoteInputs(inputs, checkParamTypes);
        ExprHandle compute = innerExpr(inputs[0]);
        return demoteOutput(compute, outputType);
      });
}

Tensor* computeTwoOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  return Compute(
      name,
      c10::fmap<DimArg>(outputShape),
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute = innerExpr(inputs[0], inputs[1]);
        return demoteOutput(compute, outputType);
      });
}

Tensor* computeTwoOperandWithAlpha(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  return Compute(
      name,
      c10::fmap<DimArg>(outputShape),
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute = innerExpr(inputs[0], inputs[2] * inputs[1]);
        return demoteOutput(compute, outputType);
      });
}

Tensor* computeConditionWithTwoOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr) {
  return Compute(
      name,
      c10::fmap<DimArg>(outputShape),
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
        };

        promoteInputs(inputs);
        // First expr is the condition, which we don't promote
        inputs.emplace(
            inputs.begin(), tensorOrConstant(inputValues[0], indices));
        ExprHandle compute = innerExpr(inputs[0], inputs[1], inputs[2]);
        return demoteOutput(compute, outputType);
      });
}

Tensor* computeThreeOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        innerExpr,
    bool promote_inputs = true) {
  return Compute(
      name,
      c10::fmap<DimArg>(outputShape),
      [inputValues, outputType, innerExpr, promote_inputs](
          const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
        };

        if (promote_inputs) {
          promoteInputs(inputs);
        }
        ExprHandle compute = innerExpr(inputs[0], inputs[1], inputs[2]);
        return demoteOutput(compute, outputType);
      });
}
Tensor* computeFourOperand(
    const std::string& name,
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    const std::function<ExprHandle(
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&)>& innerExpr) {
  return Compute(
      name,
      c10::fmap<DimArg>(outputShape),
      [inputValues, outputType, innerExpr](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices),
            tensorOrConstant(inputValues[1], indices),
            tensorOrConstant(inputValues[2], indices),
            tensorOrConstant(inputValues[3], indices),
        };

        promoteInputs(inputs);
        ExprHandle compute =
            innerExpr(inputs[0], inputs[1], inputs[2], inputs[3]);
        return demoteOutput(compute, outputType);
      });
}

std::pair<ScalarType, std::vector<BufHandle>> processCatList(
    const std::vector<BufHandle>& bufList) {
  if (bufList.size() == 0) {
    throw std::runtime_error("Empty input list is passed to aten::cat");
  }
  std::vector<BufHandle> bufInputs;
  std::vector<BufHandle> nonEmptyInputs;
  for (auto buf : bufList) {
    bufInputs.push_back(buf);
    TORCH_INTERNAL_ASSERT(buf.node()->dims().size() > 0);
    if (buf.node()->dims().size() == 1 &&
        immediateAs<int>(buf.node()->dim(0)) == 0) {
      continue;
    }
    nonEmptyInputs.push_back(buf);
  }
  ScalarType highType = bufInputs[0].dtype().scalar_type();
  for (const auto input : bufInputs) {
    auto maybe_dtype = input.dtype().scalar_type();
    highType = promoteTypes(highType, maybe_dtype);
  }
  return {highType, nonEmptyInputs};
}
Tensor* computeCatWoConditionals(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape) {
  auto input_list = c10::get<BufList>(inputs[0]);
  auto arg_dim = inputs[1];
  auto cat_info = processCatList(input_list);
  ScalarType high_type = cat_info.first;
  std::vector<BufHandle> non_empty_inputs = cat_info.second;

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

  auto output_sizes_expr = ExprHandleVectorToExprVector(outputShape);
  auto output_buf = new Buf("aten_cat", output_sizes_expr, ToDtype(high_type));
  if (non_empty_inputs.size() == 0) {
    return new Tensor(output_buf, new tensorexpr::Block({}));
  }

  int64_t concat_dim = c10::get<int64_t>(arg_dim);
  size_t norm_concat_dim =
      normalizeAndCheckIndex(concat_dim, outputShape.size());

  auto gen_code_for_input = [&](const BufHandle& inp,
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
    auto inp_buf = inp.node();
    auto load_expr = new Load(inp_buf, load_indices);
    auto load_promoted = promoteToDtype(ExprHandle(load_expr), high_type);
    Stmt* st = new Store(output_buf, store_indices, load_promoted.node());
    for (size_t i = dims.size(); i > 0; --i) {
      st = new For(for_vars[i - 1], new IntImm(0), dims[i - 1].node(), st);
    }
    return st;
  };

  Expr* concat_dim_size = nullptr;
  auto block = new tensorexpr::Block({});
  for (size_t i = 0; i < non_empty_inputs.size(); ++i) {
    auto input_dims =
        ExprVectorToExprHandleVector(non_empty_inputs[i].node()->dims());
    if (concat_dim_size == nullptr) {
      concat_dim_size = new IntImm(0);
    }
    block->append_stmt(gen_code_for_input(
        non_empty_inputs[i], i, concat_dim_size, input_dims));
    concat_dim_size =
        new Add(concat_dim_size, input_dims[norm_concat_dim].node());
  }
  return new Tensor(output_buf, IRSimplifier::simplify(block));
}

Tensor* computeCat(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    at::Device device) {
  if (device == at::kCPU && getCatWoConditionals()) {
    return computeCatWoConditionals(inputs, outputShape);
  }
  auto inputList = c10::get<BufList>(inputs[0]);
  auto argDim = inputs[1];
  auto catInfo = processCatList(inputList);
  ScalarType highType = catInfo.first;
  std::vector<BufHandle> nonEmptyInputs = catInfo.second;
  return Compute(
      "aten_cat",
      c10::fmap<DimArg>(outputShape),
      [&](const std::vector<VarHandle>& axes) {
        if (nonEmptyInputs.size() == 0) {
          return ExprHandle(0);
        }

        int64_t dim_ = c10::get<int64_t>(argDim);
        size_t dim = normalizeAndCheckIndex(dim_, axes.size());
        // Promote input types.
        // Note that we need to consider all inputs, including empty - they
        // also affect the resultant dtype.

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
            tensorOrConstant(nonEmptyInputs[0], newAxes), highType);
        size_t offset =
            dynamic_cast<const IntImm*>(nonEmptyInputs[0].node()->dim(dim))
                ->value();
        newAxes[dim] = newAxes[dim] - IntImm::make(offset);

        for (size_t ii = 1; ii < nonEmptyInputs.size(); ++ii) {
          auto input = nonEmptyInputs[ii];
          load = ifThenElse(
              CompareSelect::make(axes[dim], IntImm::make(offset), kLT),
              load,
              promoteToDtype(tensorOrConstant(input, newAxes), highType));

          offset +=
              dynamic_cast<const IntImm*>(input.node()->dim(dim))->value();
          newAxes[dim] = axes[dim] - IntImm::make(offset);
        }

        return load;
      });
}

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

Tensor* computeSoftmax(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
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

  TORCH_INTERNAL_ASSERT(inputs.size() == 3);
  auto output_dims = c10::fmap<DimArg>(outputShape);

  // We do not handle None for dims (input 1) because that is supposed to
  // be deprecated.
  TORCH_INTERNAL_ASSERT(c10::get_if<int64_t>(&inputs[1]));
  int64_t rank = valueShape(inputs[0]).size();
  size_t softmax_dim =
      normalizeAndCheckIndex(c10::get<int64_t>(inputs[1]), rank);
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

  c10::optional<Dtype> dtype = ToDtype(ScalarType::Undefined);
  if (auto d = c10::get_if<int64_t>(&inputs[2])) {
    dtype = ToDtype(static_cast<ScalarType>(*d));
  }

  auto max = Reduce(
      "aten_softmax_max",
      non_softmax_dims,
      Maximum(dtype.value()),
      [&](ParameterList& indices) {
        return tensorOrConstant(
            inputs[0], move_softmax_dim_index_to_pos(indices));
      },
      {output_dims[softmax_dim]});
  auto e =
      Compute("aten_softmax_exp", output_dims, [&](ParameterList& indices) {
        auto inp = tensorOrConstant(
            inputs[0], convert_indices_to_expr_handle(indices));
        return exp(inp - max->load(remove_softmax_dim_index(indices)));
      });
  auto sum = Reduce(
      "aten_softmax_sum",
      non_softmax_dims,
      Sum(),
      [&](ParameterList& indices) {
        return e->load(move_softmax_dim_index_to_pos(indices));
      },
      {output_dims[softmax_dim]});
  if (!log_softmax) {
    auto result =
        Compute("aten_softmax", output_dims, [&](ParameterList& indices) {
          return e->load(indices) /
              sum->load(remove_softmax_dim_index(indices));
        });
    return new Tensor(
        result->buf(),
        new tensorexpr::Block(
            {max->stmt(), e->stmt(), sum->stmt(), result->stmt()}));
  }

  auto log_sum = Compute(
      "aten_softmax_log_sum", non_softmax_dims, [&](ParameterList& indices) {
        return log(sum->load(indices));
      });
  auto result =
      Compute("aten_log_softmax", output_dims, [&](ParameterList& indices) {
        auto inp = tensorOrConstant(
            inputs[0], convert_indices_to_expr_handle(indices));
        auto non_softmax_indices = remove_softmax_dim_index(indices);
        return inp - max->load(non_softmax_indices) -
            log_sum->load(non_softmax_indices);
      });
  return new Tensor(
      result->buf(),
      new tensorexpr::Block(
          {max->stmt(),
           e->stmt(),
           sum->stmt(),
           log_sum->stmt(),
           result->stmt()}));
}

Tensor* computeSum(
    const std::vector<ArgValue>& inputs,
    const c10::optional<ScalarType>& outputType) {
  std::vector<size_t> axes;
  bool keepdim = false;
  // aten::sum takes the input tensor named self.
  auto sizes = valueShape(inputs[0]);

  int rank = sizes.size();
  if (inputs.size() > 2) {
    auto nodeAxes = c10::get<IntList>(inputs[1]);
    // Canonicalize axes: wrap around, sort and make unique.
    for (auto axis : nodeAxes) {
      axes.push_back(at::maybe_wrap_dim(axis, rank));
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    keepdim = c10::get<bool>(inputs[2]);
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
  std::vector<DimArg> outputDims;
  // Output dimensions are the complement of axes. When keepdim is set, a
  // one-sized dimension is inserted for each axis.
  for (size_t dim = 0; dim < sizes.size(); ++dim) {
    if (!std::count(axes.begin(), axes.end(), dim)) {
      outputDims.emplace_back(sizes[dim]);
    } else if (keepdim) {
      outputDims.emplace_back(1);
    }
  }

  return Reduce(
      "sum",
      outputDims,
      Sum(),
      [&](ParameterList& indices) {
        // "Squeeze" out indices inserted when keepdim is set.
        auto indices_squeezed =
            keepdim ? squeezeIndices(indices, axes) : indices;
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
        auto indexed = tensorOrConstant(inputs[0], indices_exprs);
        if (outputType) {
          return Cast::make(ToDtype(*outputType), indexed);
        } else {
          return indexed;
        }
      },
      reductionDims);
}

Tensor* computeMatmul(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  BufHandle ResultBuf("matmul", outputShape, dtype);
  const BufHandle a = c10::get<BufHandle>(inputs[0]);
  const BufHandle b = c10::get<BufHandle>(inputs[1]);

  auto size_a = a.dims();
  auto size_b = b.dims();
  const IntImm* total_size = dynamic_cast<const IntImm*>(
      IRSimplifier::simplify((size_a[0] * size_a[1] * size_b[1])).node());

  // For small sizes, where N*M*K < 1000, lower matmul to a naive 3-level
  // loopnest. The number is not tuned very carefully, and in future we should
  // fine-tune it as well as we should add more advanced native TE lowerings for
  // matmuls. For bigger sizes we generate a TE ExternalCall, which would call
  // an aten::matmul.
  // Native, even naive, lowering is beneficial when the sizes are small because
  // it allows to eliminate dispatch overhead.
  if (total_size && total_size->value() < 1000) {
    return Reduce(
        "nnc_matmul",
        {{size_a[0], "M"}, {size_b[1], "N"}},
        Sum(),
        [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
          return Load::make(a, {m, k}) * Load::make(b, {k, n});
        },
        {{size_a[1], "K"}});
  } else {
    return new Tensor(
        ResultBuf.node(),
        ExternalCall::make(ResultBuf, "nnc_aten_matmul", {a, b}, {}));
  }
}

Tensor* computeConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }

  BufHandle ResultBuf("conv", outputShape, dtype);
  BufHandle inp = c10::get<BufHandle>(inputs[0]);
  BufHandle w = c10::get<BufHandle>(inputs[1]);
  BufHandle b = c10::get<BufHandle>(inputs[2]);

  auto strides = _pair_int(inputs[3]);
  auto padding = _pair_int(inputs[4]);
  auto dilation = _pair_int(inputs[5]);

  int groups = c10::get<int64_t>(inputs[6]);

  auto inpInfo = getTensorInfo(inp);
  auto wInfo = getTensorInfo(w);
  auto bInfo = getTensorInfo(b);
  // Generate TE for depthwise convolutions.
  if (inpInfo && wInfo && bInfo &&
      conv2dIsSupported(
          *inpInfo, *wInfo, *bInfo, strides, padding, dilation, groups)) {
    return conv2d_depthwise(inp, w, b, strides[0], padding[0], groups);
  }

  // Once we have a performant TE representation for conv2d, we could use it
  // here instead of the external call!
  Stmt* s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_conv2d",
      {inp, w, b},
      {strides[0],
       strides[1],
       padding[0],
       padding[1],
       dilation[0],
       dilation[1],
       groups});
  return new Tensor(ResultBuf.node(), s);
}

Tensor* tensorexpr::computeOperandValue(
    c10::Symbol op,
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device device) {
  switch (op) {
    case aten::add: {
      auto add_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        return boolToInteger(lhs) + boolToInteger(rhs);
      };
      TORCH_INTERNAL_ASSERT(inputs.size() == 2 || inputs.size() == 3);
      return (inputs.size() > 2)
          ? computeTwoOperandWithAlpha(
                "aten_add", inputs, outputShape, outputType, add_lambda)
          : computeTwoOperand(
                "aten_add", inputs, outputShape, outputType, add_lambda);
    } break;
    case aten::sub: {
      auto sub_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        // NB: sub isn't supported on boolean, no need to promote to integer.
        return lhs - rhs;
      };
      TORCH_INTERNAL_ASSERT(inputs.size() == 2 || inputs.size() == 3);
      return (inputs.size() > 2)
          ? computeTwoOperandWithAlpha(
                "aten_sub", inputs, outputShape, outputType, sub_lambda)
          : computeTwoOperand(
                "aten_sub", inputs, outputShape, outputType, sub_lambda);
    } break;
    case aten::mul: {
      return computeTwoOperand(
          "aten_mul",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) * boolToInteger(rhs);
          });
    } break;
    case aten::div: {
      return computeTwoOperand(
          "aten_div",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return promoteIntegerToDefaultType(lhs) /
                promoteIntegerToDefaultType(rhs);
          });
    } break;

    case aten::__and__: {
      return computeTwoOperand(
          "aten_and",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) & boolToInteger(rhs);
          });
    } break;

    case aten::__or__: {
      return computeTwoOperand(
          "aten_or",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) | boolToInteger(rhs);
          });
    } break;

    case aten::__xor__: {
      return computeTwoOperand(
          "aten_xor",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) ^ boolToInteger(rhs);
          });
    } break;

    case aten::__lshift__: {
      return computeTwoOperand(
          "aten_lshift",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs << rhs;
          });
    } break;

    case aten::__rshift__: {
      return computeTwoOperand(
          "aten_rshift",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs >> rhs;
          });
    } break;
    case aten::eq: {
      return computeTwoOperand(
          "aten_eq",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs == rhs);
          });
    } break;

    case aten::ne: {
      return computeTwoOperand(
          "aten_ne",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs != rhs);
          });
    } break;
    case aten::ge: {
      return computeTwoOperand(
          "aten_ge",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs >= rhs);
          });
    } break;

    case aten::gt: {
      return computeTwoOperand(
          "aten_gt",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs > rhs);
          });
    } break;

    case aten::le: {
      return computeTwoOperand(
          "aten_le",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs <= rhs);
          });
    } break;

    case aten::lt: {
      return computeTwoOperand(
          "aten_lt",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs < rhs);
          });
    } break;

    case aten::min: {
      return computeTwoOperand(
          "aten_min",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Min::make(boolToInteger(lhs), boolToInteger(rhs), false);
          });
    } break;

    case aten::max: {
      return computeTwoOperand(
          "aten_max",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Max::make(boolToInteger(lhs), boolToInteger(rhs), false);
          });
    } break;
    case aten::masked_fill: {
      return computeThreeOperand(
          "aten_masked_fill",
          inputs,
          outputShape,
          outputType,
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
      if (c10::get_if<ArgNone>(&inputs[1])) {
        noMin = true;
      }

      if (c10::get_if<ArgNone>(&inputs[2])) {
        noMax = true;
      }

      return computeThreeOperand(
          "aten_clamp",
          inputs,
          outputShape,
          outputType,
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
              return clamp(cmin, cmax, in);
            }
          },
          false /* promote_inputs */);
    } break;
    case aten::addcmul: {
      return computeFourOperand(
          "aten_addcmul",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a0,
             const ExprHandle& a1,
             const ExprHandle& a2,
             const ExprHandle& a3) { return a0 + a3 * a1 * a2; });
    } break;
    case aten::sigmoid: {
      return computeOneOperand(
          "aten_sigmoid",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return sigmoid(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::reciprocal: {
      return computeOneOperand(
          "aten_reciprocal",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return ExprHandle(1.0f) / a; });
    } break;

    case aten::neg: {
      return computeOneOperand(
          "aten_neg", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return ExprHandle(-0) - a;
          });
    } break;

    case aten::isnan: {
      return computeOneOperand(
          "aten_isnan",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            if (!a.dtype().is_floating_point()) {
              return IntImm::make(0);
            }
            return isnan(a);
          });
    } break;

    case aten::relu: {
      return computeOneOperand(
          "aten_relu",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            auto zero = Cast::make(a.dtype(), 0);
            return CompareSelect::make(a, zero, zero, a, kLT);
          });
    } break;

    case aten::gelu: {
      return computeOneOperand(
          "aten_gelu",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            auto m_sqrt1_2 = Cast::make(a.dtype(), M_SQRT1_2);
            auto one = Cast::make(a.dtype(), 1.);
            auto point_five = Cast::make(a.dtype(), .5);
            return a * point_five * (one + erf(a * m_sqrt1_2));
          });
    } break;
    case aten::batch_norm: {
      bool hasWeight = true;
      bool hasBias = true;

      if (c10::get_if<ArgNone>(&inputs[1])) {
        hasWeight = false;
      }

      if (c10::get_if<ArgNone>(&inputs[2])) {
        hasBias = false;
      }

      return Compute(
          "aten_batch_norm",
          c10::fmap<DimArg>(outputShape),
          [&](const std::vector<VarHandle>& axes) {
            TORCH_INTERNAL_ASSERT(axes.size() >= 2);
            // axes: N, C, H, W
            std::vector<ExprHandle> indices(axes.begin(), axes.end());
            ExprHandle c = indices[1];

            // Parameter list:
            // input, weight, bias, mean, var, training, momentum, eps,
            // cudnn_enabled
            std::vector<ExprHandle> exprInputs = {
                tensorOrConstant(inputs[0], indices), // input
                tensorOrConstant(inputs[3], {c}), // mean
                tensorOrConstant(inputs[4], {c}), // var
                constant(inputs[7]) // eps
            };

            if (hasWeight) {
              exprInputs.push_back(tensorOrConstant(inputs[1], {c}));
            }
            if (hasBias) {
              exprInputs.push_back(tensorOrConstant(inputs[2], {c}));
            }
            promoteInputs(exprInputs);

            ExprHandle input = exprInputs[0];
            ExprHandle mean = exprInputs[1];
            ExprHandle var = exprInputs[2];
            ExprHandle eps = exprInputs[3];
            ExprHandle weight = FloatImm::make(1);
            ExprHandle bias = FloatImm::make(0);

            if (hasWeight) {
              weight = exprInputs[4];
            }
            // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
            if (hasBias) {
              bias = exprInputs[5];
            }

            // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
            auto inv_var = rsqrt(var + eps);
            auto alpha = inv_var * weight;
            auto beta = bias - mean * alpha;
            auto output = input * alpha + beta;
            return demoteOutput(output, outputType);
          });
    } break;
    case aten::log: {
      return computeOneOperand(
          "aten_log", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return log(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::log10: {
      return computeOneOperand(
          "aten_log10",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return log10(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::log1p: {
      return computeOneOperand(
          "aten_log1p",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return log1p(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::log2: {
      return computeOneOperand(
          "aten_log2",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return log2(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::exp: {
      return computeOneOperand(
          "aten_exp", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return exp(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::expm1: {
      return computeOneOperand(
          "aten_expm1",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return expm1(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::erf: {
      return computeOneOperand(
          "aten_erf", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return erf(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::erfc: {
      return computeOneOperand(
          "aten_erfc",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return erfc(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::cos: {
      return computeOneOperand(
          "aten_cos", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return cos(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::sin: {
      return computeOneOperand(
          "aten_sin", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return sin(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::tan: {
      return computeOneOperand(
          "aten_tan", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return tan(promoteIntegerToDefaultType(a));
          });
    } break;
    case aten::type_as: {
      const BufHandle rhs = c10::get<BufHandle>(inputs[1]);
      auto dtype = rhs.dtype();
      return computeOneOperand(
          "aten_type_as",
          inputs,
          outputShape,
          outputType,
          [dtype](const ExprHandle& lhs) { return Cast::make(dtype, lhs); });
    } break;
    case aten::pow: {
      return computeTwoOperand(
          "aten_pow",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
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
          "aten_fmod",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return fmod(promoteHalfToFloat(lhs), promoteHalfToFloat(rhs));
          });
    } break;

    case aten::lerp: {
      return computeThreeOperand(
          "aten_lerp",
          inputs,
          outputShape,
          outputType,
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
        auto const& shape =
            broadcastShapes(valueShape(inputs[0]), valueShape(inputs[1]));
        return Compute(
            "aten_remainder",
            c10::fmap<DimArg>(shape),
            [&](const std::vector<VarHandle>& axes) {
              std::vector<ExprHandle> indices(axes.begin(), axes.end());
              std::vector<ExprHandle> exprInputs = {
                  tensorOrConstant(inputs[0], indices),
                  tensorOrConstant(inputs[1], indices),
              };

              promoteInputs(exprInputs);
              bool allInt = true;
              for (auto& e : exprInputs) {
                if (e.dtype().is_floating_point()) {
                  allInt = false;
                  break;
                }
              }
              if (allInt) {
                return demoteOutput(
                    imodImpl(exprInputs[0], exprInputs[1]), outputType);
              } else {
                return demoteOutput(
                    fmodImpl(exprInputs[0], exprInputs[1]), outputType);
              }
            });
      }

    } break;
    case aten::acos: {
      return computeOneOperand(
          "aten_acos",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return acos(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::asin: {
      return computeOneOperand(
          "aten_asin",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return asin(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::cosh: {
      return computeOneOperand(
          "aten_cosh",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return cosh(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::sinh: {
      return computeOneOperand(
          "aten_sinh",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return sinh(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::atan: {
      return computeOneOperand(
          "aten_atan",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return atan(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::atan2: {
      return computeTwoOperand(
          "aten_atan2",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return atan2(
                promoteIntegerToDefaultType(lhs),
                promoteIntegerToDefaultType(rhs));
          });
    } break;

    case aten::tanh: {
      return computeOneOperand(
          "aten_tanh",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return tanh(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::hardtanh: {
      return computeThreeOperand(
          "aten_hardtanh",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a,
             const ExprHandle& min_val,
             const ExprHandle& max_val) {
            auto mm = CompareSelect::make(a, min_val, min_val, a, kLT);
            return CompareSelect::make(mm, max_val, max_val, mm, kGT);
          });
    } break;
    case aten::hardswish: {
      return computeOneOperand(
          "aten_hardswish",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            //  x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0
            auto zero = Cast::make(a.dtype(), 0.);
            auto three = Cast::make(a.dtype(), 3.);
            auto six = Cast::make(a.dtype(), 6.);

            return a * clamp(zero, six, a + three) / six;
          });
    } break;
    case aten::hardshrink: {
      return computeTwoOperand(
          "aten_hardshrink",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a, const ExprHandle& lambd) {
            auto pos_clambd = Cast::make(a.dtype(), lambd);
            auto neg_clambd =
                Cast::make(a.dtype(), ExprHandle(-0)) - pos_clambd;
            auto zero = Cast::make(a.dtype(), 0);
            auto mm = CompareSelect::make(a, neg_clambd, a, zero, kLT);
            return CompareSelect::make(a, pos_clambd, a, mm, kGT);
          });
    } break;
    case aten::sqrt: {
      return computeOneOperand(
          "aten_sqrt",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return tensorexpr::sqrt(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::rsqrt: {
      return computeOneOperand(
          "aten_rsqrt",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return rsqrt(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::abs: {
      return computeOneOperand(
          "aten_abs",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return tensorexpr::abs(promoteHalfToFloat(a));
          },
          kIntegralTypes | kFloatingPointTypes | kBoolType);
    } break;

    case aten::ceil: {
      return computeOneOperand(
          "aten_ceil",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return ceil(a); });
    } break;

    case aten::floor: {
      return computeOneOperand(
          "aten_floor",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return floor(a); });
    } break;

    case aten::round: {
      return computeOneOperand(
          "aten_round",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return round(a); });
    } break;

    case aten::trunc: {
      return computeOneOperand(
          "aten_trunc",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return trunc(a); });
    } break;

    case aten::_cast_Float: {
      return computeOneOperand(
          "aten_cast_float",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return cast<float>(a); });
    } break;
    case aten::to: {
      // see handling of aten::to in tensorexpr_fuser.cpp for why we only
      // need to handle the first input
      return computeOneOperand(
          "aten_to",
          {inputs[0]},
          outputShape,
          outputType,
          [outputType](const ExprHandle& a) {
            TORCH_INTERNAL_ASSERT(outputType);
            return Cast::make(ToDtype(*outputType), a);
          });
    } break;
    case aten::threshold: {
      return computeThreeOperand(
          "aten_threshold",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a,
             const ExprHandle& threshold,
             const ExprHandle& value) {
            return ifThenElse(CompareSelect::make(a, threshold, kLE), value, a);
          });
    } break;
    case aten::where: {
      return computeConditionWithTwoOperand(
          "aten_where",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a0, const ExprHandle& a1, const ExprHandle& a2) {
            return ifThenElse(a0, a1, a2);
          });
    } break;

    case aten::frac: {
      return computeOneOperand(
          "aten_frac",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            auto aa = promoteHalfToFloat(a);
            return aa - floor(aa);
          },
          kFloatingPointTypes);
    } break;

    case aten::lgamma: {
      return computeOneOperand(
          "aten_lgamma",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return lgamma(promoteIntegerToDefaultType(a));
          });
    } break;

    case aten::rand_like: {
      return computeOneOperand(
          "aten_rand_like",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return Intrinsics::make(IntrinsicsOp::kRand, a.dtype());
          });
    } break;
    case aten::slice: {
      return Compute(
          "aten_slice",
          c10::fmap<DimArg>(outputShape),
          [&](const std::vector<VarHandle>& axes) {
            int64_t dim = c10::get<int64_t>(inputs[1]);
            ExprHandle start = constant(inputs[2]);
            ExprHandle stride = constant(inputs[4]);

            std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
            newAxes[dim] = stride * newAxes[dim] + start;
            return tensorOrConstant(inputs[0], newAxes);
          });
    }
    case aten::unsqueeze: {
      return Compute(
          "aten_unsqueeze",
          c10::fmap<DimArg>(outputShape),
          [&](const std::vector<VarHandle>& axes) {
            int64_t dim = c10::get<int64_t>(inputs[1]);
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

            return broadcast(c10::get<BufHandle>(inputs[0]), indices);
          });
    }
    case aten::t: {
      auto shape = valueShape(inputs[0]);
      if (shape.size() == 1) {
        return new Tensor(c10::get<BufHandle>(inputs[0]).node(), nullptr);
      }
      return computeOperandValue(
          aten::transpose,
          {inputs[0], (int64_t)1, (int64_t)0},
          outputShape,
          outputType);
    }
    case aten::transpose: {
      auto A = c10::get<BufHandle>(inputs[0]);
      auto start_dim =
          at::maybe_wrap_dim(c10::get<int64_t>(inputs[1]), A.ndim());
      auto to_dim = at::maybe_wrap_dim(c10::get<int64_t>(inputs[2]), A.ndim());
      return Compute(
          "aten_transpose",
          c10::fmap<DimArg>(outputShape),
          [&](std::vector<VarHandle> axes) {
            std::swap(axes[start_dim], axes[to_dim]);
            return A.load(axes);
          });
    }
    case aten::permute: {
      auto A = c10::get<BufHandle>(inputs[0]);
      auto permute_dims = c10::get<IntList>(inputs[1]);
      return Compute(
          "aten_permute",
          c10::fmap<DimArg>(outputShape),
          [&](const std::vector<VarHandle>& axes) {
            std::vector<VarHandle> new_axes;
            assert(permute_dims.size() == axes.size());
            for (auto i : permute_dims) {
              new_axes.push_back(axes[i]);
            }
            return A.load(new_axes);
          });
    }
    case aten::expand: {
      auto A = c10::get<BufHandle>(inputs[0]);
      return Compute(
          "aten_expand",
          c10::fmap<DimArg>(outputShape),
          [&](const std::vector<VarHandle>& axes) {
            std::vector<ExprHandle> indices(axes.begin(), axes.end());
            return broadcast(A, indices);
          });
    }
    case aten::mm: // aten::mm is a subset of aten::matmul where both inputs are
                   // rank 2
    case aten::matmul: {
      return computeMatmul(inputs, outputShape, outputType);
    }
    case aten::cat: {
      return computeCat(inputs, outputShape, device);
    }
    case aten::sum: {
      return computeSum(inputs, outputType);
    }
    case aten::softmax: {
      return computeSoftmax(inputs, outputShape, false);
    }
    case aten::log_softmax: {
      return computeSoftmax(inputs, outputShape, true);
    }
    case aten::conv2d: {
      return computeConv2d(inputs, outputShape, outputType);
    } break;
    default: {
      std::string msg =
          std::string("Unhandled node kind: ") + op.toQualString();
      throw malformed_input(msg);
    }
  }
}

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
  auto inputs = v->node()->inputs();
  switch (v->node()->kind()) {
    case aten::rand_like:
      hasRandom_ = true;
      // fallthrough
    case aten::add:
    case aten::sub:
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
    case aten::masked_fill:
    case aten::clamp:
    case aten::addcmul:
    case aten::sigmoid:
    case aten::reciprocal:
    case aten::neg:
    case aten::isnan:
    case aten::relu:
    case aten::hardswish:
    case aten::gelu:
    case aten::batch_norm:
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
    case aten::type_as:
    case aten::pow:
    case aten::fmod:
    case aten::lerp:
    case aten::remainder:
    case aten::acos:
    case aten::asin:
    case aten::cosh:
    case aten::sinh:
    case aten::atan:
    case aten::atan2:
    case aten::tanh:
    case aten::hardtanh:
    case aten::hardshrink:
    case aten::sqrt:
    case aten::rsqrt:
    case aten::abs:
    case aten::ceil:
    case aten::floor:
    case aten::round:
    case aten::trunc:
    case aten::_cast_Float:
    case aten::threshold:
    case aten::where:
    case aten::frac:
    case aten::lgamma:
    case aten::slice:
    case aten::unsqueeze:
    case aten::t:
    case aten::transpose:
    case aten::expand:
    case aten::permute:
    case aten::mm:
    case aten::matmul:
    case aten::cat:
    case aten::sum:
    case aten::softmax:
    case aten::log_softmax:
    case aten::conv2d: {
      std::vector<ArgValue> argInputs;
      for (auto inp : inputs) {
        argInputs.push_back(toArg(inp));
      }
      auto outputType = findDtypeForValue(v->node()->output());
      std::vector<ExprHandle> outputShape = {};
      // shape inference not implemented for sum
      if (v->node()->kind() != aten::sum) {
        outputShape = sizesForValue(v);
      }
      return computeOperandValue(
          v->node()->kind(), argInputs, outputShape, outputType, device_);
    } break;

    case aten::to: {
      std::vector<ArgValue> argInputs;
      argInputs.push_back(toArg(inputs[0]));
      auto outputType = findDtypeForValue(v->node()->output());
      std::vector<ExprHandle> outputShape = {};
      // shape inference not implemented for sum
      if (v->node()->kind() != aten::sum) {
        outputShape = sizesForValue(v);
      }
      return computeOperandValue(
          v->node()->kind(), argInputs, outputShape, outputType, device_);
    } break;

    case prim::ConstantChunk: {
      return Compute(
          "prim_constantchunk",
          c10::fmap<DimArg>(sizesForValue(v)),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            int64_t dim = n->i(attr::dim);
            int64_t chunks = n->i(attr::chunks);
            std::vector<ExprHandle> indices(axes.begin(), axes.end());
            return chunk(
                bufs_.at(n->input(0)), v->offset(), dim, chunks, indices);
          });
    } break;

    default: {
      std::string msg = std::string("Unhandled node kind: ") +
          v->node()->kind().toQualString();
      throw malformed_input(msg);
    }
  }
  return nullptr;
}

// Return the (lower, upper) loop bounds if they are constants, else nullopt.
c10::optional<std::pair<int64_t, int64_t>> loopBounds(const For* loop) {
  auto start = IRSimplifier::simplify(loop->start());
  auto stop = IRSimplifier::simplify(loop->stop());
  if (!start->isConstant() || !stop->isConstant()) {
    return c10::nullopt;
  }
  return c10::make_optional(
      std::make_pair(immediateAs<int64_t>(start), immediateAs<int64_t>(stop)));
}

// True if all the loops in this vector have equal bounds.
bool loopBoundsAllEqual(const std::vector<For*>& loops) {
  auto bounds = loopBounds(loops[0]);
  if (!bounds) {
    return false;
  }
  for (auto const& loop : loops) {
    auto next = loopBounds(loop);
    if (!next) {
      return false;
    }
    if (bounds->first != next->first || bounds->second != next->second) {
      return false;
    }
  }
  return true;
}

// Recursively fuse all the loops with matching bounds in `st`.  Stops fusing
// at any level containing non-loops or non-matching bounds.  The restriction
// on matching bounds exists to avoid inserting conditionals on the loop
// indices where none would be needed, which would significantly complicate
// vectorization.
void fuseAllLoops(Stmt* st) {
  if (auto block = dynamic_cast<tensorexpr::Block*>(st)) {
    std::vector<For*> loopsToFuse;
    for (auto stmt : *block) {
      auto loop = dynamic_cast<For*>(stmt);
      if (!loop) {
        // Block contains something that's not a loop.  Quit.
        return;
      }
      loopsToFuse.push_back(loop);
    }
    if (!loopBoundsAllEqual(loopsToFuse)) {
      return;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    For* fusedLoop;
    if (!LoopNest::fuseLoops(loopsToFuse, &fusedLoop)) {
      return;
    }
    fuseAllLoops(fusedLoop->body());
  }
}

Stmt* TensorExprKernel::transformLoops(BackendType backendType, Stmt* st) {
  torch::jit::tensorexpr::LoopNest l(st, bufOutputs_);
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

  // Inlining output & intermediate buffers can duplicate computation.
  // Duplicating work can slow down the program if it's not ameliorated in some
  // way, but we've empirically found that:
  // - On CPU, LLVM's CSE does a good job as long as you horizontally fuse
  //   output loops.
  // - On GPU, there's enough compute to hide the extra work, and inlining
  //   avoids synchronizing between kernels.
  l.inlineIntermediateBufs(/*allow_duplicated_work=*/true);

  // Fuse loops "horizontally".  This pass allows us to combine loops that
  // write to different output buffers, as long as they have the same bounds.
  if (backendType == kLLVMCodeGen) {
    GRAPH_DEBUG("after inline", *l.root_stmt());
    fuseAllLoops(l.root_stmt());
    GRAPH_DEBUG("after fuse", *l.root_stmt());
  }

  if (backendType == kCudaCodeGen) {
    for (auto buf : bufOutputs_) {
      std::vector<For*> loops = l.getLoopStmtsFor(buf);
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
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        For* outer;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        For* inner;
        const int kDefaultBlockSize = 512;
        if (blockSize < 0) {
          blockSize = kDefaultBlockSize;
        }
        l.splitWithMask(flattened, blockSize, &outer, &inner);
        l.setGPUBlockIndex(outer, 0);
        l.setGPUThreadIndex(inner, 0);
      } else if (loopLevels == 3) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        For* outer;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        For* inner;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        For* inner1;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
    for (auto buf : bufOutputs_) {
      const int default_fp16_blocksize = 16;
      const int default_uint8_blocksize = 32;
      int blockSize = default_fp16_blocksize;
      // We only handle looplevels == 2 for now
      if (buf->dtype().scalar_type() == ScalarType::Byte) {
        blockSize = default_uint8_blocksize;
      }
      std::vector<For*> loops = l.getLoopStmtsFor(buf);
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
      // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
      sanitized_name = sanitized_name + "_";
    }
    value_to_name[input] = sanitized_name;
    name_set.insert(sanitized_name);
  }
  input_name_map_ = std::move(value_to_name);
}

Tensor* TensorExprKernel::bindInput(const torch::jit::Value* input) {
  auto const& t = input->type();
  Tensor* result = nullptr;
  switch (t->kind()) {
    case TypeKind::TensorType: {
      auto tt = input->type()->cast<TensorType>();
      if (!input->isCompleteTensor()) {
        std::string msg = std::string("Shapes for input '%") +
            input->debugName() + "' are unknown";
        throw malformed_input(msg);
      }
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
      result = Compute(
          "input" + c10::to_string(bufs_.size() + 1),
          inputTensorDims,
          [&](const std::vector<VarHandle>& axes) {
            ExprHandle idx = 0;
            for (size_t i = 0; i < axes.size(); i++) {
              idx = idx + axes[i] * IntImm::make(*strides[i]);
            }
            return inBuffer.load(idx);
          });
      bufs_.emplace(input, result->buf());

      bufferArgs_.emplace_back(inBuffer);
      break;
    }
    case TypeKind::FloatType: {
      VarHandle v("v" + input_name_map_[input], kDouble);
      bufferArgs_.emplace_back(v);
      scalars_.emplace(input, v);
      break;
    }
    case TypeKind::BoolType: {
      VarHandle v("v" + input_name_map_[input], kBool);
      bufferArgs_.emplace_back(v);
      scalars_.emplace(input, v);
      break;
    }
    case TypeKind::IntType: {
      VarHandle v("v" + input_name_map_[input], kLong);
      bufferArgs_.emplace_back(v);
      scalars_.emplace(input, v);
      break;
    }
    default: {
      throw unsupported_dtype();
      break;
    }
  }
  return result;
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
  TORCH_INTERNAL_ASSERT(bufs_.count(v));
  const Buf* buf = bufs_.at(v);

  // No shape info is present in the graph
  if (!tt->sizes().concrete_sizes()) {
    std::string msg =
        std::string("Shapes for output '%") + v->debugName() + "' are unknown";
    throw malformed_input(msg);
  }

  TORCH_INTERNAL_ASSERT(tt->sizes().concrete_sizes());
  const auto sizes = *tt->sizes().concrete_sizes();
  std::vector<int64_t> default_strides = TensorType::contiguousStridesOf(sizes);
  TORCH_INTERNAL_ASSERT(tt->strides().concrete_sizes());
  const std::vector<int64_t> strides = *tt->strides().concrete_sizes();
  // All Tensors in NNC are layed out in default, contiguous layout.
  // If the output is also default contiguous we don't need to do anything
  if (strides == default_strides) {
    return new Tensor(buf, nullptr);
  }
  // If the tensor is not dense or overlaps, we have
  // no way of matching the profiled striding
  if (!denseAndNonOverlapping(sizes, strides)) {
    return new Tensor(buf, nullptr);
  }

  auto dims = c10::fmap<DimArg>(sizesForValue(v));
  // We need to convert the output tensor so that its values are layed
  // so that when viewed from the output strides the values are correct.
  // A contiguous Tensor of size(2, 3) with values 0-5 is layed out as:
  // [0] [1] [2] [3] [4] [5]
  // The same valued tensor with strides (2, 1) would be layed out like
  // [0] [3] [1] [4] [2] [5]
  // When we are doing the re-ordering of values into the output tensor,
  // we are iterating per-element of the input, and we are fixed
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
          auto size = sizes[stride_index];
          auto index = Div::make(absolute_position, IntImm::make(stride));
          if (size != 1) {
            absolute_position =
                Mod::make(absolute_position, IntImm::make(stride));
          }
          new_axes[stride_index] = index;
        }
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        return BufHandle(buf).load(new_axes);
      });
}

void TensorExprKernel::bindConstant(const torch::jit::Value* v) {
  if (!v->type()->cast<TensorType>()) {
    // Only Tensor constants need to be bound, scalar constants will be turned
    // into immediates in TE IR
    return;
  }
  auto const_tensor = toIValue(v)->toTensor();

  const auto& tt = v->type()->expect<TensorType>();
  const auto sizes = *tt->sizes().concrete_sizes();
  std::vector<ExprHandle> te_sizes;
  te_sizes.reserve(sizes.size());
  for (auto s : sizes) {
    te_sizes.push_back(IntImm::make(s));
  }

  const Buf* buf = new Buf(
      "const_" + v->debugName(),
      ExprHandleVectorToExprVector(te_sizes),
      ToDtype(static_cast<ScalarType>(*tt->scalarType())));

  if (!const_tensor.is_contiguous()) {
    const_tensor = const_tensor.clone().contiguous();
    unpacked_constant_tensors_.push_back(const_tensor);
  }

  constants_.push_back({buf, const_tensor.data_ptr()});
  bufs_[v] = buf;
}

void TensorExprKernel::compile() {
  KernelScope kernelScope(&kernelArena_);
  GRAPH_DUMP("TensorExprKernel graph:", graph_);

  device_ = *pickDeviceType(graph_->inputs());

  // Block to collect the Stmts corresponding to all tensors.
  auto block = new Block({});

  // Bind inputs to buffers.
  nInputs_ = graph_->inputs().size();
  genInputDebugNames();
  for (auto const& input : graph_->inputs()) {
    if (Tensor* t = bindInput(input)) {
      block->append_stmt(t->stmt());
    }
  }

  // Bind nodes to tensor compute expressions.
  for (auto const& n : graph_->nodes()) {
    if (n->kind() == prim::ListConstruct) {
      continue;
    } else if (n->kind() == prim::Constant) {
      bindConstant(n->output());
      continue;
    } else {
      for (auto const& output : n->outputs()) {
        if (output->hasUses()) {
          Tensor* t = computeValue(output);
          bufs_.emplace(output, t->buf());
          // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
          block->append_stmt(t->stmt());
        }
      }
    }
    if (hasRandom_ && hasBroadcast_) {
      throw std::runtime_error(
          "Cannot support broadcast and random within one kernel");
    }
  }

  // Move output operands from `bufs_` to `bufOutputs_`
  for (const auto& output : graph_->outputs()) {
    if (!bufs_.count(output)) {
      throw malformed_input("cannot find output Tensor");
    }
    // The "strided" tensor will be incorrect if used in NNC,
    // since NNC views it as contiguous. Only convert it to the right
    // strides at the end of the kernel (if already contiguous it's a no-op)
    Tensor* properly_strided_output = convertOutputToCorrectStrides(output);
    if (properly_strided_output->stmt()) {
      block->append_stmt(properly_strided_output->stmt());
    }
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    bufs_[output] = properly_strided_output->buf();
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

    bufOutputs_.insert(bufs_.at(output));
    bufferArgs_.emplace_back(BufHandle(bufs_.at(output)));
    tensorOutputTensorOptions_.emplace_back(
        c10::TensorOptions(tensorType(bufs_.at(output))).device(device_));
    bufs_.erase(output);
  }

  for (auto c : constants_) {
    bufferArgs_.emplace_back(BufHandle(c.buf));
  }

  BackendType backendType = inferBackendTypeFromDevice(device_);
  Stmt* stmt = transformLoops(backendType, block);

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
  // TODO: preallocate `runArgs` during compilation and fill in values where
  // possible (e.g. for constant tensors)
  std::vector<CodeGen::CallArg> runArgs;
  runArgs.reserve(inputs.size() + bufOutputs_.size());

  for (const auto& input : inputs) {
    if (input.isInt()) {
      runArgs.emplace_back(input.toInt());
    } else if (input.isDouble()) {
      runArgs.emplace_back(input.toDouble());
    } else if (input.isTensor()) {
      runArgs.emplace_back(input.toTensor().data_ptr());
    }
  }

  for (size_t i = 0, e = bufOutputs_.size(); i < e; ++i) {
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

  for (auto c : constants_) {
    runArgs.emplace_back(c.ptr);
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

void TensorExprKernel::runFast(
    const std::vector<void*>& inputs,
    const std::vector<void*>& outputs) {
  KernelScope kernelScope(&kernelArena_);

  std::vector<void*> args(inputs);
  args.reserve(inputs.size() + outputs.size() + constants_.size());
  args.insert(args.end(), outputs.begin(), outputs.end());

  // TODO: we can consider preallocating and pre-filling the args vector.
  for (auto c : constants_) {
    args.push_back(c.ptr);
  }

  // Call the kernel.
  codegen_->call_raw(args);
}
