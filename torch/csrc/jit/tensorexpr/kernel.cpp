#include <c10/util/variant.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/TensorGeometry.h>
#include <c10/util/irange.h>
#include <c10/util/string_utils.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {
namespace tensorexpr {

std::string buildErrorMessage(const std::string& s) {
  static const std::string generic_error_message =
      "This error occured in the fuser. You can turn off the fuser with "
      "torch.jit.enable_fusion(False).";
  if (s.empty()) {
    return generic_error_message;
  }
  if (s.back() == '.') {
    return s + " " + generic_error_message;
  }
  return s + ". " + generic_error_message;
}

static int te_cuda_pointwise_loop_levels = -1;
static int te_cuda_pointwise_block_count = -1;
static int te_cuda_pointwise_block_size = -1;
static bool fallback_allowed = false;
static bool te_generate_block_code = false;
static bool te_must_use_llvm_on_cpu = true;
static bool cat_wo_conditionals = true; // NOLINT
static bool opt_conditionals = false; // NOLINT

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

bool& getOptConditionals() {
  return opt_conditionals;
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

c10::optional<at::Device> pickDeviceType(const std::shared_ptr<Graph>& graph) {
  c10::optional<at::Device> device = c10::nullopt;
  for (auto const& node : graph->nodes()) {
    for (auto const& input : node->inputs()) {
      if (auto tt = input->type()->cast<TensorType>()) {
        if (auto inputDevice = tt->device()) {
          TORCH_INTERNAL_ASSERT(
              !device || *device == *inputDevice,
              buildErrorMessage(
                  "Different devices specified for inputs to the fuser."));
          device = inputDevice;
        }
      }
    }
  }
  TORCH_INTERNAL_ASSERT(
      device,
      buildErrorMessage("Could not find device in fuser graph inputs."));
  return device;
}

// If v is a Tensor with concretely-known sizes and dtype, return them, else
// nullopt.
c10::optional<TensorInfo> getTensorInfoJit(torch::jit::Value* v) {
  auto const& it = v->type()->cast<TensorType>();

  c10::ScalarType dtype = c10::ScalarType::Float;

  if (!it) {
    return c10::nullopt;
  }
  if (!it->isComplete()) {
    return c10::nullopt;
  }
  if (it->scalarType()) {
    // TODO: ideally we should be strict here and return nullopt if the dtype is
    // absent in the JIT IR. We're assuming a default Float dtype for now, until
    // dtype propagation is implemented.
    dtype = *it->scalarType();
  }
  auto concrete_sizes = it->sizes().concrete_sizes();
  if (!concrete_sizes) {
    return c10::nullopt;
  }
  return TensorInfo{*concrete_sizes, dtype};
}
std::vector<int64_t> _pair_int(IValue v) {
  if (v.isIntList()) {
    return v.toIntVector();
  } else {
    return {v.toInt(), v.toInt()};
  }
}

static bool isContiguous(const torch::jit::Value* v) {
  auto const& tt = v->type()->cast<TensorType>();
  if (!tt) {
    return false;
  }
  if (!tt->isComplete()) {
    return false;
  }
  auto const& sizes = tt->sizes().concrete_sizes();
  auto const& strides = tt->strides().concrete_sizes();
  if (!sizes || !strides) {
    return false;
  }
  return *strides == TensorType::contiguousStridesOf(*sizes);
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

  // All inputs should be contiguous so no transposition is required.
  if (!isContiguous(node->input(0)) || !isContiguous(node->input(1)) ||
      !isContiguous(node->input(2))) {
    GRAPH_DEBUG("conv2dIsSupported: some inputs are not contiguous");
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

  // Inputs should be contiguous, or the TE will needlessly transpose them.
  if (!isContiguous(node->input(0)) || !isContiguous(node->input(1))) {
    GRAPH_DEBUG("matmulIsSupported: Input shapes are not contiguous");
    return false;
  }

  return true;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

static at::ScalarType tensorType(BufPtr b) {
  return static_cast<at::ScalarType>(b->dtype().scalar_type());
}

ExprHandle TensorExprKernel::constant(const torch::jit::Value* v) {
  if (v->node()->kind() == prim::Constant) {
    auto val = toIValue(v).value();
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
    auto val = toIValue(v).value();
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
  for (const auto i : c10::irange(*shape.size())) {
    dims.push_back(*shape[i]);
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
    if (tt->sizes().concrete_sizes()) {
      return sizesFromVaryingShape(tt->sizes());
    }
  }

  if (v->type()->isSubtypeOf(FloatType::get()) ||
      v->type()->isSubtypeOf(IntType::get())) {
    return {int64_t{1}};
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
    case aten::relu6:
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
    case aten::hardsigmoid:
    case aten::hardswish:
    case aten::softplus:
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
    case aten::sign:
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
      for (const auto idx : c10::irange(2)) {
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
      for (const auto idx : c10::irange(3)) {
        torch::jit::Value* inp = v->node()->input(idx);
        shapes.push_back(sizesForValue(inp));
      }
      return broadcastShapesMut(shapes);
    }

    case aten::addcmul: {
      std::vector<std::vector<ExprHandle>> shapes;
      for (const auto idx : c10::irange(4)) {
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

      TORCH_INTERNAL_ASSERT(
          n->input(1)->node()->kind() == prim::Constant,
          buildErrorMessage(
              "aten::cat op's dim input is not constant in fuser."));
      int64_t dim = n->input(1)->node()->i(attr::value);
      auto shape = sizesForValue(inputs[0]);
      auto norm_dim = normalizeAndCheckIndex(dim, shape.size());
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
      std::string msg =
          std::string("Unhandled node kind (in inferSizesForValue): ") +
          v->node()->kind().toQualString();
      throw malformed_input(msg);
    }
  }
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

RegisterNNCLoweringFunction aten_sub(
    "aten::sub",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      auto sub_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        // NB: sub isn't supported on boolean, no need to promote to integer.
        return lhs - rhs;
      };
      TORCH_INTERNAL_ASSERT(
          inputs.size() == 2 || inputs.size() == 3,
          buildErrorMessage("Invalid number of input operands"));
      return (inputs.size() > 2)
          ? computeTwoOperandWithAlpha(
                "aten_sub", inputs, outputShape, outputType, sub_lambda)
          : computeTwoOperand(
                "aten_sub", inputs, outputShape, outputType, sub_lambda);
    });

RegisterNNCLoweringFunction aten_mul(
    "aten::mul",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_mul",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) * boolToInteger(rhs);
          });
    });

RegisterNNCLoweringFunction aten_div(
    "aten::div",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_div",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return promoteIntegerToDefaultType(lhs) /
                promoteIntegerToDefaultType(rhs);
          });
    });

RegisterNNCLoweringFunction aten___and__(
    "aten::__and__",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_and",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) & boolToInteger(rhs);
          });
    });

RegisterNNCLoweringFunction aten___or__(
    "aten::__or__",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_or",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) | boolToInteger(rhs);
          });
    });

RegisterNNCLoweringFunction aten___xor__(
    "aten::__xor__",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_xor",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return boolToInteger(lhs) ^ boolToInteger(rhs);
          });
    });

RegisterNNCLoweringFunction aten___lshift__(
    "aten::__lshift__",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_lshift",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs << rhs;
          });
    });

RegisterNNCLoweringFunction aten___rshift__(
    "aten::__rshift__",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_rshift",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs >> rhs;
          });
    });

RegisterNNCLoweringFunction aten_eq(
    "aten::eq",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_eq",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs == rhs);
          });
    });

RegisterNNCLoweringFunction aten_ne(
    "aten::ne",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_ne",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs != rhs);
          });
    });

RegisterNNCLoweringFunction aten_ge(
    "aten::ge",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_ge",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs >= rhs);
          });
    });

RegisterNNCLoweringFunction aten_gt(
    "aten::gt",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_gt",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs > rhs);
          });
    });

RegisterNNCLoweringFunction aten_le(
    "aten::le",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_le",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs <= rhs);
          });
    });

RegisterNNCLoweringFunction aten_lt(
    "aten::lt",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_lt",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return cast<bool>(lhs < rhs);
          });
    });

RegisterNNCLoweringFunction aten_min(
    "aten::min",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_min",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Min::make(boolToInteger(lhs), boolToInteger(rhs), false);
          });
    });

RegisterNNCLoweringFunction aten_max(
    "aten::max",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_max",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Max::make(boolToInteger(lhs), boolToInteger(rhs), false);
          });
    });

RegisterNNCLoweringFunction aten_masked_fill(
    "aten::masked_fill",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });
RegisterNNCLoweringFunction aten_clamp(
    "aten::clamp",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_addcmul(
    "aten::addcmul",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeFourOperand(
          "aten_addcmul",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a0,
             const ExprHandle& a1,
             const ExprHandle& a2,
             const ExprHandle& a3) { return a0 + a3 * a1 * a2; });
    });

RegisterNNCLoweringFunction aten_sigmoid(
    "aten::sigmoid",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_sigmoid",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return sigmoid(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_reciprocal(
    "aten::reciprocal",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_reciprocal",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return ExprHandle(1.0f) / a; });
    });

RegisterNNCLoweringFunction aten_neg(
    "aten::neg",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_neg", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return ExprHandle(-0) - a;
          });
    });

RegisterNNCLoweringFunction aten_isnan(
    "aten::isnan",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_relu(
    "aten::relu",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_relu",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            auto zero = Cast::make(a.dtype(), 0);
            return CompareSelect::make(a, zero, zero, a, kLT);
          });
    });

RegisterNNCLoweringFunction aten_leaky_relu(
    "aten::leaky_relu",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_leaky_relu",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a, const ExprHandle& negative_slope) {
            auto neg_slope = Cast::make(a.dtype(), negative_slope);
            auto zero = Cast::make(a.dtype(), 0);
            auto one = Cast::make(a.dtype(), 1);
            auto cs = CompareSelect::make(a, zero, one, neg_slope, kGT);
            return a * cs;
          });
    });

RegisterNNCLoweringFunction aten_relu6(
    "aten::relu6",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_relu6",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            auto zero = Cast::make(a.dtype(), 0);
            auto six = Cast::make(a.dtype(), 6.);
            return clamp(zero, six, a);
          });
    });

RegisterNNCLoweringFunction aten_gelu(
    "aten::gelu",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_batch_norm(
    "aten::batch_norm",
    computeBatchNorm);

RegisterNNCLoweringFunction aten_log(
    "aten::log",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_log", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return log(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_log10(
    "aten::log10",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_log10",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return log10(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_log1p(
    "aten::log1p",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_log1p",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return log1p(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_log2(
    "aten::log2",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_log2",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return log2(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_exp(
    "aten::exp",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_exp", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return exp(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_expm1(
    "aten::expm1",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_expm1",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return expm1(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_erf(
    "aten::erf",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_erf", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return erf(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_erfc(
    "aten::erfc",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_erfc",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return erfc(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_cos(
    "aten::cos",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_cos", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return cos(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_sin(
    "aten::sin",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_sin", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return sin(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_tan(
    "aten::tan",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_tan", inputs, outputShape, outputType, [](const ExprHandle& a) {
            return tan(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_type_as(
    "aten::type_as",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      const BufHandle rhs = c10::get<BufHandle>(inputs[1]);
      auto dtype = rhs.dtype();
      return computeOneOperand(
          "aten_type_as",
          inputs,
          outputShape,
          outputType,
          [dtype](const ExprHandle& lhs) { return Cast::make(dtype, lhs); });
    });

RegisterNNCLoweringFunction aten_pow(
    "aten::pow",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_fmod(
    "aten::fmod",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTwoOperand(
          "aten_fmod",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return fmod(promoteHalfToFloat(lhs), promoteHalfToFloat(rhs));
          });
    });

RegisterNNCLoweringFunction aten_lerp(
    "aten::lerp",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeThreeOperand(
          "aten_lerp",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a,
             const ExprHandle& end,
             const ExprHandle& weight) { return a + weight * (end - a); });
    });

RegisterNNCLoweringFunction aten_remainder(
    "aten::remainder",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction prim_ConstantChunk(
    "prim::ConstantChunk",
    computeChunk);

RegisterNNCLoweringFunction aten_acos(
    "aten::acos",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_acos",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return acos(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_asin(
    "aten::asin",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_asin",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return asin(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_cosh(
    "aten::cosh",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_cosh",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return cosh(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_sinh(
    "aten::sinh",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_sinh",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return sinh(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_atan(
    "aten::atan",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_atan",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return atan(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_atan2(
    "aten::atan2",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_tanh(
    "aten::tanh",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_tanh",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return tanh(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_hardtanh(
    "aten::hardtanh",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_softplus(
    "aten::softplus",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeThreeOperand(
          "aten_softplus",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a,
             const ExprHandle& beta,
             const ExprHandle& threshold) {
            auto beta_promoted = Cast::make(a.dtype(), beta);
            auto threshold_promoted = Cast::make(a.dtype(), threshold);
            auto beta_a = beta_promoted * a;
            return CompareSelect::make(
                beta_a,
                threshold_promoted,
                a,
                log1p(exp(beta_a)) / beta_promoted,
                kGT);
          });
    });

RegisterNNCLoweringFunction aten_hardsigmoid(
    "aten::hardsigmoid",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_hardsigmoid",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            auto zero = Cast::make(a.dtype(), 0.0);
            auto three = Cast::make(a.dtype(), 3.0);
            auto six = Cast::make(a.dtype(), 6.0);
            return clamp(zero, six, a + three) / six;
          });
    });

RegisterNNCLoweringFunction aten_hardswish(
    "aten::hardswish",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_hardshrink(
    "aten::hardshrink",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_sqrt(
    "aten::sqrt",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_sqrt",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return tensorexpr::sqrt(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_rsqrt(
    "aten::rsqrt",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_rsqrt",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return rsqrt(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_abs(
    "aten::abs",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_abs",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return tensorexpr::abs(promoteHalfToFloat(a));
          },
          kIntegralTypes | kFloatingPointTypes | kBoolType);
    });

RegisterNNCLoweringFunction aten_sign(
    "aten::sign",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) { return computeSign(inputs, outputShape); });

RegisterNNCLoweringFunction aten_ceil(
    "aten::ceil",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_ceil",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return ceil(a); });
    });

RegisterNNCLoweringFunction aten_floor(
    "aten::floor",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_floor",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return floor(a); });
    });

RegisterNNCLoweringFunction aten_round(
    "aten::round",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_round",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return round(a); });
    });

RegisterNNCLoweringFunction aten_trunc(
    "aten::trunc",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_trunc",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return trunc(a); });
    });

RegisterNNCLoweringFunction aten__cast_Float(
    "aten::_cast_Float",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_cast_float",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) { return cast<float>(a); });
    });

RegisterNNCLoweringFunction aten_to(
    "aten::to",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      // see handling of aten::to in tensorexpr_fuser.cpp for why we only
      // need to handle the first input
      return computeOneOperand(
          "aten_to",
          {inputs[0]},
          outputShape,
          outputType,
          [outputType](const ExprHandle& a) {
            TORCH_INTERNAL_ASSERT(
                outputType, buildErrorMessage("Output type is null."));
            return Cast::make(ToDtype(*outputType), a);
          });
    });

RegisterNNCLoweringFunction aten_threshold(
    "aten::threshold",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_where(
    "aten::where",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeConditionWithTwoOperand(
          "aten_where",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a0, const ExprHandle& a1, const ExprHandle& a2) {
            return ifThenElse(a0, a1, a2);
          });
    });

RegisterNNCLoweringFunction aten_frac(
    "aten::frac",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });

RegisterNNCLoweringFunction aten_lgamma(
    "aten::lgamma",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_lgamma",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return lgamma(promoteIntegerToDefaultType(a));
          });
    });

RegisterNNCLoweringFunction aten_rand_like(
    "aten::rand_like",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeOneOperand(
          "aten_rand_like",
          inputs,
          outputShape,
          outputType,
          [](const ExprHandle& a) {
            return Intrinsics::make(IntrinsicsOp::kRand, a.dtype());
          });
    });

RegisterNNCLoweringFunction aten_slice(
    "aten::slice",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return Compute(
          "aten_slice",
          c10::fmap<DimArg>(outputShape),
          [&](const std::vector<VarHandle>& axes) {
            int64_t dim =
                at::maybe_wrap_dim(c10::get<int64_t>(inputs[1]), axes.size());
            ExprHandle start = constant(inputs[2]);
            ExprHandle stride = constant(inputs[4]);

            std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
            newAxes[dim] = stride * newAxes[dim] + start;
            return tensorOrConstant(inputs[0], newAxes);
          });
    });
RegisterNNCLoweringFunction aten_unsqueeze(
    "aten::unsqueeze",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
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
    });
RegisterNNCLoweringFunction aten_t(
    "aten::t",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeTranspose(
          {inputs[0], (int64_t)1, (int64_t)0}, outputShape, outputType, device);
    });
RegisterNNCLoweringFunction aten_transpose("aten::transpose", computeTranspose);
RegisterNNCLoweringFunction aten_permute(
    "aten::permute",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      auto A = c10::get<BufHandle>(inputs[0]);
      // Trivial case of 0-dim tensors: just a copy of the input
      if (A.ndim() == 0) {
        return Compute(
            "aten_permute",
            c10::fmap<DimArg>(outputShape),
            [&](const std::vector<VarHandle>& axes) {
              std::vector<ExprHandle> empty_indices;
              return A.load(empty_indices);
            });
      }
      auto permute_dims = c10::get<IntList>(inputs[1]);
      return Compute(
          "aten_permute",
          c10::fmap<DimArg>(outputShape),
          [&](const std::vector<VarHandle>& axes) {
            std::vector<VarHandle> new_axes;
            new_axes.resize(axes.size());
            assert(permute_dims.size() == axes.size());
            for (unsigned i = 0; i < axes.size(); i++) {
              auto new_dim = at::maybe_wrap_dim(permute_dims[i], A.ndim());
              new_axes[new_dim] = axes[i];
            }
            return A.load(new_axes);
          });
    });
RegisterNNCLoweringFunction aten_expand("aten::expand", computeExpand);
RegisterNNCLoweringFunction aten_expand_as("aten::expand_as", computeExpand);

RegisterNNCLoweringFunction aten_view("aten::view", computeReshape);
RegisterNNCLoweringFunction aten_reshape("aten::reshape", computeReshape);

// aten::mm is a subset of aten::matmul where both inputs are rank 2
RegisterNNCLoweringFunction aten_mm("aten::mm", computeMatmul);
RegisterNNCLoweringFunction aten_matmul("aten::matmul", computeMatmul);

RegisterNNCLoweringFunction aten_cat("aten::cat", computeCat);

RegisterNNCLoweringFunction aten_sum("aten::sum", computeSum);

RegisterNNCLoweringFunction aten_softmax(
    "aten::softmax",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeSoftmax(inputs, outputShape, false);
    });

RegisterNNCLoweringFunction aten_log_softmax(
    "aten::log_softmax",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      return computeSoftmax(inputs, outputShape, true);
    });

RegisterNNCLoweringFunction aten_conv2d("aten::conv2d", computeConv2d);

RegisterNNCLoweringFunction aten_addmm("aten::addmm", computeAddMM);

RegisterNNCLoweringFunction aten_mean("aten::mean", computeMean);

RegisterNNCLoweringFunction aten_adaptive_avg_pool2d(
    "aten::adaptive_avg_pool2d",
    computeAdaptiveAvgPool2d);

RegisterNNCLoweringFunction aten_add(
    "aten::add",
    [](const std::vector<ArgValue>& inputs,
       const std::vector<ExprHandle>& outputShape,
       const c10::optional<ScalarType>& outputType,
       at::Device device) {
      auto add_lambda = [](const ExprHandle& lhs, const ExprHandle& rhs) {
        return boolToInteger(lhs) + boolToInteger(rhs);
      };
      TORCH_INTERNAL_ASSERT(
          inputs.size() == 2 || inputs.size() == 3,
          buildErrorMessage("Invalid number of input operands"));
      return (inputs.size() > 2)
          ? computeTwoOperandWithAlpha(
                "aten_add", inputs, outputShape, outputType, add_lambda)
          : computeTwoOperand(
                "aten_add", inputs, outputShape, outputType, add_lambda);
    });

c10::optional<ScalarType> findDtypeForValue(const torch::jit::Value* v) {
  if (v->type()->kind() == TypeKind::TensorType) {
    auto tt = v->type()->cast<TensorType>();
    if (tt->scalarType()) {
      return static_cast<ScalarType>(*tt->scalarType());
    }
  }
  return c10::nullopt;
}

Tensor TensorExprKernel::computeValue(const torch::jit::Value* v) {
  auto inputs = v->node()->inputs();
  auto op = v->node()->kind();

  if (op == aten::rand_like) {
    hasRandom_ = true;
  }

  std::vector<ArgValue> argInputs;
  if (op == prim::ConstantChunk) {
    auto const& n = v->node();
    argInputs.push_back(toArg(inputs[0]));
    argInputs.push_back((int64_t)v->offset());
    argInputs.push_back(n->i(attr::dim));
    argInputs.push_back(n->i(attr::chunks));
  } else if (op == aten::to) {
    argInputs.push_back(toArg(inputs[0]));
  } else {
    for (auto inp : inputs) {
      argInputs.push_back(toArg(inp));
    }
  }
  auto outputType = findDtypeForValue(v);
  std::vector<ExprHandle> outputShape = sizesForValue(v);

  if (NNCLoweringFunction custom_lowering = getCustomLoweringFor(op)) {
    return custom_lowering(argInputs, outputShape, outputType, device_);
  }
  if (NNCLoweringFunction lowering =
          getStandardLoweringFor(op.toQualString())) {
    return lowering(argInputs, outputShape, outputType, device_);
  }
  std::string msg = std::string("Unhandled node kind (in computeValue): ") +
      op.toQualString();
  throw malformed_input(msg);
}

// Return the (lower, upper) loop bounds if they are constants, else nullopt.
c10::optional<std::pair<int64_t, int64_t>> loopBounds(ForPtr loop) {
  auto start = IRSimplifier::simplify(loop->start());
  auto stop = IRSimplifier::simplify(loop->stop());
  if (!start->isConstant() || !stop->isConstant()) {
    return c10::nullopt;
  }
  return c10::make_optional(
      std::make_pair(immediateAs<int64_t>(start), immediateAs<int64_t>(stop)));
}

// True if all the loops in this vector have equal bounds.
bool loopBoundsAllEqual(const std::vector<ForPtr>& loops) {
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
void fuseAllLoops(StmtPtr st) {
  if (auto block = to<tensorexpr::Block>(st)) {
    std::vector<ForPtr> loopsToFuse;
    for (auto stmt : *block) {
      auto loop = to<For>(stmt);
      if (!loop) {
        // Block contains something that's not a loop.  Quit.
        return;
      }
      loopsToFuse.push_back(loop);
    }
    if (loopsToFuse.empty()) {
      return;
    }
    if (!loopBoundsAllEqual(loopsToFuse)) {
      return;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ForPtr fusedLoop;
    if (!LoopNest::fuseLoops(loopsToFuse, &fusedLoop)) {
      return;
    }
    fuseAllLoops(fusedLoop->body());
  }
}

// Compute the trip count of a loop if it is a constant.
c10::optional<int64_t> tripCount(ForPtr loop) {
  auto tc = IRSimplifier::simplify(
      cast<int64_t>(ExprHandle(loop->stop()) - ExprHandle(loop->start())));
  if (auto val = to<LongImm>(tc.node())) {
    return val->value();
  }
  return c10::nullopt;
}

// Prune innermost loops until iterations satisfies a minimum grain size.
static void pruneByGrainSize(std::vector<ForPtr>& loops) {
  constexpr int64_t minGrainSize = 32768;
  int64_t grainSize = 1;
  for (int64_t i = loops.size(); i > 0; i--) {
    auto tc = tripCount(loops[i - 1]);
    if (!tc) {
      break;
    }
    grainSize *= *tc;
    if (grainSize < minGrainSize) {
      loops.pop_back();
    }
  }
}

// Retain enough outermost loops to fill the number of threads.
static void pruneByThreadCount(std::vector<ForPtr>& loops) {
  int64_t trips = 1;
  auto threads = at::get_num_threads();
  auto it = loops.begin();
  for (; it != loops.end(); it++) {
    if (trips >= threads) {
      break;
    }
    auto tc = tripCount(*it);
    if (!tc) {
      break;
    }
    trips *= *tc;
  }
  loops.erase(it, loops.end());
}

// Flatten and parallelize outer loops, subject to a minimum number of elements
// in the inner loop, and a maximum level of thread-level parallelism in the
// outer loops.
template <typename Bufs>
static void parallelizeOuterLoops(LoopNest& l, Bufs&& bufs) {
  for (auto const& buf : bufs) {
    auto loops = l.getLoopStmtsFor(buf);
    pruneByGrainSize(loops);
    pruneByThreadCount(loops);

    // There are no loops to parallelize; give up.
    if (loops.size() == 0) {
      continue;
    }
    // The loop nest contains a reduction; give up.
    auto reductions = NodeFinder<ReduceOp>::find(loops[0]);
    if (reductions.size() > 0) {
      continue;
    }
    // The loop nest has loop carried dependences; give up.
    if (LoopNest::hasLoopCarriedDependence(loops[0])) {
      continue;
    }
    // Try to flatten the outer loops and parallelize them if successful.
    ForPtr flattened = nullptr;
    if (loops.size() == 1) {
      flattened = loops[0];
    } else {
      LoopNest::flatten(loops, &flattened);
    }
    if (flattened) {
      flattened->set_parallel();
    }
  }
}

StmtPtr TensorExprKernel::transformLoops(BackendType backendType, StmtPtr st) {
  torch::jit::tensorexpr::LoopNest l(st, bufOutputs_);
  LoopNest::sanitizeNames(l.root_stmt());
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
  l.simplify();
  GRAPH_DEBUG("after simplify", *l.root_stmt());

  // Inlining output & intermediate buffers can duplicate computation.
  // Duplicating work can slow down the program if it's not ameliorated in some
  // way, but we've empirically found that:
  // - On CPU, LLVM's CSE does a good job as long as you horizontally fuse
  //   output loops.
  // - On GPU, there's enough compute to hide the extra work, and inlining
  //   avoids synchronizing between kernels.
  l.inlineIntermediateBufs(/*allow_duplicated_work=*/true);
  GRAPH_DEBUG("after inline", *l.root_stmt());

  // Optimizing conditionals needs to be performed after inlining because
  // inlining wouldn't work once the loops are split. Also, it has to be
  // performed before loop fusion because loop fusion introduces cases where
  // multiple conditionals are in the same loop and this optimization does not
  // handle such cases yet.
  if (getOptConditionals()) {
    l.optimizeConditionals();
    GRAPH_DEBUG("after optimizing conditionals: ", *l.root_stmt());
  }

  // Fuse loops "horizontally".  This pass allows us to combine loops that
  // write to different output buffers, as long as they have the same bounds.
  if (backendType == kLLVMCodeGen) {
    fuseAllLoops(l.root_stmt());
    GRAPH_DEBUG("after fuse", *l.root_stmt());
    parallelizeOuterLoops(l, bufOutputs_);
    GRAPH_DEBUG("after parallelize", *l.root_stmt());
  }

  if (backendType == kCudaCodeGen) {
    for (auto buf : bufOutputs_) {
      std::vector<ForPtr> loops = l.getLoopStmtsFor(buf);
      if (loops.empty()) {
        // This happens when Buf is 0-dim
        continue;
      }
      ForPtr flattened = nullptr;
      LoopNest::flatten(loops, &flattened);
      assert(flattened);

      int loopLevels = getTECudaPointwiseLoopLevels();
      const int kDefaultLoopLevels = 2;
      loopLevels = (loopLevels > 0) ? loopLevels : kDefaultLoopLevels;
      int blockCount = getTECudaPointwiseBlockCount();
      int blockSize = getTECudaPointwiseBlockSize();

      if (loopLevels == 2) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ForPtr inner;
        const int kDefaultBlockSize = 512;
        if (blockSize < 0) {
          blockSize = kDefaultBlockSize;
        }
        LoopNest::splitWithMask(flattened, blockSize, &inner);
        flattened->set_gpu_block_index(0);
        inner->set_gpu_thread_index(0);
      } else if (loopLevels == 3) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ForPtr inner;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        ForPtr inner1;
        // TODO: change the number of microprocessors
        const int kDefaultBlockCount = 1280;
        const int kDefaultBlockSize = 256;
        blockCount = (blockCount > 0) ? blockCount : kDefaultBlockCount;
        blockSize = (blockSize > 0) ? blockSize : kDefaultBlockSize;
        LoopNest::splitWithMask(flattened, blockCount * blockSize, &inner);
        LoopNest::splitWithMask(inner, blockSize, &inner1);
        inner->set_gpu_block_index(0);
        inner1->set_gpu_thread_index(0);
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
      std::vector<ForPtr> loops = l.getLoopStmtsFor(buf);
      TORCH_INTERNAL_ASSERT(
          !loops.empty(),
          buildErrorMessage(
              "No loops found for the buffer " + buf->name_hint() +
              " in the fuser."));
      ForPtr flattened = nullptr;
      LoopNest::flatten(loops, &flattened);
      assert(flattened);

      ForPtr inner = nullptr;
      LoopNest::splitWithMask(flattened, blockSize, &inner);
      flattened->set_gpu_block_index(0);
      inner->set_gpu_thread_index(0);
      flattened->set_buffer_map(block_analysis->getBufferMap());
    }
  }

  if (pre_alloc_) {
    auto interm_bufs = l.getIntermediateBufs();
    preAllocIntermediateBufs(interm_bufs);
    l.prepareForCodegen(interm_bufs);
  } else {
    l.prepareForCodegen();
  }

  GRAPH_DEBUG("after prepareForCodegen", *l.root_stmt());
  l.simplify();
  GRAPH_DEBUG("after simplification", *l.root_stmt());

  if (backendType == kLLVMCodeGen && !hasReduction) {
    l.vectorizeInnerLoops();
    GRAPH_DEBUG("after vectorization", *l.root_stmt());
  }

  StmtPtr stmt = l.root_stmt();
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
      sanitized_name.append("_");
    }
    value_to_name[input] = sanitized_name;
    name_set.insert(sanitized_name);
  }
  input_name_map_ = std::move(value_to_name);
}

template <typename T>
static std::vector<ExprHandle> toExprHandles(const std::vector<T>& sizes) {
  std::vector<ExprHandle> dims;
  dims.reserve(sizes.size());
  for (auto const& size : sizes) {
    dims.emplace_back(size);
  }
  return dims;
}

Tensor TensorExprKernel::bindInput(const torch::jit::Value* input) {
  auto const& t = input->type();
  Tensor result(nullptr, nullptr);
  switch (t->kind()) {
    case TypeKind::TensorType: {
      auto tt = input->type()->cast<TensorType>();
      if (!input->isCompleteTensor()) {
        std::string msg = std::string("Shapes for input '%") +
            input->debugName() + "' are unknown";
        throw malformed_input(msg);
      }
      if (isContiguous(input)) {
        BufHandle inBuffer(
            "t" + input_name_map_[input],
            toExprHandles(*tt->sizes().concrete_sizes()),
            ToDtype(static_cast<ScalarType>(*tt->scalarType())));
        bufs_.emplace(input, inBuffer.node());
        bufferArgs_.emplace_back(inBuffer);
        break;
      }
      BufHandle inBuffer(
          "t" + input_name_map_[input],
          {0},
          ToDtype(static_cast<ScalarType>(*tt->scalarType())));
      std::vector<DimArg> inputTensorDims;
      for (size_t i = 0; i < *tt->sizes().size(); i++) {
        auto const size = *tt->sizes()[i];
        inputTensorDims.emplace_back(DimArg(size, "i" + c10::to_string(i)));
      }
      auto const strides = tt->strides();
      result = Compute(
          "input" + c10::to_string(bufs_.size() + 1),
          inputTensorDims,
          [&](const std::vector<VarHandle>& axes) {
            ExprHandle idx = 0;
            for (size_t i = 0; i < axes.size(); i++) {
              idx = idx + axes[i] * *strides[i];
            }
            return inBuffer.load(idx);
          });
      bufs_.emplace(input, result.buf());
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
      throw unsupported_dtype(t->repr_str());
      break;
    }
  }
  return result;
}

NNCLoweringFunction TensorExprKernel::getCustomLoweringFor(
    c10::Symbol op) const {
  if (custom_lowerings_.count(op))
    return custom_lowerings_.at(op);
  return nullptr;
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

Tensor TensorExprKernel::convertOutputToCorrectStrides(torch::jit::Value* v) {
  const TensorTypePtr& tt = v->type()->expect<TensorType>();
  TORCH_INTERNAL_ASSERT(
      bufs_.count(v),
      buildErrorMessage(
          "Ouput tensor has no corresponding bufs in the fuser."));
  BufPtr buf = bufs_.at(v);

  // No shape info is present in the graph
  if (!tt->sizes().concrete_sizes()) {
    std::string msg =
        std::string("Shapes for output '%") + v->debugName() + "' are unknown";
    throw malformed_input(msg);
  }

  TORCH_INTERNAL_ASSERT(
      tt->sizes().concrete_sizes(),
      buildErrorMessage("Output shapes are unknown."));
  auto sizes = *tt->sizes().concrete_sizes();
  std::vector<int64_t> default_strides = TensorType::contiguousStridesOf(sizes);
  if (!tt->strides().concrete_sizes()) {
    return Tensor(buf, nullptr);
  }
  TORCH_INTERNAL_ASSERT(
      tt->strides().concrete_sizes(),
      buildErrorMessage("Output strides are unknown."));
  const std::vector<int64_t> strides = *tt->strides().concrete_sizes();
  // All Tensors in NNC are layed out in default, contiguous layout.
  // If the output is also default contiguous we don't need to do anything
  if (strides == default_strides) {
    return Tensor(buf, nullptr);
  }
  // If the tensor is not dense or overlaps, we have
  // no way of matching the profiled striding
  if (!denseAndNonOverlapping(sizes, strides)) {
    return Tensor(buf, nullptr);
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

  auto zero = LongImm::make(0);
  return Compute(
      "output_1", dims, [&](const std::vector<VarHandle>& axes_input) {
        std::vector<ExprHandle> axes(axes_input.begin(), axes_input.end());
        auto absolute_position = ExprHandle(immLike(axes[0], 0));
        for (size_t i = 0; i < axes.size(); ++i) {
          absolute_position = absolute_position +
              (ExprHandle(immLike(axes[i], default_strides[i])) * axes[i]);
        }
        std::vector<size_t> sorted_stride_indices =
            reverse_sort_indices(strides);
        std::vector<ExprHandle> new_axes(sorted_stride_indices.size());
        for (size_t stride_index : sorted_stride_indices) {
          auto size = sizes[stride_index];
          auto index = zero;
          if (size != 1) {
            auto stride = strides[stride_index];
            index = absolute_position /
                ExprHandle(immLike(absolute_position, stride));
            absolute_position = absolute_position %
                ExprHandle(immLike(absolute_position, stride));
          }
          new_axes[stride_index] = index;
        }
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
  auto sizes = *tt->sizes().concrete_sizes();
  std::vector<ExprHandle> te_sizes;
  te_sizes.reserve(sizes.size());
  for (auto s : sizes) {
    te_sizes.push_back(s);
  }

  BufPtr buf = alloc<Buf>(
      "const_" + sanitizeName(v->debugName()),
      ExprHandleVectorToExprVector(te_sizes),
      ToDtype(static_cast<ScalarType>(*tt->scalarType())));

  if (!const_tensor.is_contiguous()) {
    const_tensor = const_tensor.clone().contiguous();
    unpacked_constant_tensors_.push_back(const_tensor);
  }

  constants_.push_back({buf, const_tensor.data_ptr()});
  bufs_[v] = buf;
}

void TensorExprKernel::preAllocIntermediateBufs(
    std::unordered_set<BufPtr>& interm_bufs) {
  std::vector<std::pair<BufPtr, void*>> allocated_bufs;
  for (auto it = interm_bufs.begin(); it != interm_bufs.end();) {
    // Check if buf shape is static and compute its size if static.
    auto buf = *it;
    bool is_static = true;
    size_t size =
        elementSize(buf->dtype().scalar_type()) * buf->dtype().lanes();
    for (auto& d : buf->dims()) {
      if (!d->isConstant()) {
        is_static = false;
        break;
      }
      size = size * (*intValue(d));
    }
    // Only allocate memory for static bufs.
    if (!is_static) {
      ++it;
      continue;
    }
    auto bp = (void*)malloc(size);
    if (!bp) {
      ++it;
      continue;
    }
    allocated_bufs.emplace_back(buf, bp);
    it = interm_bufs.erase(it);
  }
  std::sort(
      allocated_bufs.begin(),
      allocated_bufs.end(),
      [](const auto& a, const auto& b) {
        return a.first->name_hint() > b.first->name_hint();
      });
  for (auto& a : allocated_bufs) {
    constants_.push_back({a.first, a.second});
  }
}

void TensorExprKernel::compile() {
  GRAPH_DUMP("TensorExprKernel graph:", graph_);

  device_ = *pickDeviceType(graph_);
  OptimizeCat(graph_);

  // Block to collect the Stmts corresponding to all tensors.
  auto block = alloc<Block>(std::vector<StmtPtr>({}));

  // Bind inputs to buffers.
  nInputs_ = graph_->inputs().size();
  genInputDebugNames();
  for (auto const& input : graph_->inputs()) {
    Tensor t = bindInput(input);
    if (t.stmt()) {
      block->append_stmt(t.stmt());
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
          Tensor t = computeValue(output);
          bufs_.emplace(output, t.buf());
          block->append_stmt(t.stmt());
        }
      }
    }
    if (hasRandom_ && hasBroadcast_) {
      throw std::runtime_error(
          "Cannot support broadcast and random within one kernel");
    }
  }

  // Move output operands from `bufs_` to `bufOutputs_`
  for (auto& output : graph_->outputs()) {
    if (!bufs_.count(output)) {
      throw malformed_input("cannot find output Tensor");
    }
    // The "strided" tensor will be incorrect if used in NNC,
    // since NNC views it as contiguous. Only convert it to the right
    // strides at the end of the kernel (if already contiguous it's a no-op)
    Tensor properly_strided_output = convertOutputToCorrectStrides(output);
    if (properly_strided_output.stmt()) {
      block->append_stmt(properly_strided_output.stmt());
    }
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    bufs_[output] = properly_strided_output.buf();
    const auto& tt = output->type()->expect<TensorType>();
    auto sizes = *tt->sizes().concrete_sizes();
    tensorOutputSizes_.push_back(sizes);
    auto strides = tt->strides().concrete_sizes();

    // If the tensor is not dense or overlaps, we have
    // no way of matching the profiled striding
    if (strides && denseAndNonOverlapping(sizes, *strides)) {
      tensorOutputStrides_.push_back(*strides);
    } else {
      tensorOutputStrides_.push_back(TensorType::contiguousStridesOf(sizes));
    }

    bufOutputs_.insert(bufs_.at(output));
    bufferArgs_.emplace_back(BufHandle(bufs_.at(output)));
    tensorOutputTensorOptions_.emplace_back(
        c10::TensorOptions(tensorType(bufs_.at(output))).device(device_));
    bufs_.erase(output);
  }

  BackendType backendType = inferBackendTypeFromDevice(device_);
  StmtPtr stmt = transformLoops(backendType, block);

  for (auto c : constants_) {
    bufferArgs_.emplace_back(BufHandle(c.buf));
  }

  // Generate code.
  codegen_ = CreateCodeGen(
      getCodeGenName(backendType),
      stmt,
      bufferArgs_,
      device_,
      SubgraphUtils::generateNameForGraph(graph_));
}

TensorExprKernel::TensorExprKernel(
    const std::shared_ptr<Graph>& subgraph,
    std::unordered_map<c10::Symbol, NNCLoweringFunction> custom_lowerings,
    bool pre_alloc /*= false*/)
    : graph_(subgraph),
      code_(subgraph, ""),
      custom_lowerings_(std::move(custom_lowerings)),
      pre_alloc_(pre_alloc) {
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

  for (auto& input : inputs) {
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

StmtPtr TensorExprKernel::getCodeGenStmt() {
  return codegen_->stmt();
}

void TensorExprKernel::runKernel(Stack& stack) {
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
