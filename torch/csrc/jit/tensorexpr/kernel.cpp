#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/constant_folder.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/schedule.h>

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {
namespace tensorexpr {

static int te_cuda_pointwise_loop_levels = -1;
static int te_cuda_pointwise_block_count = -1;
static int te_cuda_pointwise_block_size = -1;

int& GetTECudaPointwiseLoopLevels() {
  return te_cuda_pointwise_loop_levels;
}

int& GetTECudaPointwiseBlockCount() {
  return te_cuda_pointwise_block_count;
}

int& GetTECudaPointwiseBlockSize() {
  return te_cuda_pointwise_block_size;
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

static at::ScalarType tensorType(Tensor* t) {
  return static_cast<at::ScalarType>(t->body()->dtype().scalar_type());
}

static std::vector<ExprHandle> texprSizes(const c10::VaryingShape& shape) {
  std::vector<ExprHandle> dims;
  for (size_t i = 0; i < *shape.size(); i++) {
    dims.push_back(IntImm::make(*shape[i]));
  }
  return dims;
}

static std::vector<DimArg> texprDims(const torch::jit::Value* v) {
  CHECK(v->type()->kind() == TypeKind::TensorType);
  auto tt = v->type()->cast<TensorType>();
  std::vector<DimArg> dimArgs;
  int i = 0;
  for (auto const& s : texprSizes(tt->sizes())) {
    dimArgs.emplace_back(DimArg(s, "i" + std::to_string(i++)));
  }
  return dimArgs;
}

template <typename T>
int64_t bufferSize(T t) {
  int64_t size = 1;
  for (int i = 0; i < t.ndim(); i++) {
    size *= t.dim(i).template AsNode<IntImm>()->value();
  }
  return size;
}

ExprHandle TensorExprKernel::constant(const torch::jit::Value* v) {
  if (v->node()->kind() == prim::Constant) {
    const auto val = toIValue(v).value();
    if (val.isDouble()) {
      return FloatImm::make(static_cast<float>(val.toDouble()));
    } else if (val.isInt()) {
      return IntImm::make(val.toInt());
    } else if (val.isNone()) {
      // This is just a placeholder so we don't throw.  None-handling
      // is operator-specific and should be handled properly in
      // the operator-specific lowering code.
      return IntImm::make(0);
    } else {
      LOG(FATAL) << "Unhandled constant datatype";
    }
  }
  CHECK(scalars_.count(v->unique())) << "Couldn't find scalar value";
  return scalars_.at(v->unique());
}

void TensorExprKernel::promoteInputs(std::vector<ExprHandle>& inputs) {
  if (inputs.empty()) {
    return;
  }

  // Find the highest type among the inputs.
  ScalarType highType = inputs[0].dtype().scalar_type();
  for (const auto input : inputs) {
    ScalarType iType = input.dtype().scalar_type();
    if (iType == ScalarType::Bool) {
      continue;
    }
    highType = promoteTypes(highType, iType);
  }

  for (ExprHandle& e : inputs) {
    if (e.dtype().scalar_type() == ScalarType::Bool) {
      continue;
    }

    if (e.dtype().scalar_type() == highType) {
      continue;
    }

    switch (highType) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    e = cast<Type>(e);        \
    break;
      AT_FORALL_SCALAR_TYPES_AND(Half, TYPE_CASE);
#undef TYPE_CASE
      default:
        LOG(FATAL) << "Unsupported datatype: " << highType;
    }
  }
}

ExprHandle TensorExprKernel::demoteOutput(
    const ExprHandle& e,
    const torch::jit::Value* v) {
  CHECK(v->type()->kind() == TypeKind::TensorType);
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
      return e;
    default:
      LOG(FATAL) << "Unsupported datatype";
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

static std::pair<std::vector<ExprHandle>, bool> broadcastShapes(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  bool broadcast = false;
  auto at = a.rbegin();
  auto bt = b.rbegin();
  std::vector<ExprHandle> ret;
  while (at != a.rend() || bt != b.rend()) {
    if (at == a.rend()) {
      broadcast = true;
      ret.push_back(*bt++);
      continue;
    }
    if (bt == b.rend()) {
      broadcast = true;
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
        broadcast = true;
      }
    }
    ret.push_back(dim);
    at++;
    bt++;
  }
  std::reverse(ret.begin(), ret.end());
  return {ret, broadcast};
}

template <typename... Args>
static std::pair<std::vector<ExprHandle>, bool> broadcastShapes(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b,
    Args... args) {
  auto const& res = broadcastShapes(a, b);
  auto const& res2 = broadcastShapes(res.first, args...);
  return {res2.first, res.second || res2.second};
}

std::vector<ExprHandle> TensorExprKernel::valueShape(
    const torch::jit::Value* v) {
  auto it = tensors_.find(v->unique());
  if (it == tensors_.end()) {
    return {};
  }
  return ExprVectorToExprHandleVector(it->second->dims());
}

Tensor* TensorExprKernel::ComputeOneOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(const ExprHandle&)>& inner_expr) {
  auto const& n = v->node();
  auto const& shape = valueShape(n->inputs()[0]);
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, inner_expr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], axes)};

        promoteInputs(inputs);
        ExprHandle compute = inner_expr(inputs[0]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::ComputeTwoOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        inner_expr) {
  auto const& n = v->node();
  auto const& res =
      broadcastShapes(valueShape(n->inputs()[0]), valueShape(n->inputs()[1]));
  auto const& shape = res.first;
  hasBroadcast_ |= res.second;
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, inner_expr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], axes),
            tensorOrConstant(n->inputs()[1], axes),
        };

        promoteInputs(inputs);
        ExprHandle compute = inner_expr(inputs[0], inputs[1]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::ComputeTwoOperandWithAlpha(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
        inner_expr) {
  auto const& n = v->node();
  auto const& res =
      broadcastShapes(valueShape(n->inputs()[0]), valueShape(n->inputs()[1]));
  auto const& shape = res.first;
  hasBroadcast_ |= res.second;
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, inner_expr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], axes),
            tensorOrConstant(n->inputs()[1], axes),
            tensorOrConstant(n->inputs()[2], axes),
        };

        promoteInputs(inputs);
        ExprHandle compute = inner_expr(inputs[0], inputs[2] * inputs[1]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::ComputeConditionWithTwoOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        inner_expr) {
  auto const& n = v->node();
  auto const& res = broadcastShapes(
      valueShape(n->inputs()[0]),
      valueShape(n->inputs()[1]),
      valueShape(n->inputs()[2]));
  auto const& shape = res.first;
  hasBroadcast_ |= res.second;
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, inner_expr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[1], axes),
            tensorOrConstant(n->inputs()[2], axes),
        };

        promoteInputs(inputs);
        // First expr is the condition, which we don't promote
        inputs.emplace(inputs.begin(), tensorOrConstant(n->inputs()[0], axes));
        ExprHandle compute = inner_expr(inputs[0], inputs[1], inputs[2]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::ComputeThreeOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<
        ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
        inner_expr) {
  auto const& n = v->node();
  auto const& res = broadcastShapes(
      valueShape(n->inputs()[0]),
      valueShape(n->inputs()[1]),
      valueShape(n->inputs()[2]));
  auto const& shape = res.first;
  hasBroadcast_ |= res.second;
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, inner_expr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], axes),
            tensorOrConstant(n->inputs()[1], axes),
            tensorOrConstant(n->inputs()[2], axes),
        };

        promoteInputs(inputs);
        ExprHandle compute = inner_expr(inputs[0], inputs[1], inputs[2]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::ComputeFourOperand(
    const std::string& name,
    const torch::jit::Value* v,
    const std::function<ExprHandle(
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&,
        const ExprHandle&)>& inner_expr) {
  auto const& n = v->node();
  auto const& res = broadcastShapes(
      valueShape(n->inputs()[0]),
      valueShape(n->inputs()[1]),
      valueShape(n->inputs()[2]),
      valueShape(n->inputs()[3]));
  auto const& shape = res.first;
  hasBroadcast_ |= res.second;
  return Compute(
      name,
      c10::fmap<DimArg>(shape),
      [this, v, inner_expr](const std::vector<VarHandle>& axes) {
        auto const& n = v->node();
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(n->inputs()[0], axes),
            tensorOrConstant(n->inputs()[1], axes),
            tensorOrConstant(n->inputs()[2], axes),
            tensorOrConstant(n->inputs()[3], axes),
        };

        promoteInputs(inputs);
        ExprHandle compute =
            inner_expr(inputs[0], inputs[1], inputs[2], inputs[3]);
        return demoteOutput(compute, n->output());
      });
}

Tensor* TensorExprKernel::ComputeValue(const torch::jit::Value* v) {
  switch (v->node()->kind()) {
    case aten::add: {
      return ComputeTwoOperandWithAlpha(
          "aten_add", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs + rhs;
          });
    } break;

    case aten::_cast_Float: {
      return ComputeOneOperand("aten_cast_float", v, [](const ExprHandle& a) {
        return cast<float>(a);
      });
    } break;

    case aten::sub: {
      return ComputeTwoOperandWithAlpha(
          "aten_sub", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs - rhs;
          });
    } break;

    case aten::mul: {
      return ComputeTwoOperand(
          "aten_mul", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs * rhs;
          });
    } break;

    case aten::div: {
      return ComputeTwoOperand(
          "aten_div", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs / rhs;
          });
    } break;

    case aten::__and__: {
      return ComputeTwoOperand(
          "aten_and", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs & rhs;
          });
    } break;

    case aten::__or__: {
      return ComputeTwoOperand(
          "aten_or", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs | rhs;
          });
    } break;

    case aten::__xor__: {
      return ComputeTwoOperand(
          "aten_xor", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs ^ rhs;
          });
    } break;

    case aten::__lshift__: {
      return ComputeTwoOperand(
          "aten_lshift", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs << rhs;
          });
    } break;

    case aten::__rshift__: {
      return ComputeTwoOperand(
          "aten_rshift", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs >> rhs;
          });
    } break;

    case aten::addcmul: {
      return ComputeFourOperand(
          "aten_addcmul",
          v,
          [](const ExprHandle& a0,
             const ExprHandle& a1,
             const ExprHandle& a2,
             const ExprHandle& a3) { return a0 + a3 * a1 * a2; });
    } break;

    case aten::eq: {
      return ComputeTwoOperand(
          "aten_eq", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs == rhs;
          });
    } break;

    case aten::ne: {
      return ComputeTwoOperand(
          "aten_ne", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs != rhs;
          });
    } break;
    case aten::ge: {
      return ComputeTwoOperand(
          "aten_ge", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs >= rhs;
          });
    } break;

    case aten::gt: {
      return ComputeTwoOperand(
          "aten_gt", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs > rhs;
          });
    } break;

    case aten::le: {
      return ComputeTwoOperand(
          "aten_le", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs <= rhs;
          });
    } break;

    case aten::lt: {
      return ComputeTwoOperand(
          "aten_lt", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs < rhs;
          });
    } break;

    case aten::min: {
      return ComputeTwoOperand(
          "aten_min", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Min::make(lhs, rhs, false);
          });
    } break;

    case aten::max: {
      return ComputeTwoOperand(
          "aten_max", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Max::make(lhs, rhs, false);
          });
    } break;

    case aten::clamp: {
      bool no_min = false;
      bool no_max = false;
      if (v->node()->input(1)->node()->kind() == prim::Constant) {
        const auto val = toIValue(v->node()->input(1)).value();
        if (val.isNone()) {
          no_min = true;
        }
      }

      if (v->node()->input(2)->node()->kind() == prim::Constant) {
        const auto val = toIValue(v->node()->input(2)).value();
        if (val.isNone()) {
          no_max = true;
        }
      }

      return ComputeThreeOperand(
          "aten_clamp",
          v,
          [no_min, no_max](
              const ExprHandle& in,
              const ExprHandle& min,
              const ExprHandle& max) {
            if (no_min && no_max) {
              return in;
            } else if (no_min) {
              return CompareSelect::make(in, max, max, in, kGT);
            } else if (no_max) {
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
      return ComputeOneOperand("aten_sigmoid", v, [](const ExprHandle& a) {
        return ExprHandle(1.0f) /
            (ExprHandle(1.0f) + exp(ExprHandle(-0.0f) - a));
      });
    } break;

    case aten::reciprocal: {
      return ComputeOneOperand("aten_reciprocal", v, [](const ExprHandle& a) {
        return ExprHandle(1.0f) / a;
      });
    } break;

    case aten::neg: {
      return ComputeOneOperand("aten_neg", v, [](const ExprHandle& a) {
        return ExprHandle(-0) - a;
      });
    } break;

    case aten::relu: {
      return ComputeOneOperand("aten_relu", v, [](const ExprHandle& a) {
        return Max::make(a, 0, false);
      });
    } break;

    case aten::log: {
      return ComputeOneOperand(
          "aten_log", v, [](const ExprHandle& a) { return log(a); });
    } break;

    case aten::log10: {
      return ComputeOneOperand(
          "aten_log10", v, [](const ExprHandle& a) { return log10(a); });
    } break;

    case aten::log2: {
      return ComputeOneOperand(
          "aten_log2", v, [](const ExprHandle& a) { return log2(a); });
    } break;

    case aten::exp: {
      return ComputeOneOperand(
          "aten_exp", v, [](const ExprHandle& a) { return exp(a); });
    } break;

    case aten::expm1: {
      return ComputeOneOperand(
          "aten_expm1", v, [](const ExprHandle& a) { return expm1(a); });
    } break;

    case aten::erf: {
      return ComputeOneOperand(
          "aten_erf", v, [](const ExprHandle& a) { return erf(a); });
    } break;

    case aten::erfc: {
      return ComputeOneOperand(
          "aten_erfc", v, [](const ExprHandle& a) { return erfc(a); });
    } break;

    case aten::cos: {
      return ComputeOneOperand(
          "aten_cos", v, [](const ExprHandle& a) { return cos(a); });
    } break;

    case aten::sin: {
      return ComputeOneOperand(
          "aten_sin", v, [](const ExprHandle& a) { return sin(a); });
    } break;

    case aten::tan: {
      return ComputeOneOperand(
          "aten_tan", v, [](const ExprHandle& a) { return tan(a); });
    } break;

    case aten::type_as: {
      return ComputeTwoOperand(
          "aten_type_as", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return Cast::make(rhs.dtype(), lhs);
          });
    } break;

    case aten::rand_like: {
      hasRandom_ = true;
      return ComputeOneOperand("aten_rand_like", v, [](const ExprHandle& a) {
        return Intrinsics::make(IntrinsicsOp::kRand, a.dtype());
      });
    } break;

    case aten::pow: {
      return ComputeTwoOperand(
          "aten_pow", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            const FloatImm* float_imm = rhs.AsNode<FloatImm>();
            if (float_imm) {
              float imm = float_imm->value();
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

            const Cast* float_cast = rhs.AsNode<Cast>();
            if (float_cast) {
              const IntImm* int_imm =
                  dynamic_cast<const IntImm*>(float_cast->src_value());
              if (int_imm) {
                float imm = static_cast<float>(int_imm->value());
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
      return ComputeTwoOperand(
          "aten_fmod", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return fmod(lhs, rhs);
          });
    } break;

    case aten::lerp: {
      return ComputeThreeOperand(
          "aten_lerp",
          v,
          [](const ExprHandle& a,
             const ExprHandle& end,
             const ExprHandle& weight) { return a + weight * (end - a); });
    } break;
    case aten::remainder: {
      return ComputeTwoOperand(
          "aten_remainder",
          v,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return fmod((rhs + fmod(lhs, rhs)), rhs);
          });

    } break;

    case aten::acos: {
      return ComputeOneOperand(
          "aten_acos", v, [](const ExprHandle& a) { return acos(a); });
    } break;

    case aten::asin: {
      return ComputeOneOperand(
          "aten_asin", v, [](const ExprHandle& a) { return asin(a); });
    } break;

    case aten::cosh: {
      return ComputeOneOperand(
          "aten_cosh", v, [](const ExprHandle& a) { return cosh(a); });
    } break;

    case aten::sinh: {
      return ComputeOneOperand(
          "aten_sinh", v, [](const ExprHandle& a) { return sinh(a); });
    } break;

    case aten::atan: {
      return ComputeOneOperand(
          "aten_atan", v, [](const ExprHandle& a) { return atan(a); });
    } break;

    case aten::atan2: {
      return ComputeTwoOperand(
          "aten_atan2", v, [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return atan2(lhs, rhs);
          });
    } break;

    case aten::tanh: {
      return ComputeOneOperand("aten_tanh", v, [](const ExprHandle& a) {
        // return
        // (ExprHandle(-.67436811832e-5f)+(ExprHandle(.2468149110712040f)+(ExprHandle(.583691066395175e-1f)+ExprHandle(.3357335044280075e-1f)*a)*a)*a)/(ExprHandle(.2464845986383725f)+(ExprHandle(.609347197060491e-1f)+(ExprHandle(.1086202599228572f)+ExprHandle(.2874707922475963e-1f)*a)*a)*a);
        return tanh(a);
      });
    } break;

    case aten::sqrt: {
      return ComputeOneOperand(
          "aten_sqrt", v, [](const ExprHandle& a) { return sqrt(a); });
    } break;

    case aten::rsqrt: {
      return ComputeOneOperand(
          "aten_rsqrt", v, [](const ExprHandle& a) { return rsqrt(a); });
    } break;

    case aten::abs: {
      return ComputeOneOperand(
          "aten_abs", v, [](const ExprHandle& a) { return fabs(a); });
    } break;

    case aten::ceil: {
      return ComputeOneOperand(
          "aten_ceil", v, [](const ExprHandle& a) { return ceil(a); });
    } break;

    case aten::floor: {
      return ComputeOneOperand(
          "aten_floor", v, [](const ExprHandle& a) { return floor(a); });
    } break;

    case aten::round: {
      return ComputeOneOperand(
          "aten_round", v, [](const ExprHandle& a) { return round(a); });
    } break;

    case aten::trunc: {
      return ComputeOneOperand(
          "aten_trunc", v, [](const ExprHandle& a) { return trunc(a); });
    } break;

    case aten::threshold: {
      return ComputeThreeOperand(
          "aten_threshold",
          v,
          [](const ExprHandle& a,
             const ExprHandle& threshold,
             const ExprHandle& value) {
            return ifThenElse(CompareSelect::make(a, threshold, kGT), a, value);
          });
    } break;

    case aten::where: {
      return ComputeConditionWithTwoOperand(
          "aten_where",
          v,
          [](const ExprHandle& a0, const ExprHandle& a1, const ExprHandle& a2) {
            return ifThenElse(a0, a1, a2);
          });
    } break;

    case aten::frac: {
      return ComputeOneOperand(
          "aten_frac", v, [](const ExprHandle& a) { return a - floor(a); });
    } break;

    case aten::lgamma: {
      return ComputeOneOperand(
          "aten_lgamma", v, [](const ExprHandle& a) { return lgamma(a); });
    } break;

    case prim::ConstantChunk: {
      return Compute(
          "prim_constantchunk",
          texprDims(v),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            int64_t dim = n->i(attr::dim);
            int64_t chunks = n->i(attr::chunks);
            return chunk(
                tensors_.at(n->inputs()[0]->unique()),
                v->offset(),
                dim,
                chunks,
                axes);
          });
    }

    case aten::cat: {
      return Compute(
          "aten_cat",
          texprDims(v),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            auto inputs = n->inputs()[0]->node()->inputs();
            size_t dim = n->inputs()[1]->node()->i(attr::value);

            std::vector<ExprHandle> new_axes(axes.begin(), axes.end());
            ExprHandle load = tensorOrConstant(inputs[0], new_axes);
            size_t offset = bufferSizes(tensors_.at(inputs[0]->unique()))[dim];
            new_axes[dim] = new_axes[dim] - IntImm::make(offset);

            for (size_t ii = 1; ii < inputs.size(); ++ii) {
              load = ifThenElse(
                  CompareSelect::make(axes[dim], IntImm::make(offset), kLT),
                  load,
                  tensorOrConstant(inputs[ii], new_axes));
              offset += bufferSizes(tensors_.at(inputs[ii]->unique()))[dim];
              new_axes[dim] = axes[dim] - IntImm::make(offset);
            }

            return load;
          });
    }

    case aten::slice: {
      return Compute(
          "aten_slice",
          texprDims(v),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            int dim = constant(n->inputs()[1]).AsNode<IntImm>()->value();
            ExprHandle start = constant(n->inputs()[2]);
            ExprHandle stride = constant(n->inputs()[4]);

            std::vector<ExprHandle> new_axes(axes.begin(), axes.end());
            new_axes[dim] = stride * new_axes[dim] + start;
            return tensorOrConstant(n->inputs()[0], new_axes);
          });
    }

    case aten::unsqueeze: {
      return Compute(
          "aten_unsqueeze",
          texprDims(v),
          [this, v](const std::vector<VarHandle>& axes) {
            auto const& n = v->node();
            int64_t dim = constant(n->inputs()[1]).AsNode<IntImm>()->value();
            if (dim < 0) {
              CHECK(axes.size() > 0);
              dim += axes.size() - 1;
            }

            std::vector<ExprHandle> new_axes(axes.begin(), axes.end());
            new_axes.erase(new_axes.begin() + dim);
            return tensorOrConstant(n->inputs()[0], new_axes);
          });
    }

    case aten::_sigmoid_backward: {
      return ComputeTwoOperand(
          "aten_sigmoid_backward",
          v,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs * rhs * (ExprHandle(1.0f) - rhs);
          });
    }

    case aten::_tanh_backward: {
      return ComputeTwoOperand(
          "aten_tanh_backward",
          v,
          [](const ExprHandle& lhs, const ExprHandle& rhs) {
            return lhs * (ExprHandle(1.0f) - rhs * rhs);
          });
    }

    default: {
      throw std::runtime_error("Unhandled node kind");
    }
  }
}

void TensorExprKernel::LowerToBackend(BackendType backend_type) {
  std::vector<Tensor*> tensor_outputs(tensor_outputs_);

  if (backend_type == BackendType::kCudaCodeGen) {
    for (size_t tensor_idx = 0; tensor_idx < tensor_outputs_.size();
         tensor_idx++) {
      Tensor* tensor = tensor_outputs_[tensor_idx];
      ExprHandle total_count = ExprHandle(tensor->dim(0));
      for (int i = 1; i < tensor->ndim(); i++) {
        const IntImm* total_count_i = total_count.AsNode<IntImm>();
        const IntImm* tensor_dim_i =
            dynamic_cast<const IntImm*>(tensor->dim(i));
        if (total_count_i && tensor_dim_i) {
          // TODO: switch to real constant folding when it is available.
          total_count =
              ExprHandle(total_count_i->value() * tensor_dim_i->value());
        } else {
          total_count = total_count * ExprHandle(tensor->dim(i));
        }
      }
      // Flatten the index for GPU kernels.
      // TODO: move this to fusing axis when it is ready.
      Tensor* new_out = Compute(
          tensor->func_var()->name_hint() + "_flat",
          {total_count},
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
      tensor_outputs[tensor_idx] = new_out;
    }
  }

  torch::jit::tensorexpr::schedule::LoopNest l(tensor_outputs);

  // Compute non-output tensors_ inline
  for (auto& p : tensors_) {
    if (!l.hasLoopBodyFor(p.second)) {
      continue;
    }
    Stmt* loop = l.getLoopBodyFor(p.second);
    if (torch::jit::tensorexpr::HasRand(loop).has_rand()) {
      l.ComputeInlineWithRandom(loop);
    } else {
      l.ComputeInline(loop);
    }
  }
  if (backend_type == kCudaCodeGen) {
    for (size_t i = 0; i < tensor_outputs_.size(); i++) {
      l.ComputeInline(l.getLoopBodyFor(tensor_outputs_[i]));

      Tensor* tensor = tensor_outputs[i];
      const Var* index = tensor->arg(0);
      int loop_levels = GetTECudaPointwiseLoopLevels();
      const int kDefaultLoopLevels = 2;
      loop_levels = (loop_levels > 0) ? loop_levels : kDefaultLoopLevels;
      int block_count = GetTECudaPointwiseBlockCount();
      int block_size = GetTECudaPointwiseBlockSize();

      if (loop_levels == 2) {
        For* outer;
        For* inner;
        const int kDefaultBlockSize = 512;
        if (block_size < 0) {
          block_size = kDefaultBlockSize;
        }
        std::vector<For*> loops = l.getLoopStmtsFor(tensor);
        l.SplitWithMask(loops[0], block_size, &outer, &inner);
        l.SetGPUBlockIndex(outer, 0);
        l.SetGPUThreadIndex(inner, 0);
      } else if (loop_levels == 3) {
        For* outer;
        For* inner;
        For* inner_1;
        For* inner_2;
        // TODO: change the number of microprocessors
        const int kDefaultBlockCount = 1280;
        const int kDefaultBlockSize = 256;
        block_count = (block_count > 0) ? block_count : kDefaultBlockCount;
        block_size = (block_size > 0) ? block_size : kDefaultBlockSize;
        std::vector<For*> loops = l.getLoopStmtsFor(tensor);
        l.SplitWithMask(loops[0], block_count * block_size, &outer, &inner);
        l.SplitWithMask(inner, block_size, &inner_1, &inner_2);
        l.SetGPUBlockIndex(inner_1, 0);
        l.SetGPUThreadIndex(inner_2, 0);
      } else {
        throw std::runtime_error(
            "Invalid loop-level: " + std::to_string(loop_levels));
      }
    }
  } else if (backend_type == kLLVMCodeGen) {
    l.ApplyInlines();

    std::vector<For*> inner_loops;
    std::vector<For*> worklist;

    // Find outer-most For loops
    if (For* root_f = dynamic_cast<For*>(l.root_stmt())) {
      worklist.push_back(root_f);
    } else if (Block* body = dynamic_cast<Block*>(l.root_stmt())) {
      std::vector<Block*> blocks = {body};
      while (blocks.size()) {
        Block *b = blocks.back();
        blocks.pop_back();

        for (Stmt* s : b->stmts()) {
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

      bool contains_subloops = false;
      if (Block* body = dynamic_cast<Block*>(f->body())) {
        for (Stmt* s2 : body->stmts()) {
          if (For* f2 = dynamic_cast<For*>(s2)) {
            contains_subloops = true;
            worklist.push_back(f2);
          }
        }
      }

      if (!contains_subloops) {
        inner_loops.push_back(f);
      }
    }

    // Vectorize inner loops.
    for (For* loop : inner_loops) {
      For* outer1;
      For* split1;
      For* tail1;

      l.SplitWithTail(loop, 8, &outer1, &split1, &tail1);
      l.Vectorize(split1);

      if (tail1) {
        For* outer2;
        For* split2;
        For* tail2;
        l.SplitWithTail(tail1, 4, &outer2, &split2, &tail2);
        l.Vectorize(split2);
      }
    }
  }

  l.ApplyInlines();
  Stmt* stmt = l.root_stmt();

  ConstantFolder constant_folder;
  stmt = stmt->accept_mutator(&constant_folder);

  // Set up formal params (inputs, then outputs) for kernel.
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
  for (auto& o : tensor_outputs) {
    params.emplace_back(o);
  }

  // Generate code.
  std::string codegen_name;
  switch (backend_type_) {
    case kCudaCodeGen:
      codegen_name = "cuda_codegen";
      break;
    case kLLVMCodeGen:
      codegen_name = "llvm_codegen";
      break;
    case kSimpleIREval:
      codegen_name = "simple_ir_eval";
      break;
    default:
      throw std::runtime_error(
          "invalid backend type: " +
          std::to_string(static_cast<int>(backend_type_)));
  }
  codegen_ = CreateCodeGen(codegen_name, stmt, params);
}

template <typename T>
static bool isValidPrimProperty(const c10::optional<T>& a, T b) {
  return !a.has_value() || *a == b;
}

static bool isValidVaryingShape(
    const c10::VaryingShape& vs,
    at::IntArrayRef sz) {
  if (!vs.size().has_value()) {
    // TODO: does it make sense to have kernels with completely unspecified
    // shapes/strides
    return true;
  }

  if (*vs.size() != sz.size()) {
    return false;
  }

  for (size_t i = 0; i < sz.size(); i++) {
    if (!isValidPrimProperty(vs[i], sz[i])) {
      return false;
    }
  }
  return true;
}

static void checkInputs(const at::ArrayRef<IValue>& inputs, std::vector<TypePtr>& input_types) {
  TORCH_INTERNAL_ASSERT(
      inputs.size() == input_types.size(),
      "number of actual inputs don't match with the number of inputs to a subgraph");
  for (size_t i = 0; i < inputs.size(); i++) {
    // enable this to debug the asserts below
    GRAPH_DEBUG(
        "Comparing input ",
        i,
        " ivalue ",
        inputs[i],
        " against type ",
        *input_types[i]);
    if (inputs[i].isTensor()) {
      auto t = inputs[i].toTensor();
      TORCH_INTERNAL_ASSERT(
          t.defined(), "input ", i, " can't be an undefined tensor!");
      auto tt = input_types[i]->cast<TensorType>();
      TORCH_INTERNAL_ASSERT(tt, "input ", i, " expected to be a tensor!");
      TORCH_INTERNAL_ASSERT(
          isValidPrimProperty(tt->scalarType(), t.scalar_type()),
          "input ",
          i,
          " scalar types don't match");
      // TODO: do we need an extra check to make sure the device is specified
      TORCH_INTERNAL_ASSERT(
          isValidPrimProperty(tt->device(), t.device()),
          "input ",
          i,
          " device types don't match");
      TORCH_INTERNAL_ASSERT(
          isValidVaryingShape(tt->sizes(), t.sizes()),
          "input ",
          i,
          " sizes don't match");
      TORCH_INTERNAL_ASSERT(
          isValidVaryingShape(tt->strides(), t.strides()),
          "input ",
          i,
          " strides don't match");
    } else if (inputs[i].isInt()) {
      TORCH_INTERNAL_ASSERT(
          input_types[i]->cast<IntType>(), "type of ", i, " isn't an int!");
    } else if (inputs[i].isDouble()) {
      TORCH_INTERNAL_ASSERT(
          input_types[i]->cast<FloatType>(), "type of ", i, " isn't an int!");
    } else {
      // TODO: cover more IValue types
      // TODO: make it a hard error
    }
  }
}

void TensorExprKernel::PickAndCheckBackendType(
    const at::ArrayRef<IValue>& inputs) {
  checkInputs(inputs, input_types_);

  at::Device device = [&inputs]() {
    for (auto const& input : inputs) {
      if (input.isTensor()) {
        return input.toTensor().device();
      }
    }
    throw std::runtime_error("No tensor inputs");
  }();
  BackendType backend_type = BackendType::kUninitialized;
  if (device.type() == at::kCUDA) {
    backend_type = kCudaCodeGen;
  } else if (device.type() == at::kCPU) {
#ifdef ENABLE_LLVM
    backend_type = kLLVMCodeGen;
#else
    backend_type = kSimpleIREval;
    ;
#endif
  } else {
    throw std::runtime_error("Invalid device type");
  }

  if (backend_type_ == kUninitialized) {
    backend_type_ = backend_type;
    device_ = device;
    LowerToBackend(backend_type);
  } else if (backend_type_ != backend_type) {
    // TODO: if we have to support muliptole backends with the same subgraph,
    // we need to add kernel caching.
    throw std::runtime_error(
        "Inconsistent backend_type: " + std::to_string(backend_type_) + " vs " +
        std::to_string(backend_type));
  }
}

void TensorExprKernel::CodeGenRun(
    const std::vector<CodeGen::CallArg>& run_args) {
  switch (backend_type_) {
    case kSimpleIREval:
    case kLLVMCodeGen:
    case kCudaCodeGen:
      codegen_->call(run_args);
      break;
    default:
      throw std::runtime_error(
          "Invalid backend type: " + std::to_string(backend_type_));
  }
}

ExprHandle TensorExprKernel::createInputIndexExpr(
    const Buffer& buffer,
    const std::vector<VarHandle>& axes,
    const c10::VaryingShape& sizes,
    const c10::VaryingStrides& strides,
    const c10::VaryingStrides& contiguity,
    const std::unordered_map<int64_t, VarHandle>& sizeVars) {
  TORCH_CHECK(
      axes.size() == strides.size(), "strides and axes are not the same size");

  std::vector<ShapeArg> strideArgs;
  std::vector<ShapeArg> sizeArgs;
  ExprHandle stride = 1;
  ExprHandle index = 0;
  CHECK(axes.size() > 0);
  size_t n = axes.size() - 1;

  for (size_t i = 0; i < axes.size(); i++) {
    // For discontiguous tensors, create a parameter to represent stride.
    if (!*contiguity[i]) {
      VarHandle v = VarHandle{
          "stride_" + buffer.data()->name_hint() + "_" + std::to_string(i),
          kInt};
      strideArgs.emplace_back(n - i, v);
      stride = v;
    }

    // If size is dynamic (indicated by negative value) create a size param.
    ExprHandle size;
    auto sizeVal = *sizes[n - i];
    if (sizeVal < 0) {
      auto it = sizeVars.find(sizeVal);
      TORCH_CHECK(it != sizeVars.end());
      auto const& v = it->second;
      sizeArgs.emplace_back(n - i, v);
      size = v;
    } else {
      size = int32_t{sizeVal};
    }

    index = index + axes[n - i] * stride;
    stride = stride * size;
  }

  kernelArgs_.emplace_back(buffer, std::move(sizeArgs), std::move(strideArgs));
  return buffer(index);
}

void TensorExprKernel::bindInput(const torch::jit::Value* input) {
  auto const& t = input->type();
  switch (t->kind()) {
    case TypeKind::TensorType: {
      auto tt = input->type()->cast<TensorType>();
      Buffer in_buffer(
          "t" + input->debugName(),
          ToDtype(static_cast<ScalarType>(*tt->scalarType())),
          {0});
      std::vector<DimArg> inputTensorDims;
      std::unordered_map<int64_t, VarHandle> sizeVars;
      for (size_t i = 0; i < *tt->sizes().size(); i++) {
        auto const& size = *tt->sizes()[i];
        if (size < 0) {
          VarHandle v(
              "size_" + std::to_string(input->unique()) + "_" +
                  std::to_string(i),
              kInt);
          sizeVars.emplace(size, v);
          inputTensorDims.emplace_back(v);
        } else {
          inputTensorDims.emplace_back(
              DimArg(IntImm::make(size), "i" + std::to_string(i)));
        }
      }
#ifdef DYNAMIC_SHAPES
      tensors_.emplace(
          input->unique(),
          Compute(
              "input",
              inputTensorDims,
              [&](const std::vector<VarHandle>& axes) {
                return createInputIndexExpr(
                    in_buffer,
                    axes,
                    tt->sizes(),
                    tt->strides(),
                    tt->contiguity(),
                    sizeVars);
              }));
#else
      auto const& strides = tt->strides();
      tensors_.emplace(
          input->unique(),
          Compute(
              "input",
              inputTensorDims,
              [&](const std::vector<VarHandle>& axes) {
                ExprHandle idx = 0;
                for (size_t i = 0; i < axes.size(); i++) {
                  idx = idx + axes[i] * IntImm::make(*strides[i]);
                }
                return in_buffer(idx);
              }));
      kernelArgs_.emplace_back(
          in_buffer, std::vector<ShapeArg>(), std::vector<ShapeArg>());
#endif
      break;
    }
    case TypeKind::FloatType: {
      VarHandle v("v" + input->debugName(), kFloat);
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
      LOG(FATAL) << "Unhandled input type: " << *t;
      break;
    }
  }
}

void TensorExprKernel::compile() {
  KernelScope kernel_scope(&kernel_arena_);

  // Bind inputs to buffers.
  n_inputs_ = graph_->inputs().size();
  for (auto const& input : graph_->inputs()) {
    bindInput(input);
    input_types_.push_back(input->type());
  }

  // Bind nodes to tensor compute expressions.
  for (auto const& n : graph_->nodes()) {
    if (n->kind() == prim::Constant || n->kind() == prim::ListConstruct) {
      continue;
    } else {
      for (auto const& output : n->outputs()) {
        if (output->hasUses()) {
          tensors_.emplace(output->unique(), ComputeValue(output));
        }
      }
    }
    if (hasRandom_ && hasBroadcast_) {
      throw std::runtime_error(
          "Cannot support broadcast and random within one kernel");
    }
  }

  // Move output operands from `tensors_` to `tensor_outputs_`
  for (const auto& output : graph_->outputs()) {
    CHECK(tensors_.count(output->unique())) << "Output must be a tensor";
    tensor_outputs_.emplace_back(tensors_.at(output->unique()));
    tensors_.erase(output->unique());
  }
}

TensorExprKernel::TensorExprKernel(const std::shared_ptr<Graph>& subgraph)
    : graph_(subgraph), code_(subgraph) {
  try {
    compile();
  } catch (...) {
    fallback_ = true;
  }
}

void TensorExprKernel::run(Stack& stack) {
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

void TensorExprKernel::runKernel(Stack& stack) {
  KernelScope kernel_scope(&kernel_arena_);
  // Set up arguments (inputs, then outputs) for kernel call.
  auto inputs = last(stack, n_inputs_);
  PickAndCheckBackendType(inputs);

  std::map<const Expr*, int32_t> varToSize;

  std::vector<CodeGen::CallArg> run_args;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto const& input = inputs[i];
    if (input.isInt()) {
      run_args.emplace_back((int32_t)input.toInt());
    } else if (input.isDouble()) {
      run_args.emplace_back((float)input.toDouble());
    } else if (input.isTensor()) {
      auto const& tensor = input.toTensor();
      run_args.emplace_back(tensor.data_ptr());
      for (auto const& size : kernelArgs_[i].sizes()) {
        int32_t s = tensor.sizes()[size.idx];
        run_args.emplace_back(s);
        varToSize[size.var.node()] = s;
      }
      for (auto const& stride : kernelArgs_[i].strides()) {
        int32_t s = tensor.strides()[stride.idx];
        run_args.emplace_back(s);
      }
    }
  }

  std::vector<at::Tensor> outputs;
  for (auto& o : tensor_outputs_) {
    std::vector<int64_t> tensorSize;
    for (const Expr* dim : o->dims()) {
      auto it = varToSize.find(dim);
      if (it != varToSize.end()) {
        tensorSize.push_back(it->second);
      } else {
        const IntImm* s = dynamic_cast<const IntImm*>(dim);
        TORCH_CHECK(s);
        tensorSize.push_back(s->value());
      }
    }

    outputs.push_back(at::empty(
        tensorSize, c10::TensorOptions(tensorType(o)).device(device_)));
    run_args.emplace_back(outputs.back().data_ptr());
  }

  // Call the kernel.
  CodeGenRun(run_args);

  // Update the stack.
  drop(stack, n_inputs_);
  for (auto& o : outputs) {
    push_one(stack, std::move(o));
  }
}
