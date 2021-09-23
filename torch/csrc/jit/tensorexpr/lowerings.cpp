#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>

namespace torch {
namespace jit {
namespace tensorexpr {

std::unordered_map<std::string, NNCLoweringFunction>& getNNCLoweringRegistry() {
  static std::unordered_map<std::string, NNCLoweringFunction>
      lowering_registry_;
  return lowering_registry_;
}

NNCLoweringFunction getStandardLoweringFor(const std::string& op) {
  const auto& lowerings = getNNCLoweringRegistry();
  if (lowerings.count(op))
    return lowerings.at(op);
  return nullptr;
}

namespace {
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

} // namespace

} // namespace tensorexpr
} // namespace jit
} // namespace torch
