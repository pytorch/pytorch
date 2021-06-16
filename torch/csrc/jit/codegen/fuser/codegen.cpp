#include <torch/csrc/jit/codegen/fuser/codegen.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/fuser/compiler.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/codegen/fuser/tensor_info.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/codegen/fuser/cpu/resource_strings.h>
#include <torch/csrc/jit/codegen/fuser/cuda/resource_strings.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// Template for computing the offset into the tensor to access a value
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto dim_calc = CodeTemplate(R"(
//printf("tensor ${tensor} sizes[${d}] = %d, strides[${d}] = %d\n", ${tensor}.sizes[${d}],${tensor}.strides[${d}]);
size_t ${tensor}_dimIndex${d} = ${tensor}_linearIndex ${mod_sizes};
${tensor}_offset += ${tensor}_dimIndex${d} ${times_stride};
)");

static std::string valueName(const Value* n) {
  return "n" + c10::to_string(n->unique());
}

static std::string scalarValue(const int64_t v) {
  return c10::to_string(v);
}

static std::string scalarValue(const bool v) {
  return c10::to_string(v);
}

// Note: The NAN, NEG_INFINITY and POS_INFINITY strings map to device-specific
// implementations of these special values. These macros are found in the
// resource strings for each device.
static std::string scalarValue(const double v) {
  std::ostringstream out;
  if (std::isnan(v)) {
    out << "NAN";
  } else if (std::isinf(v)) {
    if (v < 0) {
      out << "NEG_INFINITY";
    } else {
      out << "POS_INFINITY";
    }
  } else {
    out << std::setprecision(16) << v;
  }
  return out.str();
}

// Note: Half is special-cased to avoid returning at::Half
static const char* scalarTypeName(const at::ScalarType type) {
  if (type == at::ScalarType::Half) {
    return "half";
  }
  if (type == at::ScalarType::BFloat16) {
    return "__nv_bfloat16";
  }

  switch (type) {
#define DEFINE_CASE(ctype, name) \
  case at::ScalarType::name:     \
    return #ctype;
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      throw std::runtime_error("unknown scalar type");
  }
}

static const char* calcScalarTypeName(const at::ScalarType type) {
  if (type == at::ScalarType::Half) {
    return "float";
  }
  if (type == at::ScalarType::BFloat16) {
    return "float";
  }
  return scalarTypeName(type);
}

static std::string variableType(const std::shared_ptr<c10::Type>& t) {
  if (t->kind() == TypeKind::IntType) {
    return "int64_t";
  } else if (t->kind() == TypeKind::FloatType) {
    return "double";
  } else if (t->kind() == TypeKind::BoolType) {
    return "bool";
  } else if (auto scalar_type = t->expectRef<TensorType>().scalarType()) {
    return calcScalarTypeName(*scalar_type);
  }
  // something went wrong with the type analysis during shape propagation
  throw std::runtime_error(
      "unknown scalar type during JIT fusion code generation");
}

static std::string typeCastedValueName(
    const std::shared_ptr<c10::Type>& t,
    const at::ScalarType outtype,
    const std::string& vn) {
  if (t->kind() == TypeKind::IntType || t->kind() == TypeKind::BoolType) {
    if (!isIntegralType(outtype, /*includeBool=*/false)) {
      return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
    }
    return vn;
  } else if (t->kind() == TypeKind::FloatType) {
    // We don't guard this on anything because in our type system for scalars,
    // there is not a distinction between `float` and `double`, however there
    // *is* a distinction in tensor scalar types. We conservatively insert a
    // cast here, which may end up being a no-op if the tensor's scalar type
    // is `double`.
    return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
  } else if (t->kind() == TypeKind::NoneType) {
    // Support None value for optional arguments like memory format
    return vn;
  } else if (auto scalar_type = t->expectRef<TensorType>().scalarType()) {
    if (*scalar_type != outtype) {
      return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
    }
    return vn;
  }
  // something went wrong with the type analysis during shape propagation
  throw std::runtime_error(
      "unknown scalar type during JIT fusion code generation");
}

// Writes RHS of special handling "simple mappable" ops
static std::string encodeSpecialRHS(const Node* n, TemplateEnv& env) {
  // special case for clamp fusion on missing min/max inputs
  // Note: It may seem unusual to have the bounds as the first case below,
  // this is so that if min or max is NaN, they are "ignored"
  // and when the input is NaN, the output is, too
  if (n->kind() == aten::clamp) {
    const auto min = n->input(1);
    const auto max = n->input(2);
    env.s("0", valueName(n->input(0)));

    if (!min->node()->mustBeNone() && !max->node()->mustBeNone()) {
      env.s("1", valueName(min));
      env.s("2", valueName(max));
      return format("(${0} < ${1} ? ${1} : (${0} > ${2}? ${2} : ${0}))", env);
    } else if (min->node()->mustBeNone()) {
      env.s("1", valueName(max));
      return format("(${0} > ${1} ? ${1} : ${0})", env);
    } else if (max->node()->mustBeNone()) {
      env.s("1", valueName(min));
      return format("(${0} < ${1} ? ${1} : ${0})", env);
    } else {
      throw std::runtime_error(
          "At least one of 'min' or 'max' must not be None");
    }
  } else {
    throw std::runtime_error("Cannot encode RHS of the node, op not supported");
  }
}

// This struct specifies a template for dispatching specific aten:: operators.
// The current variants of RHS code selection we support are for double and
// float output values. For example, an aten::log operator which is assigned
// to a float value would emit logf(), whereas an aten::log operator which is
// assigned to a double would emit log().
struct RHSTemplate {
  // Common case: float and double dispatch are identical
  RHSTemplate(const char* for_float)
      : for_float(for_float), for_double(for_float) {}

  RHSTemplate(const char* for_float, const char* for_double)
      : for_float(for_float), for_double(for_double) {}

  const char* for_float;
  const char* for_double;
};

// Writes "simple mappable" ops
static std::string encodeRHS(const Node* n) {
  static std::unordered_map<NodeKind, RHSTemplate> simple_map_ops = {
      // unary
      {aten::_cast_Float, "static_cast<float>(${0})"},
      {aten::abs, "fabs(${0})"},
      {aten::sigmoid, {"1.f / (1.f + expf(-${0}))", "1. / (1. + exp(-${0}))"}},
      {aten::relu, "${0} < 0 ? 0.f : ${0} "},
      {aten::threshold,
       "${0} <= ${1} ? static_cast<decltype(${0})>(${2}) : ${0} "},
      {aten::log, {"logf(${0})", "log(${0})"}},
      {aten::log10, {"log10f(${0})", "log10(${0})"}},
      {aten::log1p, {"log1pf(${0})", "log1p(${0})"}},
      {aten::log2, {"log2f(${0})", "log2(${0})"}},
      {aten::lgamma, {"lgammaf(${0})", "lgamma(${0})"}},
      {aten::exp, {"expf(${0})", "exp(${0})"}},
      {aten::expm1, {"expm1f(${0})", "expm1(${0})"}},
      {aten::erf, {"erff(${0})", "erf(${0})"}},
      {aten::erfc, {"erfcf(${0})", "erfc(${0})"}},
      {aten::cos, {"cosf(${0})", "cos(${0})"}},
      {aten::acos, {"acosf(${0})", "acos(${0})"}},
      {aten::cosh, {"coshf(${0})", "cosh(${0})"}},
      {aten::sin, {"sinf(${0})", "sin(${0})"}},
      {aten::asin, {"asinf(${0})", "asin(${0})"}},
      {aten::sinh, {"sinhf(${0})", "sinh(${0})"}},
      {aten::tan, {"tanf(${0})", "tan(${0})"}},
      {aten::atan, {"atanf(${0})", "atan(${0})"}},
      {aten::tanh, {"tanhf(${0})", "tanh(${0})"}},
      {aten::sqrt, {"sqrtf(${0})", "sqrt(${0})"}},
      {aten::rsqrt, {"rsqrtf(${0})", "rsqrt(${0})"}},
      {aten::ceil, {"ceilf(${0})", "ceil(${0})"}},
      {aten::floor, {"floorf(${0})", "floor(${0})"}},
      {aten::round, {"roundf(${0})", "round(${0})"}},
      {aten::trunc, {"truncf(${0})", "trunc(${0})"}},
      {aten::frac, {"${0} - truncf(${0})", "${0} - trunc(${0})"}},
      {aten::reciprocal, {"1.f/(${0})", "1./(${0})"}},
      {aten::neg, "-${0}"},
      // simple binary
      {aten::atan2, "atan2(${0}, ${1})"},
      {aten::min,
       "isnan(${0}) ? ${0} : (isnan(${1}) ? ${1} : (${0} < ${1} ? ${0} : ${1}))"},
      {aten::max,
       "isnan(${0}) ? ${0} : (isnan(${1}) ? ${1} : (${0} < ${1} ? ${1} : ${0}))"},

      // binary with other
      // TODO: some of these ops will not get generated because
      // we only work on float inputs/outputs, but they are here to record
      // that they are valid mappable ops once we handle more type

      {aten::__and__, "${0} && ${1}"},
      {aten::__lshift__, "${0} << ${1}"},
      {aten::__or__, "${0} || ${1}"},
      {aten::__rshift__, "${0} >> ${1}"},
      {aten::__xor__, "${0} ^ ${1}"},
      {aten::addcmul, "${0} + ${3} * ${1} * ${2}"},
      {aten::div, "${0} / ${1}"},
      {aten::eq, "${0_nocast} == ${1_nocast}"},
      {aten::fmod, "fmodf(${0}, ${1})"},
      {aten::ge, "(${0_nocast} >= ${1_nocast})"},
      {aten::gt, "${0_nocast} > ${1_nocast}"},
      {aten::le, "(${0_nocast} <= ${1_nocast})"},
      {aten::lt, "${0_nocast} < ${1_nocast}"},
      {aten::lerp, "${0} + ${2} * (${1} - ${0})"},
      {aten::type_as, "(${0})"},
      {aten::mul, "${0} * ${1}"},
      {aten::ne, "${0_nocast} != ${1_nocast}"},
      {aten::remainder, "fmod((${1} + fmod(${0}, ${1})), ${1})"},
      {aten::pow, {"powf(${0}, ${1})", "pow(${0}, ${1})"}},

      // alpha
      {aten::add, "${0} + ${2}*${1}"},
      {aten::sub, "(${0} - ${2}*${1})"},
      {aten::rand_like, "uniform(rnd())"},

      // where
      {aten::where, "(${0} ? ${1} : ${2})"},
  };

  TemplateEnv env;

  if (simple_map_ops.find(n->kind()) == simple_map_ops.end()) {
    return encodeSpecialRHS(n, env);
  } else {
    size_t i = 0;

    auto outtype = n->output()->type()->expectRef<TensorType>().scalarType();
    TORCH_INTERNAL_ASSERT(outtype);

    for (auto in : n->inputs()) {
      // PyTorch converts (scalar) argument types to result before applying the
      // operator e.g. 1.4-torch.tensor(3) = -2
      env.s(
          c10::to_string(i),
          typeCastedValueName(in->type(), *outtype, valueName(in)));
      // Uncasted operands only used for comparison operators
      env.s(c10::to_string(i) + "_nocast", valueName(in));
      i++;
    }

    const auto& templ = simple_map_ops.at(n->kind());
    const char* str = nullptr;
    if (*outtype == at::kFloat) {
      str = templ.for_float;
    } else {
      str = templ.for_double;
    }
    AT_ASSERT(str);
    return format(str, env);
  }
}

static void emitIndexingFor(
    std::ostream& out,
    const std::string& tensor,
    const int ndim,
    const bool last_is_cont) {
  TemplateEnv env;
  env.s("tensor", tensor);
  out << format("IndexType ${tensor}_offset = 0;\n", env);
  out << format("IndexType ${tensor}_linearIndex = linearIndex;\n", env);
  for (int d = ndim - 1; d >= 0; --d) {
    env.d("d", d);
    env.s("mod_sizes", d > 0 ? format("% ${tensor}.sizes[${d}]", env) : "");
    env.s(
        "times_stride",
        (d < ndim - 1 || !last_is_cont)
            ? format("* ${tensor}.strides[${d}]", env)
            : "");
    out << dim_calc.format(env);
    if (d > 0) {
      out << format("${tensor}_linearIndex /= ${tensor}.sizes[${d}];\n", env);
    }
  }
}

static void emitCheckFor(
    std::ostream& out,
    const std::string& tensor,
    const int ndim,
    const TensorDesc& desc) {
  TemplateEnv env;
  env.s("tensor", tensor);
  env.s("scalar_type", scalarTypeName(desc.scalar_type));

  // allocate buffer to load 4
  out << format("${scalar_type} ${tensor}_buf[4];\n", env);

  // check if last dim is contiguous
  if (!desc.lastIsContiguous()) {
    out << "flag_vec4 = false;\n";
    return;
  }

  // disable on dtype > 4 bytes for performance
  if (at::elementSize(desc.scalar_type) > 4) {
    out << "flag_vec4 = false;\n";
    return;
  }

  // last dim size multiple of 4, other dim stride multiple of 4
  for (int d = ndim - 1; d >= 0; --d) {
    env.d("d", d);
    if (d == ndim - 1) {
      // last dim stride already checked above at compile time
      out << format(
          "if(${tensor}.sizes[${d}] % 4 != 0) flag_vec4 = false;\n", env);
    } else {
      out << format(
          "if(${tensor}.strides[${d}] % 4 != 0) flag_vec4 = false;\n", env);
    }
  }

  // pointer aligned
  out << format(
      "if(((uint64_t) ${tensor}.data) % (4 * sizeof(${scalar_type})) != 0) flag_vec4 = false;\n",
      env);
}

// TODO: handle cases where we need to generate > 2^32 element tensors
std::string generateKernel(
    const std::string& name,
    const Graph& graph,
    const std::vector<std::pair<const Value*, const c10::optional<TensorDesc>>>&
        inputs,
    const std::vector<std::pair<const Value*, const TensorDesc>>& outputs,
    const bool use_cuda) {
  TemplateEnv env;
  env.s("kernelName", name);
  env.s(
      "IndexType",
      "unsigned int"); // Note: not uint32_t to avoid including cstdint

  std::stringstream tensorChecks;
  std::stringstream body;
  std::stringstream body_vec4;
  std::stringstream load;
  std::stringstream store;
  std::stringstream tensorOffsets;
  std::vector<std::string> formals;
  std::vector<std::string> argument_loads;

  // Lambda for writing arguments
  auto emitFormal = [&](const Value* n, const TensorDesc& desc) {
    env.d(
        "formal_index",
        formals.size() +
            1); // + 1 because the first argument is the linearIndex
    std::string tensor =
        "t" +
        c10::to_string(
            formals.size()); // can't be unique() because Param may be an output
    const auto nDim = desc.nDim();
    emitCheckFor(tensorChecks, tensor, nDim, desc);
    emitIndexingFor(tensorOffsets, tensor, nDim, desc.lastIsContiguous());
    env.s("tensor", tensor);
    env.d("nDim", nDim);
    env.s("scalar_type", scalarTypeName(desc.scalar_type));
    formals.push_back(
        format("const TensorInfo<${scalar_type},${nDim}> ${tensor}", env));
    argument_loads.push_back(format(
        "*static_cast<TensorInfo<${scalar_type},${nDim}>*>(args[${formal_index}])",
        env));
  };

  auto emitScalarFormal = [&](const Value* n) {
    env.d(
        "formal_index",
        formals.size() +
            1); // + 1 because the first argument is the linearIndex
    std::string scalar =
        "s" +
        c10::to_string(
            formals.size()); // can't be unique() because Param may be an output
    env.d(
        "formal_index",
        formals.size() +
            1); // + 1 because the first argument is the linearIndex
    env.s("scalar", scalar);
    env.s("scalar_type", variableType(n->type()));
    formals.push_back(format("${scalar_type} ${scalar}", env));
    argument_loads.push_back(
        format("*static_cast<${scalar_type}*>(args[${formal_index}])", env));
  };

  // Writes input parameters
  for (const auto& input : inputs) {
    if (input.second.has_value()) {
      emitFormal(input.first, *input.second);
    } else {
      emitScalarFormal(input.first);
    }
  }

  // Writes output parameters
  for (const auto& output : outputs) {
    emitFormal(output.first, output.second);
  }

  // Acquires input values
  bool has_half_tensor = false;
  bool has_bfloat_tensor = false;
  size_t formal_count = 0;
  for (const auto& input : inputs) {
    auto p = input.first;
    env.s("node", valueName(p));
    env.d("formal", formal_count++);

    // Acquires and converts (if needed) inputs
    // Note: conversion from half is only supported for CUDA kernels.
    //  The conversion immediately converts fp16 inputs to float.
    //  Access for other types is common to CUDA and CPU kernels.
    if (input.second.has_value()) {
      const auto is_half = input.second.has_value() &&
          ((*input.second).scalar_type == at::ScalarType::Half);
      const auto is_bfloat = input.second.has_value() &&
          ((*input.second).scalar_type == at::ScalarType::BFloat16);
      const auto is_bool = input.second.has_value() &&
          ((*input.second).scalar_type == at::ScalarType::Bool);
      if (is_half) {
        AT_ASSERT(use_cuda);
        env.s(
            "access",
            format("__half2float(t${formal}.data[t${formal}_offset])", env));
        env.s("access_vec4", format("__half2float(t${formal}_buf[i])", env));
        has_half_tensor = true;
      } else if (is_bfloat) {
        AT_ASSERT(use_cuda);
        env.s(
            "access",
            format(
                "__bfloat162float(t${formal}.data[t${formal}_offset])", env));
        env.s(
            "access_vec4", format("__bfloat162float(t${formal}_buf[i])", env));
        has_bfloat_tensor = true;
      } else if (use_cuda) {
        // No __ldg overload for bool
        if (is_bool) {
          env.s("access", format("t${formal}.data[t${formal}_offset]", env));
        } else {
          env.s(
              "access",
              format("__ldg(&t${formal}.data[t${formal}_offset])", env));
        }
        env.s("access_vec4", format("t${formal}_buf[i]", env));
      } else {
        env.s("access", format("t${formal}.data[t${formal}_offset]", env));
        env.s("access_vec4", format("t${formal}_buf[i]", env));
      }
      env.s("lhs_type", calcScalarTypeName(input.second.value().scalar_type));

      // load input in vectorized code path
      auto ele_size = at::elementSize((*input.second).scalar_type);
      if (ele_size == 1) {
        env.s(
            "load4",
            format(
                "*(reinterpret_cast<float*>(t${formal}_buf)) = *(reinterpret_cast<float*>(t${formal}.data + t${formal}_offset))",
                env));
      } else if (ele_size == 2) {
        env.s(
            "load4",
            format(
                "*(reinterpret_cast<float2*>(t${formal}_buf)) = *(reinterpret_cast<float2*>(t${formal}.data + t${formal}_offset))",
                env));
      } else if (ele_size == 4) {
        env.s(
            "load4",
            format(
                "*(reinterpret_cast<float4*>(t${formal}_buf)) = *(reinterpret_cast<float4*>(t${formal}.data + t${formal}_offset))",
                env));
      } else {
        env.s(
            "load4",
            format(
                "for(int i = 0; i<4; i++) t${formal}_buf[i] = t${formal}.data[t${formal}_offset + i]",
                env));
      }
      load << format("${load4};\n", env);

    } else {
      env.s("access", format("s${formal}", env));
      env.s("access_vec4", format("s${formal}", env));
      env.s("lhs_type", variableType(input.first->type()));
    }
    body << format("${lhs_type} ${node} = ${access};\n", env);
    body_vec4 << format("${lhs_type} ${node} = ${access_vec4};\n", env);
  }

  bool has_random = false;
  // Generates code for intermediate nodes
  // Note: Concat and Chunk are implicitly generated
  // Note: Random number generation is only supported for CUDA kernels.
  // Note: Constant None node is ignored and we will handle it in the
  //       places where the constant None node is used
  // Note: No need to iterate over reference as n is a pointer
  for (const auto n : graph.nodes()) {
    static_assert(std::is_pointer<decltype(n)>::value, "n must be a pointer");
    // Note: FusedConcat nodes work by narrowing the output Tensors before the
    // kernel runs
    if (n->kind() == prim::FusedConcat)
      continue;
    if (n->kind() == prim::ConstantChunk)
      continue;
    if (n->mustBeNone())
      continue;
    if (n->kind() == aten::rand_like) {
      AT_ASSERT(use_cuda);
      has_random = true;
    }
    // Always emit double for prim::Constant. This will be narrowed later based
    // on either:
    //  - Tensor-Scalar operator type rules
    //  - Math function rules
    if (n->kind() == prim::Constant) {
      const auto val = toIValue(n->output()).value();
      std::string rhs;
      if (val.isDouble()) {
        rhs = scalarValue(val.toDouble());
      } else if (val.isBool()) {
        rhs = scalarValue(val.toBool());
      } else {
        AT_ASSERT(val.isInt());
        rhs = scalarValue(val.toInt());
      }
      env.s("node", valueName(n->output()));
      env.s("rhs", rhs);
      env.s("lhs_type", variableType(n->output()->type()));
    } else {
      env.s("node", valueName(n->output()));
      env.s("rhs", encodeRHS(n));
      env.s("lhs_type", variableType(n->output()->type()));
    }

    body << format("${lhs_type} ${node} = ${rhs};\n", env);
    body_vec4 << format("${lhs_type} ${node} = ${rhs};\n", env);
  }

  // Generates writes to output tensors
  for (const auto& output : outputs) {
    env.d("formal", formal_count++);
    env.s("access", format("t${formal}.data[t${formal}_offset]", env));
    env.s("access_vec4", format("t${formal}_buf[i]", env));
    env.s("node", valueName(output.first));

    // Acquires and converts (if needed) outputs
    // Note: conversion to half is only supported for CUDA kernels.
    const auto is_half = (output.second.scalar_type == at::ScalarType::Half);
    const auto is_bfloat =
        (output.second.scalar_type == at::ScalarType::BFloat16);
    if (is_half) {
      AT_ASSERT(use_cuda);
      body << format("${access} = __float2half(${node});\n", env);
      body_vec4 << format("${access_vec4} = __float2half(${node});\n", env);
      has_half_tensor = true;
    } else if (is_bfloat) {
      AT_ASSERT(use_cuda);
      body << format("${access} = __float2bfloat16(${node});\n", env);
      body_vec4 << format("${access_vec4} = __float2bfloat16(${node});\n", env);
      has_bfloat_tensor = true;
    } else {
      body << format("${access} = ${node};\n", env);
      body_vec4 << format("${access_vec4} = ${node};\n", env);
    }

    // store output in vectorized code path
    auto ele_size = at::elementSize(output.second.scalar_type);
    if (ele_size == 1) {
      env.s(
          "store4",
          format(
              "*(reinterpret_cast<float*>(t${formal}.data + t${formal}_offset)) = *(reinterpret_cast<float*>(t${formal}_buf))",
              env));
    } else if (ele_size == 2) {
      env.s(
          "store4",
          format(
              "*(reinterpret_cast<float2*>(t${formal}.data + t${formal}_offset)) = *(reinterpret_cast<float2*>(t${formal}_buf))",
              env));
    } else if (ele_size == 4) {
      env.s(
          "store4",
          format(
              "*(reinterpret_cast<float4*>(t${formal}.data + t${formal}_offset)) = *(reinterpret_cast<float4*>(t${formal}_buf))",
              env));
    } else {
      env.s(
          "store4",
          format(
              "for(int i = 0; i<4; i++) t${formal}.data[t${formal}_offset + i] = t${formal}_buf[i]",
              env));
    }
    store << format("${store4};\n", env);
  }

  // Includes headers
  // Note: CUDA kernels support halfs and random generation, CPU kernels do not
  if (has_half_tensor) {
    env.s("HalfHeader", cuda::half_support_literal);
  } else {
    env.s("HalfHeader", "");
  }
  if (has_bfloat_tensor) {
    env.s("BFloat16Header", cuda::bfloat16_support_literal);
  } else {
    env.s("BFloat16Header", "");
  }

  if (has_random) {
    env.s("RandHeader", cuda::rand_support_literal);
    env.s("RandParam", cuda::rand_param);
    env.s("RandInit", cuda::rand_init);
  } else {
    env.s("RandHeader", "");
    env.s("RandParam", "");
    env.s("RandInit", "");
  }

  // HIP headers must be included until precompiled header feature is available
  // clang-format off
#ifdef __HIP_PLATFORM_HCC__
#if ROCM_VERSION < 40200
  if (use_cuda && has_half_tensor) {
    env.s("RuntimeHeader", R"(
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
)");
  } else if (use_cuda) {
    env.s("RuntimeHeader", R"(
#include <hip/hip_runtime.h>
)");
  }
#else
  // Still need the key defined, but empty.
  env.s("RuntimeHeader", R"()");
#endif
#endif
  // clang-format on

  // Instantiates the CUDA or CPU-specific templates
  env.s("tensorOffsets", tensorOffsets.str());
  env.s("tensorChecks", tensorChecks.str());
  env.s("kernelBody", body.str());
  env.s("kernelBody_vec4", body_vec4.str());
  env.s("kernelLoad", load.str());
  env.s("kernelStore", store.str());
  env.v("formals", formals);
  env.v("argument_loads", argument_loads);
  std::string code_string;
  if (use_cuda) {
    env.s("type_declarations", cuda::type_declarations_template.format(env));
    code_string = cuda::cuda_compilation_unit_template.format(env);
  } else {
    env.s("type_declarations", cpu::type_declarations_template.format(env));
    code_string = cpu::cpu_compilation_unit_template.format(env);
  }

  if (debugFuser()) {
    std::cerr << "fusion code:" << code_string << std::endl;
  }
  return code_string;
}

} // namespace fuser
} // namespace jit
} // namespace torch
