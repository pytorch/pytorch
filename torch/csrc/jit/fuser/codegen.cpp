#include <torch/csrc/jit/fuser/codegen.h>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/code_template.h>
#include <torch/csrc/jit/fuser/compiler.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/tensor_info.h>
#include <torch/csrc/jit/ir.h>

#include <torch/csrc/jit/fuser/cpu/resource_strings.h>
#include <torch/csrc/jit/fuser/cuda/resource_strings.h>

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
static auto dim_calc = CodeTemplate(R"(
//printf("tensor ${tensor} sizes[${d}] = %d, strides[${d}] = %d\n", ${tensor}.sizes[${d}],${tensor}.strides[${d}]);
size_t ${tensor}_dimIndex${d} = ${tensor}_linearIndex ${mod_sizes};
${tensor}_offset += ${tensor}_dimIndex${d} ${times_stride};
)");

static std::string valueName(const Value* n) {
  return "n" + std::to_string(n->unique());
}

static std::string scalarValue(const int64_t v) {
  return std::to_string(v);
}

static std::string scalarValue(const bool v) {
  return std::to_string(v);
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
    out << std::scientific << v << "f";
  }
  return out.str();
}

// Note: Half is special-cased to avoid returning at::Half
static const char* scalarTypeName(const at::ScalarType type) {
  if (type == at::ScalarType::Half) {
    return "half";
  }

  switch (type) {
#define DEFINE_CASE(ctype, name, _) \
  case at::ScalarType::name:        \
    return #ctype;
    AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      throw std::runtime_error("unknown scalar type");
  }
}

static const char* calcScalarTypeName(const at::ScalarType type) {
  if (type == at::ScalarType::Half) {
    return "float";
  }
  return scalarTypeName(type);
}

static std::string variableType(const std::shared_ptr<c10::Type>& t) {
  if (t->kind() == TypeKind::IntType) {
    return "int";
  } else if (t->kind() == TypeKind::FloatType) {
    return "float";
  } else if (t->kind() == TypeKind::BoolType) {
    return "bool";
  } else if (t->kind() == TypeKind::DimensionedTensorType) {
    auto const tt = t->cast<DimensionedTensorType>();
    return calcScalarTypeName(tt->scalarType());
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
    if (!isIntegralType(outtype)) {
      return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
    }
    return vn;
  } else if (t->kind() == TypeKind::FloatType) {
    if (!isFloatingType(outtype)) {
      return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
    }
    return vn;
  } else if (t->kind() == TypeKind::DimensionedTensorType) {
    auto const tt = t->cast<DimensionedTensorType>();
    if (tt->scalarType() != outtype) {
      return std::string("((") + calcScalarTypeName(outtype) + ") " + vn + ")";
    }
    return vn;
  }
  // something went wrong with the type analysis during shape propagation
  throw std::runtime_error(
      "unknown scalar type during JIT fusion code generation");
}

// Writes "simple mappable" ops
static std::string encodeRHS(const Node* n) {
  static std::unordered_map<NodeKind, std::string> simple_map_ops = {
      // unary
      {aten::_cast_Float, "static_cast<float>(${0})"},
      {aten::abs, "fabs(${0})"},
      {aten::sigmoid, "1.f / (1.f + expf(-${0}))"},
      {aten::relu, "${0} < 0 ? 0.f : ${0} "},
      {aten::threshold,
       "${0} <= ${1} ? static_cast<decltype(${0})>(${2}) : ${0} "},
      {aten::log, "logf(${0})"},
      {aten::log10, "log10f(${0})"},
      {aten::log1p, "log1pf(${0})"},
      {aten::log2, "log2f(${0})"},
      {aten::lgamma, "lgammaf(${0})"},
      {aten::exp, "expf(${0})"},
      {aten::expm1, "expm1f(${0})"},
      {aten::erf, "erff(${0})"},
      {aten::erfc, "erfcf(${0})"},
      {aten::cos, "cosf(${0})"},
      {aten::acos, "acosf(${0})"},
      {aten::cosh, "coshf(${0})"},
      {aten::sin, "sinf(${0})"},
      {aten::asin, "asinf(${0})"},
      {aten::sinh, "sinhf(${0})"},
      {aten::tan, "tanf(${0})"},
      {aten::atan, "atanf(${0})"},
      {aten::tanh, "tanhf(${0})"},
      {aten::sqrt, "sqrtf(${0})"},
      {aten::rsqrt, "rsqrtf(${0})"},
      {aten::ceil, "ceilf(${0})"},
      {aten::floor, "floorf(${0})"},
      {aten::round, "roundf(${0})"},
      {aten::trunc, "truncf(${0})"},
      {aten::frac, "fracf(${0})"},
      {aten::reciprocal, "1.f/(${0})"},
      {aten::neg, "-${0}"},
      // simple binary
      {aten::atan2, "atan2(${0}, ${1})"},
      {aten::min, "fminf(${0}, ${1})"},
      {aten::max, "fmaxf(${0}, ${1})"},

      // binary with other
      // TODO: some of these ops will not get generated because
      // we only work on float inputs/outputs, but they are here to record
      // that they are valid mappable ops once we handle more type

      {aten::__and__, "${0} && ${1}"},
      {aten::__lshift__, "${0} << ${1}"},
      {aten::__or__, "${0} || ${1}"},
      {aten::__rshift__, "${0} >> ${1}"},
      {aten::__xor__, "${0} ^ ${1}"},
      {aten::div, "${cast_0} / ${cast_1}"},
      {aten::eq, "${0} == ${1}"},
      {aten::fmod, "fmodf(${cast_0}, ${cast_1})"},
      {aten::ge, "(${0} >= ${1})"},
      {aten::gt, "${0} > ${1}"},
      {aten::le, "(${0} <= ${1})"},
      {aten::lt, "${0} < ${1}"},
      {aten::type_as, "(${cast_0})"},
      {aten::mul, "${cast_0} * ${cast_1}"},
      {aten::ne, "${0} != ${1}"},
      {aten::remainder, "remainderf(${0}, ${1})"},
      {aten::pow, "powf(${cast_0}, ${cast_1})"},

      // alpha
      {aten::add, "${cast_0} + ${cast_2}*${cast_1}"},
      {aten::sub, "(${cast_0} - ${cast_2}*${cast_1})"},
      {aten::rand_like, "uniform(rnd())"},

      // min, max
      // It may seem unusual to have the bounds as the first case below,
      // this is so that if min or max is NaN, they are "ignored"
      // and when the input is NaN, the output is, too
      {aten::clamp, "(${0}<${1}?${1}:(${0}>${2}?${2}:${0}))"},

      // where
      {aten::where, "(${0} ? ${1} : ${2})"},

      // simple derivatives
      {aten::_sigmoid_backward, "${0} * ${1} * (1.f - ${1})"},
      {aten::_tanh_backward, "${0} * (1.f - ${1} * ${1})"},
  };

  if (n->kind() == prim::Constant) {
    const auto val = toIValue(n->output()).value();
    if (val.isDouble()) {
      return scalarValue(val.toDouble());
    } else if (val.isBool()) {
      return scalarValue(val.toBool());
    } else {
      AT_ASSERT(val.isInt());
      return scalarValue(val.toInt());
    }
  }

  TemplateEnv env;
  size_t i = 0;
  auto outtype =
      n->output()->type()->expect<c10::DimensionedTensorType const>()->scalarType();
  for (auto in : n->inputs()) {
    // PyTorch converts (scalar) argument types to result before applying the
    // operator e.g. 1.4-torch.tensor(3) = -2
    env.s(std::to_string(i), valueName(in));
    env.s(
        std::string("cast_") + std::to_string(i),
        typeCastedValueName(in->type(), outtype, valueName(in)));
    i++;
  }

  const auto& str = simple_map_ops.at(n->kind());
  return format(str, env);
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

// TODO: handle cases where we need to generate > 2^32 element tensors
std::string generateKernel(
    const std::string& name,
    const Graph& graph,
    const std::vector<std::pair<const Value*, const TensorDesc>>& inputs,
    const std::vector<std::pair<const Value*, const TensorDesc>>& outputs,
    const bool use_cuda) {
  TemplateEnv env;
  env.s("kernelName", name);
  env.s(
      "IndexType",
      "unsigned int"); // Note: not uint32_t to avoid including cstdint

  std::stringstream body;
  std::stringstream tensorOffsets;
  std::vector<std::string> formals;
  std::vector<std::string> argument_loads;

  // Lambda for writing arguments
  auto emitFormal = [&](const Value* n, const TensorDesc& desc) {
    std::string tensor =
        "t" +
        std::to_string(
            formals.size()); // can't be unique() because Param may be an output
    const auto nDim = desc.nDim();
    emitIndexingFor(tensorOffsets, tensor, nDim, desc.lastIsContiguous());
    env.s("tensor", tensor);
    env.d(
        "formal_index",
        formals.size() +
            1); // + 1 because the first argument is the linearIndex
    env.d("nDim", nDim);
    env.s("scalar_type", scalarTypeName(desc.scalar_type));
    formals.push_back(
        format("TensorInfo<${scalar_type},${nDim}> ${tensor}", env));
    argument_loads.push_back(format(
        "*static_cast<TensorInfo<${scalar_type},${nDim}>*>(args[${formal_index}])",
        env));
  };

  // Writes input parameters
  for (const auto& input : inputs) {
    emitFormal(input.first, input.second);
  }

  // Writes output parameters
  for (const auto& output : outputs) {
    emitFormal(output.first, output.second);
  }

  // Acquires input values
  bool has_half_tensor = false;
  size_t formal_count = 0;
  for (const auto& input : inputs) {
    auto p = input.first;
    env.s("node", valueName(p));
    env.d("formal", formal_count++);

    // Acquires and converts (if needed) inputs
    // Note: conversion from half is only supported for CUDA kernels.
    //  The conversion immediately converts fp16 inputs to float.
    //  Access for other types is common to CUDA and CPU kernels.
    const auto is_half = (input.second.scalar_type == at::ScalarType::Half);
    if (is_half) {
      AT_ASSERT(use_cuda);
      env.s(
          "access",
          format("__half2float(t${formal}.data[t${formal}_offset])", env));
      has_half_tensor = true;
    } else {
      env.s("access", format("t${formal}.data[t${formal}_offset]", env));
    }
    env.s("lhs_type", calcScalarTypeName(input.second.scalar_type));

    body << format("${lhs_type} ${node} = ${access};\n", env);
  }

  bool has_random = false;
  // Generates code for intermediate nodes
  // Note: Concat and Chunk are implicitly generated
  // Note: Random number generation is only supported for CUDA kernels.
  for (const auto& n : graph.nodes()) {
    // Note: FusedConcat nodes work by narrowing the output Tensors before the
    // kernel runs
    if (n->kind() == prim::FusedConcat)
      continue;
    if (n->kind() == prim::ConstantChunk)
      continue;
    if (n->kind() == aten::rand_like) {
      AT_ASSERT(use_cuda);
      has_random = true;
    }
    env.s("node", valueName(n->output()));
    env.s("rhs", encodeRHS(n));
    env.s("lhs_type", variableType(n->output()->type()));
    body << format("${lhs_type} ${node} = ${rhs};\n", env);
  }

  // Generates writes to output tensors
  for (const auto& output : outputs) {
    env.d("formal", formal_count++);
    env.s("access", format("t${formal}.data[t${formal}_offset]", env));
    env.s("node", valueName(output.first));

    // Acquires and converts (if needed) outputs
    // Note: conversion to half is only supported for CUDA kernels.
    const auto is_half = (output.second.scalar_type == at::ScalarType::Half);
    if (is_half) {
      AT_ASSERT(use_cuda);
      body << format("${access} = __float2half(${node});\n", env);
      has_half_tensor = true;
    } else {
      body << format("${access} = ${node};\n", env);
    }
  }

  // Includes headers
  // Note: CUDA kernels support halfs and random generation, CPU kernels do not
  if (has_half_tensor) {
    env.s("HalfHeader", cuda::half_support_literal);
  } else {
    env.s("HalfHeader", "");
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

  // Insantiates the CUDA or CPU-specific templates
  env.s("tensorOffsets", tensorOffsets.str());
  env.s("kernelBody", body.str());
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
