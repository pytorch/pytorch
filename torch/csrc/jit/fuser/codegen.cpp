#include "torch/csrc/jit/fuser/codegen.h"

#include "ATen/ATen.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/fuser/config.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/tensor_info.h"

#if USE_CUDA_FUSER
  #include "torch/csrc/jit/fuser/cuda/resource_strings.h"
#endif 

#if USE_CPU_FUSER
  #include "torch/csrc/jit/fuser/cpu/resource_strings.h"
#endif 

#include <tuple>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <vector>
#include <cmath>

namespace torch { namespace jit { namespace fuser {

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

  switch(type) {
    #define DEFINE_CASE(ctype,name,_) \
      case at::ScalarType::name: return #ctype;
    AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(DEFINE_CASE)
    #undef DEFINE_CASE
    default:
      throw std::runtime_error("unknown scalar type");
  }
}

// Writes "simple mappable" ops
static std::string encodeRHS(const Node* n) {
  static std::unordered_map<NodeKind, std::string> simple_map_ops = {
    // unary
    {aten::abs, "absf(${0})"},
    {aten::sigmoid, "1.f / (1.f + expf(-${0}))"},
    {aten::relu, "${0} < 0 ? 0.f : ${0} "},
    {aten::log, "logf(${0})"},
    {aten::log10, "log10f(${0})"},
    {aten::log1p, "log1pf(${0})"},
    {aten::log2,  "log2f(${0})"},
    {aten::lgamma, "lgammaf(${0})"},
    {aten::exp, "expf(${0})"},
    {aten::expm1, "expm1f(${0})"},
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
    {aten::reciprocal, "reciprocalf(${0})"},
    {aten::neg, "-${0}"},
    //simple binary
    {aten::atan2, "atan2(${0}, ${1})"},
    {aten::min, "fminf(${0}, ${1})"},
    {aten::max, "fmaxf(${0}, ${1})"},

    //binary with other
    // TODO: some of these ops will not get generated because
    // we only work on float inputs/outputs, but they are here to record
    // that they are valid mappable ops once we handle more type

    {aten::__and__, "${0} && ${1}"},
    {aten::__lshift__, "${0} << ${1}"},
    {aten::__or__, "${0} || ${1}"},
    {aten::__rshift__, "${0} >> ${1}"},
    {aten::__xor__, "${0} ^ ${1}"},
    {aten::div, "${0} / ${1}"},
    {aten::eq, "${0} == ${1}"},
    {aten::fmod, "fmodf(${0}, ${1})"},
    {aten::ge, "(${0} >= ${1})"},
    {aten::gt, "${0} > ${1}"},
    {aten::le, "(${0} <= ${1})"},
    {aten::lt, "${0} < ${1}"},
    {aten::type_as, "(${0})"}, //everything is implicitly convertible to float
    {aten::mul, "${0} * ${1}"},
    {aten::ne, "${0} != ${1}"},
    {aten::remainder, "remainderf(${0}, ${1})"},
    {aten::pow, "powf(${0}, ${1})"},

    //alpha
    {aten::add, "${0} + ${2}*${1}"},
    {aten::sub, "(${0} - ${2}*${1})"},
    {aten::rand_like, "uniform(rnd())"},

    // min, max
    // It may seem unusual to have the bounds as the first case below,
    // this is so that if min or max is NaN, they are "ignored"
    // and when the input is NaN, the output is, too
    {aten::clamp, "(${0}<${1}?${1}:(${0}>${2}?${2}:${0}))"},

    // simple derivatives
    {aten::_sigmoid_backward, "${0} * ${1} * (1.f - ${1})"},
    {aten::_tanh_backward,    "${0} * (1.f - ${1} * ${1})"},
  };

  if (n->kind() == prim::Constant) {
    const auto val = toIValue(n->output()).value();
    if (val.isDouble()) {
      return scalarValue(val.toDouble());
    } else {
      JIT_ASSERT(val.isInt());
      return scalarValue(val.toInt());
    }
  }

  TemplateEnv env;
  size_t i = 0;
  for(auto in : n->inputs()) {
    env.s(std::to_string(i++), valueName(in));
  }

  const auto & str = simple_map_ops.at(n->kind());
  return format(str, env);
}

// If there is a single user of a node and it's a chunk operation, returns
//  that user. Returns nullptr otherwise.
static Node* usedInFusedChunk(const Value* input) {
  auto uses = input->uses();
  if (uses.size() == 1) {
    Node *user = uses[0].user;
    if (user->kind() == prim::ConstantChunk) {
      return user;
    }
  }
  return nullptr;
}

static void emitIndexingFor(
  std::ostream& out
, const std::string& tensor
, const int ndim
, const bool last_is_cont) {
  TemplateEnv env;
  env.s("tensor",tensor);
  out << format("IndexType ${tensor}_offset = 0;\n",env);
  out << format("IndexType ${tensor}_linearIndex = linearIndex;\n",env);
  for (int d = ndim - 1; d >= 0; --d) {
    env.d("d",d);
    env.s("mod_sizes", d > 0 ? format("% ${tensor}.sizes[${d}]",env) : "");
    env.s("times_stride",(d < ndim - 1 || !last_is_cont) ?
      format("* ${tensor}.strides[${d}]",env) : "");
    out << dim_calc.format(env);
    if (d > 0) {
      out << format("${tensor}_linearIndex /= ${tensor}.sizes[${d}];\n",env);
    }
  }
}

// TODO: handle cases where we need to generate > 2^32 element tensors
std::tuple<
  std::string
, std::vector<PartitionDesc>
, std::vector<PartitionDesc>
, bool> 
generateKernel(
  const std::string& name
, const Graph& graph
, const std::vector<TensorDesc>& input_desc
, const std::vector<TensorDesc>& output_desc
, const bool use_cuda) {
  TemplateEnv env;
  env.s("kernelName", name);
  env.s("IndexType","unsigned int"); // Note: not uint32_t to avoid including cstdint

  std::stringstream body;
  std::stringstream tensorOffsets;
  std::vector<std::string> formals;
  std::vector<std::string> argument_loads;

  // Lambda for writing arguments
  auto emitFormal = [&](const Value* n, const TensorDesc& desc) {
    std::string tensor = "t" + std::to_string(formals.size()); //can't be unique() because Param may be an output
    const auto nDim = desc.nDim();
    emitIndexingFor(tensorOffsets, tensor, nDim,  desc.lastIsContiguous());
    env.s("tensor", tensor);
    env.d("formal_index", formals.size() + 1); // + 1 because the first argument is the linearIndex
    env.d("nDim", nDim);
    env.s("scalar_type", scalarTypeName(desc.scalar_type));
    formals.push_back(format("TensorInfo<${scalar_type},${nDim}> ${tensor}", env));
    argument_loads.push_back(format("*static_cast<TensorInfo<${scalar_type},${nDim}>*>(args[${formal_index}])", env));
  };

  // Writes input parameters and creates flattened inputs
  std::vector<PartitionDesc> chunk_desc;
  std::vector<std::pair<const Value*, const TensorDesc&>> flat_inputs;
  {
    size_t input_index = 0;
    for(const auto& p : graph.inputs()) {
      if (const Node* chunk = usedInFusedChunk(p)) {
        int64_t dim = chunk->i(attr::dim);
        int64_t chunks = chunk->i(attr::chunks);
        chunk_desc.emplace_back(input_desc[input_index++], chunks, dim);
        for (const auto* o : chunk->outputs()) {
          flat_inputs.emplace_back(o, *chunk_desc.back().subTensorDesc());
        }
      } else {
        chunk_desc.emplace_back();
        flat_inputs.emplace_back(p, input_desc[input_index++]);
      }
    }
    for (const auto& input : flat_inputs) {
      emitFormal(input.first, input.second);
    }
  }

  // Writes output parameters and creates flattened outputs
  std::vector<PartitionDesc> concat_desc;
  std::vector<std::pair<const Value*, TensorDesc>> flat_output_nodes;
  {
    size_t i = 0;
    for (const auto& o : graph.outputs()) {
      const auto& desc = output_desc[i++];
      if (o->node()->kind() != prim::FusedConcat) {
        emitFormal(o, desc);
        concat_desc.emplace_back();
        flat_output_nodes.emplace_back(o, desc);
      } else {
        const auto cat = o->node();
        concat_desc.emplace_back(desc, cat->inputs().size(), cat->i(attr::dim));
        for(const auto& c : cat->inputs()) {
          emitFormal(c, *concat_desc.back().subTensorDesc());
          flat_output_nodes.emplace_back(c, desc);
        }
      }
    }
  }

  // Acquires input values
  bool has_half_tensor = false;
  size_t formal_count = 0;
  for (const auto input : flat_inputs) {
    auto p = input.first;
    env.s("node", valueName(p));
    env.d("formal", formal_count++);

    // Acquires and converts (if needed) inputs
    // Note: conversion from half is only supported for CUDA kernels.
    //  The conversion immediately converts fp16 inputs to float.
    //  Access for other types is common to CUDA and CPU kernels.
    const auto is_half = (input.second.scalar_type == at::ScalarType::Half);
    if (is_half) {
      JIT_ASSERT(use_cuda);
      env.s(
        "access"
      , format("__half2float(t${formal}.data[t${formal}_offset])", env));
      has_half_tensor = true;
    } else {
      env.s("access", format("t${formal}.data[t${formal}_offset]", env));
    }

    //TODO: actual type propagation rather than relying on auto..
    body << format("auto ${node} = ${access};\n", env);
  }

  bool has_random = false;
  // Generates code for intermediate nodes
  // Note: Concat and Chunk are implicitly generated
  // Note: Random number generation is only supported for CUDA kernels. 
  for (const auto& n : graph.nodes()) {
    // Note: FusedConcat nodes work by narrowing the output Tensors before the kernel runs
    if (n->kind() == prim::FusedConcat) continue;
    if (n->kind() == prim::ConstantChunk) continue;
    if (n->kind() == aten::rand_like) {
      JIT_ASSERT(use_cuda);
      has_random = true;
    }
    env.s("node", valueName(n->output()));
    env.s("rhs", encodeRHS(n));
    body << format("auto ${node} = ${rhs};\n",env);
  }

  // Generates writes to output tensors
  for (const auto& output : flat_output_nodes) {
    const auto& o = output.first;
    env.d("formal", formal_count++);
    env.s("access", format("t${formal}.data[t${formal}_offset]",env));
    env.s("node", valueName(o));

    // Acquires and converts (if needed) outputs
    // Note: conversion to half is only supported for CUDA kernels.
    const auto is_half = (output.second.scalar_type == at::ScalarType::Half);
    if (is_half) {
      JIT_ASSERT(use_cuda);
      body << format("${access} = __float2half(${node});\n",env);
      has_half_tensor = true;
    } else {
      body << format("${access} = ${node};\n",env);
    }
  }

  // Includes headers
  // Note: CUDA kernels support halfs and random generation, CPU kernels do not
  #if USE_CUDA_FUSER
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
  #endif // USE_CUDA_FUSER

  // Insantiates the CUDA or CPU-specific templates
  env.s("tensorOffsets", tensorOffsets.str());
  env.s("kernelBody", body.str());
  env.v("formals", formals);
  env.v("argument_loads", argument_loads);
  std::string code_string;
  if (use_cuda) {
    #if USE_CUDA_FUSER
      env.s("type_declarations", cuda::type_declarations_template.format(env));
      code_string = cuda::cuda_compilation_unit_template.format(env);
    #else
      throw std::runtime_error("CUDA Fusion requested but not supported.");
    #endif // USE_CUDA_FUSER
  } else {
    #if USE_CPU_FUSER
      env.s("type_declarations", cpu::type_declarations_template.format(env));
      code_string = cpu::cpu_compilation_unit_template.format(env);
    #else
      throw std::runtime_error("CPU Fusion requested but not supported");
    #endif // USE_CPU_FUSER
  }

  return std::make_tuple(code_string, std::move(chunk_desc), std::move(concat_desc), has_random);
}

} // namespace fuser
} // namespace jit
} // namespace torch
