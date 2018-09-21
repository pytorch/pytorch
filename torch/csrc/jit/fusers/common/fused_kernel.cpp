#include "torch/csrc/jit/fusers/common/fused_kernel.h"

#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/cpu/resource_strings.h"
#include "torch/csrc/jit/fusers/cuda/resource_strings.h"
#include "torch/csrc/jit/fusers/common/partition_desc.h"
#include "torch/csrc/jit/fusers/common/tensor_desc.h"
#include "torch/csrc/jit/fusers/common/tensor_info.h"

#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/assertions.h"

#include "ATen/ATen.h"

#if USE_CUDA_FUSER
  #include "THC/THCTensorRandom.h"
  #include "THC/THCGenerator.hpp"
  THCGenerator* THCRandom_getGenerator(THCState* state);
#endif // USE_CUDA_FUSER

#include <tuple>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <vector>

namespace torch { namespace jit {

// curDimIndex = linearId % sizes[i]; // % sizes[i] is not needed for d == 0, because we already guard for numel outside the index calculation
// offset += curDimIndex*strides[i]; // *strides[i] is optional if list_is_cont becaause strides.back() == 1
// linearId /= sizes[i];
auto dim_calc = CodeTemplate(R"(
//printf("tensor ${tensor} sizes[${d}] = %d, strides[${d}] = %d\n", ${tensor}.sizes[${d}],${tensor}.strides[${d}]);
size_t ${tensor}_dimIndex${d} = ${tensor}_linearIndex ${mod_sizes};
${tensor}_offset += ${tensor}_dimIndex${d} ${times_stride};
)");

// XXX: this code assumes that inputs are 32-bit addressable
static uint32_t computeNumel(at::ArrayRef<int64_t> sizes) {
  uint32_t result = 1;
  if (sizes.size() == 0) {
    return 1; // scalar tensor
  }
  for (int64_t size : sizes) {
    result *= size;
  }
  return result;
}

// XXX: Assumes that after at::chunk, all inputs are the same size
static std::vector<int64_t> computeMapSize(
    const at::Tensor& tensor,
    const PartitionDesc& chunkDesc) {
  std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
  // Should have been checked in graph fuser
  JIT_ASSERT(sizes[chunkDesc.dim] % chunkDesc.nSubtensors == 0);
  sizes[chunkDesc.dim] /= chunkDesc.nSubtensors;
  return sizes;
}

// Tries to compress sizes and strides according to cont. Emits the result t
// c_sizes, c_strides and throws an error on failure (if can't compress)
static void compressContiguous(
  at::IntList sizes
, at::IntList strides
, const std::vector<bool> & cont
, uint32_t* c_sizes
, uint32_t* c_strides) {
  size_t compressed_dims = 0;
  size_t cur = 0;
  size_t ndim = sizes.size();
  while (cur < ndim) {
    size_t total_size = sizes[cur];
    cur++;
    while (cont[cur-1] && cur < ndim) {
      JIT_ASSERT(strides[cur-1] == sizes[cur]*strides[cur]);
      total_size *= sizes[cur];
      cur++;
    }
   // cur starts pointing at the beginning of run to compress
   // cur ends one _after_ the terminating false or end of list.
   // total_size is the size of all dimensions [begin,end)
   // examples:
   // f = not cont.
   // t = cont.
   // x = don't care, including past end of list
   // s = start of cur
   // e = end of cur


   // f x x x
   // s e

   //  t f x x
   //  s   e

   //  t t f x
   //  s     e

    c_sizes[compressed_dims] = total_size;
    c_strides[compressed_dims] = strides[cur-1];
    compressed_dims++;
  }
  if (ndim > 0) {
    JIT_ASSERT(!cont.back() || strides.back() == 1);
  }
}

void FusedKernel::launch_with_tensors(
  at::ArrayRef<at::Tensor> inputs
, at::ArrayRef<at::Tensor> outputs) {
  at::DeviceGuard device_guard(inputs);
  JIT_ASSERT(inputs.size() == input_desc.size());
  JIT_ASSERT(outputs.size() == output_desc.size());
  size_t flat_inputs_size = 0;
  size_t flat_outputs_size = 0;
  for (auto& c : chunk_desc)
    flat_inputs_size += c.nSubtensors;
  for (auto& c : concat_desc)
    flat_outputs_size += c.nSubtensors;
  // XXX: this code assumes that inputs are 32-bit addressable
  // XXX: this code assumes that all inputs are of the same size
  JIT_ASSERT(inputs[0].numel() <= std::numeric_limits<uint32_t>::max());

  // Compute map_size, numel from the first input
  at::IntList map_size;
  uint32_t numel;
  std::vector<int64_t> keep_alive_size;
  if (chunk_desc[0].isNoop()) {
    map_size = inputs[0].sizes();
    numel = inputs[0].numel();
  } else {
    keep_alive_size = computeMapSize(inputs[0], chunk_desc[0]);
    map_size = keep_alive_size;
    numel = computeNumel(map_size);
  }

  // Compute the storage needed to store TensorInfo structs for inputs and outputs.
  size_t uncompressedDim = input_desc.at(0).contiguity.size();
  size_t maxPossibleTensorInfoSize = sizeof(TensorInfo) + 2 * sizeof(uint32_t) * uncompressedDim;
  size_t maxPossibleBufferSize = maxPossibleTensorInfoSize * (flat_inputs_size + flat_outputs_size);
  std::vector<char> buffer(maxPossibleBufferSize);
  char* buffer_next = buffer.data();
  // A vector of arguments to the kernel. It's (numel, *input_descs, *output_descs)
  std::vector<void*> arguments;
  arguments.reserve(3 + flat_inputs_size + flat_outputs_size);
  auto addTensorInfoRaw = [&](TensorDesc & desc, void* data_ptr, at::IntList sizes, at::IntList strides) {
    size_t nDim = desc.nDim(); // NOTE: this is the compressed dim
    JIT_ASSERT(nDim <= uncompressedDim); // We'd overflow the space otherwise
    auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
    ti->data = data_ptr;
    compressContiguous(sizes, strides, desc.contiguity, ti->sizes(nDim), ti->strides(nDim));
    buffer_next += maxPossibleTensorInfoSize;
    arguments.push_back(ti);
  };
  // Asserts that t's dims can be compressed in the same way as in desc
  // (that's what the kernel assumes), and appends it to the arguments vector.
  auto addTensorInfo = [&](TensorDesc & desc, const at::Tensor & t) {
    addTensorInfoRaw(desc, t.data_ptr(), t.sizes(), t.strides());
  };
  arguments.push_back(&numel);
  for (size_t i = 0; i < input_desc.size(); ++i) {
    auto & chunk = chunk_desc[i];
    const at::Tensor& tensor = inputs[i];
    if (chunk.isNoop()) {
      addTensorInfo(input_desc[i], tensor);
    } else {
      size_t chunk_offset = map_size[chunk.dim] * tensor.stride(chunk.dim) * elementSize(tensor.type().scalarType());
      char * data_ptr = reinterpret_cast<char*>(tensor.data_ptr());
      for (size_t chunks = 0; chunks < chunk.nSubtensors; ++chunks) {
        addTensorInfoRaw(*chunk.subtensorDesc, data_ptr, map_size, tensor.strides());
        data_ptr += chunk_offset;
      }
    }
  }
  for (size_t i = 0; i < output_desc.size(); ++i) {
    auto & c = concat_desc[i];
    at::Tensor o = outputs[i];
    if (c.isNoop()) {
      o.resize_(map_size);
      addTensorInfo(output_desc[i], outputs[i]);
    } else {
      size_t small_size = map_size[c.dim];
      std::vector<int64_t> concat_size(map_size.begin(), map_size.end());
      concat_size[c.dim] = small_size * c.nSubtensors;
      o.resize_(concat_size);
      size_t offset = 0;
      for(size_t j = 0; j < c.nSubtensors; ++j) {
        // because the concatenated_output stays live, the underlying data
        // in this view remains live through the end of this function
        // so there is not need to hold onto this tensor
        auto view = o.narrow(c.dim, offset, small_size);
        addTensorInfo(*c.subtensorDesc, view);
        offset += small_size;
      }
    }
  }

  // If the kernel call contains a random op, we need to pass in random seeds as
  // well.
  #if USE_CUDA_FUSER
    if (has_random && this->backend() == at::Backend::CUDA) {
      auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
      uint64_t offset =
          gen_->state.philox_seed_offset.fetch_add(this->get_rand_offset(numel));
      arguments.push_back(&gen_->state.initial_seed);
      arguments.push_back(&offset);
    }
  #endif // USE_CUDA_FUSER
  
  launch_raw(numel, arguments.data());
}

void FusedKernel::launch(
  at::ArrayRef<at::Tensor> inputs
, std::vector<at::Tensor> & outputs) {
  at::DeviceGuard guard(inputs.back());
  JIT_ASSERT(inputs.size() > 0);
  auto & ref_type = inputs[0].type();
  outputs.clear();
  outputs.reserve(outputDescriptors().size());
  for(auto & od : outputDescriptors()) {
    outputs.push_back(ref_type.toScalarType(od.scalar_type).tensor());
  }

  launch_with_tensors(inputs, outputs);
}

static std::string valueName(Value * n) {
  return "n" + std::to_string(n->unique());
}

static std::string scalarValue(int64_t v) {
  return std::to_string(v);
}

static std::string scalarValue(double v) {
  std::ostringstream out;
  out << std::scientific << v << "f";
  return out.str();
}

static const char * scalarTypeName(at::ScalarType type) {
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

static std::string encodeRHS(Node* n) {
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
    auto val = toIValue(n->output()).value();
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

static Node* usedInFusedChunk(Value* input) {
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
, int ndim
, bool last_is_cont) {
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

// Returns: (input chunk metadata, output concat metadata, is_random)
std::tuple<
    std::vector<PartitionDesc>
  , std::vector<PartitionDesc>
  , bool> 
  emitCompilationUnit(
    std::ostream& out
  , const std::string& name
  , AnnotatedGraph& agraph
  , bool use_cuda) {
  bool has_random = false;
  Graph& subgraph = *agraph.graph;
  TemplateEnv env;
  env.s("kernelName", name);
  // TODO: handle cases where we need to generate > 2^32 element tensors
  env.s("IndexType","unsigned int"); //avoiding slow header includes to get uint32_t

  std::stringstream body;
  std::stringstream tensorOffsets;
  std::vector<std::string> formals;
  std::vector<std::string> argument_loads;
  auto emitFormal = [&](Value * n, const TensorDesc & desc) {
    std::string tensor = "t" + std::to_string(formals.size()); //can't be unique() because Param may be an output
    size_t nDim = desc.nDim();
    emitIndexingFor(tensorOffsets, tensor, nDim,  desc.lastIsContiguous());
    env.s("tensor",tensor);
    env.d("formal_index", formals.size() + 1); // + 1 because the first argument is the linearIndex
    env.d("nDim",nDim);
    env.s("scalar_type",scalarTypeName(desc.scalar_type));
    formals.push_back(format("TensorInfo<${scalar_type},${nDim}> ${tensor}", env));
    argument_loads.push_back(format("*static_cast<TensorInfo<${scalar_type},${nDim}>*>(args[${formal_index}])", env));
  };

  std::vector<PartitionDesc> chunk_desc;
  std::vector<std::pair<Value*,TensorDesc&>> flat_inputs;
  {
    size_t input_index = 0;
    for(auto p : subgraph.inputs()) {
      if (Node * chunk = usedInFusedChunk(p)) {
        int64_t dim = chunk->i(attr::dim);
        int64_t chunks = chunk->i(attr::chunks);
        chunk_desc.emplace_back(agraph.input_desc[input_index++], chunks, dim);
        for (auto * o : chunk->outputs()) {
          flat_inputs.emplace_back(o, *chunk_desc.back().subtensorDesc);
        }
      } else {
        chunk_desc.emplace_back();
        flat_inputs.emplace_back(p, agraph.input_desc[input_index++]);
      }
    }
    for (auto & input : flat_inputs) {
      emitFormal(input.first, input.second);
    }
  }

  std::vector<PartitionDesc> concat_desc;
  std::vector<std::pair<Value*,TensorDesc>> flat_output_nodes;
  {
    size_t i = 0;
    for(auto o : subgraph.outputs()) {
      auto & desc = agraph.output_desc[i++];
      if(o->node()->kind() != prim::FusedConcat) {
        emitFormal(o, desc);
        concat_desc.emplace_back();
        flat_output_nodes.emplace_back(o, desc);
      } else {
        auto cat = o->node();
        concat_desc.emplace_back(desc, cat->inputs().size(), cat->i(attr::dim));
        for(auto c : cat->inputs()) {
          emitFormal(c, *concat_desc.back().subtensorDesc);
          flat_output_nodes.emplace_back(c, desc);
        }
      }
    }
  }

  #if USE_CUDA_FUSER
    bool has_half_tensor = false;
  #endif // USE_CUDA_FUSER
  size_t formal_count = 0;
  for(auto input : flat_inputs) {
    auto p = input.first;
    env.s("node", valueName(p));
    env.d("formal", formal_count++);

    // Acquires and converts (if needed) inputs
    bool is_half = input.second.scalar_type == at::ScalarType::Half;
    if (is_half) {
      AT_ASSERT(use_cuda);
      #if USE_CUDA_FUSER
        env.s(
          "access"
        , format("__half2float(t${formal}.data[t${formal}_offset])", env));
        has_half_tensor = true;
      #endif // USE_CUDA_FUSER
    } else {
      env.s("access", format("t${formal}.data[t${formal}_offset]", env));
    }

    //TODO: actual type propagation rather than relying on auto..
    body << format("auto ${node} = ${access};\n", env);
  }

  for (auto n : subgraph.nodes()) {
    // FusedConcat nodes work by narrowing the output Tensors before the kernel runs
    if (n->kind() == prim::FusedConcat)
      continue;
    if (n->kind() == prim::ConstantChunk)
      continue;
    if (n->kind() == aten::rand_like) {
      has_random = true;
      if (!use_cuda)
        throw std::runtime_error("Fusion doesn't support rand on CPU");
    }
    env.s("node",valueName(n->output()));
    env.s("rhs", encodeRHS(n));
    body << format("auto ${node} = ${rhs};\n",env);
  }

  for (auto output : flat_output_nodes) {
    auto o = output.first;
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    env.s("node",valueName(o));

    // Acquires and converts (if needed) outputs
    bool is_half = output.second.scalar_type == at::ScalarType::Half;
    if (is_half) {
      AT_ASSERT(use_cuda);
      #if USE_CUDA_FUSER
        body << format("${access} = __float2half(${node});\n",env);
        has_half_tensor = true;
      #endif // USE_CUDA_FUSER
    } else {
      body << format("${access} = ${node};\n",env);
    }
  }

  // Includes half support if any half tensors are involved
  #if USE_CUDA_FUSER
    if (has_half_tensor) {
      env.s("HalfHeader", cudafuser::half_support_literal);
    } else {
      env.s("HalfHeader", "");
    }

    if (has_random) {
      env.s("RandHeader", cudafuser::rand_support_literal);
      env.s("RandParam", cudafuser::rand_param);
      env.s("RandInit", cudafuser::rand_init);
    } else {
      env.s("RandHeader", "");
      env.s("RandParam", "");
      env.s("RandInit", "");
    }
  #endif // USE_CUDA_FUSER

  env.s("tensorOffsets", tensorOffsets.str());
  env.s("kernelBody", body.str());
  env.v("formals", formals);
  env.v("argument_loads", argument_loads);
  if (use_cuda) {
    #if USE_CUDA_FUSER
      env.s("type_declarations", cudafuser::type_declarations_template.format(env));
      out << cudafuser::cuda_compilation_unit_template.format(env);
    #else
      throw std::runtime_error("CUDA Fusion requested but not supported.");
    #endif // USE_CUDA_FUSER
  } else {
    env.s("type_declarations", cpufuser::type_declarations_template.format(env));
    out << cpufuser::cpu_compilation_unit_template.format(env);
  }

  return std::make_tuple(std::move(chunk_desc), std::move(concat_desc), has_random);
}

} // namespace jit
} // namespace torch
