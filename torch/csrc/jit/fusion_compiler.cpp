#ifndef _WIN32
#include "torch/csrc/jit/fusion_compiler.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/custom_operator.h"

#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/variable_tensor_functions.h"
#include "torch/csrc/utils/hash.h"
#include <torch/csrc/jit/assertions.h>

#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"
#include "ATen/WrapDimUtils.h"

#ifdef USE_CUDA
#include "ATen/cuda/CUDAContext.h"
#include "THC/THC.h"
#include <THC/THCGenerator.hpp>
#include "torch/csrc/cuda/cuda_check.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <iostream>
#include <dlfcn.h>
#include <unistd.h>

#ifdef USE_CUDA
THCGenerator* THCRandom_getGenerator(THCState* state);
#endif

namespace torch { namespace jit {

std::vector<bool> TensorDesc::findContiguous(
    const at::IntList& sizes,
    const at::IntList& strides) {
  JIT_ASSERT(sizes.size() == strides.size());
  std::vector<bool> cont(sizes.size());
  for(size_t i = 0; i < sizes.size(); ++i) {
    int64_t expected_stride = (i + 1 < sizes.size()) ? sizes[i+1]*strides[i+1] : 1;
    cont[i] = strides[i] == expected_stride;
  }
  return cont;
}

// Descriptor for chunk-ing an input tensor into subtensors
// OR concat-ing an output tensor from subtensors
struct PartitionDesc {
  size_t nSubtensors; // == 1 for tensors that should not be operated on via chunk/cat
  size_t dim; // dimension along which the chunk/concat occurs
  std::unique_ptr<TensorDesc> subtensorDesc; // descriptor for the subtensor, if it exists
  PartitionDesc()
  : nSubtensors(1), dim(0) {}

  PartitionDesc(const TensorDesc & desc, size_t nSubtensors, size_t dim)
  : nSubtensors(nSubtensors), dim(dim) {
    JIT_ASSERT(nSubtensors > 1);
    std::vector<bool> cont = desc.contiguity;
    if(dim > 0) {
      // when we narrow the concatenated output/chunked input
      // we make the size[dim] smaller while keeping the stride[dim] the same,
      // meaning: stride[dim - 1] != stride[dim]*size[dim]
      // so dim - 1 is no longer contiguous
      cont[dim - 1] = false;
    }
    subtensorDesc.reset(new TensorDesc(desc.scalar_type, cont));
  }

  bool isNoop() const {
    return nSubtensors == 1;
  }
};

struct FusedKernel {
  TH_DISALLOW_COPY_AND_ASSIGN(FusedKernel);

  FusedKernel(const std::string & name, AnnotatedGraph & agraph);
  virtual ~FusedKernel() = default;

  // expects outputs to be pre-allocated
  void launch_with_tensors(at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs);

  // creates new tensors for outputs
  void launch(at::ArrayRef<at::Tensor> inputs, std::vector<at::Tensor> & outputs);
  const std::vector<TensorDesc> & outputDescriptors() const {
    return output_desc;
  }
protected:
  virtual at::Backend backend() const = 0;

  // arguments is a list of pointers to the arguments for the compiled CUDA/CPU
  // code.
  // The format of arguments is suitable for directly passing to a call to
  // cuLaunchKernel as the kernel arguments.
  // Currently the first argument is a pointer to numel (for passing to
  // CUDA code), and the remainder are pointers to the TensorInfo<T> structs
  // that compiled code uses to load Tensor data.
  // launch_with_tensors handles packing at::Tensors into this arguments array.
  // CPU code uses the same convension so that launch_with_tensors can be shared.
  virtual void launch_raw(uint32_t numel, void ** arguments) = 0;

  virtual uint64_t get_rand_offset(uint32_t numel) = 0;
  bool has_random;
  std::string name;
  // We keep these around for debugging
  std::string compilation_unit;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;

  // same size as output_desc, describes whether
  // an output is actually a concatenation of
  // many subtensors that the fusion group produces
  std::vector<PartitionDesc> concat_desc;

  // same size as input_desc, describes whether an
  // input should be broken into subtensors (chunks)
  // to be consumed by the fusion group
  std::vector<PartitionDesc> chunk_desc;
};


namespace {

#ifdef USE_CUDA

static int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

#endif

Node* usedInFusedChunk(Value * input) {
  auto uses = input->uses();
  if (uses.size() == 1) {
    Node *user = uses[0].user;
    if (user->kind() == prim::ConstantChunk) {
      return user;
    }
  }
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
// Code generation

namespace codegen {

/*with type_as not checking type of its input, a fusion group can have non-fp32 tensor as input.
Correct code for this case is generated, however, nvrtc does not know how to handle int*_t integer types,
so typedefs help it handle those cases*/

auto type_declarations_template = CodeTemplate(R"(
#if defined(__CUDACC_RTC__)
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int  int16_t;
typedef long long int int64_t;
${HalfHeader}
${RandHeader}
#endif
typedef ${IndexType} IndexType;
template<typename T, size_t N>
struct TensorInfo {
  T * data;
  IndexType sizes[N];
  IndexType strides[N];
};
template<typename T>
struct TensorInfo<T, 0> {
  T * data;
};
)");

// We rewrite the code for philox RNG from curand as nvrtc couldn't resolve the
// curand header correctly.
constexpr auto rand_support_literal = R"(

  class Philox {
  public:
    __device__ inline Philox(unsigned long long seed,
                             unsigned long long subsequence,
                             unsigned long long offset) {
      key.x = (unsigned int)seed;
      key.y = (unsigned int)(seed >> 32);
      counter = make_uint4(0, 0, 0, 0);
      counter.z = (unsigned int)(subsequence);
      counter.w = (unsigned int)(subsequence >> 32);
      STATE = 0;
      incr_n(offset / 4);
    }

    __device__ inline unsigned long operator()() {
      if(STATE == 0) {
        uint4 counter_ = counter;
        uint2 key_ = key;
        for(int i = 0; i < 9; i++) {
          counter_ = single_round(counter_, key_);
          key_.x += (kPhilox10A); key_.y += (kPhilox10B);
        }
        output = single_round(counter_, key_);
        incr();
      }
      unsigned long ret;
      switch(STATE) {
        case 0: ret = output.x; break;
        case 1: ret = output.y; break;
        case 2: ret = output.z; break;
        case 3: ret = output.w; break;
      }
      STATE = (STATE + 1) % 4;
      return ret;
    }

  private:
    uint4 counter;
    uint4 output;
    uint2 key;
    unsigned int STATE;
    __device__ inline void incr_n(unsigned long long n) {
      unsigned int nlo = (unsigned int)(n);
      unsigned int nhi = (unsigned int)(n >> 32);
      counter.x += nlo;
      if (counter.x < nlo)
        nhi++;
      counter.y += nhi;
      if (nhi <= counter.y)
        return;
      if (++counter.z)
        return;
      ++counter.w;
    }
    __device__ inline void incr() {
      if (++counter.x)
        return;
      if (++counter.y)
        return;
      if (++counter.z)
        return;
      ++counter.w;
    }
    __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                      unsigned int *result_high) {
      *result_high = __umulhi(a, b);
      return a*b;
    }

    __device__ inline uint4 single_round(uint4 ctr, uint2 key) {
      unsigned int hi0;
      unsigned int hi1;
      unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
      unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);

      uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
      return ret;
    }

    static const unsigned long kPhilox10A = 0x9E3779B9;
    static const unsigned long kPhilox10B = 0xBB67AE85;
    static const unsigned long kPhiloxSA = 0xD2511F53;
    static const unsigned long kPhiloxSB = 0xCD9E8D57;
  };

  // Inverse of 2^32.
  #define M_RAN_INVM32 2.3283064e-10f
  __device__  __inline__ float uniform(unsigned int x) {
    return x * M_RAN_INVM32;
  }
)";

constexpr auto rand_param = ",unsigned long long seed, unsigned long long offset";
constexpr auto rand_init = R"(
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Philox rnd(seed, idx, offset);
)";
auto cuda_compilation_unit_template = CodeTemplate(R"(
${type_declarations}

extern "C" __global__
void ${kernelName}(IndexType totalElements, ${formals} ${RandParam}) {
  ${RandInit}
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
        linearIndex < totalElements;
        linearIndex += gridDim.x * blockDim.x) {
      // Convert `linearIndex` into an offset of tensor:
      ${tensorOffsets}
      // calculate the results
      ${kernelBody}
    }
}
)");

auto cpu_compilation_unit_template = CodeTemplate(R"(
#include <cstddef>
#include <cstdint>
#include <math.h>
${type_declarations}

#define OMP_THRESHOLD 100000
static void ${kernelName}_kernel(IndexType totalElements, ${formals}) {
  #pragma omp parallel for if(totalElements > OMP_THRESHOLD)
  for (IndexType linearIndex = 0;
        linearIndex < totalElements;
        linearIndex += 1) {
      // Convert `linearIndex` into an offset of tensor:
      ${tensorOffsets}
      // calculate the results
      ${kernelBody}
    }
}

extern "C"
void ${kernelName}(IndexType totalElements, void ** args) {
  ${kernelName}_kernel(totalElements ${,argument_loads});
}
)");

// This snippet enables half support in the jit. Following the pattern for
// reductions, fp16 input data is immediately upconverted to float
// with __half2float(). All mathematical operations are done on float
// values, and if needed the intermediate float representation is
// converted to half with __float2half() when writing to a half tensor.
constexpr auto half_support_literal  = R"(
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#if defined(__cplusplus)
  struct __align__(2) __half {
    __host__ __device__ __half() { }

  protected:
    unsigned short __x;
  };

  /* All intrinsic functions are only available to nvcc compilers */
  #if defined(__CUDACC__)
    /* Definitions of intrinsics */
    __device__ __half __float2half(const float f) {
      __half val;
      asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
      return val;
    }

    __device__ float __half2float(const __half h) {
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
      return val;
    }
  #endif /* defined(__CUDACC__) */
#endif /* defined(__cplusplus) */
#undef __HALF_TO_US
#undef __HALF_TO_CUS

typedef __half half;
)";

// curDimIndex = linearId % sizes[i]; // % sizes[i] is not needed for d == 0, because we already guard for numel outside the index calculation
// offset += curDimIndex*strides[i]; // *strides[i] is optional if list_is_cont becaause strides.back() == 1
// linearId /= sizes[i];
auto dim_calc = CodeTemplate(R"(
//printf("tensor ${tensor} sizes[${d}] = %d, strides[${d}] = %d\n", ${tensor}.sizes[${d}],${tensor}.strides[${d}]);
size_t ${tensor}_dimIndex${d} = ${tensor}_linearIndex ${mod_sizes};
${tensor}_offset += ${tensor}_dimIndex${d} ${times_stride};
)");

static void emitIndexingFor(std::ostream & out, const std::string & tensor, int ndim, bool last_is_cont) {
  TemplateEnv env;
  env.s("tensor",tensor);
  out << format("IndexType ${tensor}_offset = 0;\n",env);
  out << format("IndexType ${tensor}_linearIndex = linearIndex;\n",env);
  for(int d = ndim - 1; d >= 0; --d) {
    env.d("d",d);
    env.s("mod_sizes", d > 0 ? format("% ${tensor}.sizes[${d}]",env) : "");
    env.s("times_stride",(d < ndim - 1 || !last_is_cont) ?
      format("* ${tensor}.strides[${d}]",env) : "");
    out << dim_calc.format(env);
    if(d > 0) {
      out << format("${tensor}_linearIndex /= ${tensor}.sizes[${d}];\n",env);
    }
  }
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

std::string encodeRHS(Node * n) {
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

// Returns: (input chunk metadata, output concat metadata, is_random)
std::tuple<std::vector<PartitionDesc>,std::vector<PartitionDesc>,bool> emitCompilationUnit(
    std::ostream& out,
    const std::string& name,
    AnnotatedGraph& agraph,
    bool use_cuda) {
  bool has_random = false;
  Graph& subgraph = *agraph.graph;
  TemplateEnv env;
  env.s("kernelName",name);
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
    formals.push_back(format("TensorInfo<${scalar_type},${nDim}> ${tensor}",env));
    argument_loads.push_back(format("*static_cast<TensorInfo<${scalar_type},${nDim}>*>(args[${formal_index}])",env));
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

  bool has_half_tensor = false;
  size_t formal_count = 0;
  for(auto input : flat_inputs) {
    auto p = input.first;
    env.s("node",valueName(p));
    env.d("formal",formal_count++);

    // Acquires and converts (if needed) inputs
    bool is_half = input.second.scalar_type == at::ScalarType::Half;
    if (is_half) {
      AT_ASSERT(use_cuda);
      env.s(
        "access"
      , format("__half2float(t${formal}.data[t${formal}_offset])", env));
      has_half_tensor = true;
    } else {
      env.s("access", format("t${formal}.data[t${formal}_offset]", env));
    }

    //TODO: actual type propagation rather than relying on auto..
    body << format("auto ${node} = ${access};\n",env);
  }

  for(auto n : subgraph.nodes()) {
    // FusedConcat nodes work by narrowing the output Tensors before the kernel runs
    if (n->kind() == prim::FusedConcat)
      continue;
    if (n->kind() == prim::ConstantChunk)
      continue;
    if(n->kind() == aten::rand_like) {
      has_random = true;
      if(!use_cuda)
        throw std::runtime_error("Fusion doesn't support rand on CPU");
    }
    env.s("node",valueName(n->output()));
    env.s("rhs", encodeRHS(n));
    body << format("auto ${node} = ${rhs};\n",env);
  }

  for(auto output : flat_output_nodes) {
    auto o = output.first;
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    env.s("node",valueName(o));

    // Acquires and converts (if needed) outputs
    bool is_half = output.second.scalar_type == at::ScalarType::Half;
    if (is_half) {
      AT_ASSERT(use_cuda);
      body << format("${access} = __float2half(${node});\n",env);
      has_half_tensor = true;
    } else {
      body << format("${access} = ${node};\n",env);
    }
  }

  // Includes half support if any half tensors are involved
  if (has_half_tensor) {
    env.s("HalfHeader", half_support_literal);
  } else {
    env.s("HalfHeader", "");
  }

  if (has_random) {
    env.s("RandHeader", rand_support_literal);
    env.s("RandParam", rand_param);
    env.s("RandInit", rand_init);
  } else {
    env.s("RandHeader", "");
    env.s("RandParam", "");
    env.s("RandInit", "");
  }

  env.s("tensorOffsets",tensorOffsets.str());
  env.s("kernelBody",body.str());
  env.v("formals",formals);
  env.v("argument_loads",argument_loads);
  env.s("type_declarations", type_declarations_template.format(env));
  if(use_cuda) {
    out << cuda_compilation_unit_template.format(env);
  } else {
    out << cpu_compilation_unit_template.format(env);
  }

  return std::make_tuple(std::move(chunk_desc), std::move(concat_desc), has_random);
}

////////////////////////////////////////////////////////////////////////////////

} // codegen namespace
} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////
// CompiledFunctionFunction

// Host-side view of TensorInfo (that visivle for the kernel is defined above).
// Note dims[0] - we need to dynamically allocate the dims.
struct TensorInfo {
  void * data;
#pragma GCC diagnostic ignored "-Wpedantic"
  uint32_t sizes_strides[0];
#pragma GCC diagnostic pop

  uint32_t* sizes(size_t nDim) { return &sizes_strides[0]; }
  uint32_t* strides(size_t nDim) { return &sizes_strides[nDim]; }
};

FusedKernel::FusedKernel(const std::string & name, AnnotatedGraph & agraph)
  : name(name)
  , input_desc(agraph.input_desc)
  , output_desc(agraph.output_desc) {}

namespace {

// Tries to compress sizes and strides according to cont. Emits the result t
// c_sizes, c_strides and throws an error on failure (if can't compress)
void compressContiguous(
    at::IntList sizes,
    at::IntList strides,
    const std::vector<bool> & cont,
    uint32_t * c_sizes,
    uint32_t * c_strides) {
  size_t compressed_dims = 0;
  size_t cur = 0;
  size_t ndim = sizes.size();
  while(cur < ndim) {
    size_t total_size = sizes[cur];
    cur++;
    while(cont[cur-1] && cur < ndim) {
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

} // anonymous namespace

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

void FusedKernel::launch_with_tensors(at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs) {
  at::DeviceGuard device_guard(inputs);
  JIT_ASSERT(inputs.size() == input_desc.size());
  JIT_ASSERT(outputs.size() == output_desc.size());
  size_t flat_inputs_size = 0;
  size_t flat_outputs_size = 0;
  for(auto & c : chunk_desc)
    flat_inputs_size += c.nSubtensors;
  for(auto & c : concat_desc)
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
  char * buffer_next = buffer.data();
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
    if(c.isNoop()) {
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
  #ifdef USE_CUDA
  if(has_random && this->backend() == at::Backend::CUDA) {
    auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
    uint64_t offset =
        gen_->state.philox_seed_offset.fetch_add(this->get_rand_offset(numel));
    arguments.push_back(&gen_->state.initial_seed);
    arguments.push_back(&offset);
  }
  #endif

  launch_raw(numel, arguments.data());
}

void FusedKernel::launch(at::ArrayRef<at::Tensor> inputs, std::vector<at::Tensor> & outputs) {
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

////////////////////////////////////////////////////////////////////////////////
// CUDAFusedKernel

#ifdef USE_CUDA

void checkCUDAVersion(const cudaDeviceProp & prop) {
  if ((prop.major >= 6 && CUDA_VERSION < 8000) ||
      (prop.major >= 7 && CUDA_VERSION < 9000)) {
    std::stringstream err_string;
    err_string << "In CUDAFusedKernel, PyTorch compiled with insufficient CUDA version: "
         << CUDA_VERSION << " for the current GPU device " << prop.name
         << " with device capability " << prop.major << "." << prop.minor;
    throw std::runtime_error(err_string.str());
  }
}

struct CUDAFusedKernel : public FusedKernel {
  CUDAFusedKernel(const std::string & name, AnnotatedGraph & agraph)
  : FusedKernel(name, agraph) {
    at::DeviceGuard device_guard(agraph.device);

    TORCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, agraph.device));
    checkCUDAVersion(prop);

    std::stringstream cu;
    std::tie(chunk_desc, concat_desc, has_random) = codegen::emitCompilationUnit(cu, name, agraph, true);
    compilation_unit = cu.str();
    nvrtcProgram program;
    TORCH_NVRTC_CHECK(nvrtcCreateProgram(&program, compilation_unit.c_str(), nullptr, 0, nullptr, nullptr));

    std::string compute = "--gpu-architecture=compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
    std::vector<const char *> args = {"--std=c++11", compute.c_str(), "-default-device"};
    nvrtcResult result = nvrtcCompileProgram(program, args.size(), args.data());
    if (result == NVRTC_ERROR_COMPILATION) {
      size_t logsize;
      nvrtcGetProgramLogSize(program, &logsize);
      std::vector<char> log(logsize);
      nvrtcGetProgramLog(program, log.data());
      cu << log.data();
      throw std::runtime_error(cu.str());
    }
    ResourceGuard holdProgram([&] {
      TORCH_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    });
    TORCH_NVRTC_CHECK(result);

    size_t ptx_size;
    TORCH_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
    ptx.resize(ptx_size);
    TORCH_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
//need an active context for cuModuleLoadData
    CUcontext pctx = 0;
    TORCH_CU_CHECK(cuCtxGetCurrent(&pctx));
    if (!pctx) {
       std::unique_lock<std::mutex> cudaFreeMutexLock(
           *(THCCachingAllocator_getCudaFreeMutex()));
       cudaFree(0);
    }
    TORCH_CU_CHECK(cuModuleLoadData(&module, ptx.data()));
    TORCH_CU_CHECK(cuModuleGetFunction(&function, module, name.c_str()));

    TORCH_CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks, function, 128, 0));
    maxBlocks *= prop.multiProcessorCount;
  }
  virtual ~CUDAFusedKernel() override {
    TORCH_CU_CHECK(cuModuleUnload(module));
  }
protected:
  virtual at::Backend backend() const override {
    return at::Backend::CUDA;
  }
  virtual uint64_t get_rand_offset(uint32_t numel) override {
     int numBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));
     return 4 * (ceil(numel/(4 * blockSize * numBlocks)) + 1);
  }
  virtual void launch_raw(uint32_t numel, void ** arguments) override {
     int numBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));

     //std::cout << "maxBlocks = " << maxBlocks << " needed blocks: " << ceilDiv(numel,blockSize)
     //          << " numblocks =  " << numBlocks;

     // it is possible that this is the first cuda call on this thread
     // so make sure we initialize the Driver API's context
     // cudaFree(0) accomplishes this.
     CUcontext pctx = 0;
     TORCH_CU_CHECK(cuCtxGetCurrent(&pctx));
     if (!pctx) {
        std::unique_lock<std::mutex> cudaFreeMutexLock(
            *(THCCachingAllocator_getCudaFreeMutex()));
        cudaFree(0);
     }
     CUstream stream = at::cuda::getCurrentCUDAStream();
     TORCH_CU_CHECK(cuLaunchKernel(
       function,
       numBlocks, 1, 1,
       blockSize, 1, 1,
       0, stream,
       arguments,
       nullptr));
  }
  std::vector<char> ptx;
  CUmodule module;
  CUfunction function;

  // we record prop/device so if they are availiable for launch heuristics
  // querying at launch is too slow for device properties.
  int device;
  cudaDeviceProp prop;
  int blockSize = 128;
  int maxBlocks;
};

#endif

////////////////////////////////////////////////////////////////////////////////
// CPUFusedKernel

struct TempFile {
  TH_DISALLOW_COPY_AND_ASSIGN(TempFile);
  TempFile(const std::string & t, int suffix) {
    // mkstemps edits its first argument in places
    // so we make a copy of the string here, including null terminator
    std::vector<char> tt(t.c_str(), t.c_str() + t.size() + 1);
    int fd = mkstemps(tt.data(), suffix);
    JIT_ASSERT(fd != -1);
    file_ = fdopen(fd, "r+");

    // - 1 becuase tt.size() includes the null terminator,
    // but std::string does not expect one
    name_ = std::string(tt.begin(), tt.end() - 1);
  }
  const std::string & name() const {
    return name_;
  }
  void sync() {
    fflush(file_);
  }
  void write(const std::string & str) {
    size_t result = fwrite(str.c_str(), 1, str.size(), file_);
    JIT_ASSERT(str.size() == result);
  }
  FILE* file()  {
    return file_;
  }
  ~TempFile() {
    if(file_ != nullptr) {
      // unlink first to ensure another mkstemps doesn't
      // race between close and unlink
      unlink(name_.c_str());
      fclose(file_);
    }
  }
private:
  FILE * file_ = nullptr;
  std::string name_;
};

static void* checkDL(void * x) {
  if(!x) {
    AT_ERROR("error in dlopen or dlsym: ", dlerror());
  }
  return x;
}

struct DynamicLibrary {
  TH_DISALLOW_COPY_AND_ASSIGN(DynamicLibrary);
  DynamicLibrary(const char * name) {
    handle = checkDL(dlopen(name, RTLD_LOCAL | RTLD_NOW));
  }
  void * sym(const char * name) {
    JIT_ASSERT(handle);
    return checkDL(dlsym(handle, name));
  }
  ~DynamicLibrary() {
    if(!handle) return;
    dlclose(handle);
  }
private:
  void * handle = nullptr;
};

static const std::string so_template = "/tmp/pytorch_fuserXXXXXX.so";
static const std::string cpp_template = "/tmp/pytorch_fuserXXXXXX.cpp";

// NB: -march=native not supported on PPC64 g++.  It's a bit annoying
// to do a configure-style test to decide whether or not the g++
// actually supports it or not, so we heuristically use the host
// compiler to predict if the runtime compiler supports the option we
// want.  This probably won't work if you're cross-compiling.
// NB: -march=native is disabled because it has caused problems where
// compiler and assembler do not agree on what native instruction they
// understand for AVX512. When we need better CPU performance this
// optimization can be re-enabled by tracking down the platforms where
// this error occurs and only selectively disabling it.
static const std::string compile_string =
  "\"${cxx}\" -O3 -g "
#ifndef __PPC64__
//  "-march=native "
#endif
  "-std=c++11 -fPIC ${fopenmp} -shared \"${cpp_file}\" -o \"${so_file}\" -lm";

static void runCompiler(FusionCompilerConfig & config, const std::string & cpp_file, const std::string & so_file) {
  TemplateEnv env;
  env.s("cxx", config.cxx);
  env.s("fopenmp", config.openmp ? "-fopenmp" : "");
  env.s("cpp_file",cpp_file);
  env.s("so_file",so_file);
  std::string result = format(compile_string,env);
  int r = system(result.c_str());
  if(config.openmp && r != 0) {
    std::cerr << "warning: pytorch jit fuser failed to compile with openmp, trying without it...\n";
    config.openmp = false; // disable for future compiles
    return runCompiler(config, cpp_file, so_file);
  }
  JIT_ASSERTM(r == 0, "Failed to compile a fused CPU kernel");
}


static const std::string disas_string =
  "objdump -M  intel -d \"${so_file}\"";
static void disas(const std::string & so_file) {
  TemplateEnv env;
  env.s("so_file", so_file);
  std::string cmd = format(disas_string, env);
  int r = system(cmd.c_str());
  JIT_ASSERT(r == 0);
}

struct CPUFusedKernel : public FusedKernel {
  CPUFusedKernel(const std::string & name, AnnotatedGraph & agraph, FusionCompilerConfig & config)
  : FusedKernel(name, agraph) {
    TempFile so_file(so_template, 3);
    TempFile cpp_file(cpp_template, 4);

    std::stringstream cu;
    std::tie(chunk_desc, concat_desc, has_random) = codegen::emitCompilationUnit(cu, name, agraph, false);
    JIT_ASSERT(!has_random);
    compilation_unit = cu.str();
    cpp_file.write(compilation_unit);
    cpp_file.sync();
    runCompiler(config, cpp_file.name(), so_file.name());
    if(config.debug) {
      disas(so_file.name());
    }
    so_lib.reset(new DynamicLibrary(so_file.name().c_str()));
#pragma GCC diagnostic ignored "-Wpedantic"
    kernel = reinterpret_cast<void(*)(uint32_t, void**)>(so_lib->sym(name.c_str()));
#pragma GCC diagnostic pop
  }
protected:
  virtual at::Backend backend() const override {
    return at::Backend::CPU;
  }
  virtual uint64_t get_rand_offset(uint32_t numel) override {
     return numel;
  }
  virtual void launch_raw(uint32_t numel, void ** arguments) override {
    kernel(numel, arguments);
  }
  std::unique_ptr<DynamicLibrary> so_lib;
  void (*kernel)(uint32_t, void**) = nullptr;
};

////////////////////////////////////////////////////////////////////////////////
// FusedKernelCache

// Note [Run-time shape checking code]
// There are multiple assumptions that our codegen makes, which we can't check
// in the fusion pass, because we don't have the shape information. Most notably,
// that all values (post-input-chunk, and pre-output-concat) have the same shape
// (hereinafter referred to as map size). One way to check this would be to run
// shape propagation for every size configuration we get as an input, but that
// requires a full graph traversal, and might incur unnecessary overhead. The code
// below uses a few nice properties of broadcasting rules and their interactions with
// pointwise operations, and takes a smarter approach, to quickly verify validity of
// the kernel.
//
// Notation:
//   - a.s when a is a tensor is a shorthand for a.shape.
//   - B is a shorthand for the broadcasting/expanding function. It is used as a
//     vararg function.
//   - E is a shorthand for expand function.
//   - Every pointwise operation can be equivalently rewritten as
//     f(a, b) = f^(E(a, B(a.s, b.s)), E(b, B(a.s, b.s))),
//     where f^ is a non-broadcasting verison of f.
//   - A set of inputs that are used to produce a certain graph output is referred to
//     as the output's broadcasting group (see Lemma 2. for explanation why).
//
// Lemma 1. Set of lists of integers (shapes) + { _|_ (bottom/error marker) }, with the
//          operation of broadcasting (returning bottom upon shape mismatch) forms a monoid.
//          In simpler terms: broadcasting is associative, i.e. B(a, B(b, c)) == B(B(a, b), c).
//
// Proof.   Satisfies all monoid laws:
//            - Closed under broadcasting (trivial)
//            - Empty shape is the identity element: B(a, []) == B([], a) == a
//            - Associativity: A simple visual proof is that you can expand 3 tensors
//                at the same time by stacking their sizes (with alignment to the right),
//                just as you'd do in the case of 2 tensors, but with an intermediate
//                (the algorithm ends up being pretty much the same).
//
// Lemma 2. Shape of an output of an arbitrary DAG of pointwise ops depends only on the set
//          of inputs used in this DAG and is equal to B([i.shape for i in used_inputs]).
//
// Proof.   Let G be any DAG of pointwise ops and < be any valid topological
//          ordering on nodes of G. Proof by induction over <.
//          Base case (graph input):
//            Trivial (input is also an output).
//          Step (n = f(q, r)):
//            Let QS (RS) be the set of shapes of inputs that q (r) depends on.
//            Note that the set of inputs that n depends on is exactly QS + RS.
//            shape(n) == shape(f(q, r))
//                          (def of f)
//                     == shape(f^(E(q, B(q.s, r.s)), E(r, B(q.s, r.s))))
//                          (output shape of f^ is equal to either of argument shapes)
//                     == shape(E(q, B(q.s, r.s)))
//                          (property of expand)
//                     == B(q.s, r.s)
//                          (induction assumption)
//                     == B(B(QS...), B(RS...))
//                          (Lemma 1.)
//                     == B(QS..., RS...)
//                          (repeated shapes don't matter for broadcasting)
//                     == B((QS + RS)...)
//
// Lemma 3. Expands are distributive over pointwise ops, i.e. E(f(a, b), s) = f(E(a, s), E(b, s))
// Lemma 4. Expands can be collapsed, i.e. E(E(x, s1), s2) = E(x, B(s1, s2)).
// Proof.   A simple exercise for the reader :)
//
// Theorem. If all (pre-concat-)outputs have equal shapes, then we can push the expands to
//          (post-chunk-)inputs, and have all intermediates of the same shape
//          (no broadcasting happening in the body).
//
// Proof.   Using the above lemmas we can easily show that a graph with a single output
//          can be easily rewritten by taking the shape given by B applied to all input
//          shapes, expanding inputs to it, and using only non-broadcasting operations.
//          Example:
//
//          let d = f(a, b) in
//          let e = h(b, c) in
//          g(d, e)
//
//          (By def. of broadcasting pointwise ops applied to g, f and h)
//          (Lemma 2. for a closed formula for the size of g = gs)
//
//          let gs = B(a.s, b.s, c.s) in
//          let d' = E(f^(E(a, B(a.s, b.s)), E(b, B(a.s, b.s))), gs) in
//          let e' = E(h^(E(b, B(b.s, c.s)), E(c, B(b.s, c.s))), gs) in
//          g^(d', e')
//
//          (Lemma 3.)
//
//          let gs = B(a.s, b.s, c.s) in
//          let d' = f^(E(E(a, B(a.s, b.s)), gs), E(E(b, B(a.s, b.s)), gs)) in
//          let e' = h^(E(E(b, B(b.s, c.s)), gs), E(E(c, B(b.s, c.s)), gs)) in
//          g^(d', e')
//
//          (Lemma 4. + Lemma 1. to simplify broadcasting function)
//
//          let gs = B(a.s, b.s, c.s) in
//          let d' = f^(E(a, gs), E(b, gs)) in
//          let e' = h^(E(b, gs), E(c, gs)) in
//          g^(d', e')
//
//          (Simple rewrite)
//
//          let gs = B(a.s, b.s, c.s) in
//          let a' = E(a, gs) in
//          let b' = E(b, gs) in
//          let c' = E(c, gs) in
//          let d' = f^(a', b') in
//          let e' = h^(b', c') in
//          g^(d', e')
//
//          This example can be easily formalized to arbitrary DAGs using induction
//          over topological ordering, similar to Lemma 2. Now, if broadcasting groups
//          for all outputs have the same shape, then performing an expand to this size
//          on all inputs will ensure that all intermediates on all paths to outputs
//          will have the same shape, proving that the body of the kernel is valid.
//
//          This shows the part until post-chunk-inputs. Extending it to pre-chunk-inputs
//          is straightforward (needs a simple lemma for moving expands through chunks).

// Register implementations of fused operators, so that we can reuse the fused graph
// to generate fallback code.
RegisterOperators reg_fused_operators({
  Operator(
    prim::FusedConcat,
    [](Node* node) {
      int64_t dim = node->i(attr::dim);
      int64_t num_inputs = node->inputs().size();
      return [dim, num_inputs](Stack& stack) {
        auto result = at::cat(
          fmap(last(stack, num_inputs), [](const IValue& i) { return i.toTensor(); }),
          dim
        );
        drop(stack, num_inputs);
        pack(stack, std::move(result));
        return 0;
      };
    })
});

FusedKernelCache::FusedKernelCache(FusionCompiler& compiler, std::shared_ptr<Graph> _graph, int device)
  : device(device)
  , fallback_code(_graph)
  , compiler(compiler)
  , graph(std::move(_graph))
  , input_broadcast_groups(getInputBroadcastGroups())
  , input_chunks(getInputChunkDescriptors())
  , kernels() {}

std::atomic<size_t> FusedKernelCache::next_kernel_id {0};

auto FusedKernelCache::getInputChunkDescriptors() -> std::vector<PartitionInfo> {
  std::vector<PartitionInfo> descs;
  descs.reserve(graph->inputs().size());
  for (Value * input : graph->inputs()) {
    if (Node * chunk = usedInFusedChunk(input)) {
      descs.emplace_back(chunk->i(attr::chunks), chunk->i(attr::dim));
    } else {
      descs.emplace_back(1, 0);
    }
  }
  return descs;
}

// NB: this vector is really a set, but we want to keep it contiguous in memory for faster access
static std::vector<int64_t> getInputDependencies(Value* output) {
  // Run a DFS traversal to find all inputs that affect a given output value
  std::vector<Value*> queue { output };
  std::unordered_set<Value*> inputs;
  std::unordered_set<Value*> seen;
  while (!queue.empty()) {
    Value * val = queue.back(); queue.pop_back();
    Node * producer = val->node();
    if (producer->kind() == prim::Param) {
      inputs.insert(val);
      continue;
    }
    for (Value * input : producer->inputs()) {
      if (/*bool inserted = */seen.insert(input).second) {
        queue.push_back(input);
      }
    }
  }

  // Convert Value* into offsets into the graph's input list
  std::vector<int64_t> offsets;
  offsets.reserve(inputs.size());
  for (Value * input : inputs) {
    offsets.push_back(input->offset());
  }
  std::sort(offsets.begin(), offsets.end());
  return offsets;
}

std::vector<std::vector<int64_t>> FusedKernelCache::getInputBroadcastGroups() {
  std::unordered_set<std::vector<int64_t>, torch::hash<std::vector<int64_t>>> broadcast_groups;
  for (Value * output : graph->outputs()) {
    broadcast_groups.insert(getInputDependencies(output));
  }
  return std::vector<std::vector<int64_t>>{ broadcast_groups.begin(), broadcast_groups.end() };
}

void FusedKernelCache::run(Stack& stack) {
  int64_t num_inputs = graph->inputs().size();
  auto args = fmap(last(stack, num_inputs), [](const IValue& i) {
                return i.toTensor();
              });

  auto maybe_map_size = canRunKernel(args);
  if (!maybe_map_size) {
    return runFallback(stack);
  }
  expandArgs(args, *maybe_map_size);

  FusedKernelArgSpec spec { args };
  auto it = kernels.find(spec);
  if (it == kernels.end()) {
    std::tie(it, std::ignore) = kernels.emplace(spec, compileSpec(spec, *maybe_map_size));
  }
  auto & fn = it->second;

  std::vector<at::Tensor> outputs;
  fn->launch(args, outputs);
  drop(stack, num_inputs);
  stack.insert(stack.end(), std::make_move_iterator(outputs.begin()),
                            std::make_move_iterator(outputs.end()));
}

at::optional<std::vector<int64_t>> FusedKernelCache::getMapSize(at::TensorList args, at::IntList arg_subset) {
  int64_t dim_after_broadcast = 0;
  for (int64_t arg_idx : arg_subset) {
    dim_after_broadcast = std::max(dim_after_broadcast, args[arg_idx].dim());
  }
  // TODO: this keeps reallocating map_size at every iteration, but we know
  // exactly how much storage do we need, so this could be fixed in-place at
  // every step. We're just missing a few functions for ATen, but the fix
  // should be straightforward.
  // NB: we leave this uninitialized, because an empty size is trivially
  // broadcastable to any other size.
  std::vector<int64_t> map_size;
  for (size_t i = 0; i < arg_subset.size(); ++i) {
    auto & arg = args.at(arg_subset[i]);
    auto & chunk_desc = input_chunks.at(arg_subset[i]);
    if (chunk_desc.nSubtensors == 1) {
      try {
        map_size = at::infer_size(map_size, arg.sizes());
      } catch (std::exception& e) {
        return at::nullopt;
      }
    } else {
      auto tensor_sizes = arg.sizes().vec();
      int64_t num_chunks = chunk_desc.nSubtensors;
      int64_t dim = at::maybe_wrap_dim(chunk_desc.dim, tensor_sizes.size());
      if (tensor_sizes[dim] % num_chunks != 0) {
        return at::nullopt;
      }
      tensor_sizes[dim] /= num_chunks;
      try {
        map_size = at::infer_size(map_size, tensor_sizes);
      } catch (std::exception& e) {
        return at::nullopt;
      }
    }
  }

  return {map_size};
}

// See Note [Run-time shape checking code] for more explanation on the algorithm.
at::optional<std::vector<int64_t>> FusedKernelCache::canRunKernel(at::TensorList args) {
  AT_CHECK(args.size() == input_chunks.size(),
           "Expected ", input_chunks.size(), " arguments, but got ", args.size());

  at::optional<std::vector<int64_t>> map_size;
  for (const auto & broadcast_group : input_broadcast_groups) {
    if (!map_size) {
      map_size = getMapSize(args, broadcast_group);
      if (!map_size) {
        return at::nullopt;
      }
    } else {
      auto group_map_size = getMapSize(args, broadcast_group);
      // NB: this checks that group_map_size is defined AND equal to map_size
      if (map_size != group_map_size) {
        return at::nullopt;
      }
    }
  }
  return map_size;
}

void FusedKernelCache::runFallback(Stack& stack) {
  InterpreterState(fallback_code).runOneStage(stack);
}

// NB: args are mutated in this call. map_size is mutated too, but is restored to its original
// value before this function returns (it's an optimization).
void FusedKernelCache::expandArgs(std::vector<at::Tensor>& args, std::vector<int64_t>& map_size) {
  for (size_t i = 0; i < args.size(); ++i) {
    auto & arg = args[i];
    auto & pdesc = input_chunks[i];
    if (pdesc.nSubtensors == 1) {
      if (arg.sizes().equals(map_size)) continue;
      arg = arg.expand(map_size);
    } else {
      map_size.at(pdesc.dim) *= pdesc.nSubtensors;
      if (!arg.sizes().equals(map_size)) {
        arg = arg.expand(map_size);
      }
      map_size.at(pdesc.dim) /= pdesc.nSubtensors;
    }
  }
}

std::unique_ptr<FusedKernel> FusedKernelCache::compileSpec(
      const FusedKernelArgSpec& spec, const std::vector<int64_t>& map_size) {
  AnnotatedGraph agraph {*graph, device};

  agraph.input_desc = spec.descs();
  // XXX: this assumes that fused kernels only operate on floating-point values inside
  at::optional<at::ScalarType> scalar_type;
  for (TensorDesc& desc : agraph.input_desc) {
    if (isFloatingType(desc.scalar_type)) {
      scalar_type = desc.scalar_type;
      break;
    }
  }
  JIT_ASSERT(scalar_type);

  for (Value * output : graph->outputs()) {
    std::vector<int64_t> sizes = map_size;
    if (output->node()->kind() == prim::FusedConcat) {
      sizes.at(output->node()->i(attr::dim)) *= output->node()->inputs().size();
    }
    auto type = CompleteTensorType::create(*scalar_type, device, sizes);
    agraph.output_desc.emplace_back(std::move(type));
  }

  std::string name = "kernel_" + std::to_string(next_kernel_id++);
  FusedKernel * raw_func;
  if (device != kCPUDevice) {
#ifdef USE_CUDA
    raw_func = new CUDAFusedKernel(name, agraph);
#else
    throw std::runtime_error("cannot compile a CUDA fusion group, CUDA is not enabled.");
#endif
  } else {
    JIT_ASSERT(compiler.canCompileOnCPU());
    raw_func = new CPUFusedKernel(name, agraph, compiler.config_);
  }
  return std::unique_ptr<FusedKernel>(raw_func);
}

////////////////////////////////////////////////////////////////////////////////
// FusionCompiler

std::shared_ptr<FusedKernelCache> FusionCompiler::getOrCompile(Node* fusion_group) {
  int device = fusion_group->i(attr::device);
  if (device == kCPUDevice) {
    JIT_ASSERT(canCompileOnCPU());
  } else {
#ifndef USE_CUDA
    throw std::runtime_error("cannot compile a CUDA fusion group - CUDA is not enabled.");
#endif
  }
  auto graph = fusion_group->g(attr::Subgraph)->copy();
  EraseShapeInformation(*graph);
  std::stringstream key;
  key << "device " << device << "\n";
  key << *graph << "\n";
  std::string key_ = key.str();
  auto it = cache_map.find(key_);
  if (it == cache_map.end()) {
    std::tie(it, std::ignore) = cache_map.emplace(key_, std::make_shared<FusedKernelCache>(*this, graph, device));
  }
  return it->second;
}

std::vector<at::Tensor> FusionCompiler::debugLaunchGraph(Graph & graph, int device, at::ArrayRef<at::Tensor> inputs) {
  auto wrapper_graph = std::make_shared<Graph>();
  Node * fusion_group = wrapper_graph->insertNode(wrapper_graph->createFusionGroup(device));
  fusion_group->g_(attr::Subgraph, graph.copy());
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }
  auto cache = getOrCompile(fusion_group);
  Stack stack = fmap<IValue>(inputs);
  cache->run(stack);
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

static const std::string check_exists_string =
  "which '${program}' > /dev/null";

static bool programExists(const std::string & program) {
  TemplateEnv env;
  env.s("program", program);
  std::string cmd = format(check_exists_string, env);
  return 0 == system(cmd.c_str());
}

FusionCompiler::FusionCompiler() {
  const char * cxx_env = getenv("CXX");
  if(cxx_env != nullptr) {
    config_.cxx = cxx_env;
  }
  if(!programExists(config_.cxx)) {
    config_.cxx = "";
  }
  const char * debug_env = getenv("PYTORCH_FUSION_DEBUG");
  config_.debug = debug_env && atoi(debug_env) != 0;
}

//TODO: thread safety
FusionCompiler & sharedFusionCompiler() {
  static FusionCompiler compiler;
  return compiler;
}

}}

# else
// dummy implementations for windows

#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "ATen/ATen.h"
#ifdef USE_CUDA
#include "torch/csrc/cuda/cuda_check.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iostream>

namespace torch { namespace jit {

struct FusedKernel {
  char padding;
};

FusedKernelCache::FusedKernelCache(FusionCompiler& compiler, std::shared_ptr<Graph> graph, int device)
  : compiler(compiler) {}
void FusedKernelCache::run(Stack& inputs) {}
void FusedKernelCache::runFallback(Stack& stack) {}
void FusedKernelCache::expandArgs(std::vector<at::Tensor>& args, std::vector<int64_t>& map_size) {}
at::optional<std::vector<int64_t>> FusedKernelCache::canRunKernel(at::TensorList args) { return at::nullopt; }
at::optional<std::vector<int64_t>> FusedKernelCache::getMapSize(at::TensorList args, at::IntList arg_subset) { return at::nullopt; }
std::vector<std::vector<int64_t>> FusedKernelCache::getInputBroadcastGroups() { return {}; }
auto FusedKernelCache::getInputChunkDescriptors() -> std::vector<PartitionInfo> { return {}; }
std::unique_ptr<FusedKernel> FusedKernelCache::compileSpec(
      const FusedKernelArgSpec& spec, const std::vector<int64_t>& map_size) { return nullptr; }
std::atomic<size_t> FusedKernelCache::next_kernel_id {0};

FusionCompiler::FusionCompiler() {}
std::shared_ptr<FusedKernelCache> FusionCompiler::getOrCompile(Node* fusion_group) { return nullptr; }
std::vector<at::Tensor> FusionCompiler::debugLaunchGraph(Graph & graph, int device, at::ArrayRef<at::Tensor> inputs) { return {}; }

FusionCompiler & sharedFusionCompiler() {
  throw std::runtime_error("NYI: fuser is not supported on Windows.");
}

}}

# endif
