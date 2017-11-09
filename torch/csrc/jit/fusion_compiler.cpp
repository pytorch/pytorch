#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "ATen/ATen.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iostream>

namespace torch { namespace jit {

std::unordered_map<NodeKind, std::string> simple_map_ops = {
  // unary
  {kabs, "absf(${0})"},
  {ksigmoid, "1.f / (1.f + expf(-${0}))"},
  {klog, "logf(${0})"},
  {klog1p, "log1pf(${0})"},
  {klgamma, "lgammaf(${0})"},
  {kexp, "expf(${0})"},
  {kcos, "cosf(${0})"},
  {kacos, "acosf(${0})"},
  {kcosh, "coshf(${0})"},
  {ksin, "sinf(${0})"},
  {kasin, "asinf(${0})"},
  {ksinh, "sinhf(${0})"},
  {ktan, "tanf(${0})"},
  {katan, "atanf(${0})"},
  {ktanh, "tanhf(${0})"},
  {ksqrt, "sqrtf(${0})"},
  {krsqrt, "rsqrtf(${0})"},
  {kceil, "ceilf(${0})"},
  {kfloor, "floorf(${0})"},
  {kround, "roundf(${0})"},
  {ktrunc, "truncf(${0})"},
  {kfrac, "fracf(${0})"},
  {kreciprocal, "reciprocalf(${0})"},
  {kneg, "-${0}"},
  //simple binary
  {katan2, "atan2(${0}, ${1})"},
  {kmin, "fminf(${0}, ${1})"},
  {kmax, "fmaxf(${0}, ${1})"},

  //binary with other
  // TODO: some of these ops will not get generated because
  // we only work on float inputs/outputs, but they are here to record
  // that they are valid mappable ops once we handle more type
  {k__and__, "${0} && ${1}"},
  {k__lshift__, "${0} << ${1}"},
  {k__or__, "${0} || ${1}"},
  {k__rshift__, "${0} >> ${1}"},
  {k__xor__, "${0} ^ ${1}"},
  {kdiv, "${0} / ${1}"},
  {keq, "${0} == ${1}"},
  {kfmod, "fmodf(${0}, ${1})"},
  {kge, "${0} >= ${1})"},
  {kgt, "${0} > ${1}"},
  {kle, "${0} <= ${1})"},
  {klt, "${0} < ${1}"},
  {kmul, "${0} * ${1}"},
  {kne, "${0} != ${1}"},
  {kremainder, "remainderf(${0}, ${1})"},
  {kpow, "powf(${0}, ${1})"},

  //alpha
  {kadd, "${0} + ${alpha}*${1}"},
  {ksub, "${0} - ${alpha}*${1})"},

  // special
  {klerp, "${0} + ${weight}*(${1} - ${0})"},
  {kclamp, "min(max(${0},${min}),${max})"},

};

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

namespace {

static int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

std::ostream& operator<<(std::ostream & out, const TensorDesc & d) {
  out << d.scalar_type << "[";
  for(auto b : d.contiguity)
    out << b << ";";
  out << "]";
  return out;
}

// We're using three CUDA APIs, so define a few helpers for error handling
static void nvrtcCheck(nvrtcResult result,const char * file, int line) {
  if(result != NVRTC_SUCCESS) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << nvrtcGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define JIT_NVRTC_CHECK(result) nvrtcCheck(result,__FILE__,__LINE__);

static void cuCheck(CUresult result, const char * file, int line) {
  if(result != CUDA_SUCCESS) {
    const char * str;
    cuGetErrorString(result, &str);
    std::stringstream ss;
    ss << file << ":" << line << ": " << str;
    throw std::runtime_error(ss.str());
  }
}
#define JIT_CU_CHECK(result) cuCheck(result,__FILE__,__LINE__);

static void cudaCheck(cudaError_t result, const char * file, int line) {
  if(result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define JIT_CUDA_CHECK(result) cudaCheck(result,__FILE__,__LINE__);

////////////////////////////////////////////////////////////////////////////////
// Code generation

namespace codegen {

auto compilation_unit_template = CodeTemplate(R"(
typedef ${IndexType} IndexType;
template<typename T, size_t N>
struct TensorInfo {
  T * data;
  IndexType sizes[N];
  IndexType strides[N];
};

extern "C" __global__
void ${kernelName}(IndexType totalElements, ${formals}) {
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

// curDimIndex = linearId % sizes[i]; // % sizes[i] is not needed for d == 0, because we already guard for numel outside the index calculation
// offset += curDimIndex*strides[i]; // *strides[i] is optional if list_is_cont becaause strides.back() == 1
// linearId /= sizes[i];
auto dim_calc = CodeTemplate(R"(
//printf("tensor ${tensor} sizes[${d}] = %d, strides[${d}] = %d\n", ${tensor}.sizes[${d}],${tensor}.strides[${d}]);
size_t ${tensor}_dimIndex${d} = ${tensor}_linearIndex ${mod_sizes};
${tensor}_offset += ${tensor}_dimIndex${d} ${times_stride};
)");

void emitIndexingFor(std::ostream & out, const std::string & tensor, int ndim, bool last_is_cont) {
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

std::string nodeName(Node * n) {
  return "n" + std::to_string(n->unique());
}

 std::string scalarValue(const at::Tensor & t) {
  auto s =  at::Scalar(t);
  return (s.isIntegral()) ?
    std::to_string(s.toLong()) :
    std::to_string(s.toDouble());
}

const char * scalarTypeName(at::ScalarType type) {
  switch(type) {
    #define DEFINE_CASE(ctype,name,_) \
      case at::ScalarType::name: return #ctype;
    AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
    #undef DEFINE_CASE
    default:
      throw new std::runtime_error("unknown scalar type");
  }
}

std::string encodeRHS(Node * n) {
  TemplateEnv env;
  size_t i = 0;
  for(auto in : n->inputs()) {
    env.s(std::to_string(i++),nodeName(in));
  }
  // ops like div have a / b or a / 2 with the constant having the attribute other
  // so we add other as an input if it is present
  // 'pow' is the same but uses exponent as the attribute, so we handle that here as well
  if(n->hasAttribute(kother) || n->hasAttribute(kexponent)) {
    env.s(std::to_string(i), scalarValue(n->t(kother)));
  }
  // we also add any other scalar tensors to the env for special ops
  for(auto a : n->attributeNames()) {
    if(n->kindOf(a) == AttributeKind::t) {
      auto v = n->t(a);
      if(v.dim() == 0) {
        env.s(symbolToString(a), scalarValue(v));
      }
    }
  }
  const auto & str = simple_map_ops.at(n->kind());
  return format(str, env);
}

std::vector<ConcatDesc> emitCompilationUnit(std::ostream & out,
                                            const std::string & name,
                                            AnnotatedGraph & agraph) {
  Graph& subgraph = *agraph.graph;
  TemplateEnv env;
  env.s("kernelName",name);
  // TODO: handle cases where we need to generate > 2^32 element tensors
  env.s("IndexType","unsigned int"); //avoiding slow header includes to get uint32_t

  std::stringstream body;
  std::stringstream tensorOffsets;
  std::vector<std::string> formals;
  auto emitFormal = [&](Node * n, const TensorDesc & desc) {
    std::string tensor = "t" + std::to_string(formals.size()); //can't be unique() because Param may be an output
    size_t nDim = desc.nDim();
    emitIndexingFor(tensorOffsets, tensor, nDim,  desc.lastIsContiguous());
    env.s("tensor",tensor);
    env.d("nDim",nDim);
    env.s("scalar_type",scalarTypeName(desc.scalar_type));
    formals.push_back(format("TensorInfo<${scalar_type},${nDim}> ${tensor}",env));
  };
  {
    size_t i = 0;
    for(auto p : subgraph.inputs())
      emitFormal(p,agraph.input_desc[i++]);
  }
  std::vector<ConcatDesc> concat_desc;
  std::vector<Node*> flat_output_nodes;
  {
    size_t i = 0;
    for(auto o : subgraph.outputs()) {
      auto & desc = agraph.output_desc[i++];
      if(o->kind() != kcat) {
        emitFormal(o, desc);
        concat_desc.emplace_back();
        flat_output_nodes.push_back(o);
      } else {
        size_t nInputs = o->inputs().size();
        concat_desc.emplace_back(desc, nInputs, o->i(kdim));
        for(auto c : o->inputs()) {
          emitFormal(c, *concat_desc.back().subtensorDesc);
          flat_output_nodes.push_back(c);
        }
      }
    }
  }
  size_t formal_count = 0;
  for(auto p : subgraph.inputs()) {
    env.s("node",nodeName(p));
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    //TODO: actual type propagation rather than relying on auto..
    body << format("auto ${node} = ${access};\n",env);
  }
  for(auto n : subgraph.nodes()) {
    if(n->kind() == kcat)
      continue; // Concat nodes by narrowing the output Tensors before the kernel runs
    env.s("node",nodeName(n));
    env.s("rhs", encodeRHS(n));
    body << format("auto ${node} = ${rhs};\n",env);
  }
  for(auto o : flat_output_nodes) {
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    env.s("node",nodeName(o));
    body << format("${access} = ${node};\n",env);
  }
  env.s("tensorOffsets",tensorOffsets.str());
  env.s("kernelBody",body.str());
  env.v("formals",formals);
  out << compilation_unit_template.format(env);
  return concat_desc;
}

////////////////////////////////////////////////////////////////////////////////

} // codegen namespace
} // anonymous namespace

// Host-side view of TensorInfo (that visivle for the kernel is defined above).
// Note dims[0] - we need to dynamically allocate the dims.
struct TensorInfo {
  void * data;
  uint32_t sizes_strides[0];

  uint32_t* sizes(size_t nDim) { return &sizes_strides[0]; }
  uint32_t* strides(size_t nDim) { return &sizes_strides[nDim]; }
};

CompiledFusionFunction::CompiledFusionFunction(const std::string & name, AnnotatedGraph & agraph)
  : name(name)
  , input_desc(agraph.input_desc)
  , output_desc(agraph.output_desc) {
  JIT_CUDA_CHECK(cudaGetDevice(&device));
  JIT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::stringstream cu;
  concat_desc = codegen::emitCompilationUnit(cu, name, agraph);
  compilation_unit = cu.str();
  nvrtcProgram program;
  JIT_NVRTC_CHECK(nvrtcCreateProgram(&program, compilation_unit.c_str(), NULL, 0, nullptr, nullptr));

  std::string compute = "--gpu-architecture=compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
  std::vector<const char *> args = {"--std=c++11", compute.c_str()};
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
    JIT_NVRTC_CHECK(nvrtcDestroyProgram(&program));
  });
  JIT_NVRTC_CHECK(result);

  size_t ptx_size;
  JIT_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  JIT_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));

  JIT_CU_CHECK(cuModuleLoadData(&module, ptx.data()));
  JIT_CU_CHECK(cuModuleGetFunction(&function, module, name.c_str()));

  JIT_CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks, function, 128, 0));
  maxBlocks *= prop.multiProcessorCount;
}

CompiledFusionFunction::~CompiledFusionFunction() {
  JIT_CU_CHECK(cuModuleUnload(module));
}

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
  JIT_ASSERT(!cont.back() || strides.back() == 1);
}

} // anonymous namespace

void CompiledFusionFunction::launch(at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs) {
  AutoGPU gpu_guard(inputs);
  JIT_ASSERT(inputs.size() == input_desc.size());
  JIT_ASSERT(outputs.size() == output_desc.size());
  size_t flat_outputs_size = 0;
  for(auto & c : concat_desc)
    flat_outputs_size += c.nSubtensors;
  // XXX: this code assumes that inputs are 32-bit addressable
  // XXX: this code assumes that all inputs are of the same size
  JIT_ASSERT(inputs[0].numel() <= std::numeric_limits<uint32_t>::max());
  uint32_t numel = inputs[0].numel();
  at::IntList map_size = inputs[0].sizes();
  // Compute the storage needed to store TensorInfo structs for inputs and outputs.
  size_t uncompressedDim = input_desc.at(0).contiguity.size();
  size_t maxPossibleTensorInfoSize = sizeof(TensorInfo) + 2 * sizeof(uint32_t) * uncompressedDim;
  size_t maxPossibleBufferSize = maxPossibleTensorInfoSize * (inputs.size() + flat_outputs_size);
  std::vector<char> buffer(maxPossibleBufferSize);
  char * buffer_next = buffer.data();
  // A vector of arguments to the kernel. It's (numel, *input_descs, *output_descs)
  std::vector<void*> arguments;
  arguments.reserve(1 + inputs.size() + flat_outputs_size);
  // Asserts that t's dims can be compressed in the same way as in desc
  // (that's what the kernel assumes), and appends it to the arguments vector.
  auto addTensorInfo = [&](TensorDesc & desc, const at::Tensor & t) {
    size_t nDim = desc.nDim(); // NOTE: this is the compressed dim
    JIT_ASSERT(nDim <= uncompressedDim); // We'd overflow the space otherwise
    auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
    ti->data = t.data_ptr();
    compressContiguous(t.sizes(), t.strides(), desc.contiguity, ti->sizes(nDim), ti->strides(nDim));
    buffer_next += maxPossibleTensorInfoSize;
    arguments.push_back(ti);
  };
  arguments.push_back(&numel);
  for (std::size_t i = 0; i < input_desc.size(); ++i)
    addTensorInfo(input_desc[i], inputs[i]);
  for (std::size_t i = 0; i < output_desc.size(); ++i) {
    auto & c = concat_desc[i];
    at::Tensor o = outputs[i];
    if(c.nSubtensors == 1) {
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
  launch(numel, arguments.data());
}

void CompiledFusionFunction::launch(uint32_t numel, void ** arguments) {
  int numBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));
  //std::cout << "maxBlocks = " << maxBlocks << " needed blocks: " << ceilDiv(numel,blockSize)
  //          << " numblocks =  " << numBlocks;

  // it is possible that this is the first cuda call on this thread
  // so make sure we initialize the Driver API's context
  // cudaFree(0) accomplishes this.
  cudaFree(0);

  JIT_CU_CHECK(cuLaunchKernel(
    function,
    numBlocks, 1, 1,
    blockSize, 1, 1,
    0, nullptr,
    arguments,
    nullptr));
}

std::shared_ptr<CompiledFusionFunction> FusionCompiler::getOrCompile(AnnotatedGraph & agraph) {
  std::stringstream key;
  key << *agraph.graph << "\n";
  int device;
  JIT_CUDA_CHECK(cudaGetDevice(&device));
  key << "Device " << device << "\n";
  for(auto & i : agraph.input_desc)
    key << i << "\n";
  for(auto & i : agraph.output_desc)
    key << i << "\n";
  std::string key_ = key.str();

  auto it = cache.find(key_);
  if (it == cache.end()) {
    std::string name = "kernel_" + std::to_string(cache.size());
    auto func = std::make_shared<CompiledFusionFunction>(name, agraph);
    it = cache.emplace(key_, std::move(func)).first;
  }
  return it->second;
}

std::shared_ptr<CompiledFusionFunction> FusionCompiler::getOrCompile(Graph & graph) {
  AnnotatedGraph agraph { &graph };
  for(auto & input : graph.inputs()) {
    agraph.input_desc.emplace_back(input->type()->expect<TensorType>());
  }
  for(auto & output : graph.outputs()) {
    agraph.output_desc.emplace_back(output->type()->expect<TensorType>());
  }
  return getOrCompile(agraph);
}

void FusionCompiler::debugLaunchGraph(Graph & graph, at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs) {
  AnnotatedGraph agraph { &graph };
  for(auto & i : inputs) {
    agraph.input_desc.emplace_back(i);
  }
  for(auto & i : outputs) {
    agraph.output_desc.emplace_back(i);
  }
  auto func = getOrCompile(agraph);
  func->launch(inputs, outputs);
}

//TODO: thread safety
FusionCompiler & sharedFusionCompiler() {
  static FusionCompiler compiler;
  return compiler;
}

}}
