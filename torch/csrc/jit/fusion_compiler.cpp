#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/DisallowCopy.h"
#include "torch/csrc/jit/code_template.h"
#include "ATen/ATen.h"
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <assert.h>
#include <iostream>

namespace torch { namespace jit {

void findContiguous(
  at::IntList sizes,
  at::IntList strides,
  std::vector<bool> & cont) {
  assert(sizes.size() == strides.size());
  cont.resize(sizes.size());
  for(size_t i = 0; i < sizes.size(); ++i) {
    int64_t expected_stride = (i + 1 < sizes.size()) ? sizes[i+1]*strides[i+1] : 1;
    cont[i] = strides[i] == expected_stride;
  }
}

// compress dimensions when the tensor is marked as cont
// anytime we do a compression, we assert that it is valid for this particular tensor.
static void compressContiguous(
  at::IntList sizes,
  at::IntList strides,
  const std::vector<bool> & cont,
  uint32_t * c_sizes,
  uint32_t * c_strides) {
  size_t c = 0;
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

    c_sizes[c] = total_size;
    c_strides[c] = strides[cur-1];
    c++;
  }
  JIT_ASSERT(!cont.back() || strides.back() == 1);
}

static auto compilation_unit_template = CodeTemplate(R"(
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
static auto dim_calc = CodeTemplate(R"(
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

static std::ostream& operator<<(std::ostream & out, const TensorDesc & d) {
  out << d.scalar_type << "[";
  for(auto b : d.contiguity)
    out << b << ";";
  out << "]";
  return out;
}

static std::string nodeName(Node * n) {
  return "n"+std::to_string(n->unique());
}

static std::unordered_map<std::string,std::string> simple_map_ops = {
  {"Sigmoid","1.f / (1.f + expf(-${0}))"},
  {"Tanh","tanh(${0})"},
  {"Mul","${0} * ${1}"},
  {"Add","${0} + ${1}"},
};

const char * toCString(at::ScalarType type) {
  switch(type) {
    #define DEFINE_CASE(ctype,name,_) \
      case at::ScalarType::name: return #ctype;
    AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
    #undef DEFINE_CASE
    default:
      throw new std::runtime_error("unknown scalar type");
  }
}

void emitCompilationUnit(std::ostream & out,
  const std::string & name,
  AnnotatedGraph & agraph) {
  Graph& subgraph = *agraph.graph;
  TemplateEnv env;
  env.s("kernelName",name);
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
    env.s("scalar_type",toCString(desc.scalar_type));
    formals.push_back(format("TensorInfo<${scalar_type},${nDim}> ${tensor}",env));
  };
  {
    size_t i = 0;
    for(auto p : subgraph.inputs())
      emitFormal(p,agraph.input_desc[i++]);
  }
  {
    size_t i = 0;
    for(auto o : subgraph.outputs())
      emitFormal(o,agraph.output_desc[i++]);
  }
  size_t formal_count = 0;
  for(auto p : subgraph.inputs()) {
    env.s("node",nodeName(p));
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    // TODO: auto doesn't work so we need to do shape/type inference
    body << format("auto ${node} = ${access};\n",env);
  }
  for(auto n_ : subgraph.nodes()) {
    if(n_->kind() == NodeKind::Return)
      continue; //TODO: remove when return is not in list
    auto n = n_->cast<SimpleMap>();
    JIT_ASSERT(n);
    size_t i = 0;
    for(auto in : n->inputs()) {
      env.s(std::to_string(i++),nodeName(in));
    }
    env.s("node",nodeName(n));
    env.s("rhs",format(simple_map_ops.at(n->op),env));
    body << format("auto ${node} = ${rhs};\n",env);
  }
  for(auto o : subgraph.outputs()) {
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    env.s("node",nodeName(o));
    body << format("${access} = ${node};\n",env);
  }
  env.s("tensorOffsets",tensorOffsets.str());
  env.s("kernelBody",body.str());
  env.v("formals",formals);
  out << compilation_unit_template.format(env);
}

static void nvrtcCheck(nvrtcResult result,const char * file, int line) {
  if(result != NVRTC_SUCCESS) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << nvrtcGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define JIT_NVRTC_CHECK(result) \
  nvrtcCheck(result,__FILE__,__LINE__);

#define JIT_CU_CHECK(result) \
  cuCheck(result,__FILE__,__LINE__);
static void cuCheck(CUresult result, const char * file, int line) {
  if(result != CUDA_SUCCESS) {
    const char * str;
    cuGetErrorString(result, &str);
    std::stringstream ss;
    ss << file << ":" << line << ": " << str;
    throw std::runtime_error(ss.str());
  }
}
#define JIT_CU_CHECK(result) \
  cuCheck(result,__FILE__,__LINE__);

static void cudaCheck(cudaError_t result, const char * file, int line) {
  if(result != cudaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << cudaGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define JIT_CUDA_CHECK(result) \
    cudaCheck(result,__FILE__,__LINE__);

static int cielDiv(int a, int b) {
  return (a + b - 1) / b;
}

//host-side view of TensorInfo
//note dims[0], because we need to dynamically allocate the dims
struct TensorInfo {
  void * data;
  uint32_t dims[0];
  uint32_t* sizes(size_t nDim) {
    return &dims[0];
  }
  uint32_t* strides(size_t nDim) {
    return &dims[nDim];
  }
};

CompiledFusionFunction::CompiledFusionFunction(const std::string & name, AnnotatedGraph & agraph)
: name(name), input_desc(agraph.input_desc), output_desc(agraph.output_desc) {
  std::stringstream cu;
  emitCompilationUnit(cu, name, agraph);
  compliation_unit = cu.str();
  nvrtcProgram program;
  JIT_NVRTC_CHECK(nvrtcCreateProgram(&program,compliation_unit.c_str(), NULL, 0, nullptr, nullptr));
  cudaDeviceProp deviceProp;
  JIT_CUDA_CHECK(cudaGetDevice(&device));
  JIT_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
  std::string compute = "--gpu-architecture=compute_" + std::to_string(deviceProp.major) + std::to_string(deviceProp.minor);
  std::vector<const char *> args = {"--std=c++11", compute.c_str()};
  nvrtcResult result = nvrtcCompileProgram(program, args.size(), args.data());
  if(result == NVRTC_ERROR_COMPILATION) {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
    cu << log.data();
    throw std::runtime_error(cu.str());
  }
  JIT_NVRTC_CHECK(result);
  size_t ptx_size;
  JIT_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  JIT_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
  JIT_NVRTC_CHECK(nvrtcDestroyProgram(&program));

  JIT_CU_CHECK(cuModuleLoadData(&module, ptx.data()));
  JIT_CU_CHECK(cuModuleGetFunction(&function, module, name.c_str()));

  JIT_CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks, function, 128, 0));
  JIT_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  maxBlocks *= prop.multiProcessorCount;
}
CompiledFusionFunction::~CompiledFusionFunction() {
  JIT_CU_CHECK(cuModuleUnload(module));
}

void CompiledFusionFunction::launch(at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs) {
  JIT_ASSERT(inputs.size() == input_desc.size());
  JIT_ASSERT(outputs.size() == output_desc.size());
  size_t uncompressedDim = input_desc.at(0).contiguity.size();
  uint32_t numel = inputs[0].numel();
  size_t maxPossibleTensorInfoSize = sizeof(TensorInfo) + 2*sizeof(uint32_t)*uncompressedDim;
  size_t maxPossibleBufferSize = maxPossibleTensorInfoSize * (inputs.size() + outputs.size());
  std::vector<char> buffer(maxPossibleBufferSize);
  char * buffer_next = buffer.data();
  std::vector<void*> arguments;
  arguments.reserve(1 + inputs.size() + outputs.size());
  auto addTensorInfo = [&](TensorDesc & desc, const at::Tensor & t) {
    size_t nDim = desc.nDim(); //the compressed dim
    auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
    ti->data = t.data_ptr();
    compressContiguous(t.sizes(), t.strides(), desc.contiguity, ti->sizes(nDim), ti->strides(nDim));
    buffer_next += maxPossibleTensorInfoSize;
    arguments.push_back(ti);
  };
  arguments.push_back(&numel);
  {
    size_t i = 0;
    for(auto & desc : input_desc) {
      addTensorInfo(desc,inputs[i++]);
    }
  }
  {
    size_t i = 0;
    for(auto & desc : output_desc) {
      addTensorInfo(desc,outputs[i++]);
    }
  }
  launch(numel, arguments.data());
}

void CompiledFusionFunction::launch(uint32_t numel, void ** arguments) {
int numBlocks = std::min(maxBlocks,cielDiv(numel,blockSize));
  //std::cout << "maxBlocks = " << maxBlocks << " needed blocks: " << cielDiv(numel,blockSize)
  //          << " numblocks =  " << numBlocks;

  JIT_CU_CHECK(cuLaunchKernel(
    function,
    numBlocks, 1, 1,
    blockSize, 1, 1,
    0, nullptr,
    arguments,
    nullptr));
}




FusionCompiler::FusionCompiler() {}
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
  if(it == cache.end()) {
    std::string name = "kernel_" + std::to_string(cache.size());
    auto func = std::make_shared<CompiledFusionFunction>(name,agraph);
    it = cache.emplace(key_,std::move(func)).first;
  }
  return it->second;
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

}}
