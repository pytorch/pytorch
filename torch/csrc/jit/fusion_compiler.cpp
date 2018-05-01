#ifndef _WIN32
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "ATen/ATen.h"
#ifdef WITH_CUDA
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
#include <dlfcn.h>
#include <unistd.h>

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

namespace {

#ifdef WITH_CUDA

static int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

#endif

std::ostream& operator<<(std::ostream & out, const TensorDesc & d) {
  out << d.scalar_type << "[";
  for(auto b : d.contiguity)
    out << b << ";";
  out << "]";
  return out;
}

////////////////////////////////////////////////////////////////////////////////
// Code generation

namespace codegen {

auto type_declarations_template = CodeTemplate(R"(
typedef ${IndexType} IndexType;
template<typename T, size_t N>
struct TensorInfo {
  T * data;
  IndexType sizes[N];
  IndexType strides[N];
};
)");

auto cuda_compilation_unit_template = CodeTemplate(R"(
${type_declarations}

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

auto cpu_compilation_unit_template = CodeTemplate(R"(
#include <cstddef>
#include <math.h>
#include <iostream>
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

std::string valueName(Value * n) {
  return "n" + std::to_string(n->unique());
}

 std::string scalarValue(const at::Tensor & t) {
  auto s =  at::Scalar(t);
  return (s.isIntegral()) ?
    std::to_string(s.toLong()) :
    (std::to_string(s.toDouble()) + "f");
}

const char * scalarTypeName(at::ScalarType type) {
  switch(type) {
    #define DEFINE_CASE(ctype,name,_) \
      case at::ScalarType::name: return #ctype;
    AT_FORALL_SCALAR_TYPES(DEFINE_CASE)
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
    {aten::ge, "${0} >= ${1})"},
    {aten::gt, "${0} > ${1}"},
    {aten::le, "${0} <= ${1})"},
    {aten::lt, "${0} < ${1}"},
    {aten::mul, "${0} * ${1}"},
    {aten::ne, "${0} != ${1}"},
    {aten::remainder, "remainderf(${0}, ${1})"},
    {aten::pow, "powf(${0}, ${1})"},

    //alpha
    {aten::add, "${0} + ${alpha}*${1}"},
    {aten::sub, "(${0} - ${alpha}*${1})"},

    // special
    {aten::lerp, "${0} + ${weight}*(${1} - ${0})"},
    {aten::clamp, "min(max(${0},${min}),${max})"},

    // simple derivatives
    {aten::_sigmoid_backward, "${0} * ${1} * (1.f - ${1})"},
    {aten::_tanh_backward,    "${0} * (1.f - ${1} * ${1})"},
  };


  TemplateEnv env;
  size_t i = 0;
  for(auto in : n->inputs()) {
    env.s(std::to_string(i++),valueName(in));
  }
  // ops like div have a / b or a / 2 with the constant having the attribute other
  // so we add other as an input if it is present
  // 'pow' is the same but uses exponent as the attribute, so we handle that here as well
  if(n->hasAttribute(attr::other) || n->hasAttribute(attr::exponent)) {
    env.s(std::to_string(i), scalarValue(n->t(attr::other)));
  }
  // we also add any other scalar tensors to the env for special ops
  for(auto a : n->attributeNames()) {
    if(n->kindOf(a) == AttributeKind::t) {
      auto v = n->t(a);
      if(v.dim() == 0) {
        JIT_ASSERT(a.is_attr());
        env.s(a.toUnqualString(), scalarValue(v));
      }
    }
  }
  const auto & str = simple_map_ops.at(n->kind());
  return format(str, env);
}

std::vector<ConcatDesc> emitCompilationUnit(std::ostream & out,
                                            const std::string & name,
                                            AnnotatedGraph & agraph,
                                            bool use_cuda) {
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
  {
    size_t i = 0;
    for(auto p : subgraph.inputs())
      emitFormal(p,agraph.input_desc[i++]);
  }
  std::vector<ConcatDesc> concat_desc;
  std::vector<Value*> flat_output_nodes;
  {
    size_t i = 0;
    for(auto o : subgraph.outputs()) {
      auto & desc = agraph.output_desc[i++];
      if(o->node()->kind() != aten::cat) {
        emitFormal(o, desc);
        concat_desc.emplace_back();
        flat_output_nodes.push_back(o);
      } else {
        auto cat = o->node();
        size_t nInputs = cat->inputs().size();
        concat_desc.emplace_back(desc, nInputs, cat->i(attr::dim));
        for(auto c : cat->inputs()) {
          emitFormal(c, *concat_desc.back().subtensorDesc);
          flat_output_nodes.push_back(c);
        }
      }
    }
  }
  size_t formal_count = 0;
  for(auto p : subgraph.inputs()) {
    env.s("node",valueName(p));
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    //TODO: actual type propagation rather than relying on auto..
    body << format("auto ${node} = ${access};\n",env);
  }
  for(auto n : subgraph.nodes()) {
    if(n->kind() == aten::cat)
      continue; // Concat nodes by narrowing the output Tensors before the kernel runs
    env.s("node",valueName(n->output()));
    env.s("rhs", encodeRHS(n));
    body << format("auto ${node} = ${rhs};\n",env);
  }
  for(auto o : flat_output_nodes) {
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    env.s("node",valueName(o));
    body << format("${access} = ${node};\n",env);
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
  return concat_desc;
}

////////////////////////////////////////////////////////////////////////////////

} // codegen namespace
} // anonymous namespace

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

CompiledFusionFunction::CompiledFusionFunction(const std::string & name, AnnotatedGraph & agraph)
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
  JIT_ASSERT(!cont.back() || strides.back() == 1);
}

} // anonymous namespace

void CompiledFusionFunction::launch_with_tensors(at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs) {
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
  launch_raw(numel, arguments.data());
}

void CompiledFusionFunction::launch(at::ArrayRef<at::Tensor> inputs, std::vector<at::Tensor> & outputs) {
  AutoGPU guard(inputs.back());
  outputs.clear();
  outputs.reserve(outputDescriptors().size());
  for(auto & od : outputDescriptors()) {
    outputs.push_back(at::getType(backend(),od.scalar_type).tensor());
  }
  launch_with_tensors(inputs, outputs);
}

#ifdef WITH_CUDA

void checkCUDAVersion(const cudaDeviceProp & prop) {
  if ((prop.major >= 6 && CUDA_VERSION < 8000) ||
      (prop.major >= 7 && CUDA_VERSION < 9000)) {
    std::stringstream err_string;
    err_string << "In CompiledFusionFunction, PyTorch compiled with insufficient CUDA version: "
         << CUDA_VERSION << " for the current GPU device " << prop.name
         << " with device capability " << prop.major << "." << prop.minor;
    throw std::runtime_error(err_string.str());
  }
}

struct CUDAFusionFunction : public CompiledFusionFunction {
  CUDAFusionFunction(const std::string & name, AnnotatedGraph & agraph)
  : CompiledFusionFunction(name, agraph) {
    AutoGPU gpu_guard(agraph.device);

    TORCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, agraph.device));
    checkCUDAVersion(prop);

    std::stringstream cu;
    concat_desc = codegen::emitCompilationUnit(cu, name, agraph, true);
    compilation_unit = cu.str();
    nvrtcProgram program;
    TORCH_NVRTC_CHECK(nvrtcCreateProgram(&program, compilation_unit.c_str(), NULL, 0, nullptr, nullptr));

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
      TORCH_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    });
    TORCH_NVRTC_CHECK(result);

    size_t ptx_size;
    TORCH_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
    ptx.resize(ptx_size);
    TORCH_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));

    TORCH_CU_CHECK(cuModuleLoadData(&module, ptx.data()));
    TORCH_CU_CHECK(cuModuleGetFunction(&function, module, name.c_str()));

    TORCH_CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks, function, 128, 0));
    maxBlocks *= prop.multiProcessorCount;
  }
  virtual ~CUDAFusionFunction() override {
    TORCH_CU_CHECK(cuModuleUnload(module));
  }
protected:
  virtual at::Backend backend() const override {
    return at::kCUDA;
  }
  virtual void launch_raw(uint32_t numel, void ** arguments) override {
     int numBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));
     //std::cout << "maxBlocks = " << maxBlocks << " needed blocks: " << ceilDiv(numel,blockSize)
     //          << " numblocks =  " << numBlocks;

     // it is possible that this is the first cuda call on this thread
     // so make sure we initialize the Driver API's context
     // cudaFree(0) accomplishes this.
     cudaFree(0);

     TORCH_CU_CHECK(cuLaunchKernel(
       function,
       numBlocks, 1, 1,
       blockSize, 1, 1,
       0, nullptr,
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
    barf("error in dlopen or dlsym: %s", dlerror());
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
    int r = dlclose(handle);
    if(r) {
      barf("error in dlclose: %s", dlerror());
    }
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
static const std::string compile_string =
  "\"${cxx}\" -O3 -g "
#ifndef __PPC64__
  "-march=native "
#endif
  "-std=c++11 -fPIC ${fopenmp} -shared \"${cpp_file}\" -o \"${so_file}\"";

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
  JIT_ASSERT(r == 0);
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

struct CPUFusionFunction : public CompiledFusionFunction {
  CPUFusionFunction(const std::string & name, AnnotatedGraph & agraph, FusionCompilerConfig & config)
  : CompiledFusionFunction(name, agraph) {
    TempFile so_file(so_template, 3);
    TempFile cpp_file(cpp_template, 4);

    std::stringstream cu;
    concat_desc = codegen::emitCompilationUnit(cu, name, agraph, false);
    compilation_unit = cu.str();
    cpp_file.write(compilation_unit);
    cpp_file.sync();
    runCompiler(config, cpp_file.name(), so_file.name());
    if(config.debug) {
      std::cout << compilation_unit << "\n";
      disas(so_file.name());
    }
    so_lib.reset(new DynamicLibrary(so_file.name().c_str()));
#pragma GCC diagnostic ignored "-Wpedantic"
    kernel = reinterpret_cast<void(*)(uint32_t, void**)>(so_lib->sym(name.c_str()));
#pragma GCC diagnostic pop
  }
protected:
  virtual at::Backend backend() const override {
    return at::kCPU;
  }
  virtual void launch_raw(uint32_t numel, void ** arguments) override {
    kernel(numel, arguments);
  }
  std::unique_ptr<DynamicLibrary> so_lib;
  void (*kernel)(uint32_t, void**) = nullptr;
};

std::shared_ptr<CompiledFusionFunction> FusionCompiler::getOrCompile(AnnotatedGraph & agraph) {
  std::stringstream key;
  key << *agraph.graph << "\n";
  key << "device " << agraph.device << "\n";
  for(auto & i : agraph.input_desc)
    key << i << "\n";
  for(auto & i : agraph.output_desc)
    key << i << "\n";
  std::string key_ = key.str();

  auto it = cache.find(key_);
  if (it == cache.end()) {
    std::string name = "kernel_" + std::to_string(cache.size());
    CompiledFusionFunction * raw_func;
    if(agraph.device != kCPUDevice) {
#ifdef WITH_CUDA
      raw_func = new CUDAFusionFunction(name, agraph);
#else
      throw std::runtime_error("cannot compile a CUDA fusion group, CUDA is not enabled.");
#endif
    } else {
      JIT_ASSERT(canCompileOnCPU());
      raw_func = new CPUFusionFunction(name, agraph, config_);
    }
    it = cache.emplace(key_, std::shared_ptr<CompiledFusionFunction>(raw_func)).first;
  }
  return it->second;
}

std::shared_ptr<CompiledFusionFunction> FusionCompiler::getOrCompile(Node* fusion_group) {
  auto & graph = *fusion_group->g(attr::Subgraph);
  AnnotatedGraph agraph(graph, fusion_group->i(attr::device));
  for(auto & input : graph.inputs()) {
    auto t = input->type()->expect<TensorType>();
    agraph.input_desc.emplace_back(t);
  }
  for(auto & output : graph.outputs()) {
    auto t = output->type()->expect<TensorType>();
    agraph.output_desc.emplace_back(t);
  }
  return getOrCompile(agraph);
}


std::shared_ptr<CompiledFusionFunction> FusionCompiler::getOrCompile(Graph & graph,
                                                     int device,
                                                     at::ArrayRef<at::Tensor> inputs,
                                                     at::ArrayRef<at::Tensor> outputs) {
  AnnotatedGraph agraph(graph, device);
  for(auto & i : inputs) {
   agraph.input_desc.emplace_back(i);
  }
  for(auto & i : outputs) {
   agraph.output_desc.emplace_back(i);
  }
  return getOrCompile(agraph);
}

void FusionCompiler::debugLaunchGraph(Graph & graph, int device, at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs) {
  auto func = getOrCompile(graph, device, inputs, outputs);
  func->launch_with_tensors(inputs, outputs);
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
#ifdef WITH_CUDA
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

CompiledFusionFunction::CompiledFusionFunction(const std::string & name, AnnotatedGraph & agraph) {}

void CompiledFusionFunction::launch_with_tensors(at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs) {}

void CompiledFusionFunction::launch(at::ArrayRef<at::Tensor> inputs, std::vector<at::Tensor> & outputs) {}

std::shared_ptr<CompiledFusionFunction> FusionCompiler::getOrCompile(AnnotatedGraph & agraph) {
  return nullptr;
}

std::shared_ptr<CompiledFusionFunction> FusionCompiler::getOrCompile(Node* fusion_group) {
  return nullptr;
}


std::shared_ptr<CompiledFusionFunction> FusionCompiler::getOrCompile(Graph & graph,
                                                     int device,
                                                     at::ArrayRef<at::Tensor> inputs,
                                                     at::ArrayRef<at::Tensor> outputs) {
  return nullptr;
}

void FusionCompiler::debugLaunchGraph(Graph & graph, int device, at::ArrayRef<at::Tensor> inputs, at::ArrayRef<at::Tensor> outputs) {}

FusionCompiler::FusionCompiler() {}

FusionCompiler & sharedFusionCompiler() {
  throw std::runtime_error("NYI: fuser is not supported on Windows.");
}

}}

# endif
