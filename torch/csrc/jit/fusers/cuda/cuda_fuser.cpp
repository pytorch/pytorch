#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)

#include "torch/csrc/jit/fusers/cuda/cuda_fuser.h"
#include "torch/csrc/jit/fusers/cuda/resource_strings.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/variable_tensor_functions.h"
#include "torch/csrc/jit/assertions.h"

#include "ATen/ATen.h"
#include "ATen/DeviceGuard.h"
#include "ATen/cuda/CUDAContext.h"
#include "THC/THC.h"
#include "THC/THCGenerator.hpp"
#include "torch/csrc/cuda/cuda_check.h"

#include "nvrtc.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iostream>
#include <dlfcn.h>

namespace torch { namespace jit { namespace cudafuser {

// Note: there is only one CUDAFuser
static CUDAFuser fuser;

CUDAFuser& getCUDAFuser() {
  return fuser;
}

static const std::string check_exists_string =
  "which '${program}' > /dev/null";

static bool programExists(const std::string& program) {
  TemplateEnv env;
  env.s("program", program);
  std::string cmd = format(check_exists_string, env);
  return 0 == system(cmd.c_str());
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

static std::string valueName(Value* n) {
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

static const char* scalarTypeName(at::ScalarType type) {
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

std::string encodeRHS(Node* n) {
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

  const auto& str = simple_map_ops.at(n->kind());
  return format(str, env);
}

std::pair<std::vector<ConcatDesc>, bool> emitCompilationUnit(
  std::ostream& out
, const std::string& name
, AnnotatedGraph& agraph
, bool use_cuda) {
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
  auto emitFormal = [&](Value* n, const TensorDesc& desc) {
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
      if(o->node()->kind() != prim::FusedConcat) {
        emitFormal(o, desc);
        concat_desc.emplace_back();
        flat_output_nodes.push_back(o);
      } else {
        auto cat = o->node();
        concat_desc.emplace_back(desc, cat->inputs().size(), cat->i(attr::dim));
        for(auto c : cat->inputs()) {
          emitFormal(c, *concat_desc.back().subtensorDesc);
          flat_output_nodes.push_back(c);
        }
      }
    }
  }

  bool has_half_tensor = false;
  size_t formal_count = 0;
  for(auto p : subgraph.inputs()) {
    env.s("node",valueName(p));
    env.d("formal",formal_count++);

    // Acquires and converts (if needed) inputs
    auto pt = p->type()->cast<TensorType>();
    if (use_cuda && pt && pt->scalarType() == at::ScalarType::Half) {
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
    if(n->kind() == aten::rand_like) {
      has_random = true;
      if(!use_cuda)
        throw std::runtime_error("Fusion doesn't support rand on CPU");
    }
    env.s("node",valueName(n->output()));
    env.s("rhs", encodeRHS(n));
    body << format("auto ${node} = ${rhs};\n",env);
  }

  for(auto o : flat_output_nodes) {
    env.d("formal",formal_count++);
    env.s("access",format("t${formal}.data[t${formal}_offset]",env));
    env.s("node",valueName(o));

    // Acquires and converts (if needed) outputs
    auto ot = o->type()->cast<TensorType>();
    if (use_cuda && ot && ot->scalarType() == at::ScalarType::Half) {
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
  out << cuda_compilation_unit_template.format(env);

  return std::make_pair(std::move(concat_desc), has_random);
}

CUDAFuser::CUDAFuser() {
  const char* cxx_env = getenv("CXX");
  if (cxx_env != nullptr) config_.cxx = cxx_env;
  if (!programExists(config_.cxx)) config_.cxx = "";
  const char* debug_env = getenv("PYTORCH_FUSION_DEBUG");
  config_.debug = debug_env && atoi(debug_env) != 0;
}

static void checkCUDAVersion(const cudaDeviceProp& prop) {
  if ((prop.major >= 6 && CUDA_VERSION < 8000) ||
      (prop.major >= 7 && CUDA_VERSION < 9000)) {
    std::stringstream err_string;
    err_string << "PyTorch compiled with insufficient CUDA version for fusion: "
         << CUDA_VERSION << " for the current GPU device " << prop.name
         << " with device capability " << prop.major << "." << prop.minor;
    throw std::runtime_error(err_string.str());
  }
}

CUDAFusionFunction* CUDAFuser::constructFusionFunction(
  std::string name
, AnnotatedGraph& agraph) {
  CUDAFusionFunction* func = new CUDAFusionFunction(name, agraph);

  at::DeviceGuard device_guard(agraph.device);

  CUDA_ASSERT(cudaGetDeviceProperties(&func->prop, agraph.device));
  checkCUDAVersion(func->prop);

  std::stringstream cu;
  auto ret = emitCompilationUnit(cu, name, agraph, true);
  func->concat_desc = std::move(ret.first);
  func->has_random = ret.second;
  func->compilation_unit = cu.str();
  nvrtcProgram program;
  NVRTC_ASSERT(nvrtcCreateProgram(
    &program
  , func->compilation_unit.c_str()
  , NULL
  , 0
  , nullptr
  , nullptr));

  std::string compute = "--gpu-architecture=compute_" + std::to_string(func->prop.major) + std::to_string(func->prop.minor);
  std::vector<const char *> args = {"--std=c++11", compute.c_str(), "-default-device"};
  nvrtcResult result = nvrtcCompileProgram(program, args.size(), args.data());
  
  // Fails on error
  // Note: special cases compilation error
  if (result == NVRTC_ERROR_COMPILATION) {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
    cu << log.data();
    throw std::runtime_error(cu.str());
  }
  NVRTC_ASSERT(result);

  ResourceGuard holdProgram([&] {
    NVRTC_ASSERT(nvrtcDestroyProgram(&program));
  });

  size_t ptx_size;
  NVRTC_ASSERT(nvrtcGetPTXSize(program, &ptx_size));
  func->ptx.resize(ptx_size);
  NVRTC_ASSERT(nvrtcGetPTX(program, func->ptx.data()));
  
  CU_ASSERT(cuModuleLoadData(&func->module, func->ptx.data()));
  CU_ASSERT(cuModuleGetFunction(&func->function, func->module, func->name.c_str()));

  CU_ASSERT(cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &func->maxBlocks
  , func->function
  , 128
  , 0));
  func->maxBlocks *= func->prop.multiProcessorCount;

  return func;
}

std::shared_ptr<CUDAFusionFunction> CUDAFuser::getOrCompile(
  AnnotatedGraph& agraph) {
  
  JIT_ASSERT(agraph.device != kCPUDevice);

  // Constructs key (for caching)
  // TODO: abstract caching
  std::stringstream key;
  key << *agraph.graph << "\n";
  key << "device " << agraph.device << "\n";
  for (auto& i : agraph.input_desc) key << i << "\n";
  for (auto& i : agraph.output_desc) key << i << "\n";
  std::string key_ = key.str();

  auto it = cache.find(key_);
  if (it == cache.end()) {  
    std::string name = "kernel_" + std::to_string(cache.size());
    CUDAFusionFunction* raw_func = constructFusionFunction(name, agraph);
    it = cache.emplace(key_, std::shared_ptr<CUDAFusionFunction>(raw_func)).first;
  }

  return it->second;
}

void CUDAFuser::debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs
, at::ArrayRef<at::Tensor> outputs) {
  AnnotatedGraph agraph(graph, device);
  for (auto& i : inputs) agraph.input_desc.emplace_back(i);
  for (auto& i : outputs) agraph.output_desc.emplace_back(i);
  auto func = getOrCompile(agraph);
  func->launch_with_tensors(inputs, outputs);
}

} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
