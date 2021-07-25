#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/fuser/cuda/fused_kernel.h>
#include <torch/csrc/jit/resource_guard.h>

#include <nvfuser_resources/block_reduction.h>
#include <nvfuser_resources/broadcast.h>
#include <nvfuser_resources/fp16_support.h>
#include <nvfuser_resources/grid_reduction.h>
#include <nvfuser_resources/helpers.h>
#include <nvfuser_resources/random_numbers.h>
#include <nvfuser_resources/tensor.h>

#include <fstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace executor_utils {

std::string kernelPreamble() {
  std::stringstream ss;

#ifndef __HIP_PLATFORM_HCC__
  ss << nvfuser_resources::fp16_support_cu;
#else
  ss << R"(
#ifndef __noinline__
#define __noinline__ __attribute__((noinline))
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef assert
#define assert(expr) ((void)0)
#endif
#ifndef __align__
#define __align__(x) __attribute__((aligned(x)))
#endif
  )";
#endif

  ss << nvfuser_resources::tensor_cu;
  ss << nvfuser_resources::random_numbers_cu;
  ss << nvfuser_resources::helpers_cu;
  ss << nvfuser_resources::block_reduction_cu;
  ss << nvfuser_resources::grid_reduction_cu;
  ss << nvfuser_resources::broadcast_cu;

  return ss.str();
}

namespace {

// return false if arg's type, number of dimensions, and device, doesn't match
// param and provided c10:device
bool validateKernelArgTensor(
    const at::Tensor& arg,
    const Val* param,
    const c10::Device& device,
    std::stringstream& msg) {
  // Arg is a tensor. Param must be a tensor too.
  if (*param->getValType() != ValType::TensorView) {
    msg << "Argument is a tensor, but the parameter is not.\n";
    return false;
  }

  // Check the rank of the tensors.
  size_t arg_dim = arg.dim();
  // Note: This requires current Fusion to be active.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t param_dim =
      TensorDomain::noReductions(param->as<TensorView>()->getRootDomain())
          .size();
  // see [Note - broadcast support in integration]
  // Because of broadcasting support handled in integration, we relax the rank
  // check as necessary.
  if (arg_dim > param_dim) {
    msg << "Argument tensor's rank is " << arg_dim << ", but the parameter is "
        << param_dim << "\n";
    return false;
  }

  if (arg.device() != device) {
    msg << "Argument is on device that is not compiled for."
        << "\n";
    return false;
  }
  // Check element type
  at::ScalarType arg_data_type = arg.scalar_type();
  DataType param_data_type = *param->getDataType();
  bool match = false;
  switch (arg_data_type) {
    case at::ScalarType::Half:
      match = param_data_type == DataType::Half;
      break;
    case at::ScalarType::Float:
      match = param_data_type == DataType::Float;
      break;
    case at::ScalarType::Bool:
      match = param_data_type == DataType::Bool;
      break;
    default:
      msg << "Argument element type, " << arg_data_type << ", is not supported."
          << "\n";
      return false;
  }
  if (!match)
    msg << "Argument element type is " << arg_data_type
        << ", but the parameter is " << param_data_type << "\n";
  return match;
}

// Return false if  arg_type doesn't match the type in param
bool validateKernelArgScalar(
    const c10::TypePtr& arg_type,
    const Val* param,
    std::stringstream& msg) {
  if (!param->isScalar()) {
    msg << "Argument is a scalar, but the parameter is not."
        << "\n";
    return false;
  }
  DataType param_type = *param->getDataType();
  bool match = false;
  switch (arg_type->kind()) {
    case c10::TypeKind::IntType:
      match = param_type == DataType::Int;
      break;
    case c10::TypeKind::FloatType:
      match = param_type == DataType::Float;
      break;
    case c10::TypeKind::BoolType:
      match = param_type == DataType::Bool;
      break;
    default:
      match = false;
  }
  if (!match) {
    msg << "Argument type is " << *arg_type << ", but the parameter is "
        << param_type << "\n";
  }
  return match;
}

// Return false if arg and param don't match up and if arg's device (if a
// tensor) doesn't match provided device
bool validateKernelArg(
    const c10::IValue& arg,
    const Val* param,
    const c10::Device& device,
    std::stringstream& msg) {
  if (arg.isTensor()) {
    return validateKernelArgTensor(arg.toTensor(), param, device, msg);
  } else {
    return validateKernelArgScalar(arg.type(), param, msg);
  }
}

} // namespace

void validateKernelInputs(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    const c10::Device& device) {
  FUSER_PERF_SCOPE("validateKernelInputs");

  // This is necessary as we were traversing the fusion graph later in the check
  FusionGuard fg(fusion);
  // Check inputs
  TORCH_INTERNAL_ASSERT(
      inputs.size() == fusion->inputs().size(),
      "Wrong number of kernel inputs.");

  std::stringstream msg;
  bool mismatch = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const IValue& arg = inputs[i];
    const Val* param = fusion->inputs()[i];
    mismatch = !validateKernelArg(arg, param, device, msg) || mismatch;
  }
  TORCH_INTERNAL_ASSERT(
      !mismatch, "Found one or more invalid arguments: ", msg.str());
}

void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    const c10::Device& device) {
  FUSER_PERF_SCOPE("validateKernelOutputs");

  TORCH_INTERNAL_ASSERT(
      fusion->outputs().size() != 0,
      "Kernel should have at least one output tensor.");

  TORCH_INTERNAL_ASSERT(
      outputs.size() == fusion->outputs().size(),
      "Wrong number of kernel outputs.");

  std::stringstream msg;
  bool mismatch = false;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const at::Tensor& arg = outputs[i];
    const Val* param = fusion->outputs()[i];
    mismatch = !validateKernelArg(arg, param, device, msg) || mismatch;
  }
  TORCH_INTERNAL_ASSERT(
      !mismatch, "Found one or more invalid arguments: ", msg.str());
}

StatefulExpressionEvaluator statefulBindInputs(
    const at::ArrayRef<IValue>& aten_inputs,
    Fusion* fusion,
    GpuLower* lower) {
  FUSER_PERF_SCOPE("statefulBindInputs");

  TORCH_INTERNAL_ASSERT(
      fusion->inputs().size() == aten_inputs.size(),
      "Something went wrong configuring launch. Inputs no longer match.");

  auto fusion_inputs = fusion->inputs();
  StatefulExpressionEvaluator evaluator(fusion);

  // This should probably move to EvaluationContext as we may want to bind
  // input values frequently. Bind fusion input values to runtime values.
  for (const auto i : c10::irange(fusion->inputs().size())) {
    if (fusion->inputs()[i]->getValType() == ValType::TensorView) {
      TensorView* cg_tensor = fusion->inputs()[i]->as<TensorView>();

      TORCH_INTERNAL_ASSERT(
          aten_inputs[i].isTensor(),
          "Something went wrong configuring launch. Inputs no longer match.");

      auto aten_tensor = aten_inputs[i].toTensor();
      auto root_dom = TensorDomain::noReductions(cg_tensor->getRootDomain());
      TORCH_INTERNAL_ASSERT(
          aten_tensor.ndimension() == (int64_t)root_dom.size(),
          "Something went wrong configuring launch. Inputs no longer match.");

      for (const auto dim : c10::irange(root_dom.size())) {
        evaluator.safeBind(
            root_dom[dim]->extent(), aten_tensor.sizes()[dim], lower);
      }
    } else if (
        fusion->inputs()[i]->getValType().value() == ValType::Scalar &&
        fusion->inputs()[i]->getDataType().value() == DataType::Int) {
      TORCH_INTERNAL_ASSERT(
          aten_inputs[i].type()->kind() == c10::TypeKind::IntType);
      evaluator.safeBind(fusion->inputs()[i], aten_inputs[i].toInt(), lower);
    }
  }
  return evaluator;
}

NvrtcFunction nvrtcCompile(
    const std::string& code,
    const std::string& func_name,
    int id) {
  FUSER_PERF_SCOPE("NVRTC");

  // lazily construct context if non-existing yet;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(nullptr);
  }

  const auto prop = at::cuda::getCurrentDeviceProperties();

  int major = 0, minor = 0;
  bool compile_to_sass = false;
  codegenOutputQuery(prop, major, minor, compile_to_sass);

  nvrtcProgram program; // NOLINT(cppcoreguidelines-init-variables)

  {
    FUSER_PERF_SCOPE("nvrtcCreateProgram");
    AT_CUDA_NVRTC_CHECK(at::globalContext().getNVRTC().nvrtcCreateProgram(
        &program, code.c_str(), nullptr, 0, nullptr, nullptr));
  }

  ResourceGuard holdProgram([&] {
    FUSER_PERF_SCOPE("nvrtcDestroyProgram");
    AT_CUDA_NVRTC_CHECK(
        at::globalContext().getNVRTC().nvrtcDestroyProgram(&program));
  });

#ifdef __HIP_PLATFORM_HCC__
  std::vector<const char*> args = {"--std=c++14"};
#if ROCM_VERSION >= 40200
  args.push_back("-hip-pch");
#endif
#else
  const std::string compute = std::string("--gpu-architecture=") +
#if CUDA_VERSION >= 11010
      // CUDA 11.1 allows going directly to SASS (sm_) instead of PTX (compute_)
      // which gives better backwards compatibility to work on older driver,
      // (since older driver doesn't necessrily recognize PTX emitted by new
      // toolkit);
      // Meanwhile, for forward compatibility (future device with
      // `unsupported_arch==True`), since SASS are not necessarily compatible,
      // we fallback to PTX instead.
      (compile_to_sass ? "sm_" : "compute_") +
#else
      "compute_" +
#endif
      std::to_string(major) + std::to_string(minor);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};
#endif

  const char* disable_fma = getenv("PYTORCH_CUDA_FUSER_DISABLE_FMA");
  // int disable_fma_flag = disable_fma ? atoi(disable_fma) : 0;
  if (disable_fma && atoi(disable_fma)) {
#ifdef __HIP_PLATFORM_HCC__
    TORCH_WARN_ONCE(
        "PYTORCH_CUDA_FUSER_DISABLE_FMA is not supported on ROCm, ignoring");
#else
    args.push_back("--fmad=false");
#endif
  }

  const char* ptxas_opt_level = getenv("PYTORCH_CUDA_FUSER_JIT_OPT_LEVEL");
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  uint32_t jit_opt_level;

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<CUjit_option> options;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<void*> option_vals;

  if (ptxas_opt_level) {
    int val = atoi(ptxas_opt_level);
    if (val <= 4 && val >= 0) {
      jit_opt_level = static_cast<uint32_t>(val);
      options.push_back(CU_JIT_OPTIMIZATION_LEVEL);
      option_vals.emplace_back(&jit_opt_level);
    } else {
      TORCH_WARN_ONCE(
          "acceptable range for PYTORCH_CUDA_FUSER_JIT_OPT_LEVEL is between 0 and 4, but received ",
          jit_opt_level,
          ", ignoring the option");
    }
  }

  at::globalContext().getNVRTC().nvrtcAddNameExpression(
      program, func_name.c_str());

  {
    FUSER_PERF_SCOPE("nvrtcCompileProgram");

    const auto result = at::globalContext().getNVRTC().nvrtcCompileProgram(
        program, args.size(), args.data());

    if (result != NVRTC_SUCCESS) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t logsize;
      at::globalContext().getNVRTC().nvrtcGetProgramLogSize(program, &logsize);
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      std::vector<char> log(logsize);
      at::globalContext().getNVRTC().nvrtcGetProgramLog(program, log.data());

      TORCH_INTERNAL_ASSERT(
          false, code.c_str(), "\nCUDA NVRTC compile error: ", log.data());
    }

    AT_CUDA_NVRTC_CHECK(result);
  }

  const char* lowered_kernel_name = nullptr;
  at::globalContext().getNVRTC().nvrtcGetLoweredName(
      program, func_name.c_str(), &lowered_kernel_name);

  size_t ptx_size = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<char> ptx;

  {
    FUSER_PERF_SCOPE("get PTX");
#if CUDA_VERSION >= 11010
    // compile_to_sass determines whether we are generating SASS or PTX, hence
    // the different API.
    const auto getSize = compile_to_sass
        ? at::globalContext().getNVRTC().nvrtcGetCUBINSize
        : at::globalContext().getNVRTC().nvrtcGetPTXSize;
    const auto getFunc = compile_to_sass
        ? at::globalContext().getNVRTC().nvrtcGetCUBIN
        : at::globalContext().getNVRTC().nvrtcGetPTX;
#else
    const auto getSize = at::globalContext().getNVRTC().nvrtcGetPTXSize;
    const auto getFunc = at::globalContext().getNVRTC().nvrtcGetPTX;
#endif
    AT_CUDA_NVRTC_CHECK(getSize(program, &ptx_size));
    ptx.resize(ptx_size);
    AT_CUDA_NVRTC_CHECK(getFunc(program, ptx.data()));
  }

  NvrtcFunction compiled_kernel_;

  // TODO: We do go through different code path, should investigate whether this
  // has an impact on generated binary.
  const char* prefix_env = getenv("PYTORCH_CUDA_FUSER_CUBIN");
#ifndef __HIP_PLATFORM_HCC__
  if (prefix_env) {
#if CUDA_VERSION >= 11010
    TORCH_CHECK(
        !compile_to_sass,
        "PYTORCH_NVFUSER_CUBIN cannot be used when compile direct to SASS. Please set PYTORCH_NVFUSER_CUBIN to empty");
#endif
    FUSER_PERF_SCOPE("load CUBIN");

    // Output ptx file
    std::stringstream ptx_file_name;
    ptx_file_name << prefix_env << "_" << id << ".ptx";
    std::ofstream myPtxFile(ptx_file_name.str().c_str(), std::ios::out);
    if (myPtxFile.is_open()) {
      myPtxFile.write(ptx.data(), ptx.size());
      myPtxFile.close();
    }

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    CUlinkState linkState;

    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkCreate(
        0, nullptr, nullptr, &linkState));

    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkAddData(
        linkState,
        CU_JIT_INPUT_PTX,
        ptx.data(),
        ptx_size,
        "compiling PTX",
        options.size(),
        options.data(),
        option_vals.data()));

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t cubinSize;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void* cubin;
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkComplete(
        linkState, &cubin, &cubinSize));

    // Output binary file
    std::stringstream cubin_file_name;
    cubin_file_name << prefix_env << "_" << id << ".cubin";

    std::ofstream myCubinFile(
        cubin_file_name.str().c_str(), std::ios::out | std::ios::binary);

    if (myCubinFile.is_open()) {
      myCubinFile.write(static_cast<const char*>(cubin), cubinSize);
      myCubinFile.close();
    }
    // load compiled cubin
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadData(
        &(compiled_kernel_.module), cubin));
  } else {
    FUSER_PERF_SCOPE("load PTX");

    // load ptx directly
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadDataEx(
        &(compiled_kernel_.module),
        ptx.data(),
        options.size(),
        options.data(),
        option_vals.data()));
  }
#else
  // load ptx directly
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadData(
      &(compiled_kernel_.module), ptx.data()));

#endif
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleGetFunction(
      &(compiled_kernel_.function),
      compiled_kernel_.module,
      lowered_kernel_name));

  return compiled_kernel_;
}

} // namespace executor_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
