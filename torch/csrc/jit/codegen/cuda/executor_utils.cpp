#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <torch/csrc/jit/resource_guard.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_resource_strings.h>

#include <torch/csrc/jit/codegen/cuda/executor_utils.h>

#include <fstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace executor_utils {

std::string kernelPreamble() {
  std::stringstream ss;
  ss << code_template_tensor_struct << "\n"
     << code_fp16_support << "\n"
     << code_random_number_gen << "\n"
     << code_helper_funcs << "\n"
     << code_template_block_reduction << "\n"
     << code_template_grid_reduction << "\n"
     << code_template_block_broadcast << "\n";
  return ss.str();
}

namespace {

bool validateKernelArgTensor(
    const at::Tensor& arg,
    const Val* param,
    c10::Device device,
    std::stringstream& msg) {
  // Arg is a tensor. Param must be a tensor too.
  if (*param->getValType() != ValType::TensorView) {
    msg << "Argument is a tensor, but the parameter is not.";
    return false;
  }

  // Check the rank of the tensors.
  size_t arg_dim = arg.dim();
  // Note: This requires current Fusion to be active.
  size_t param_dim =
      TensorDomain::noReductions(param->as<TensorView>()->getRootDomain())
          .size();
  // see [Note - broadcast support in integration]
  // Because of broadcasting support handled in integration, we relax the rank
  // check as necessary.
  if (arg_dim > param_dim) {
    msg << "Argument tensor's rank is " << arg_dim << ", but the parameter is "
        << param_dim;
    return false;
  }

  if (arg.device() != device) {
    msg << "Argument is on device that is not compiled for";
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
      msg << "Argument element type, " << arg_data_type
          << ", is not supported.";
      return false;
  }
  if (!match)
    msg << "Argument element type is " << arg_data_type
        << ", but the parameter is " << param_data_type;
  return match;
}

bool validateKernelArgScalar(
    const c10::TypePtr& arg_type,
    const Val* param,
    std::stringstream& msg) {
  if (!param->isScalar()) {
    msg << "Argument is a scalar, but the parameter is not.";
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
        << param_type;
  }
  return match;
}

bool validateKernelArg(
    const c10::IValue& arg,
    const Val* param,
    c10::Device device,
    std::stringstream& msg) {
  if (arg.type()->kind() != c10::TypeKind::TensorType) {
    return validateKernelArgScalar(arg.type(), param, msg);
  } else {
    return validateKernelArgTensor(arg.toTensor(), param, device, msg);
  }
}

} // namespace

void validateKernelInputs(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    c10::Device device) {
  // This is necessary as we were traversing the fusion graph later in the check
  FusionGuard fg(fusion);
  // Check inputs
  TORCH_INTERNAL_ASSERT(
      inputs.size() == fusion->inputs().size(),
      "Wrong number of kernel inputs.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    const IValue& arg = inputs[i];
    const Val* param = fusion->inputs()[i];
    std::stringstream msg;
    TORCH_INTERNAL_ASSERT(
        validateKernelArg(arg, param, device, msg),
        "Input argument at position ",
        i,
        " is invalid; ",
        msg.str());
  }
}

void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    c10::Device device) {
  TORCH_INTERNAL_ASSERT(
      fusion->outputs().size() != 0,
      "Kernel should have at least one output tensor.");

  TORCH_INTERNAL_ASSERT(
      outputs.size() == fusion->outputs().size(),
      "Wrong number of kernel outputs.");
  for (size_t i = 0; i < outputs.size(); ++i) {
    const at::Tensor& arg = outputs[i];
    const Val* param = fusion->outputs()[i];
    std::stringstream msg;
    TORCH_INTERNAL_ASSERT(
        validateKernelArgTensor(arg, param, device, msg),
        "Output argument at position ",
        i,
        " is invalid; ",
        msg.str());
  }
}

void safeBind(
    EvaluationContext& ec,
    const Val* value,
    Int::ScalarType concrete_value) {
  auto already_concrete_val = ec.concreteValue(value);

  if (already_concrete_val.has_value()) {
    TORCH_INTERNAL_ASSERT(
        concrete_value == already_concrete_val.value(),
        "Tried to bind ",
        value,
        " to ",
        " concrete value, but it's already set to ",
        already_concrete_val.value());
  } else {
    ec.bind(value, concrete_value);
  }
}

EvaluationContext bindInputs(
    const at::ArrayRef<IValue>& aten_inputs,
    Fusion* fusion) {
  TORCH_INTERNAL_ASSERT(
      fusion->inputs().size() == aten_inputs.size(),
      "Something went wrong configuring launch. Inputs no longer match.");

  auto fusion_inputs = fusion->inputs();
  EvaluationContext eval_context(fusion);

  // This should probably move to EvaluationContext as we may want to bind
  // input values frequently. Bind fusion input values to runtime values.
  for (size_t i = 0; i < fusion->inputs().size(); i++) {
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

      for (size_t dim = 0; dim < root_dom.size(); dim++) {
        safeBind(
            eval_context, root_dom[dim]->extent(), aten_tensor.sizes()[dim]);
      }
    }
  }
  return eval_context;
}

NvrtcFunction nvrtcCompile(
    const std::string& code,
    const std::string& func_name,
    int id) {
  // lazily construct context if non-existing yet;
  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(nullptr);
  }

  const auto prop = at::cuda::getCurrentDeviceProperties();
  int nvrtc_major, nvrtc_minor;
  AT_CUDA_NVRTC_CHECK(
      at::globalContext().getNVRTC().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

  // Short-circuits if NVRTC version too low
  TORCH_INTERNAL_ASSERT(nvrtc_major >= 6);
  // Major and minor is determined by device properties and
  // possibly "downcompiled" to a lower (compatible) compute architecture
  // based on the NVRTC version
  const int major = prop->major;
  const int minor = prop->minor;
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(at::globalContext().getNVRTC().nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));
  ResourceGuard holdProgram([&] {
    AT_CUDA_NVRTC_CHECK(
        at::globalContext().getNVRTC().nvrtcDestroyProgram(&program));
  });

  const std::string compute = "--gpu-architecture=compute_" +
      std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};

  at::globalContext().getNVRTC().nvrtcAddNameExpression(
      program, func_name.c_str());
  const auto result = at::globalContext().getNVRTC().nvrtcCompileProgram(
      program, args.size(), args.data());

  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    at::globalContext().getNVRTC().nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    at::globalContext().getNVRTC().nvrtcGetProgramLog(program, log.data());

    TORCH_INTERNAL_ASSERT(
        false, code.c_str(), "\nCUDA NVRTC compile error: ", log.data());
  }
  const char* lowered_kernel_name;
  at::globalContext().getNVRTC().nvrtcGetLoweredName(
      program, func_name.c_str(), &lowered_kernel_name);

  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(
      at::globalContext().getNVRTC().nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx;
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(
      at::globalContext().getNVRTC().nvrtcGetPTX(program, ptx.data()));

  NvrtcFunction compiled_kernel_;

  // TODO: We do go through different code path, should investigate whether this
  // has an impact on generated binary.
  const char* prefix_env = getenv("PYTORCH_CUDA_FUSER_CUBIN");
  if (prefix_env) {
    // Output ptx file
    std::stringstream ptx_file_name;
    ptx_file_name << prefix_env << "_" << id << ".ptx";
    std::ofstream myPtxFile(ptx_file_name.str().c_str(), std::ios::out);
    if (myPtxFile.is_open()) {
      myPtxFile.write(ptx.data(), ptx.size());
      myPtxFile.close();
    }

    CUlinkState linkState;

    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkCreate(
        0, nullptr, nullptr, &linkState));

    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkAddData(
        linkState,
        CU_JIT_INPUT_PTX,
        ptx.data(),
        ptx_size,
        "compiling PTX",
        0,
        nullptr,
        nullptr));

    size_t cubinSize;
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
    // load ptx directly
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadData(
        &(compiled_kernel_.module), ptx.data()));
  }
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
