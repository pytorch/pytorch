#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <iostream>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/jit/resource_guard.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

constexpr auto CG_NAMESPACE = "CudaCodeGen";
constexpr auto KERNEL_NAME = "kernel";

namespace {
// See NOTE [ USE OF NVRTC AND DRIVER API ]
static const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

static int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

std::pair<std::string, std::string> codeGeneration(Fusion& fusion) {
  std::stringstream str_stream;

  str_stream << "namespace " << CG_NAMESPACE << " {\n" << typeinfo << "\n";
  std::stringstream cdg;
  GPULower gpulw(&fusion);
  gpulw.printKernel(str_stream, KERNEL_NAME);
  str_stream << "\n} // namespace";

  std::string func_name = std::string(CG_NAMESPACE) + "::" + KERNEL_NAME;

  return std::make_pair(func_name, str_stream.str());
};

void prepare_argument(
    std::vector<void*>& arguments,
    std::vector<Tensor<float>>& tensor_args,
    const at::Tensor& val) {
  tensor_args.emplace_back();
  Tensor<float>& t = tensor_args.back();
  // passing address, type doesn't really matter here;
  t.data = static_cast<float*>(val.data_ptr());

  for (decltype(val.dim()) i{0}; i < val.dim(); i++) {
    t.size[i] = val.sizes()[i];
    t.stride[i] = val.strides()[i];
  }

  arguments.push_back(&(tensor_args.back()));
};

void prepare_argument(
    std::vector<void*>& arguments,
    std::vector<Tensor<float>>& tensor_args,
    std::vector<int>& int_args,
    std::vector<float>& float_args,
    const IValue& val) {
  if (val.isTensor()) {
    prepare_argument(arguments, tensor_args, val.toTensor());
  } else if (val.isDouble()) {
    float_args.push_back(val.to<float>());
    arguments.push_back(&(float_args.back()));
  } else if (val.isInt()) {
    int_args.push_back(val.to<int>());
    arguments.push_back(&(int_args.back()));
  } else {
    TORCH_CHECK(false, "Not supported input IValue encounted.");
  }
};

} // namespace

void compileKernel(Fusion& fusion, CudaKernel& entry) {
  // generating cuda code;
  std::string code;
  std::string func_name;
  std::tie(func_name, code) = codeGeneration(fusion);

  // vvv NVRTC COMPILATION vvv

  // lazily construct context if non-existing yet;
  CUcontext pctx = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(0);
  }

  // set device for the operation;
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(entry.device_);

  const auto prop = at::cuda::getCurrentDeviceProperties();
  int nvrtc_major, nvrtc_minor;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

  // Short-circuits if NVRTC version too low
  AT_ASSERT(nvrtc_major >= 6);
  // Major and minor is determined by device properties and
  // possibly "downcompiled" to a lower (compatible) compute architecture
  // based on the NVRTC version
  int major, minor;
  major = prop->major;
  minor = prop->minor;
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
      &program, code.c_str(), nullptr, 0, nullptr, nullptr));
  ResourceGuard holdProgram(
      [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });

  const std::string compute = "--gpu-architecture=compute_" +
      std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> args = {
      "--std=c++11", compute.c_str(), "-default-device"};

  nvrtc().nvrtcAddNameExpression(program, func_name.c_str());
  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    nvrtc().nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtc().nvrtcGetProgramLog(program, log.data());

    TORCH_INTERNAL_ASSERT(
        false, "CUDA NVRTC compile error: ", log.data(), "\n", code.c_str());
  }
  const char* lowered_kernel_name;
  nvrtc().nvrtcGetLoweredName(program, func_name.c_str(), &lowered_kernel_name);

  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx;
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx.data()));

  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&(entry.module_), ptx.data()));
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleGetFunction(
      &(entry.function_), entry.module_, lowered_kernel_name));
  AT_CUDA_DRIVER_CHECK(nvrtc().cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &entry.max_blocks_, entry.function_, 128, 0));
  entry.max_blocks_ *= prop->multiProcessorCount;
}

void runKernel(
    CudaKernel& entry,
    const at::ArrayRef<IValue>& inputs,
    std::vector<at::Tensor>& outputs) {
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(entry.device_);
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO: Proper API to establish reasonable launch configurations;
  // Naive launch config;
  size_t numel = outputs[0].numel();
  const auto nBlocks = std::min(entry.max_blocks_, ceilDiv(numel, 128));

  // TODO: Proper API to tranform JIT I/O Tensor to CodeGen I/O Tensor
  std::vector<void*> arguments;

  // TODO: There are better ways to do this;
  // argument holder;
  // host code, `T` in `Tensor<T>` doesn't really matter, as we only interact
  // with the address; Just put a float here to simply the argument holder.
  auto max_capacity = inputs.size() + outputs.size();
  std::vector<Tensor<float>> tensor_args;
  std::vector<int> int_args;
  std::vector<float> float_args;
  tensor_args.reserve(max_capacity);
  int_args.reserve(max_capacity);
  float_args.reserve(max_capacity);

  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (auto& input : inputs) {
    prepare_argument(arguments, tensor_args, int_args, float_args, input);
  }
  for (auto& output : outputs) {
    prepare_argument(arguments, tensor_args, output);
  }

  // launch kernel;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      entry.function_,
      nBlocks,
      1,
      1,
      128,
      1,
      1,
      0,
      stream,
      arguments.data(),
      nullptr));

  // Resets device (see at::DeviceGuard notes above)
  at::cuda::set_device(prior_device);

  /*
    for (auto& output : outputs) {
      output.fill_(0.24);
    }
   */
}

// WARNING:
// This function is here for testing purposes only
void runTestKernel(
    CudaKernel& entry,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs) {
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(entry.device_);
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO: Proper API to tranform JIT I/O Tensor to CodeGen I/O Tensor
  std::vector<void*> arguments;

  // TODO: There are better ways to do this;
  // argument holder;
  // host code, `T` in `Tensor<T>` doesn't really matter, as we only interact
  // with the address; Just put a float here to simply the argument holder.
  auto max_capacity = inputs.size() + outputs.size();
  std::vector<Tensor<float>> tensor_args;
  std::vector<int> int_args;
  std::vector<float> float_args;
  tensor_args.reserve(max_capacity);
  int_args.reserve(max_capacity);
  float_args.reserve(max_capacity);

  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (auto& input : inputs) {
    prepare_argument(arguments, tensor_args, input);
  }
  for (auto& output : outputs) {
    prepare_argument(arguments, tensor_args, output);
  }

  // launch kernel;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      entry.function_,
      entry.grid_.x,
      entry.grid_.y,
      entry.grid_.z,
      entry.block_.x,
      entry.block_.y,
      entry.block_.z,
      0,
      stream,
      arguments.data(),
      nullptr));

  // Resets device (see at::DeviceGuard notes above)
  at::cuda::set_device(prior_device);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
