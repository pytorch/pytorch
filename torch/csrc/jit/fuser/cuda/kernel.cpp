#include <torch/csrc/jit/fuser/cuda/kernel.h>
#include <torch/csrc/jit/fuser/common/code_write.h>
#include <iostream>

#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

#define KERNEL_NAME "kernel"

namespace {
// See NOTE [ USE OF NVRTC AND DRIVER API ]
static const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

std::string codeGeneration(Fusion& fusion) {
  std::stringstream str_stream;
  CodeWrite cw(str_stream);
  cw.traverse(&fusion);
  
  return str_stream.str();
  /*
  std::string kernel_name = "";
  std::string kernel_string = "namespace Fuser {\n";
  kernel_string += Fuser::typeinfo + std::string("\n");
  kernel_string += Fuser::saxpy_codegen(kernel_name);
  kernel_string += std::string("\n}");

  std::cout << "---------------------" << std::endl;
  std::cout << kernel_string << std::endl;
  std::cout << "---------------------" << std::endl;
  auto func_name = "Fuser::" + kernel_name + "<" +
      Fuser::getTypeName<Fuser::IO_struct<float, int32_t, 4>>() + ">";
  std::cout << func_name << std::endl;
   */
};

} // namespace

void compileKernel(Fusion& fusion, CudaKernel& entry) {
  std::cout << "compiling kernel" << std::endl;

  // generating cuda code;
  std::string code = codeGeneration(fusion);

  std::cout << code << std::endl;

  std::cout << "data structure:" << std::endl;
  std::cout << typeinfo << std::endl;

  // vvv COMPILATION vvv 
  /*

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
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str());
  }
  const char *lowered_kernel_name;
  nvrtc().nvrtcGetLoweredName(program, func_name.c_str(), &lowered_kernel_name);

  ResourceGuard holdProgram(
      [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });
  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx;
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx.data()));

  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&(entry.module_), ptx.data()));
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&(entry.function_), entry.module_, lowered_kernel_    name));
  AT_CUDA_DRIVER_CHECK(nvrtc().cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &entry.maxBlocks_, entry.function_, 128, 0));
  entry.maxBlocks_ *= prop->multiProcessorCount;
  */
}

TORCH_API void runKernel(
    CudaKernel& entry,
    const at::ArrayRef<IValue>& inputs,
    std::vector<at::Tensor>& outputs) {

  for (auto& output : outputs) {
    output.fill_(0.24);
  }
}

}}}} // namespace torch::jit::fuser::cuda
