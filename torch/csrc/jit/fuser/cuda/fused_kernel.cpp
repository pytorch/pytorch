#include <torch/csrc/jit/fuser/cuda/fused_kernel.h>
#include <torch/csrc/jit/fuser/compiler.h>

#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/fuser/cpu/dynamic_library.h>
#include <torch/csrc/jit/fuser/cuda/thnvrtc.h>
#include <torch/csrc/jit/resource_guard.h>

// Note: unclear why this forward declaration is necessary
#include <THC/THCGenerator.hpp>
#include <THC/THCTensorRandom.h>
THCGenerator* THCRandom_getGenerator(THCState* state);

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// [USE OF NVRTC AND DRIVER API]
// libtorch does not directly link to either libnvrtc or libcuda because
// they require libcuda to be installed. Normal CUDA code in torch uses the cuda
// runtime libraries which can be installed even if the driver is not installed,
// but here we specifically need to use the driver API to load JIT compiled
// code. To accomplish this, we lazily link libthnvrtc which provides a struct
// THNVRTC that contains function pointers to all of the apis we need.
//
// IT IS AN ERROR TO TRY TO CALL ANY nvrtc* or cu* FUNCTION DIRECTLY.
// INSTEAD USE, e.g. nvrtc().cuLoadModule(...)
// If a function is missing add it to the list in thnvrtc.

void checkCUDAVersion(const cudaDeviceProp& prop) {
  if ((prop.major >= 6 && CUDA_VERSION < 8000) ||
      (prop.major >= 7 && CUDA_VERSION < 9000)) {
    std::stringstream err_string;
    err_string
        << "In CUDAFusedKernel, PyTorch compiled with insufficient CUDA version: "
        << CUDA_VERSION << " for the current GPU device " << prop.name
        << " with device capability " << prop.major << "." << prop.minor;
    throw std::runtime_error(err_string.str());
  }
}

#ifdef USE_DIRECT_NVRTC
std::pair<std::unique_ptr<cpu::DynamicLibrary>, THNVRTC*> loadNVRTC() {
  return std::make_pair(nullptr, torch_load_nvrtc());
}
#else
std::pair<std::unique_ptr<cpu::DynamicLibrary>, THNVRTC*> loadNVRTC() {
  std::string path = cpu::DynamicLibrary::directoryOf((void*)checkCUDAVersion);
#ifdef __APPLE__
  std::string libthnvrtc = path + "/libthnvrtc.dylib";
#else
  std::string libthnvrtc = path + "/libthnvrtc.so";
#endif
  std::unique_ptr<cpu::DynamicLibrary> libnvrtc_stub(
      new cpu::DynamicLibrary(libthnvrtc.c_str()));
  auto fn = (THNVRTC * (*)()) libnvrtc_stub->sym("torch_load_nvrtc");
  return std::make_pair(std::move(libnvrtc_stub), fn());
}
#endif

const THNVRTC& nvrtc() {
  // must hold onto DynamicLibrary otherwise it will unload
  static auto handle = loadNVRTC();
  return *handle.second;
}

// We're using three CUDA APIs, so define a few helpers for error handling
static inline void nvrtcCheck(nvrtcResult result, const char* file, int line) {
  if (result != NVRTC_SUCCESS) {
    std::stringstream ss;
    ss << file << ":" << line << ": " << nvrtc().nvrtcGetErrorString(result);
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_NVRTC_CHECK(result) nvrtcCheck(result, __FILE__, __LINE__);

static inline void cuCheck(CUresult result, const char* file, int line) {
  if (result != CUDA_SUCCESS) {
    const char* str;
    nvrtc().cuGetErrorString(result, &str);
    std::stringstream ss;
    ss << file << ":" << line << ": " << str;
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_CU_CHECK(result) cuCheck(result, __FILE__, __LINE__);

static void getMajorMinor(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor) {
  int nvrtc_major, nvrtc_minor;
  TORCH_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

  // Short-circuits if NVRTC version too low
  AT_ASSERT(nvrtc_major >= 6);

  // Major and minor is determined by device properties and
  // possibly "downcompiled" to a lower (compatible) compute architecture
  // based on the NVRTC version
  major = prop->major;
  minor = prop->minor;
  if (nvrtc_major <= 7 && prop->major > 5) { // 7 supports 2-5.x
    major = 5;
    minor = 0;
  } else if (nvrtc_major <= 8 && prop->major > 6) { // 8 supports 2-6.x
    major = 6;
    minor = 0;
  } else if (nvrtc_major <= 9 && prop->major >= 7) { // 9 supports 3-7.2
    major = 7;
    if (prop->major == 7 && prop->minor <= 2)
      minor = prop->minor;
    else
      minor = 0;
  } else if (nvrtc_major <= 10 && prop->major >= 7) { // 10 supports 3-7.5
    major = 7;
    if (prop->major == 7 && prop->minor <= 5)
      minor = prop->minor;
    else
      minor = 0;
  }
}

// Compiles the specified kernel and stores the metadata required to run it
FusedKernelCUDA::FusedKernelCUDA(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random)
    : FusedKernel(
          std::move(name),
          std::move(code),
          std::move(input_desc),
          std::move(output_desc),
          std::move(chunk_desc),
          std::move(concat_desc),
          has_random),
      device_(device) {
  // Initializes driver's API context (if necessary)
  CUcontext pctx = 0;
  TORCH_CU_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(0);
  }

  // Note: hacked at::DeviceGuard since at::DeviceGuard was failing to work
  // properly in some scenarios
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(device_);

  // Acquires device and NVRTC properties (for compile arch and occupancy
  // calculations)
  prop_ = at::cuda::getCurrentDeviceProperties();
  int major, minor;
  getMajorMinor(prop_, major, minor);

  // Creates the NVRTC program
  nvrtcProgram program;
  TORCH_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
      &program, code_.c_str(), nullptr, 0, nullptr, nullptr));

  const std::string compute = "--gpu-architecture=compute_" +
      std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> args = {
      "--std=c++11", compute.c_str(), "-default-device"};
  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result == NVRTC_ERROR_COMPILATION) {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str());
  }
  ResourceGuard holdProgram(
      [&] { TORCH_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });
  TORCH_NVRTC_CHECK(result);
  size_t ptx_size;
  TORCH_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
  ptx_.resize(ptx_size);
  TORCH_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx_.data()));

  TORCH_CU_CHECK(nvrtc().cuModuleLoadData(&module_, ptx_.data()));
  TORCH_CU_CHECK(
      nvrtc().cuModuleGetFunction(&function_, module_, name_.c_str()));

  // Computes max blocks
  TORCH_CU_CHECK(nvrtc().cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks_, function_, 128, 0));
  maxBlocks_ *= prop_->multiProcessorCount;

  // Resets device (end of hacked at::DeviceGuard)
  at::cuda::set_device(prior_device);
}

static int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

void FusedKernelCUDA::launch_raw(
    const uint32_t numel,
    std::vector<void*>& arguments) const {
  at::cuda::CUDAGuard{device_};
  // Hacked at::DeviceGuard (see note above)
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(device_);

  const auto nBlocks = std::min(maxBlocks_, ceilDiv(numel, kBlockSize));

  // Adds random state to arguments if necessary
  // Note: offset defined here so its lifetime extends to the launch
  uint64_t offset;
  if (has_random_) {
    const auto rand_offset =
        4 * (std::ceil(numel / (4.0 * kBlockSize * nBlocks)) + 1);
    auto gen = THCRandom_getGenerator(at::globalContext().getTHCState());
    offset = gen->state.philox_seed_offset.fetch_add(rand_offset);
    arguments.push_back(&gen->state.initial_seed);
    arguments.push_back(&offset);
  }

  // Launches kernel on current stream (device was set by executor)
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CU_CHECK(nvrtc().cuLaunchKernel(
      function_,
      nBlocks,
      1,
      1,
      kBlockSize,
      1,
      1,
      0,
      stream,
      arguments.data(),
      nullptr));

  // Resets device (see at::DeviceGuard notes above)
  at::cuda::set_device(prior_device);
}

FusedKernelCUDA::~FusedKernelCUDA() {
  nvrtc().cuModuleUnload(module_);
}

static std::shared_ptr<FusedKernel> createFusionKernel(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random) {
  return std::make_shared<FusedKernelCUDA>(
      device,
      std::move(name),
      std::move(code),
      std::move(input_desc),
      std::move(output_desc),
      std::move(chunk_desc),
      std::move(concat_desc),
      has_random);
}

RegisterFusionBackend reg(at::DeviceType::CUDA, createFusionKernel);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
