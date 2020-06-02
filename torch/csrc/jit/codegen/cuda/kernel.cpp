#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/kernel_resource_strings.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <torch/csrc/jit/resource_guard.h>
#include <fstream>
#include <iostream>

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

// Go through a tensor, and grab it's sizes/strides potentially broadcasted
struct ExtractSizeStride {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;

  ExtractSizeStride(
      const at::Tensor& val,
      c10::optional<at::IntArrayRef> broadcasted_size = c10::nullopt) {
    if (broadcasted_size) {
      int b_dim = (int)broadcasted_size->size();
      int o_dim = (int)val.dim();
      TORCH_CHECK(b_dim >= o_dim);
      for (int i = 0; i < b_dim; i++) {
        sizes.push_back(broadcasted_size->at(i));
        int index = i + o_dim - b_dim;
        if (index < 0) {
          strides.push_back(0);
        } else if (val.sizes()[index] == sizes[i]) {
          strides.push_back(val.strides()[index]);
        } else {
          TORCH_CHECK(
              val.sizes()[index] == 1,
              "Not compatible dimension size for broadcast");
          strides.push_back(0);
        }
      }
    } else {
      auto o_dim = val.dim();
      for (decltype(val.dim()) i{0}; i < o_dim; i++) {
        sizes.push_back(val.sizes()[i]);
        strides.push_back(val.strides()[i]);
      }
    }
  }
};

struct KernelArgumentHolder {
 private:
  std::vector<ArgAbstract*> arguments;
  std::vector<void*> void_ptrs;
  bool changed = true;

 public:
  virtual ~KernelArgumentHolder() {
    for (auto arg : arguments)
      delete arg;
  }

  // Push a tensor to the arguments
  void push(
      const at::Tensor& val,
      c10::optional<at::IntArrayRef> broadcasted_size = c10::nullopt) {
    changed = true;
    ExtractSizeStride ess(val, std::move(broadcasted_size));
    int nDims = ess.sizes.size();

    c10::ScalarType dtype = val.scalar_type();
    TensorArgAbstract* tensor_arg = getTensorArg(dtype, nDims);
    tensor_arg->setPointer(val.data_ptr());
    for (int i = 0; i < nDims; i++) {
      tensor_arg->setSize(i, ess.sizes[i]);
      tensor_arg->setStride(i, ess.strides[i]);
    }
    arguments.push_back(tensor_arg);
  }

  // Push a scalar or integer to the arguments
  void push(const IValue& val) {
    changed = true;
    TORCH_INTERNAL_ASSERT(
        val.isScalar(),
        "Tried to push an arg to run in a fused kernel, expected a scalar but got, ",
        val);
    switch (val.toScalar().type()) {
      case (c10::ScalarType::Double):
        arguments.push_back(new FloatArg((float)val.toDouble()));
        return;
      case (c10::ScalarType::Long):
        arguments.push_back(new IntArg((int)val.toInt()));
        return;
      default:
        TORCH_INTERNAL_ASSERT(
            false,
            " Tried to create argument to send to a fused kernel, but got an unexpected type.");
    }
    TORCH_INTERNAL_ASSERT(
        false,
        " Tried to create argument to send to a fused kernel, but got a non-scalar type.");
  }

  void push(const uint64_t& val) {
    arguments.push_back(new ULongArg(val));
  }

  // Create buffer, flatten arguments into it, align by 8 Bytes, return pointers
  // in the buffer
  void** getBuffer() {
    if (changed) {
      void_ptrs = std::vector<void*>(arguments.size(), nullptr);
      for (decltype(arguments.size()) i{0}; i < arguments.size(); i++)
        void_ptrs[i] = static_cast<void*>(arguments[i]->arg());
      changed = false;
    }
    return void_ptrs.data();
  }
};

std::pair<std::string, std::string> codeGeneration(Fusion& fusion) {
  std::stringstream str_stream;
  str_stream << "namespace " << CG_NAMESPACE << " {\n"
             << code_template_tensor_struct << "\n"
             << code_fp16_support << "\n"
             << code_random_number_gen << "\n"
             << code_helper_funcs << "\n"
             << code_template_block_reduction << "\n";
  std::stringstream cdg;
  GPULower gpulw(&fusion);
  gpulw.printKernel(str_stream, KERNEL_NAME);
  str_stream << "\n} // namespace";

  std::string func_name = std::string(CG_NAMESPACE) + "::" + KERNEL_NAME;
  return std::make_pair(func_name, str_stream.str());
};

} // namespace

bool NaivePWKernelArgsReq::matchKernelSize(const at::ArrayRef<IValue> inputs) {
  TORCH_INTERNAL_ASSERT(
      inputs.size() == dims_.size(),
      "wrong number of inputs feed to generated kernel!");
  for (size_t i = 0; i < dims_.size(); i++) {
    if (inputs[i].isTensor()) {
      if (inputs[i].toTensor().dim() != dims_[i]) {
        return false;
      }
    } else {
      if (dims_[i] != -1) {
        return false;
      }
    }
  }
  return true;
}

void compileKernel(Fusion& fusion, CudaKernel* entry) {
  // generating cuda code;
  std::string code;
  std::string func_name;
  std::tie(func_name, code) = codeGeneration(fusion);

  // Keep input and output reference to validate/line up arguments
  for (auto inp : fusion.inputs())
    entry->inputs.push_back(inp);

  for (auto out : fusion.outputs())
    entry->outputs.push_back(out);

  static int32_t compiled_kernel_id = 0;

  // vvv NVRTC COMPILATION vvv

  // lazily construct context if non-existing yet;
  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(nullptr);
  }

  // set device for the operation;
  at::cuda::set_device(entry->device_);
  entry->has_random_ = fusion.hasRNG();

  const auto prop = at::cuda::getCurrentDeviceProperties();
  int nvrtc_major, nvrtc_minor;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

  // Short-circuits if NVRTC version too low
  TORCH_INTERNAL_ASSERT(nvrtc_major >= 6);
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
      "--std=c++14", compute.c_str(), "-default-device"};

  nvrtc().nvrtcAddNameExpression(program, func_name.c_str());
  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result != NVRTC_SUCCESS) {
    size_t logsize;
    nvrtc().nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtc().nvrtcGetProgramLog(program, log.data());

    TORCH_INTERNAL_ASSERT(
        false, code.c_str(), "\nCUDA NVRTC compile error: ", log.data());
  }
  const char* lowered_kernel_name;
  nvrtc().nvrtcGetLoweredName(program, func_name.c_str(), &lowered_kernel_name);

  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
  std::vector<char> ptx;
  ptx.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx.data()));

  // TODO: We do go through different code path, should investigate whether this
  // has an impact on generated binary.
  const char* prefix_env = getenv("PYTORCH_CUDA_FUSER_CUBIN");
  if (prefix_env) {
    // Output ptx file
    std::stringstream ptx_file_name;
    ptx_file_name << prefix_env << "_" << compiled_kernel_id << ".ptx";
    std::ofstream myPtxFile(ptx_file_name.str().c_str(), std::ios::out);
    if (myPtxFile.is_open()) {
      myPtxFile.write(ptx.data(), ptx.size());
      myPtxFile.close();
    }

    CUlinkState linkState;

    AT_CUDA_DRIVER_CHECK(nvrtc().cuLinkCreate(0, nullptr, nullptr, &linkState));
    AT_CUDA_DRIVER_CHECK(nvrtc().cuLinkAddData(
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
    AT_CUDA_DRIVER_CHECK(nvrtc().cuLinkComplete(linkState, &cubin, &cubinSize));

    // Output binary file
    std::stringstream cubin_file_name;
    cubin_file_name << prefix_env << "_" << compiled_kernel_id << ".cubin";
    std::ofstream myCubinFile(
        cubin_file_name.str().c_str(), std::ios::out | std::ios::binary);
    if (myCubinFile.is_open()) {
      myCubinFile.write(static_cast<const char*>(cubin), cubinSize);
      myCubinFile.close();
    }

    // load compiled cubin
    AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&(entry->module_), cubin));
  } else {
    // load ptx directly
    AT_CUDA_DRIVER_CHECK(
        nvrtc().cuModuleLoadData(&(entry->module_), ptx.data()));
  }
  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleGetFunction(
      &(entry->function_), entry->module_, lowered_kernel_name));
#if defined(__HIP_PLATFORM_HCC__) && HIP_VERSION < 305
  // HIP function signature is not compatible yet
  uint32_t max_blocks;
  AT_CUDA_DRIVER_CHECK(nvrtc().hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, entry->function_, 128, 0));
  entry->max_blocks_ = max_blocks;
#else
  AT_CUDA_DRIVER_CHECK(nvrtc().cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &entry->max_blocks_, entry->function_, 128, 0));
#endif
  entry->max_blocks_ *= prop->multiProcessorCount;
}

void runKernel(
    CudaKernel* entry,
    const at::ArrayRef<IValue> inputs,
    std::vector<at::Tensor> outputs) {
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(entry->device_);
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO: Proper API to establish reasonable launch configurations;
  // Naive launch config;
  size_t numel = outputs[0].numel();

  // TODO: we can't randomly clap down this until we got striding.
  const auto nBlocks = ceilDiv(numel, 128 * entry->unroll_factor_);

  KernelArgumentHolder kernel_args;

  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (auto& input : inputs) {
    if (input.isTensor()) {
      TORCH_INTERNAL_ASSERT(
          input.toTensor().device().index() == entry->device_,
          "input to kernel on device that is not compiled for");
      kernel_args.push(input.toTensor(), outputs[0].sizes());
    } else {
      kernel_args.push(input);
    }
  }

  for (auto& output : outputs) {
    kernel_args.push(output);
  }

  // TODO: this probably won't work for us.
  if (entry->has_random_) {
    std::pair<uint64_t, uint64_t> philox_engine_inputs;
    const auto rand_offset = 4 * (std::ceil(numel / (4.0 * 128 * nBlocks)) + 1);
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      philox_engine_inputs =
          at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
              rand_offset);
    }
    kernel_args.push(philox_engine_inputs.first);
    kernel_args.push(philox_engine_inputs.second);
  }

  // launch kernel;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      entry->function_,
      nBlocks,
      1,
      1,
      128,
      1,
      1,
      0,
      stream,
      kernel_args.getBuffer(),
      nullptr));

  // Resets device (see at::DeviceGuard notes above)
  at::cuda::set_device(prior_device);
}

// WARNING:
// This function is here for testing purposes only
void runTestKernel(
    CudaKernel* entry,
    const at::ArrayRef<IValue> inputs,
    std::vector<at::Tensor> outputs) {
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(entry->device_);
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO: Proper API to establish reasonable launch configurations;
  // Naive launch config;
  TORCH_INTERNAL_ASSERT(!outputs.empty(), "No outputs set for test kernel.");
  size_t numel = outputs[0].numel();

  // TODO: we can't randomly clap down this until we got striding.
  const auto nBlocks = ceilDiv(numel, 128 * entry->unroll_factor_);

  KernelArgumentHolder kernel_args;

  auto exprs = entry->outputs[0]->fusion()->exprs(true);
  bool has_reduction = std::any_of(exprs.begin(), exprs.end(), [](Expr* expr) {
    return expr->getExprType() == ExprType::ReductionOp;
  });

  // Naive I/O setup, I'm ignoring all the potential transformation (i.e. I/O
  // allocated here from the subgraph could be, and very likely are, different
  // from I/O expected by the generated CUDA kernel.
  for (auto& input : inputs) {
    if (input.isTensor()) {
      TORCH_INTERNAL_ASSERT(
          input.toTensor().device().index() == entry->device_,
          "input to kernel on device that is not compiled for");
      TORCH_INTERNAL_ASSERT(
          !entry->outputs.empty(),
          "No output found for this kernel, aborting.");
      if (has_reduction) {
        kernel_args.push(input.toTensor());
      } else {
        kernel_args.push(input.toTensor(), outputs[0].sizes());
      }
    } else {
      kernel_args.push(input);
    }
  }

  for (auto& output : outputs) {
    kernel_args.push(output);
  }

  // TODO: this probably won't work for us.
  if (entry->has_random_) {
    std::pair<uint64_t, uint64_t> philox_engine_inputs;
    const auto rand_offset = 4 * (std::ceil(numel / (4.0 * 128 * nBlocks)) + 1);
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      philox_engine_inputs =
          at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
              rand_offset);
    }
    kernel_args.push(philox_engine_inputs.first);
    kernel_args.push(philox_engine_inputs.second);
  }

  // launch kernel;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      entry->function_,
      entry->grid_.x,
      entry->grid_.y,
      entry->grid_.z,
      entry->block_.x,
      entry->block_.y,
      entry->block_.z,
      0,
      stream,
      kernel_args.getBuffer(),
      nullptr));

  // Resets device (see at::DeviceGuard notes above)
  at::cuda::set_device(prior_device);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
