#include <torch/nativert/executor/triton/TritonKernelManager.h>

#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <c10/util/FbcodeMaps.h>
#include <c10/util/Logging.h>

namespace {
const at::cuda::NVRTC& get_nvrtc() {
  return at::globalContext().getNVRTC();
}
} // namespace

#define CU_LOG_ERROR(fn, result, ...)                   \
  {                                                     \
    LOG(ERROR) << #fn << " returned error: " << result; \
    const char* errMsg = nullptr;                       \
    get_nvrtc().cuGetErrorString(result, &errMsg);      \
    LOG(ERROR) << "cuGetErrorString: " << errMsg;       \
  }

namespace torch::nativert {

// CUDA/HIP-specific launch parameters
class CudaLaunchParams : public LaunchParams {
 public:
  int num_warps = 4;
  int shared_memory_bytes = 0;

  void parseAttributes(const Node* node) {
    parseCommonAttributes(node);
    for (const auto& attr : node->attributes()) {
      set_from_variant<int64_t>(
          num_warps, "num_warps", attr, [](auto v) { return v > 0; });
      set_from_variant<int64_t>(
          shared_memory_bytes, "shared_memory_bytes", attr, [](auto v) { return v > 0; });
    }
  }
};

// cuda kernels require an extra level of indirection
// for who knows what reason.
class CudaKernelInputs final : public KernelInputs {
 public:
  CudaKernelInputs(size_t num_args, size_t num_attrs)
      : KernelInputs(num_args, num_attrs),
        arg_ptrs_(num_args),
        global_scratch_(0) {
    inputs_.push_back(&global_scratch_);
  }
  ~CudaKernelInputs() final = default;

  void add_arg(void* arg) override {
    TORCH_CHECK(arg_idx_ < num_args_, "Too many args");
    arg_ptrs_[arg_idx_] = arg;
    inputs_[arg_idx_] = reinterpret_cast<void*>(&arg_ptrs_[arg_idx_]);
    arg_idx_++;
  }

 private:
  std::vector<void*> arg_ptrs_;
  CUdeviceptr global_scratch_;
};

class CudaTritonKernelManager final : public TritonKernelManager {
 public:
  CudaTritonKernelManager(std::string kernel_name, std::string kernel_bin_path);
  ~CudaTritonKernelManager() final;

  CudaTritonKernelManager(const CudaTritonKernelManager& other);
  CudaTritonKernelManager& operator=(const CudaTritonKernelManager& other);
  CudaTritonKernelManager(CudaTritonKernelManager&& other) noexcept;
  CudaTritonKernelManager& operator=(CudaTritonKernelManager&& other) noexcept;

  std::unique_ptr<LaunchParams> createLaunchParams(
      const Node* node) const override {
    auto params = std::make_unique<CudaLaunchParams>();
    params->parseAttributes(node);
    return params;
  }

  void launch(const LaunchParams& launch_params, void** args) final;
  std::unique_ptr<KernelInputs> create_inputs(
      size_t num_args,
      size_t num_attrs,
      const KernelInputParams& /*params*/) const final {
    return std::unique_ptr<KernelInputs>(
        new CudaKernelInputs(num_args, num_attrs));
  }

 private:
  CUfunction load();
  c10::FastMap<c10::DeviceIndex, CUfunction> cache_;
  std::vector<CUmodule> loaded_modules_;
};

CudaTritonKernelManager::CudaTritonKernelManager(
    std::string kernel_name,
    std::string kernel_bin_path)
    : TritonKernelManager(std::move(kernel_name), std::move(kernel_bin_path)) {
  TORCH_CHECK(
      at::globalContext().hasCUDA() || at::globalContext().hasHIP(),
      "cuda or hip required");
}

CudaTritonKernelManager::~CudaTritonKernelManager() {
  const auto& nvrtc = get_nvrtc();
  for (auto& mod : loaded_modules_) {
    if (CUresult err = nvrtc.cuModuleUnload(mod); err != 0) {
      CU_LOG_ERROR(nvrtc.cuModuleUnload, err);
    }
  }
}

CUfunction CudaTritonKernelManager::load() {
  const auto idx = c10::cuda::current_device();
  if (const auto res = cache_.find(idx); res != cache_.end()) {
    return res->second;
  }

  const auto& nvrtc = get_nvrtc();

  CUmodule mod_ptr = nullptr;

  if (CUresult err = nvrtc.cuModuleLoad(&mod_ptr, kernel_bin_path_.c_str());
      err != 0) {
    CU_LOG_ERROR(nvrtc.cuModuleLoad, err);
    return nullptr;
  }

  CUfunction func = nullptr;

  if (CUresult err =
          nvrtc.cuModuleGetFunction(&func, mod_ptr, kernel_name_.c_str());
      err != 0) {
    CU_LOG_ERROR(nvrtc.cuModuleGetFunction, err);
    return nullptr;
  }

  loaded_modules_.emplace_back(mod_ptr);
  return cache_.emplace(idx, func).first->second;
}

void CudaTritonKernelManager::launch(
    const LaunchParams& launch_params,
    void** args /* { ...inputs, output }*/) {
  const auto& cuda_params = static_cast<const CudaLaunchParams&>(launch_params);
  const constexpr int kThreadsPerWarp = 2 << 4;

  auto kernel_fn = load();
  TORCH_CHECK(
      kernel_fn != nullptr, "failed to load triton kernel: ", kernel_name_);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  AT_CUDA_DRIVER_CHECK(get_nvrtc().cuLaunchKernel(
      kernel_fn,
      cuda_params.grid_dims.x,
      cuda_params.grid_dims.y,
      cuda_params.grid_dims.z,
      /* blockDimX = */ kThreadsPerWarp * cuda_params.num_warps,
      /* blockDimY = */ 1,
      /* blockDimZ = */ 1,
      /* sharedMemBytes = */ cuda_params.shared_memory_bytes,
      stream,
      args,
      nullptr));
}

namespace {
std::unique_ptr<TritonKernelManager> create_cuda_triton_kernel_manager(
    std::string kernel_name,
    std::string kernel_bin_path,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] std::string kernel_launcher_bin_path) {
  return std::make_unique<CudaTritonKernelManager>(
      std::move(kernel_name), std::move(kernel_bin_path));
}
} // namespace

#ifdef USE_ROCM

C10_REGISTER_TYPED_CREATOR(
    TritonKernelManagerRegistry,
    at::kHIP,
    create_cuda_triton_kernel_manager)

#else

C10_REGISTER_TYPED_CREATOR(
    TritonKernelManagerRegistry,
    at::kCUDA,
    create_cuda_triton_kernel_manager)

#endif // USE_ROCM

} // namespace torch::nativert
