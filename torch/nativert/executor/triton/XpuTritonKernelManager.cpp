#include <torch/nativert/executor/triton/TritonKernelManager.h>

#include <c10/util/FbcodeMaps.h>
#include <c10/util/Logging.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

// Reuses the SYCL/Level-Zero module & kernel loading helpers already used by
// AOTI's standalone cpp wrapper for XPU.
#include <torch/csrc/inductor/aoti_runtime/sycl_runtime_wrappers.h>

namespace torch::nativert {

// XPU-specific launch parameters
class XpuLaunchParams : public LaunchParams {
 public:
  int num_warps = 4;
  int shared_memory_bytes = 0;

  void parseAttributes(const Node* node) {
    parseCommonAttributes(node);
    for (const auto& attr : node->attributes()) {
      set_from_variant<int64_t>(
          num_warps, "num_warps", attr, [](auto v) { return v > 0; });
      set_from_variant<int64_t>(
          shared_memory_bytes, "shared_memory_bytes", attr, [](auto v) {
            return v > 0;
          });
    }
  }
};

// Like CUDA, the SYCL kernel launch path expects each entry of the args array
// to be a pointer to the actual argument value (an extra level of
// indirection), so scalar/tensor args need their own backing storage.
class XpuKernelInputs final : public KernelInputs {
 public:
  XpuKernelInputs(size_t num_args, size_t num_attrs)
      : KernelInputs(num_args, num_attrs),
        arg_ptrs_(num_args),
        global_scratch_(nullptr),
        profile_scratch_(nullptr) {
    inputs_.push_back(&global_scratch_);
    // Some Triton kernel ABIs include profile scratch after global scratch.
    // Kernels without it ignore this trailing parameter because the compiled
    // kernel signature determines how many entries the runtime reads.
    inputs_.push_back(&profile_scratch_);
  }
  ~XpuKernelInputs() final = default;

  void add_arg(void* arg) override {
    TORCH_CHECK(arg_idx_ < num_args_, "Too many args");
    arg_ptrs_[arg_idx_] = arg;
    inputs_[arg_idx_] = reinterpret_cast<void*>(&arg_ptrs_[arg_idx_]);
    arg_idx_++;
  }

  void add_scalar_arg(const c10::IValue& value, std::string_view param_type)
      override {
    TORCH_CHECK(arg_idx_ < num_args_, "Too many args");
    inputs_[arg_idx_++] = store_scalar_arg(value, param_type);
  }

 private:
  std::vector<void*> arg_ptrs_;
  void* global_scratch_;
  void* profile_scratch_;
};

class XpuTritonKernelManager final : public TritonKernelManager {
 public:
  XpuTritonKernelManager(std::string kernel_name, std::string kernel_bin_path);
  ~XpuTritonKernelManager() final = default;

  std::unique_ptr<LaunchParams> createLaunchParams(
      const Node* node) const override {
    auto params = std::make_unique<XpuLaunchParams>();
    params->parseAttributes(node);
    return params;
  }

  void launch(const LaunchParams& launch_params, void** args) final;
  std::unique_ptr<KernelInputs> create_inputs(
      size_t num_args,
      size_t num_attrs,
      const KernelInputParams& /*params*/) const final {
    return std::unique_ptr<KernelInputs>(
        new XpuKernelInputs(num_args, num_attrs));
  }

 private:
  std::unique_ptr<sycl::kernel>& load();

  c10::FastMap<c10::DeviceIndex, std::unique_ptr<sycl::kernel>> cache_;
};

XpuTritonKernelManager::XpuTritonKernelManager(
    std::string kernel_name,
    std::string kernel_bin_path)
    : TritonKernelManager(std::move(kernel_name), std::move(kernel_bin_path)) {
}

std::unique_ptr<sycl::kernel>& XpuTritonKernelManager::load() {
  const auto idx = c10::xpu::current_device();
  if (const auto res = cache_.find(idx); res != cache_.end()) {
    return res->second;
  }

  auto kernel = loadKernel(
      kernel_bin_path_,
      kernel_name_,
      /*sharedMemBytes=*/0);
  TORCH_CHECK(
      kernel != nullptr, "failed to load triton kernel: ", kernel_name_);

  return cache_.emplace(idx, std::move(kernel)).first->second;
}

void XpuTritonKernelManager::launch(
    const LaunchParams& launch_params,
    void** args /* { ...inputs, output }*/) {
  const auto& xpu_params = static_cast<const XpuLaunchParams&>(launch_params);

  auto& kernel_fn = load();
  auto& queue = c10::xpu::getCurrentXPUStream().queue();

  launchKernel(
      kernel_fn,
      static_cast<uint32_t>(xpu_params.grid_dims.x),
      static_cast<uint32_t>(xpu_params.grid_dims.y),
      static_cast<uint32_t>(xpu_params.grid_dims.z),
      static_cast<uint32_t>(xpu_params.num_warps),
      static_cast<uint32_t>(xpu_params.shared_memory_bytes),
      args,
      &queue);
}

namespace {
std::unique_ptr<TritonKernelManager> create_xpu_triton_kernel_manager(
    std::string kernel_name,
    std::string kernel_bin_path,
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    [[maybe_unused]] std::string kernel_launcher_bin_path) {
  return std::make_unique<XpuTritonKernelManager>(
      std::move(kernel_name), std::move(kernel_bin_path));
}
} // namespace

C10_REGISTER_TYPED_CREATOR(
    TritonKernelManagerRegistry,
    at::kXPU,
    create_xpu_triton_kernel_manager)

} // namespace torch::nativert
