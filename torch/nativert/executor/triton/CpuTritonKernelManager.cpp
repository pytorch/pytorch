#include <torch/nativert/executor/triton/TritonKernelManager.h>

#include <c10/util/Exception.h>

#ifndef _WIN32
#include <dlfcn.h>
#endif // _WIN32

namespace torch::nativert {

namespace {
void* _dlopen(const char* filename) {
#if defined(_WIN32)
  return nullptr;
#else
  return dlopen(filename, RTLD_NOW | RTLD_LOCAL);
#endif
}

void* _dlsym(void* handle, const char* name) {
#if defined(_WIN32)
  return nullptr;
#else
  return dlsym(handle, name);
#endif
}

char* _dlerror() {
#if defined(_WIN32)
  TORCH_CHECK(false, "dlerror not supported on Windows");
#else
  return dlerror();
#endif
}

} // namespace

typedef void* kernel_ptr_t;
typedef void (
    *launcher_ptr_t)(uint32_t, uint32_t, uint32_t, int, void**, kernel_ptr_t);

struct DlcloseDeleter {
  void operator()(void* p) const {
    if (p) {
#if defined(_WIN32)
      TORCH_CHECK(false, "Windows is not supported");
#else
      dlclose(p);
#endif
    }
  }
};

class CpuTritonKernelManager final : public TritonKernelManager {
 public:
  CpuTritonKernelManager(
      std::string kernel_name,
      std::string kernel_bin_path,
      std::string kernel_launcher_bin_path);
  ~CpuTritonKernelManager() final = default;
  void launch(const LaunchParams& launch_params, void** args) final;

 private:
  void load();

  kernel_ptr_t kernel_fn_{nullptr};
  launcher_ptr_t launcher_fn_{nullptr};

  std::unique_ptr<void, DlcloseDeleter> kernel_handle_{nullptr};
  std::unique_ptr<void, DlcloseDeleter> launcher_handle_{nullptr};

  std::string kernel_launcher_bin_path_;
};

CpuTritonKernelManager::CpuTritonKernelManager(
    std::string kernel_name,
    std::string kernel_bin_path,
    std::string kernel_launcher_bin_path)
    : TritonKernelManager(std::move(kernel_name), std::move(kernel_bin_path)),
      kernel_launcher_bin_path_(std::move(kernel_launcher_bin_path)) {}

void CpuTritonKernelManager::load() {
  if (C10_LIKELY(kernel_fn_ != nullptr)) {
    return;
  }

  kernel_handle_.reset(_dlopen(kernel_bin_path_.c_str()));
  TORCH_CHECK(
      kernel_handle_ != nullptr,
      "could not dlopen ",
      kernel_bin_path_,
      ": ",
      _dlerror());

  launcher_handle_.reset(_dlopen(kernel_launcher_bin_path_.c_str()));
  TORCH_CHECK(
      launcher_handle_ != nullptr,
      "could not dlopen ",
      kernel_launcher_bin_path_,
      ": ",
      _dlerror());

  kernel_fn_ = _dlsym(kernel_handle_.get(), kernel_name_.c_str());
  TORCH_CHECK(
      kernel_fn_ != nullptr,
      "could not dlsym ",
      kernel_name_,
      ": ",
      _dlerror());

  launcher_fn_ = reinterpret_cast<launcher_ptr_t>(
      _dlsym(launcher_handle_.get(), "run_from_nativert"));
  TORCH_CHECK(launcher_fn_ != nullptr, "could not dlsym run: ", _dlerror());
}

void CpuTritonKernelManager::launch(
    const LaunchParams& launch_params,
    void** args /* { ...inputs, output }*/) {
  load();
  launcher_fn_(
      launch_params.grid_dims.x,
      launch_params.grid_dims.y,
      launch_params.grid_dims.z,
      launch_params.num_cpu_threads,
      args,
      kernel_fn_);
}

namespace {
std::unique_ptr<TritonKernelManager> create_cpu_triton_kernel_manager(
    std::string kernel_name,
    std::string kernel_bin_path,
    std::string kernel_launcher_bin_path) {
  return std::make_unique<CpuTritonKernelManager>(
      std::move(kernel_name),
      std::move(kernel_bin_path),
      std::move(kernel_launcher_bin_path));
}
} // namespace

C10_REGISTER_TYPED_CREATOR(
    TritonKernelManagerRegistry,
    at::kCPU,
    create_cpu_triton_kernel_manager)

} // namespace torch::nativert
