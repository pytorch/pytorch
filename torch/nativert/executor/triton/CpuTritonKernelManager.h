#pragma once

#include <torch/nativert/executor/triton/TritonKernelManager.h>

#include <c10/core/Device.h>
#include <c10/util/FbcodeMaps.h>

#ifndef _WIN32
#include <dlfcn.h>
#endif

typedef void* kernel_ptr_t;
typedef void (
    *launcher_ptr_t)(uint32_t, uint32_t, uint32_t, void**, kernel_ptr_t);

namespace torch::nativert {

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

} // namespace torch::nativert
