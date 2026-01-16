#pragma once

#include <string>

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

namespace torch::nativert {

struct GridDims {
 public:
  GridDims(int x = 1, int y = 1, int z = 1) : x(x), y(y), z(z) {}
  int x;
  int y;
  int z;
};

struct LaunchParams {
  // CPU params
  int num_cpu_threads = 0; // 0 means use all available threads
  // GPU params
  // TODO: Add more GPU autotuning parameters
  int num_warps = 4;
  int shared_memory_bytes = 0;
  GridDims grid_dims;
};

class KernelInputs {
 public:
  KernelInputs(size_t num_args, size_t num_attrs)
      : num_args_(num_args),
        inputs_(num_args + num_attrs),
        num_attrs_(num_attrs) {}
  virtual ~KernelInputs() = default;

  virtual void add_arg(void* arg) {
    TORCH_CHECK(arg_idx_ < num_args_, "Too many args");
    inputs_[arg_idx_++] = arg;
  }

  void add_attribute(void* attr) {
    TORCH_CHECK(attr_idx_ < num_attrs_, "Too many attributes");
    inputs_[num_args_ + attr_idx_++] = attr;
  }

  void** as_void() {
    return inputs_.data();
  }

 protected:
  size_t num_args_;
  size_t arg_idx_ = 0;
  std::vector<void*> inputs_;

 private:
  size_t num_attrs_;
  size_t attr_idx_ = 0;
};

class TritonKernelManager {
 public:
  TritonKernelManager(std::string kernel_name, std::string kernel_bin_path)
      : kernel_name_(std::move(kernel_name)),
        kernel_bin_path_(std::move(kernel_bin_path)) {}
  virtual ~TritonKernelManager() = default;
  virtual std::unique_ptr<KernelInputs> create_inputs(
      size_t num_args,
      size_t num_attrs) const {
    return std::make_unique<KernelInputs>(num_args, num_attrs);
  }
  virtual void launch(const LaunchParams& launch_params, void** args) = 0;

 protected:
  std::string kernel_name_, kernel_bin_path_;
};

C10_DECLARE_TYPED_REGISTRY(
    TritonKernelManagerRegistry,
    c10::DeviceType,
    TritonKernelManager,
    std::unique_ptr,
    std::string /* kernel_name */,
    std::string /* kernel_bin_path */,
    std::string /* kernel_launcher_bin_path */);

} // namespace torch::nativert
