#pragma once

#include <optional>
#include <string>
#include <vector>

#include <ATen/core/TensorBody.h>
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

// Parameters for kernel inputs, used for backend-specific initialization.
// MTIA uses kernel_param_names and kernel_param_types for fatbin compilation
// and proper scalar type casting. Other backends can ignore this struct.
struct KernelInputParams {
  std::vector<std::string> kernel_param_names;
  std::vector<std::string> kernel_param_types;
  std::vector<int64_t> output_indices;
};

struct LaunchParams {
  // CPU params
  int num_cpu_threads = 0; // 0 means use all available threads
  // GPU params
  // TODO: Add more GPU autotuning parameters
  int num_warps = 4;
  int shared_memory_bytes = 0;
  GridDims grid_dims;

  // MTIA params
  std::optional<int> mtia_tile_width;
  std::optional<int> mtia_tile_height;
  std::optional<int> mtia_base_pe;
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

  // Add a tensor argument. The default implementation just uses data_ptr(),
  // this option allows any custom logic to take the tensor directly instead
  // of just the data pointer if needed.
  virtual void add_tensor_arg(const at::Tensor& tensor) {
    add_arg(tensor.data_ptr());
  }

  virtual void add_attribute(void* attr) {
    TORCH_CHECK(attr_idx_ < num_attrs_, "Too many attributes");
    inputs_[num_args_ + attr_idx_++] = attr;
  }

  virtual void** as_void() {
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
      size_t num_attrs,
      const KernelInputParams& /*params*/) const {
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
