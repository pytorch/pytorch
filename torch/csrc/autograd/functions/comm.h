#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ATenCUDAGeneral.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace autograd {

//TODO: change it to TORCH_API when we merge the libs
struct AT_CUDA_API Scatter : public Function {
  explicit Scatter(
      std::vector<at::Device> devices,
      const c10::optional<std::vector<int64_t>>& chunk_sizes = c10::nullopt,
      int64_t dim = 0,
      const c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>>& streams =
          c10::nullopt,
      bool unsqueeze_scalars = false);
  ~Scatter() override;

  variable_list apply(variable_list&& inputs) override;

  std::vector<at::Device> devices_;
  c10::optional<std::vector<int64_t>> chunk_sizes_;
  int64_t dim_;
  c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>> streams_;
  bool unsqueeze_scalars_;
};

struct AT_CUDA_API Gather : public Function {
  explicit Gather(const at::Device& destination_device, int64_t dim = 0);
  ~Gather() override;

  variable_list apply(variable_list&& inputs) override;

  at::Device destination_device_;
  int64_t dim_;
};

} // namespace autograd
} // namespace torch
