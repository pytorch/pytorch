#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace autograd {

struct TORCH_CUDA_CU_API Scatter : public Node {
  explicit Scatter(
      std::vector<at::Device> devices,
      std::optional<std::vector<int64_t>> chunk_sizes = c10::nullopt,
      int64_t dim = 0,
      std::optional<std::vector<std::optional<at::cuda::CUDAStream>>> streams =
          c10::nullopt,
      bool unsqueeze_scalars = false);
  ~Scatter() override;

  variable_list apply(variable_list&& inputs) override;

  std::vector<at::Device> devices_;
  std::optional<std::vector<int64_t>> chunk_sizes_;
  int64_t dim_;
  std::optional<std::vector<std::optional<at::cuda::CUDAStream>>> streams_;
  bool unsqueeze_scalars_;
};

struct TORCH_CUDA_CU_API Gather : public Node {
  explicit Gather(const at::Device& destination_device, int64_t dim = 0);
  ~Gather() override;

  variable_list apply(variable_list&& inputs) override;

  at::Device destination_device_;
  int64_t dim_;
};

} // namespace autograd
} // namespace torch
