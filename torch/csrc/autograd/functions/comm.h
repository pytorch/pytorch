#pragma once

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace autograd {

struct Scatter : public Function {
  explicit Scatter(
      std::vector<at::Device> devices,
      const at::optional<std::vector<int64_t>>& chunk_sizes = at::nullopt,
      int64_t dim = 0,
      const at::optional<std::vector<at::CUDAStream>>& streams = at::nullopt,
      bool unsqueeze_scalars = false);

  variable_list apply(variable_list&& inputs) override;

  std::vector<at::Device> devices_;
  at::optional<std::vector<int64_t>> chunk_sizes_;
  int64_t dim_;
  at::optional<std::vector<at::CUDAStream>> streams_;
  bool unsqueeze_scalars_;
};

struct Gather : public Function {
  explicit Gather(const at::Device& destination_device, int64_t dim = 0);

  variable_list apply(variable_list&& inputs) override;

  at::Device destination_device_;
  int64_t dim_;
};

} // namespace autograd
} // namespace torch
