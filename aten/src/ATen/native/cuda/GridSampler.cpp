#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/GridSampler.h>
#include <utility>
#ifdef USE_ROCM
#include <ATen/core/TensorBase.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/grid_sampler_2d_backward_native.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d_backward_native.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {



Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
  launch_grid_sampler_2d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

Tensor grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  launch_grid_sampler_3d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
#ifdef USE_ROCM
      // Channels-last gradients coalesce the channel-parallel atomics
      // (6.57 ms -> 0.79 ms for large-C channels-last backward). Both kernels
      // support these strides, so allocation depends only on the input layout.
      if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
        return input.new_empty_strided_symint(input.sym_sizes(), input.sym_strides()).zero_();
      }
#endif
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_2d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(std::move(grad_input), std::move(grad_grid));
}

std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool,2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_3d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(std::move(grad_input), std::move(grad_grid));
}

}  // namespace at::native
