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
      // See Note [ROCm grid_sampler_2d backward channel-lane eligibility].
      // When the channel-lane kernel will be used on channels-last inputs, give
      // it a grad_input in the same format so its per-channel atomics are
      // contiguous across the wave rather than input.size(2)*input.size(3)
      // elements apart. On the recorded MI455X workload that is the difference
      // between 6.57 ms and 0.79 ms.
      //
      // This is deliberately narrow. Contiguous inputs keep the historical
      // contiguous grad_input, so the only observable stride change is that a
      // channels-last input now yields a channels-last gradient -- which matches
      // what most PyTorch ops already do. The CUDA path is intentionally left
      // alone: it uses the one-thread-per-(n,h,w) kernel, whose threads walk the
      // channel dimension serially, so a channels-last grad_input would not
      // coalesce anything there and would change observable strides for no gain.
      // Integer test first; see the same ordering note in GridSampler.cu.
      if (input.size(1) >= 4) {
        const bool channels_last =
            input.is_contiguous(at::MemoryFormat::ChannelsLast) &&
            grad_output.is_contiguous(at::MemoryFormat::ChannelsLast);
        // The launch extent is part of eligibility: if the kernel would fall
        // back, this must not hand the generic kernel channels-last strides.
        const int64_t nblocks_nhw =
            input.size(0) * grid.size(1) * grid.size(2);
        if (channels_last &&
            rocm_grid_sampler_2d_backward_use_channel_lane(
                input.size(1), channels_last, nblocks_nhw)) {
          return at::zeros_like(input, at::MemoryFormat::Preserve);
        }
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
