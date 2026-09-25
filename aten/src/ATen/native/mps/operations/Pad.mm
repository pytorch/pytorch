//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/Padding.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/Pad.h>
#include <c10/metal/common.h>

#include <algorithm>
#include <limits>
#include <numeric>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/constant_pad_nd_native.h>
#include <ATen/ops/reflection_pad1d_backward_native.h>
#include <ATen/ops/reflection_pad1d_native.h>
#include <ATen/ops/reflection_pad2d_backward_native.h>
#include <ATen/ops/reflection_pad2d_native.h>
#include <ATen/ops/reflection_pad3d_backward_native.h>
#include <ATen/ops/reflection_pad3d_native.h>
#include <ATen/ops/replication_pad1d_backward_native.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad2d_backward_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad3d_backward_native.h>
#include <ATen/ops/replication_pad3d_native.h>
#endif

namespace at::native {
namespace mps {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/ReplicationPad_metallib.h>
#endif

static MTLSize pad_threadgroup(id<MTLComputePipelineState> pso, NSUInteger gx, NSUInteger gy, NSUInteger gz) {
  const auto maxTPG = [pso maxTotalThreadsPerThreadgroup];
  const auto tg_x = std::min<NSUInteger>(maxTPG, gx);
  const auto tg_y = std::min<NSUInteger>(maxTPG / tg_x, gy);
  const auto tg_z = std::min<NSUInteger>(maxTPG / (tg_x * tg_y), gz);
  return MTLSizeMake(tg_x, tg_y, tg_z);
}

// Pad operations (1D/2D/3D forward and backward)
static Tensor& pad_out_template(Tensor& output,
                                const Tensor& input,
                                IntArrayRef padding,
                                const std::optional<Tensor>& grad_output_opt,
                                bool is_reflection,
                                const std::string& op_name) {
  TORCH_CHECK(padding.size() == 2 || padding.size() == 4 || padding.size() == 6,
              "invalid padding argument of size ",
              padding.size());
  const auto padding_dim = static_cast<int64_t>(padding.size() / 2);
  const Tensor& grad_output = *at::borrow_from_optional_tensor(grad_output_opt);
  const bool backward = grad_output.defined();
  if (!backward) {
    output.resize_(at::native::padding::pad_shape_check(input, padding, padding_dim, is_reflection));
  } else {
    at::native::padding::check_valid_input(input, padding, padding_dim);
    at::native::padding::pad_backward_shape_check(grad_output, input, padding, padding_dim);
    output.resize_as_(input);
  }
  if (output.numel() == 0) {
    return output;
  }
  if (backward && grad_output.numel() == 0) {
    return output.zero_();
  }

  const auto& source = backward ? grad_output : input;
  // Plain TORCH_CHECK to match the RuntimeError CPU and CUDA raise from data_ptr<T>() for this mismatch.
  TORCH_CHECK(source.scalar_type() == output.scalar_type(),
              "expected scalar type ",
              source.scalar_type(),
              " but found ",
              output.scalar_type());
  const auto& input_tensor = backward ? output : input;
  const auto& output_tensor = backward ? grad_output : output;
  const auto channel_dim = input.dim() - padding_dim - 1;
  // The _i32 kernels form pad_left + 2 * (size - 1), so leave int32 headroom beyond the offsets themselves.
  constexpr int64_t max32 = int64_t{1} << 29;
  const bool use32 = canUse32BitIndexMath(input_tensor, max32) && canUse32BitIndexMath(output_tensor, max32) &&
      std::all_of(padding.begin(), padding.end(), [](int64_t pad) { return std::abs(pad) < max32; });
  const auto width = output.size(-1);
  const auto height = padding_dim >= 2 ? output.size(-2) : 1;
  const auto grid_x = c10::metal::ceil_div(width, static_cast<int64_t>(c10::metal::ILP_PER_THREAD));
  const auto grid = MTLSizeMake(c10::checked_convert<uint32_t>(grid_x, "uint32_t"),
                                c10::checked_convert<uint32_t>(height, "uint32_t"),
                                c10::checked_convert<uint32_t>(output.numel() / (width * height), "uint32_t"));
  using namespace std::string_view_literals;
  const auto kernel_name = fmt::format("{}_pad{}d_{}_{}{}",
                                       is_reflection ? "reflection"sv : "replication"sv,
                                       padding_dim,
                                       backward ? "backward"sv : "forward"sv,
                                       scalarToMetalTypeString(source),
                                       use32 ? "_i32"sv : "_i64"sv);
  auto pso = lib.getPipelineStateForFunc(kernel_name);
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, op_name, {source, output}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtlDispatchByIndexWidth<int32_t, int64_t>(use32, [&](auto idx_tag) {
        using idx_t = typename decltype(idx_tag)::type;
        // Innermost padded dims first, then the channel stride and the batch stride (0 when unbatched).
        auto sizes = [&](const Tensor& tensor) {
          c10::metal::array<idx_t, 3> values = {1, 1, 1};
          for (const auto dim : c10::irange(padding_dim)) {
            values[dim] = static_cast<idx_t>(tensor.size(-1 - dim));
          }
          return values;
        };
        auto strides = [&](const Tensor& tensor) {
          c10::metal::array<idx_t, 5> values{};
          for (const auto dim : c10::irange(padding_dim)) {
            values[dim] = static_cast<idx_t>(tensor.stride(-1 - dim));
          }
          values[3] = static_cast<idx_t>(tensor.stride(channel_dim));
          values[4] = static_cast<idx_t>(channel_dim == 1 ? tensor.stride(0) : 0);
          return values;
        };
        c10::metal::array<idx_t, 3> left_pad{};
        for (const auto dim : c10::irange(padding_dim)) {
          left_pad[dim] = static_cast<idx_t>(padding[2 * dim]);
        }
        const PadParams<idx_t> params{
            .input_sizes = sizes(input_tensor),
            .output_sizes = sizes(output_tensor),
            .left_pad = left_pad,
            .input_strides = strides(input_tensor),
            .output_strides = strides(output_tensor),
            .channels = static_cast<idx_t>(input.size(channel_dim)),
        };
        mtl_setArgs(encoder, source, output, params);
      });
      [encoder dispatchThreads:grid threadsPerThreadgroup:pad_threadgroup(pso, grid.width, grid.height, grid.depth)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  return output;
}

static void replication_pad1d_kernel_mps(const Tensor& input_, IntArrayRef padding, const Tensor& output) {
  if (output.numel() == 0 || input_.numel() == 0) {
    return;
  }
  auto input = input_.contiguous();
  const bool output_needs_copy = !output.is_contiguous();
  auto output_buf = output_needs_copy ? at::empty(output.sizes(), output.options()) : output;
  auto output_c = output_buf;
  if (input.dim() == 2) {
    input = input.unsqueeze(0);
    output_c = output_c.unsqueeze(0);
  }
  TORCH_INTERNAL_ASSERT(input.dim() == 3 && output_c.dim() == 3);

  const auto nbatch = c10::checked_convert<int32_t>(input.size(0), "int32_t");
  const auto nplane = c10::checked_convert<int32_t>(input.size(1), "int32_t");
  const auto input_W = c10::checked_convert<int32_t>(input.size(2), "int32_t");
  const auto output_W = c10::checked_convert<int32_t>(output_c.size(2), "int32_t");
  const std::array<int32_t, 4> sizes_pad = {input_W,
                                            output_W,
                                            c10::checked_convert<int32_t>(padding[0], "int32_t"),
                                            c10::checked_convert<int32_t>(padding[1], "int32_t")};

  auto pso = lib.getPipelineStateForFunc("replication_pad1d_forward_" + scalarToMetalTypeString(input));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "replication_pad1d_forward", {input, output_c}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, input, output_c, sizes_pad);
      [encoder dispatchThreads:MTLSizeMake(output_W, nplane, nbatch)
          threadsPerThreadgroup:pad_threadgroup(pso, output_W, nplane, nbatch)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  if (output_needs_copy) {
    output.copy_(output_buf);
  }
}

static void replication_pad1d_backward_kernel_mps(const Tensor& grad_output_,
                                                  const Tensor& input,
                                                  IntArrayRef padding,
                                                  const Tensor& grad_input) {
  if (grad_input.numel() == 0 || grad_output_.numel() == 0) {
    return;
  }
  auto grad_output = grad_output_.contiguous();
  const bool grad_input_needs_copy = !grad_input.is_contiguous();
  auto grad_input_buf = grad_input_needs_copy ? at::empty(grad_input.sizes(), grad_input.options()) : grad_input;
  auto grad_input_c = grad_input_buf;
  if (input.dim() == 2) {
    grad_output = grad_output.unsqueeze(0);
    grad_input_c = grad_input_c.unsqueeze(0);
  }
  TORCH_INTERNAL_ASSERT(grad_output.dim() == 3 && grad_input_c.dim() == 3);

  const auto nbatch = c10::checked_convert<int32_t>(grad_input_c.size(0), "int32_t");
  const auto nplane = c10::checked_convert<int32_t>(grad_input_c.size(1), "int32_t");
  const auto input_W = c10::checked_convert<int32_t>(grad_input_c.size(2), "int32_t");
  const auto output_W = c10::checked_convert<int32_t>(grad_output.size(2), "int32_t");
  const std::array<int32_t, 4> sizes_pad = {input_W,
                                            output_W,
                                            c10::checked_convert<int32_t>(padding[0], "int32_t"),
                                            c10::checked_convert<int32_t>(padding[1], "int32_t")};

  auto pso = lib.getPipelineStateForFunc("replication_pad1d_backward_" + scalarToMetalTypeString(grad_input_c));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "replication_pad1d_backward", {grad_output, grad_input_c}, stream);
      auto encoder = stream->commandEncoder();
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, grad_output, grad_input_c, sizes_pad);
      [encoder dispatchThreads:MTLSizeMake(input_W, nplane, nbatch)
          threadsPerThreadgroup:pad_threadgroup(pso, input_W, nplane, nbatch)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
  if (grad_input_needs_copy) {
    grad_input.copy_(grad_input_buf);
  }
}

static Tensor crop_negative_pads(const Tensor& self, IntArrayRef pad, int64_t padding_dim) {
  const auto ndim = self.dim();
  auto cropped = self;
  for (const auto dim : c10::irange(ndim - padding_dim, ndim)) {
    const auto pad_idx = 2 * (ndim - dim - 1);
    if (pad[pad_idx] < 0) {
      cropped = cropped.narrow(dim, -pad[pad_idx], cropped.size(dim) + pad[pad_idx]);
    }
    if (pad[pad_idx + 1] < 0) {
      cropped = cropped.narrow(dim, 0, cropped.size(dim) + pad[pad_idx + 1]);
    }
  }
  return cropped;
}

static Tensor allocate_pad_output(const Tensor& self,
                                  const Tensor& cropped,
                                  IntArrayRef pad,
                                  int64_t padding_dim,
                                  bool all_pads_non_positive) {
  if (all_pads_non_positive) {
    return at::empty_like(cropped);
  }
  const auto ndim = self.dim();
  auto output_sizes = self.sizes().vec();
  for (const auto dim : c10::irange(ndim - padding_dim, ndim)) {
    const auto pad_idx = 2 * (ndim - dim - 1);
    const auto output_size = self.size(dim) + pad[pad_idx] + pad[pad_idx + 1];
    TORCH_CHECK(output_size >= 0,
                "The input size ",
                self.size(dim),
                ", plus negative padding ",
                pad[pad_idx],
                " and ",
                pad[pad_idx + 1],
                " resulted in a negative output size, which is invalid. Check dimension ",
                dim,
                " of your input.");
    output_sizes[dim] = output_size;
  }
  return at::empty(output_sizes, self.options().memory_format(self.suggest_memory_format()));
}

static bool constant_pad_dense_eligible(const Tensor& input, const Tensor& output, int64_t padding_dim) {
  if (padding_dim > 3 || !input.is_contiguous() || !output.is_contiguous()) {
    return false;
  }
  constexpr auto uint_max = std::numeric_limits<uint32_t>::max();
  const auto out_w = output.size(-1);
  const auto out_h = padding_dim >= 2 ? output.size(-2) : 1;
  const auto out_d = padding_dim >= 3 ? output.size(-3) : 1;
  const bool sizes_fit = out_w <= uint_max && out_h <= uint_max && out_d <= uint_max;
  return sizes_fit && output.numel() / (out_w * out_h) <= uint_max;
}

static void constant_pad_dense_kernel_mps(const Tensor& input,
                                          const Tensor& output,
                                          IntArrayRef pad,
                                          int64_t padding_dim,
                                          const Scalar& fill) {
  const auto in_w = input.size(-1);
  const auto in_h = padding_dim >= 2 ? input.size(-2) : 1;
  const auto in_d = padding_dim >= 3 ? input.size(-3) : 1;
  const auto out_w = output.size(-1);
  const auto out_h = padding_dim >= 2 ? output.size(-2) : 1;
  const auto out_d = padding_dim >= 3 ? output.size(-3) : 1;
  const auto left_w = padding_dim >= 1 ? std::max<int64_t>(pad[0], 0) : 0;
  const auto left_h = padding_dim >= 2 ? std::max<int64_t>(pad[2], 0) : 0;
  const auto left_d = padding_dim >= 3 ? std::max<int64_t>(pad[4], 0) : 0;
  const auto grid_x = c10::metal::ceil_div(out_w, static_cast<int64_t>(c10::metal::ILP_PER_THREAD));
  const auto grid_z = output.numel() / (out_w * out_h);
  const ConstantPadDenseParams params = {
      {static_cast<uint32_t>(in_w), static_cast<uint32_t>(in_h), static_cast<uint32_t>(in_d)},
      {static_cast<uint32_t>(out_w), static_cast<uint32_t>(out_h), static_cast<uint32_t>(out_d)},
      {static_cast<uint32_t>(left_w), static_cast<uint32_t>(left_h), static_cast<uint32_t>(left_d)}};
  auto pso = lib.getPipelineStateForFunc("constant_pad_nd_dense_" + scalarToMetalTypeString(input));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "constant_pad_nd", {input, output}, stream);
      auto encoder = stream->commandEncoder();
      auto fill_value = getMPSScalar(fill, input.scalar_type());
      [encoder setComputePipelineState:pso];
      mtl_setArgs(encoder, input, output, params, fill_value);
      [encoder dispatchThreads:MTLSizeMake(grid_x, out_h, grid_z)
          threadsPerThreadgroup:pad_threadgroup(pso, grid_x, out_h, grid_z)];
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

static void constant_pad_strided_kernel_mps(const Tensor& input,
                                            const Tensor& output,
                                            IntArrayRef pad,
                                            int64_t padding_dim,
                                            const Scalar& fill) {
  const auto ndim = output.dim();
  DimVector dim_order(ndim);
  std::iota(dim_order.begin(), dim_order.end(), 0);
  std::stable_sort(dim_order.begin(), dim_order.end(), [&](int64_t lhs, int64_t rhs) {
    return output.stride(lhs) < output.stride(rhs);
  });
  const auto inner = output.size(dim_order[0]);
  const auto grid_x = c10::metal::ceil_div(inner, static_cast<int64_t>(c10::metal::ILP_PER_THREAD));
  const auto grid_y = output.numel() / inner;
  const bool use_u32 = offsetsFitIn<uint32_t>(input, output);
  auto pso = lib.getPipelineStateForFunc(
      fmt::format("constant_pad_nd_{}{}", scalarToMetalTypeString(input), mtlIdxSuffix(use_u32)));
  auto stream = getCurrentMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    @autoreleasepool {
      getMPSProfiler().beginProfileKernel(pso, "constant_pad_nd", {input, output}, stream);
      auto encoder = stream->commandEncoder();
      auto fill_value = getMPSScalar(fill, input.scalar_type());
      [encoder setComputePipelineState:pso];
      mtlDispatchByIndexWidth<uint32_t, uint64_t>(use_u32, [&](auto idx_tag) {
        using idx_t = typename decltype(idx_tag)::type;
        ConstantPadNdParams<idx_t> params{};
        params.ndim = static_cast<uint32_t>(ndim);
        for (const auto i : c10::irange(ndim)) {
          const auto dim = dim_order[i];
          const auto pad_idx = 2 * (ndim - dim - 1);
          params.output_sizes[i] = static_cast<idx_t>(output.size(dim));
          params.input_sizes[i] = static_cast<idx_t>(input.size(dim));
          params.input_strides[i] = static_cast<idx_t>(input.stride(dim));
          params.output_strides[i] = static_cast<idx_t>(output.stride(dim));
          params.left_pad[i] = static_cast<idx_t>(dim >= ndim - padding_dim ? std::max<int64_t>(pad[pad_idx], 0) : 0);
        }
        mtl_setArgs(encoder, input, output, params, fill_value);
      });
      mtl_dispatch2DJob(encoder, pso, grid_x, grid_y);
      getMPSProfiler().endProfileKernel(pso, stream);
    }
  });
}

} // namespace mps

// 1D Reflection and Replication Padding
TORCH_IMPL_FUNC(reflection_pad1d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, std::nullopt, true, "reflection_pad1d_out_mps");
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  mps::pad_out_template(
      const_cast<Tensor&>(grad_input), input, padding, grad_output, true, "reflection_pad1d_backward_out_mps");
}

TORCH_IMPL_FUNC(replication_pad1d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::replication_pad1d_kernel_mps(input, padding, output);
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  mps::replication_pad1d_backward_kernel_mps(grad_output, input, padding, grad_input);
}

// 2D Reflection and Replication Padding
Tensor& reflection_pad2d_out_mps(const Tensor& input, IntArrayRef padding, Tensor& output) {
  return mps::pad_out_template(output, input, padding, std::nullopt, true, __func__);
}

Tensor reflection_pad2d_mps(const Tensor& input, IntArrayRef padding) {
  Tensor output = at::empty({0}, input.options());
  return mps::pad_out_template(output, input, padding, std::nullopt, true, __func__);
}

Tensor& reflection_pad2d_backward_out_mps(const Tensor& grad_output,
                                          const Tensor& input,
                                          IntArrayRef padding,
                                          Tensor& grad_input) {
  return mps::pad_out_template(grad_input, input, padding, grad_output, true, __func__);
}

Tensor reflection_pad2d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, true, __func__);
}

TORCH_IMPL_FUNC(replication_pad2d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, std::nullopt, false, "replication_pad2d_out_mps");
}

Tensor& replication_pad2d_backward_out_mps(const Tensor& grad_output,
                                           const Tensor& input,
                                           IntArrayRef padding,
                                           Tensor& grad_input) {
  return mps::pad_out_template(grad_input, input, padding, grad_output, false, __func__);
}

Tensor replication_pad2d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, false, __func__);
}

// 3D Reflection and Replication Padding
TORCH_IMPL_FUNC(reflection_pad3d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, std::nullopt, true, "reflection_pad3d_out_mps");
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, IntArrayRef padding, const Tensor& grad_input) {
  mps::pad_out_template(
      const_cast<Tensor&>(grad_input), input, padding, grad_output, true, "reflection_pad3d_backward_out_mps");
}

TORCH_IMPL_FUNC(replication_pad3d_out_mps)
(const Tensor& input, IntArrayRef padding, const Tensor& output) {
  mps::pad_out_template(const_cast<Tensor&>(output), input, padding, std::nullopt, false, "replication_pad3d_out_mps");
}

Tensor& replication_pad3d_backward_out_mps(const Tensor& grad_output,
                                           const Tensor& input,
                                           IntArrayRef padding,
                                           Tensor& grad_input) {
  return mps::pad_out_template(grad_input, input, padding, grad_output, false, __func__);
}

Tensor replication_pad3d_backward_mps(const Tensor& grad_output, const Tensor& input, IntArrayRef padding) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return mps::pad_out_template(grad_input, input, padding, grad_output, false, __func__);
}

// backward pass is explicitly handled in autograd by negating the "pad" argument
Tensor constant_pad_nd_mps(const Tensor& self, IntArrayRef pad, const Scalar& value) {
  TORCH_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size());

  const auto ndim = self.dim();
  const auto padding_dim = static_cast<int64_t>(pad.size() / 2);
  TORCH_CHECK(ndim >= padding_dim,
              "Length of pad should be no more than twice the number of dimensions of the input. Pad length is ",
              pad.size(),
              " while the input has ",
              ndim,
              " dimensions.");

  // Negative pads mean we crop the input
  const bool all_pads_non_positive = std::ranges::all_of(pad, [](int64_t p) { return p <= 0; });
  const auto cropped = mps::crop_negative_pads(self, pad, padding_dim);
  if (all_pads_non_positive && cropped.is_contiguous()) {
    return cropped.clone();
  }

  auto output = mps::allocate_pad_output(self, cropped, pad, padding_dim, all_pads_non_positive);
  if (output.numel() == 0) {
    return output;
  }
  if (cropped.numel() == 0) {
    return output.fill_(value);
  }

  const Scalar fill = all_pads_non_positive ? Scalar(0) : value;
  if (mps::constant_pad_dense_eligible(cropped, output, padding_dim)) {
    mps::constant_pad_dense_kernel_mps(cropped, output, pad, padding_dim, fill);
  } else if (output.numel() > std::numeric_limits<uint32_t>::max() || ndim > c10::metal::max_ndim) {
    return at::native::constant_pad_nd(self, pad, value);
  } else {
    mps::constant_pad_strided_kernel_mps(cropped, output, pad, padding_dim, fill);
  }
  return output;
}

} // namespace at::native
