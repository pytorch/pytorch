#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

static int64_t pool_output_size(int64_t input_size, int64_t kernel_size,
                                int64_t padding, int64_t stride) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

// ── Max Pool 2D ─────────────────────────────────────────────────
at::Tensor vulkan_max_pool2d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {

    auto input = self.contiguous();
    check_supported_float(input, "max_pool2d");
    auto orig_dtype = input.scalar_type();
    input = ensure_float32(input);
    TORCH_CHECK(input.dim() == 4, "max_pool2d requires 4D input [N,C,H,W]");

    int64_t kH = kernel_size[0];
    int64_t kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride.empty() ? kH : stride[0];
    int64_t sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
    int64_t pH = padding.empty() ? 0 : padding[0];
    int64_t pW = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pH);

    int64_t N = input.size(0), C = input.size(1);
    int64_t iH = input.size(2), iW = input.size(3);
    int64_t oH = pool_output_size(iH, kH, pH, sH);
    int64_t oW = pool_output_size(iW, kW, pW, sW);

    auto output = at::empty({N, C, oH, oW}, input.options());
    uint32_t total = static_cast<uint32_t>(N * C * oH * oW);

    if (total == 0) return output;

    struct {
        uint32_t batch_channels;
        uint32_t iH, iW, oH, oW, kH, kW, sH, sW, pH, pW;
    } params{
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(iH), static_cast<uint32_t>(iW),
        static_cast<uint32_t>(oH), static_cast<uint32_t>(oW),
        static_cast<uint32_t>(kH), static_cast<uint32_t>(kW),
        static_cast<uint32_t>(sH), static_cast<uint32_t>(sW),
        static_cast<uint32_t>(pH), static_cast<uint32_t>(pW)
    };

    uint32_t workgroups = (total + 255) / 256;
    dispatch_shader("pooling_max_pool2d_fwd",
                    shaders::pooling_max_pool2d_fwd, shaders::pooling_max_pool2d_fwd_size,
                    {input, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── Avg Pool 2D ─────────────────────────────────────────────────
at::Tensor vulkan_avg_pool2d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {

    auto input = self.contiguous();
    check_supported_float(input, "avg_pool2d");
    auto orig_dtype = input.scalar_type();
    input = ensure_float32(input);
    TORCH_CHECK(input.dim() == 4, "avg_pool2d requires 4D input [N,C,H,W]");

    int64_t kH = kernel_size[0];
    int64_t kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride.empty() ? kH : stride[0];
    int64_t sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
    int64_t pH = padding.empty() ? 0 : padding[0];
    int64_t pW = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pH);

    int64_t N = input.size(0), C = input.size(1);
    int64_t iH = input.size(2), iW = input.size(3);
    int64_t oH = pool_output_size(iH, kH, pH, sH);
    int64_t oW = pool_output_size(iW, kW, pW, sW);

    auto output = at::empty({N, C, oH, oW}, input.options());
    uint32_t total = static_cast<uint32_t>(N * C * oH * oW);

    if (total == 0) return output;

    struct {
        uint32_t batch_channels;
        uint32_t iH, iW, oH, oW, kH, kW, sH, sW, pH, pW;
        uint32_t count_include_pad_flag;
    } params{
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(iH), static_cast<uint32_t>(iW),
        static_cast<uint32_t>(oH), static_cast<uint32_t>(oW),
        static_cast<uint32_t>(kH), static_cast<uint32_t>(kW),
        static_cast<uint32_t>(sH), static_cast<uint32_t>(sW),
        static_cast<uint32_t>(pH), static_cast<uint32_t>(pW),
        count_include_pad ? 1u : 0u
    };

    uint32_t workgroups = (total + 255) / 256;
    dispatch_shader("pooling_avg_pool2d_fwd",
                    shaders::pooling_avg_pool2d_fwd, shaders::pooling_avg_pool2d_fwd_size,
                    {input, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── Adaptive Avg Pool 2D (GPU shader) ───────────────────────────
at::Tensor vulkan_adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size) {
    auto input = self.contiguous();
    TORCH_CHECK(input.dim() == 4, "adaptive_avg_pool2d requires 4D input");
    check_supported_float(input, "adaptive_avg_pool2d");
    auto orig_dtype = input.scalar_type();
    input = ensure_float32(input);

    int64_t N = input.size(0), C = input.size(1);
    int64_t iH = input.size(2), iW = input.size(3);
    int64_t oH = output_size[0], oW = output_size[1];

    auto output = at::empty({N, C, oH, oW}, input.options());
    uint32_t total = static_cast<uint32_t>(N * C * oH * oW);
    if (total == 0) return output;

    struct {
        uint32_t batch_channels;
        uint32_t iH, iW, oH, oW;
    } params{
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(iH), static_cast<uint32_t>(iW),
        static_cast<uint32_t>(oH), static_cast<uint32_t>(oW)
    };

    uint32_t workgroups = (total + 255) / 256;
    dispatch_shader("pooling_adaptive_avg_pool2d_fwd",
                    shaders::pooling_adaptive_avg_pool2d_fwd, shaders::pooling_adaptive_avg_pool2d_fwd_size,
                    {input, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── Adaptive Avg Pool 2D Backward (GPU shader) ──────────────────
at::Tensor vulkan_adaptive_avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& input) {
    auto go_c = grad_output.contiguous();
    check_supported_float(go_c, "adaptive_avg_pool2d_backward");
    auto orig_dtype = go_c.scalar_type();
    go_c = ensure_float32(go_c);

    int64_t N = input.size(0), C = input.size(1);
    int64_t iH = input.size(2), iW = input.size(3);
    int64_t oH = go_c.size(2), oW = go_c.size(3);

    auto grad_input = at::empty({N, C, iH, iW}, go_c.options());
    uint32_t numel = static_cast<uint32_t>(grad_input.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    struct {
        uint32_t numel;
        uint32_t batch_channels;
        uint32_t iH, iW, oH, oW;
    } params{
        numel,
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(iH), static_cast<uint32_t>(iW),
        static_cast<uint32_t>(oH), static_cast<uint32_t>(oW)
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("pooling_adaptive_avg_pool2d_backward_fwd",
                    shaders::pooling_adaptive_avg_pool2d_backward_fwd,
                    shaders::pooling_adaptive_avg_pool2d_backward_fwd_size,
                    {go_c, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

}} // namespace torch_vulkan::ops
