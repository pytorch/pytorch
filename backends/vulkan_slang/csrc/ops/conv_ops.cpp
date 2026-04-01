#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

static int64_t conv_output_size(int64_t input_size, int64_t kernel_size,
                                int64_t padding, int64_t stride, int64_t dilation) {
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

at::Tensor vulkan_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {

    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();

    check_supported_float(input_c, "conv2d");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);
    weight_c = ensure_float32(weight_c);
    TORCH_CHECK(input_c.dim() == 4, "conv2d requires 4D input [N,C,H,W]");
    TORCH_CHECK(weight_c.dim() == 4, "conv2d requires 4D weight [C_out,C_in/groups,kH,kW]");

    int64_t dH = dilation.size() > 0 ? dilation[0] : 1;
    int64_t dW = dilation.size() > 1 ? dilation[1] : dH;

    int64_t N = input_c.size(0);
    int64_t C_in = input_c.size(1);
    int64_t iH = input_c.size(2), iW = input_c.size(3);
    int64_t C_out = weight_c.size(0);
    int64_t kH = weight_c.size(2), kW = weight_c.size(3);
    int64_t sH = stride[0], sW = stride.size() > 1 ? stride[1] : sH;
    int64_t pH = padding[0], pW = padding.size() > 1 ? padding[1] : pH;

    int64_t oH = conv_output_size(iH, kH, pH, sH, dH);
    int64_t oW = conv_output_size(iW, kW, pW, sW, dW);

    auto output = at::empty({N, C_out, oH, oW}, input_c.options());
    uint32_t total = static_cast<uint32_t>(N * C_out * oH * oW);

    if (total == 0) return output;

    struct {
        uint32_t N;
        uint32_t C_in, C_out;
        uint32_t iH, iW;
        uint32_t oH, oW;
        uint32_t kH, kW;
        uint32_t sH, sW;
        uint32_t pH, pW;
        uint32_t groups;
        uint32_t dH, dW;
    } params{
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(C_in), static_cast<uint32_t>(C_out),
        static_cast<uint32_t>(iH), static_cast<uint32_t>(iW),
        static_cast<uint32_t>(oH), static_cast<uint32_t>(oW),
        static_cast<uint32_t>(kH), static_cast<uint32_t>(kW),
        static_cast<uint32_t>(sH), static_cast<uint32_t>(sW),
        static_cast<uint32_t>(pH), static_cast<uint32_t>(pW),
        static_cast<uint32_t>(groups),
        static_cast<uint32_t>(dH), static_cast<uint32_t>(dW)
    };

    uint32_t workgroups = (total + 255) / 256;

    dispatch_shader("conv_conv2d_fwd",
                    shaders::conv_conv2d_fwd, shaders::conv_conv2d_fwd_size,
                    {input_c, weight_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Add bias if provided
    if (bias_opt.has_value()) {
        auto bias = ensure_float32(bias_opt->contiguous());
        // bias is [C_out], need to broadcast to [N, C_out, oH, oW]
        auto bias_expanded = bias.reshape({1, C_out, 1, 1}).expand_as(output).contiguous();

        struct { uint32_t numel; } add_params{total};
        uint32_t add_wg = (total + 255) / 256;
        auto biased_output = at::empty_like(output);

        dispatch_shader("binary_add_fwd",
                        shaders::binary_add_fwd, shaders::binary_add_fwd_size,
                        {output, bias_expanded, biased_output},
                        add_wg, 1, 1,
                        &add_params, sizeof(add_params));
        return cast_from_float32(biased_output, orig_dtype);
    }

    return cast_from_float32(output, orig_dtype);
}

// ── Conv Transpose 2D (GPU shader) ──────────────────────────────
at::Tensor vulkan_conv_transpose2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation) {

    auto input_c = input.contiguous();
    auto weight_c = weight.contiguous();

    int64_t dH = dilation.size() > 0 ? dilation[0] : 1;
    int64_t dW = dilation.size() > 1 ? dilation[1] : dH;

    // CPU fallback for dilation > 1 or output_padding > 0
    int64_t opH = output_padding.size() > 0 ? output_padding[0] : 0;
    int64_t opW = output_padding.size() > 1 ? output_padding[1] : opH;
    check_supported_float(input_c, "conv_transpose2d");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);
    weight_c = ensure_float32(weight_c);
    if (dH != 1 || dW != 1 || opH != 0 || opW != 0) {
        TORCH_CHECK(false, "Vulkan conv_transpose2d: dilation>1 and output_padding>0 not implemented on GPU. ",
                    "Got dilation=(", dH, ",", dW, "), output_padding=(", opH, ",", opW, ")");
    }

    int64_t N = input_c.size(0);
    int64_t C_in = input_c.size(1);
    int64_t iH = input_c.size(2), iW = input_c.size(3);
    int64_t C_out = weight_c.size(1) * groups;  // Weight: [C_in, C_out/groups, kH, kW]
    int64_t kH = weight_c.size(2), kW = weight_c.size(3);
    int64_t sH = stride[0], sW = stride.size() > 1 ? stride[1] : sH;
    int64_t pH = padding[0], pW = padding.size() > 1 ? padding[1] : pH;

    // Transpose conv output size: oH = (iH - 1) * sH - 2 * pH + kH
    int64_t oH = (iH - 1) * sH - 2 * pH + kH;
    int64_t oW = (iW - 1) * sW - 2 * pW + kW;

    auto output = at::empty({N, C_out, oH, oW}, input_c.options());
    uint32_t total = static_cast<uint32_t>(N * C_out * oH * oW);
    if (total == 0) return output;

    struct {
        uint32_t N;
        uint32_t C_in, C_out;
        uint32_t iH, iW;
        uint32_t oH, oW;
        uint32_t kH, kW;
        uint32_t sH, sW;
        uint32_t pH, pW;
        uint32_t groups;
    } params{
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(C_in), static_cast<uint32_t>(C_out),
        static_cast<uint32_t>(iH), static_cast<uint32_t>(iW),
        static_cast<uint32_t>(oH), static_cast<uint32_t>(oW),
        static_cast<uint32_t>(kH), static_cast<uint32_t>(kW),
        static_cast<uint32_t>(sH), static_cast<uint32_t>(sW),
        static_cast<uint32_t>(pH), static_cast<uint32_t>(pW),
        static_cast<uint32_t>(groups)
    };

    uint32_t workgroups = (total + 255) / 256;
    dispatch_shader("conv_conv_transpose2d_fwd",
                    shaders::conv_conv_transpose2d_fwd, shaders::conv_conv_transpose2d_fwd_size,
                    {input_c, weight_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Add bias if provided
    if (bias_opt.has_value()) {
        auto bias = ensure_float32(bias_opt->contiguous());
        auto bias_expanded = bias.reshape({1, C_out, 1, 1}).expand_as(output).contiguous();
        struct { uint32_t numel; } add_params{total};
        uint32_t add_wg = (total + 255) / 256;
        auto biased_output = at::empty_like(output);
        dispatch_shader("binary_add_fwd",
                        shaders::binary_add_fwd, shaders::binary_add_fwd_size,
                        {output, bias_expanded, biased_output},
                        add_wg, 1, 1,
                        &add_params, sizeof(add_params));
        return cast_from_float32(biased_output, orig_dtype);
    }

    return cast_from_float32(output, orig_dtype);
}

}} // namespace torch_vulkan::ops
