#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/autograd.h>
#include <torch/library.h>
#include <cstring>

namespace torch_vulkan { namespace ops {

// ── ReLU Backward ───────────────────────────────────────────────
// grad_input = grad_output * (input > 0)
class VulkanReluFunction : public torch::autograd::Function<VulkanReluFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input) {
        ctx->save_for_backward({input});
        return vulkan_relu(input);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];

        // grad_input = grad_output * (input > 0)
        auto mask = vulkan_gt_scalar(input, at::Scalar(0.0f));
        auto mask_float = mask.to(at::kFloat);
        auto grad_input = vulkan_mul(grad_output, mask_float);

        return {grad_input};
    }
};

at::Tensor vulkan_relu_autograd(const at::Tensor& self) {
    return VulkanReluFunction::apply(self);
}

// ── MM Backward ─────────────────────────────────────────────────
// C = A @ B
// dA = dC @ B.T
// dB = A.T @ dC
class VulkanMmFunction : public torch::autograd::Function<VulkanMmFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, const at::Tensor& mat2) {
        ctx->save_for_backward({self, mat2});
        return vulkan_mm(self, mat2);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0];
        auto mat2 = saved[1];
        auto grad_output = grad_outputs[0];

        // Use mm_ex to avoid GPU permute copy from .t()
        auto grad_self = vulkan_mm_ex(grad_output, mat2, false, true);
        auto grad_mat2 = vulkan_mm_ex(self, grad_output, true, false);

        return {grad_self, grad_mat2};
    }
};

at::Tensor vulkan_mm_autograd(const at::Tensor& self, const at::Tensor& mat2) {
    return VulkanMmFunction::apply(self, mat2);
}

// ── AddMM Backward ──────────────────────────────────────────────
// out = beta * bias + alpha * (self @ mat2)
class VulkanAddmmFunction : public torch::autograd::Function<VulkanAddmmFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& bias, const at::Tensor& self,
                               const at::Tensor& mat2,
                               const at::Scalar& beta, const at::Scalar& alpha) {
        ctx->save_for_backward({self, mat2, bias});
        ctx->saved_data["beta"] = beta;
        ctx->saved_data["alpha"] = alpha;
        return vulkan_addmm(bias, self, mat2, beta, alpha);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0];
        auto mat2 = saved[1];
        auto bias = saved[2];
        auto grad_output = grad_outputs[0];
        auto alpha = ctx->saved_data["alpha"].toScalar();
        auto beta = ctx->saved_data["beta"].toScalar();

        // grad_bias = beta * sum(grad_output, dim=0)
        at::Tensor grad_bias;
        if (bias.dim() == 1) {
            grad_bias = vulkan_sum(grad_output, at::IntArrayRef({0}), false, c10::nullopt);
            if (beta.toDouble() != 1.0)
                grad_bias = vulkan_mul_scalar(grad_bias, beta);
        } else {
            grad_bias = grad_output;
            if (beta.toDouble() != 1.0)
                grad_bias = vulkan_mul_scalar(grad_bias, beta);
        }

        // grad_self = alpha * grad_output @ mat2.T
        // Use mm_ex to avoid GPU permute copy from .t()
        auto grad_self = vulkan_mm_ex(grad_output, mat2, false, true);
        if (alpha.toDouble() != 1.0)
            grad_self = vulkan_mul_scalar(grad_self, alpha);

        // grad_mat2 = alpha * self.T @ grad_output
        auto grad_mat2 = vulkan_mm_ex(self, grad_output, true, false);
        if (alpha.toDouble() != 1.0)
            grad_mat2 = vulkan_mul_scalar(grad_mat2, alpha);

        return {grad_bias, grad_self, grad_mat2, at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_addmm_autograd(const at::Tensor& bias, const at::Tensor& self,
                                   const at::Tensor& mat2,
                                   const at::Scalar& beta, const at::Scalar& alpha) {
    return VulkanAddmmFunction::apply(bias, self, mat2, beta, alpha);
}

// ── Linear Backward ─────────────────────────────────────────────
class VulkanLinearFunction : public torch::autograd::Function<VulkanLinearFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, const at::Tensor& weight,
                               const at::Tensor& bias) {
        ctx->save_for_backward({input, weight});
        ctx->saved_data["has_bias"] = bias.defined();
        return vulkan_linear(input, weight,
                             bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto grad_output = grad_outputs[0];
        bool has_bias = ctx->saved_data["has_bias"].toBool();

        // Flatten batch dims for matmul
        auto orig_shape = input.sizes().vec();
        int64_t in_features = input.size(-1);
        int64_t batch = input.numel() / in_features;
        auto input_2d = input.reshape({batch, in_features});
        auto grad_2d = grad_output.reshape({batch, weight.size(0)});

        // grad_input = grad_output @ weight
        auto grad_input = vulkan_mm(grad_2d, weight);
        grad_input = grad_input.reshape(orig_shape);

        // grad_weight = grad_output.T @ input
        // Use mm_ex to avoid GPU permute copy from .t()
        auto grad_weight = vulkan_mm_ex(grad_2d, input_2d, true, false);

        // grad_bias = sum(grad_output, dim=0...-1)
        at::Tensor grad_bias;
        if (has_bias) {
            grad_bias = vulkan_sum(grad_2d, at::IntArrayRef({0}), false, c10::nullopt);
        }

        return {grad_input, grad_weight, grad_bias};
    }
};

at::Tensor vulkan_linear_autograd(const at::Tensor& input, const at::Tensor& weight,
                                    const std::optional<at::Tensor>& bias_opt) {
    auto bias = bias_opt.has_value() ? *bias_opt : at::Tensor();
    return VulkanLinearFunction::apply(input, weight, bias);
}


// ── BMM Backward ────────────────────────────────────────────────
class VulkanBmmFunction : public torch::autograd::Function<VulkanBmmFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, const at::Tensor& mat2) {
        ctx->save_for_backward({self, mat2});
        return vulkan_bmm(self, mat2);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto self = saved[0];
        auto mat2 = saved[1];
        auto grad_output = grad_outputs[0];

        // dA = dC @ B.T
        // Use bmm_ex to avoid GPU permute copy from .transpose()
        auto grad_self = vulkan_bmm_ex(grad_output, mat2, false, true);
        // dB = A.T @ dC
        auto grad_mat2 = vulkan_bmm_ex(self, grad_output, true, false);

        return {grad_self, grad_mat2};
    }
};

at::Tensor vulkan_bmm_autograd(const at::Tensor& self, const at::Tensor& mat2) {
    return VulkanBmmFunction::apply(self, mat2);
}

// ── Sigmoid Backward ────────────────────────────────────────────
// grad_input = grad_output * sigmoid(input) * (1 - sigmoid(input))
class VulkanSigmoidFunction : public torch::autograd::Function<VulkanSigmoidFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input) {
        auto output = vulkan_sigmoid(input);
        ctx->save_for_backward({output});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto output = saved[0];
        auto grad_output = grad_outputs[0];

        auto one_minus = vulkan_rsub_scalar(output, at::Scalar(1.0f));
        auto grad_input = vulkan_mul(vulkan_mul(grad_output, output), one_minus);
        return {grad_input};
    }
};

at::Tensor vulkan_sigmoid_autograd(const at::Tensor& self) {
    return VulkanSigmoidFunction::apply(self);
}

// ── Tanh Backward ───────────────────────────────────────────────
// grad_input = grad_output * (1 - tanh(input)^2)
class VulkanTanhFunction : public torch::autograd::Function<VulkanTanhFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input) {
        auto output = vulkan_tanh(input);
        ctx->save_for_backward({output});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto output = saved[0];
        auto grad_output = grad_outputs[0];

        auto output_sq = vulkan_mul(output, output);
        auto one_minus_sq = vulkan_rsub_scalar(output_sq, at::Scalar(1.0f));
        auto grad_input = vulkan_mul(grad_output, one_minus_sq);
        return {grad_input};
    }
};

at::Tensor vulkan_tanh_autograd(const at::Tensor& self) {
    return VulkanTanhFunction::apply(self);
}

// ── GELU Backward ──────────────────────────────────────────────
// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// grad ≈ gelu'(x) computed via forward ops
class VulkanGeluFunction : public torch::autograd::Function<VulkanGeluFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, c10::string_view approximate) {
        ctx->save_for_backward({input});
        ctx->saved_data["approximate"] = std::string(approximate);
        return vulkan_gelu(input, approximate);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];

        // Use GPU gelu_backward shader
        auto approx = ctx->saved_data["approximate"].toStringRef();
        auto grad_input = vulkan_gelu_backward(grad_output, input, approx);
        return {grad_input, at::Tensor()};
    }
};

at::Tensor vulkan_gelu_autograd(const at::Tensor& self, c10::string_view approximate) {
    return VulkanGeluFunction::apply(self, approximate);
}

// ── SiLU Backward ──────────────────────────────────────────────
// silu(x) = x * sigmoid(x)
// silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
class VulkanSiluFunction : public torch::autograd::Function<VulkanSiluFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input) {
        auto sig = vulkan_sigmoid(input);
        ctx->save_for_backward({input, sig});
        return vulkan_mul(input, sig);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto sig = saved[1];
        auto grad_output = grad_outputs[0];

        // grad = sig * (1 + x * (1 - sig))
        auto one_minus_sig = vulkan_rsub_scalar(sig, at::Scalar(1.0f));
        auto x_times_one_minus_sig = vulkan_mul(input, one_minus_sig);
        auto one_plus = vulkan_add_scalar(x_times_one_minus_sig, at::Scalar(1.0f), at::Scalar(1));
        auto grad_factor = vulkan_mul(sig, one_plus);
        auto grad_input = vulkan_mul(grad_output, grad_factor);

        return {grad_input};
    }
};

at::Tensor vulkan_silu_autograd(const at::Tensor& self) {
    return VulkanSiluFunction::apply(self);
}

// ── Leaky ReLU Backward ────────────────────────────────────────
// grad_input = grad_output * (input >= 0 ? 1 : negative_slope)
class VulkanLeakyReluFunction : public torch::autograd::Function<VulkanLeakyReluFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, const at::Scalar& negative_slope) {
        ctx->save_for_backward({input});
        ctx->saved_data["negative_slope"] = negative_slope;
        return vulkan_leaky_relu(input, negative_slope);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];
        auto ns = ctx->saved_data["negative_slope"].toScalar();

        auto mask = vulkan_ge_scalar(input, at::Scalar(0.0f));
        auto mask_float = mask.to(at::kFloat);
        // grad_factor = mask + (1 - mask) * negative_slope
        auto one_minus_mask = vulkan_rsub_scalar(mask_float, at::Scalar(1.0f));
        auto scaled = vulkan_mul_scalar(one_minus_mask, ns);
        auto grad_factor = vulkan_add(mask_float, scaled, at::Scalar(1));
        auto grad_input = vulkan_mul(grad_output, grad_factor);

        return {grad_input, at::Tensor()};
    }
};

at::Tensor vulkan_leaky_relu_autograd(const at::Tensor& self, const at::Scalar& negative_slope) {
    return VulkanLeakyReluFunction::apply(self, negative_slope);
}

// ── ELU Backward ───────────────────────────────────────────────
// elu(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
// grad = grad_output * (x > 0 ? scale*input_scale : scale*input_scale*(output + alpha*scale))
// simplified for default scale=1, input_scale=1:
// grad = grad_output * (x > 0 ? 1 : output + alpha)
class VulkanEluFunction : public torch::autograd::Function<VulkanEluFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, const at::Scalar& alpha,
                               const at::Scalar& scale, const at::Scalar& input_scale) {
        auto output = vulkan_elu(input, alpha, scale, input_scale);
        ctx->save_for_backward({input, output});
        ctx->saved_data["alpha"] = alpha;
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto output = saved[1];
        auto grad_output = grad_outputs[0];
        auto alpha = ctx->saved_data["alpha"].toScalar();

        auto mask = vulkan_gt_scalar(input, at::Scalar(0.0f));
        auto mask_float = mask.to(at::kFloat);
        // For x <= 0: derivative = output + alpha
        auto neg_deriv = vulkan_add_scalar(output, alpha, 1);
        // grad_factor = mask * 1 + (1 - mask) * neg_deriv
        auto one_minus_mask = vulkan_rsub_scalar(mask_float, at::Scalar(1.0f));
        auto neg_part = vulkan_mul(one_minus_mask, neg_deriv);
        auto grad_factor = vulkan_add(mask_float, neg_part, at::Scalar(1));
        auto grad_input = vulkan_mul(grad_output, grad_factor);

        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_elu_autograd(const at::Tensor& self, const at::Scalar& alpha,
                                const at::Scalar& scale, const at::Scalar& input_scale) {
    return VulkanEluFunction::apply(self, alpha, scale, input_scale);
}

// ── Softmax Backward ───────────────────────────────────────────
// grad_input = softmax * (grad_output - sum(grad_output * softmax, dim))
class VulkanSoftmaxFunction : public torch::autograd::Function<VulkanSoftmaxFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, int64_t dim,
                               std::optional<at::ScalarType> dtype) {
        auto output = vulkan_softmax(input, dim, dtype);
        ctx->save_for_backward({output});
        ctx->saved_data["dim"] = dim;
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto output = saved[0];
        auto grad_output = grad_outputs[0];
        int64_t dim = ctx->saved_data["dim"].toInt();

        // grad_input = output * (grad_output - sum(grad_output * output, dim, keepdim))
        auto gxo = vulkan_mul(grad_output, output);
        auto sum_gxo = vulkan_sum(gxo, at::IntArrayRef({dim}), true, c10::nullopt);
        // Expand sum back to full shape
        auto sum_expanded = sum_gxo.expand_as(grad_output).contiguous();
        auto diff = vulkan_sub(grad_output, sum_expanded, at::Scalar(1));
        auto grad_input = vulkan_mul(output, diff);

        return {grad_input, at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_softmax_autograd(const at::Tensor& self, int64_t dim,
                                     std::optional<at::ScalarType> dtype) {
    return VulkanSoftmaxFunction::apply(self, dim, dtype);
}

// ── Log Softmax Backward ───────────────────────────────────────
// grad_input = grad_output - exp(log_softmax) * sum(grad_output, dim)
class VulkanLogSoftmaxFunction : public torch::autograd::Function<VulkanLogSoftmaxFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, int64_t dim,
                               std::optional<at::ScalarType> dtype) {
        auto output = vulkan_log_softmax(input, dim, dtype);
        ctx->save_for_backward({output});
        ctx->saved_data["dim"] = dim;
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto output = saved[0];
        auto grad_output = grad_outputs[0];
        int64_t dim = ctx->saved_data["dim"].toInt();

        // grad_input = grad_output - softmax * sum(grad_output, dim, keepdim)
        auto softmax = vulkan_exp(output);
        auto sum_grad = vulkan_sum(grad_output, at::IntArrayRef({dim}), true, c10::nullopt);
        auto sum_expanded = sum_grad.expand_as(grad_output).contiguous();
        auto scaled = vulkan_mul(softmax, sum_expanded);
        auto grad_input = vulkan_sub(grad_output, scaled, at::Scalar(1));

        return {grad_input, at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_log_softmax_autograd(const at::Tensor& self, int64_t dim,
                                         std::optional<at::ScalarType> dtype) {
    return VulkanLogSoftmaxFunction::apply(self, dim, dtype);
}

// ── Conv2D Backward ────────────────────────────────────────────
// Uses dedicated backward shaders for grad_input and grad_weight.
// Falls back to CPU for dilation > 1 (matching forward behavior).
class VulkanConv2dFunction : public torch::autograd::Function<VulkanConv2dFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, const at::Tensor& weight,
                               const at::Tensor& bias,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, int64_t groups) {
        ctx->save_for_backward({input, weight});
        ctx->saved_data["has_bias"] = bias.defined();
        ctx->saved_data["stride"] = stride.vec();
        ctx->saved_data["padding"] = padding.vec();
        ctx->saved_data["dilation"] = dilation.vec();
        ctx->saved_data["groups"] = groups;

        return vulkan_conv2d(input, weight,
                             bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt,
                             stride, padding, dilation, groups);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto grad_output = grad_outputs[0];

        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        auto dilation = ctx->saved_data["dilation"].toIntVector();
        int64_t groups = ctx->saved_data["groups"].toInt();
        bool has_bias = ctx->saved_data["has_bias"].toBool();

        // For dilation > 1, not implemented on GPU
        TORCH_CHECK(dilation[0] == 1 && (dilation.size() <= 1 || dilation[1] == 1),
                    "Vulkan conv2d backward: dilation > 1 is not implemented on GPU. ",
                    "Got dilation=(", dilation[0], ",", dilation.size() > 1 ? dilation[1] : 1, ")");

        auto input_c = input.contiguous();
        auto weight_c = weight.contiguous();
        auto grad_c = grad_output.contiguous();

        int64_t N = input_c.size(0);
        int64_t C_in = input_c.size(1);
        int64_t iH = input_c.size(2), iW = input_c.size(3);
        int64_t C_out = weight_c.size(0);
        int64_t kH = weight_c.size(2), kW = weight_c.size(3);
        int64_t sH = stride[0], sW = stride.size() > 1 ? stride[1] : sH;
        int64_t pH = padding[0], pW = padding.size() > 1 ? padding[1] : pH;
        int64_t oH = grad_c.size(2), oW = grad_c.size(3);

        // grad_input via shader
        auto grad_input = at::empty_like(input_c);
        {
            struct {
                uint32_t N, C_in, C_out, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, groups;
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

            uint32_t total = static_cast<uint32_t>(N * C_in * iH * iW);
            uint32_t workgroups = (total + 255) / 256;

            dispatch_shader("conv_conv2d_backward_input_fwd",
                            shaders::conv_conv2d_backward_input_fwd,
                            shaders::conv_conv2d_backward_input_fwd_size,
                            {grad_c, weight_c, grad_input},
                            workgroups, 1, 1,
                            &params, sizeof(params));
        }

        // grad_weight via shader
        auto grad_weight = at::empty_like(weight_c);
        {
            struct {
                uint32_t N, C_in, C_out, iH, iW, oH, oW, kH, kW, sH, sW, pH, pW, groups;
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

            uint32_t c_in_per_group = static_cast<uint32_t>(C_in / groups);
            uint32_t total = static_cast<uint32_t>(C_out * c_in_per_group * kH * kW);
            uint32_t workgroups = (total + 255) / 256;

            dispatch_shader("conv_conv2d_backward_weight_fwd",
                            shaders::conv_conv2d_backward_weight_fwd,
                            shaders::conv_conv2d_backward_weight_fwd_size,
                            {grad_c, input_c, grad_weight},
                            workgroups, 1, 1,
                            &params, sizeof(params));
        }

        // grad_bias
        at::Tensor grad_bias;
        if (has_bias) {
            // bias grad = sum(grad_output, dims={0,2,3}) -> [C_out]
            grad_bias = vulkan_sum(grad_c, at::IntArrayRef({0, 2, 3}), false, c10::nullopt);
        }

        return {grad_input, grad_weight, grad_bias,
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_conv2d_autograd(const at::Tensor& input, const at::Tensor& weight,
                                    const std::optional<at::Tensor>& bias_opt,
                                    at::IntArrayRef stride, at::IntArrayRef padding,
                                    at::IntArrayRef dilation, int64_t groups) {
    auto bias = bias_opt.has_value() ? *bias_opt : at::Tensor();
    return VulkanConv2dFunction::apply(input, weight, bias, stride, padding, dilation, groups);
}

// ── Max Pool 2D Backward ───────────────────────────────────────
// Uses max_pool2d_with_indices forward, then scatters grad_output using indices.
class VulkanMaxPool2dFunction : public torch::autograd::Function<VulkanMaxPool2dFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride,
                               at::IntArrayRef padding,
                               at::IntArrayRef dilation,
                               bool ceil_mode) {
        // Run forward with indices
        auto input_c = input.contiguous();

        int64_t kH = kernel_size[0];
        int64_t kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
        int64_t sH = stride.empty() ? kH : stride[0];
        int64_t sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
        int64_t pH = padding.empty() ? 0 : padding[0];
        int64_t pW = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pH);

        int64_t N = input_c.size(0), C = input_c.size(1);
        int64_t iH = input_c.size(2), iW = input_c.size(3);
        int64_t oH = (iH + 2*pH - kH) / sH + 1;
        int64_t oW = (iW + 2*pW - kW) / sW + 1;

        auto output = at::empty({N, C, oH, oW}, input_c.options());
        auto indices = at::empty({N, C, oH, oW}, input_c.options());
        uint32_t total = static_cast<uint32_t>(N * C * oH * oW);

        if (total > 0) {
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
            dispatch_shader("pooling_max_pool2d_indices_fwd",
                            shaders::pooling_max_pool2d_indices_fwd,
                            shaders::pooling_max_pool2d_indices_fwd_size,
                            {input_c, output, indices},
                            workgroups, 1, 1,
                            &params, sizeof(params), 2);
        }

        ctx->save_for_backward({indices});
        ctx->saved_data["input_sizes"] = input_c.sizes().vec();
        ctx->saved_data["kH"] = kH; ctx->saved_data["kW"] = kW;
        ctx->saved_data["sH"] = sH; ctx->saved_data["sW"] = sW;
        ctx->saved_data["pH"] = pH; ctx->saved_data["pW"] = pW;

        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto indices = saved[0];
        auto grad_output = grad_outputs[0];
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();

        // GPU shader for max_pool2d backward (gather-based)
        auto go_c = grad_output.contiguous();
        auto indices_c = indices.contiguous();
        auto grad_input = at::zeros(input_sizes, go_c.options());

        uint32_t output_numel = static_cast<uint32_t>(go_c.numel());
        uint32_t input_numel = static_cast<uint32_t>(grad_input.numel());
        if (output_numel == 0) {
            return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor()};
        }

        int64_t N_val = input_sizes[0], C_val = input_sizes[1];
        int64_t iH_val = input_sizes[2], iW_val = input_sizes[3];
        int64_t oH_val = go_c.size(2), oW_val = go_c.size(3);
        int64_t kH_val = ctx->saved_data["kH"].toInt();
        int64_t kW_val = ctx->saved_data["kW"].toInt();
        int64_t sH_val = ctx->saved_data["sH"].toInt();
        int64_t sW_val = ctx->saved_data["sW"].toInt();
        int64_t pH_val = ctx->saved_data["pH"].toInt();
        int64_t pW_val = ctx->saved_data["pW"].toInt();

        struct {
            uint32_t input_numel; uint32_t output_numel;
            uint32_t NC;
            uint32_t iH; uint32_t iW;
            uint32_t oH; uint32_t oW;
            uint32_t kH; uint32_t kW;
            uint32_t sH; uint32_t sW;
            uint32_t pH; uint32_t pW;
        } params{
            input_numel, output_numel,
            static_cast<uint32_t>(N_val * C_val),
            static_cast<uint32_t>(iH_val), static_cast<uint32_t>(iW_val),
            static_cast<uint32_t>(oH_val), static_cast<uint32_t>(oW_val),
            static_cast<uint32_t>(kH_val), static_cast<uint32_t>(kW_val),
            static_cast<uint32_t>(sH_val), static_cast<uint32_t>(sW_val),
            static_cast<uint32_t>(pH_val), static_cast<uint32_t>(pW_val)
        };
        uint32_t workgroups = (input_numel + 255) / 256;
        dispatch_shader("pooling_max_pool2d_backward_fwd",
                        shaders::pooling_max_pool2d_backward_fwd,
                        shaders::pooling_max_pool2d_backward_fwd_size,
                        {go_c, indices_c, grad_input},
                        workgroups, 1, 1,
                        &params, sizeof(params));

        return {grad_input, at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_max_pool2d_autograd(const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    return VulkanMaxPool2dFunction::apply(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// ── Avg Pool 2D Backward ───────────────────────────────────────
// grad_input = distribute grad_output evenly over pooling windows
// Falls back to CPU for simplicity (avg_pool backward is straightforward but verbose in GPU)
class VulkanAvgPool2dFunction : public torch::autograd::Function<VulkanAvgPool2dFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input,
                               at::IntArrayRef kernel_size, at::IntArrayRef stride,
                               at::IntArrayRef padding, bool ceil_mode,
                               bool count_include_pad,
                               std::optional<int64_t> divisor_override) {
        ctx->save_for_backward({input});
        ctx->saved_data["kernel_size"] = kernel_size.vec();
        ctx->saved_data["stride"] = stride.vec();
        ctx->saved_data["padding"] = padding.vec();
        ctx->saved_data["ceil_mode"] = ceil_mode;
        ctx->saved_data["count_include_pad"] = count_include_pad;
        return vulkan_avg_pool2d(input, kernel_size, stride, padding,
                                  ceil_mode, count_include_pad, divisor_override);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];

        auto kernel_size = ctx->saved_data["kernel_size"].toIntVector();
        auto stride = ctx->saved_data["stride"].toIntVector();
        auto padding = ctx->saved_data["padding"].toIntVector();
        bool ceil_mode = ctx->saved_data["ceil_mode"].toBool();
        bool count_include_pad = ctx->saved_data["count_include_pad"].toBool();

        // GPU shader for avg_pool2d backward
        auto grad_input = vulkan_avg_pool2d_backward(
            grad_output, input, kernel_size, stride, padding,
            ceil_mode, count_include_pad, c10::nullopt);
        return {grad_input,
                at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_avg_pool2d_autograd(const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode,
    bool count_include_pad, std::optional<int64_t> divisor_override) {
    return VulkanAvgPool2dFunction::apply(self, kernel_size, stride, padding,
                                           ceil_mode, count_include_pad, divisor_override);
}

// ── Adaptive Avg Pool 2D Backward ──────────────────────────────
class VulkanAdaptiveAvgPool2dFunction : public torch::autograd::Function<VulkanAdaptiveAvgPool2dFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input,
                               at::IntArrayRef output_size) {
        ctx->save_for_backward({input});
        ctx->saved_data["output_size"] = output_size.vec();
        return vulkan_adaptive_avg_pool2d(input, output_size);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];

        auto grad_input = vulkan_adaptive_avg_pool2d_backward(grad_output, input);
        return {grad_input, at::Tensor()};
    }
};

at::Tensor vulkan_adaptive_avg_pool2d_autograd(const at::Tensor& self, at::IntArrayRef output_size) {
    return VulkanAdaptiveAvgPool2dFunction::apply(self, output_size);
}

// ── Batch Norm Backward ────────────────────────────────────────
// Training mode batch norm with backward pass.
// For training, computes mean/var on-the-fly, then does backward via CPU.
class VulkanBatchNormFunction : public torch::autograd::Function<VulkanBatchNormFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input,
                               const at::Tensor& weight,
                               const at::Tensor& bias,
                               const at::Tensor& running_mean,
                               const at::Tensor& running_var,
                               bool training, double momentum, double eps) {
        if (!training) {
            // Inference mode — use existing shader, no autograd needed
            ctx->save_for_backward({});
            ctx->saved_data["training"] = false;
            return vulkan_batch_norm(input,
                weight.defined() ? std::optional<at::Tensor>(weight) : std::nullopt,
                bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt,
                running_mean.defined() ? std::optional<at::Tensor>(running_mean) : std::nullopt,
                running_var.defined() ? std::optional<at::Tensor>(running_var) : std::nullopt,
                false, momentum, eps, false);
        }

        // Training mode: compute batch statistics on GPU using existing ops
        auto input_c = input.contiguous();
        int64_t C = input_c.size(1);
        int64_t N_batch = input_c.size(0);
        int64_t spatial = 1;
        for (int64_t i = 2; i < input_c.dim(); i++) spatial *= input_c.size(i);
        int64_t M = N_batch * spatial;

        // Compute per-channel mean and var
        // Reshape to [N, C, spatial] -> permute to [C, N*spatial]
        auto in_3d = input_c.reshape({N_batch, C, spatial});
        auto in_perm = in_3d.permute({1, 0, 2}).contiguous().reshape({C, M});
        auto batch_mean = vulkan_mean(in_perm, at::IntArrayRef({1}), false, c10::nullopt);
        auto in_centered = vulkan_sub(in_perm, batch_mean.unsqueeze(1).expand({C, M}).contiguous(), 1);
        auto batch_var = vulkan_mean(vulkan_mul(in_centered, in_centered), at::IntArrayRef({1}), false, c10::nullopt);

        // Normalize: x_hat = (x - mean) / sqrt(var + eps)
        auto invstd = at::rsqrt(vulkan_add_scalar(batch_var, at::Scalar(static_cast<float>(eps)), 1));
        auto mean_ex = batch_mean.reshape({1, C, 1}).expand({N_batch, C, spatial}).contiguous();
        auto invstd_ex = invstd.reshape({1, C, 1}).expand({N_batch, C, spatial}).contiguous();
        auto x_hat = vulkan_mul(vulkan_sub(in_3d, mean_ex, 1), invstd_ex);

        // Apply affine: y = weight * x_hat + bias
        at::Tensor result = x_hat;
        if (weight.defined()) {
            auto w_ex = weight.reshape({1, C, 1}).expand({N_batch, C, spatial}).contiguous();
            result = vulkan_mul(result, w_ex);
        }
        if (bias.defined()) {
            auto b_ex = bias.reshape({1, C, 1}).expand({N_batch, C, spatial}).contiguous();
            result = vulkan_add(result, b_ex, 1);
        }

        // Update running stats on GPU
        if (running_mean.defined()) {
            // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            float mom_f = static_cast<float>(momentum);
            auto rm_scaled = vulkan_mul_scalar(running_mean, at::Scalar(1.0f - mom_f));
            auto bm_scaled = vulkan_mul_scalar(batch_mean, at::Scalar(mom_f));
            auto new_rm = vulkan_add(rm_scaled, bm_scaled, 1);
            dispatch_copy_buffer(new_rm, running_mean);

            // Unbiased var for running_var: batch_var * M / (M - 1)
            auto unbiased_var = vulkan_mul_scalar(batch_var, at::Scalar(static_cast<float>(M) / (M - 1)));
            auto rv_scaled = vulkan_mul_scalar(running_var, at::Scalar(1.0f - mom_f));
            auto bv_scaled = vulkan_mul_scalar(unbiased_var, at::Scalar(mom_f));
            auto new_rv = vulkan_add(rv_scaled, bv_scaled, 1);
            dispatch_copy_buffer(new_rv, running_var);
        }

        ctx->save_for_backward({input, weight});
        ctx->saved_data["training"] = true;
        ctx->saved_data["eps"] = eps;

        return result.reshape(input.sizes());
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        if (!ctx->saved_data["training"].toBool()) {
            return {at::Tensor(), at::Tensor(), at::Tensor(),
                    at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
        }

        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto grad_output = grad_outputs[0];
        double eps = ctx->saved_data["eps"].toDouble();

        // Use GPU batch_norm_backward
        auto [grad_input, grad_weight, grad_bias] =
            vulkan_native_batch_norm_backward(grad_output, input,
                weight.defined() ? std::optional<at::Tensor>(weight) : std::nullopt,
                std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                true, eps, {true, weight.defined(), false});

        return {grad_input, grad_weight, grad_bias,
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_batch_norm_autograd(
    const at::Tensor& input, const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::Tensor>& running_mean_opt,
    const std::optional<at::Tensor>& running_var_opt,
    bool training, double momentum, double eps, bool cudnn_enabled) {
    auto weight = weight_opt.has_value() ? *weight_opt : at::Tensor();
    auto bias = bias_opt.has_value() ? *bias_opt : at::Tensor();
    auto rm = running_mean_opt.has_value() ? *running_mean_opt : at::Tensor();
    auto rv = running_var_opt.has_value() ? *running_var_opt : at::Tensor();
    return VulkanBatchNormFunction::apply(input, weight, bias, rm, rv,
                                           training, momentum, eps);
}

// ── Embedding Backward ─────────────────────────────────────────
// Scatters grad_output into grad_weight using indices.
class VulkanEmbeddingFunction : public torch::autograd::Function<VulkanEmbeddingFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& weight, const at::Tensor& indices,
                               int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
        ctx->save_for_backward({indices});
        ctx->saved_data["num_embeddings"] = weight.size(0);
        ctx->saved_data["embedding_dim"] = weight.size(1);
        ctx->saved_data["padding_idx"] = padding_idx;
        return vulkan_embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto indices = saved[0];
        auto grad_output = grad_outputs[0];
        int64_t num_embeddings = ctx->saved_data["num_embeddings"].toInt();
        int64_t padding_idx = ctx->saved_data["padding_idx"].toInt();

        // Use GPU embedding_dense_backward
        auto grad_weight = vulkan_embedding_dense_backward(
            grad_output, indices, num_embeddings, padding_idx, false);
        return {grad_weight,
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_embedding_autograd(const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    return VulkanEmbeddingFunction::apply(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

// ── Layer Norm Backward ────────────────────────────────────────
// CPU fallback for backward pass
class VulkanLayerNormFunction : public torch::autograd::Function<VulkanLayerNormFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, at::IntArrayRef normalized_shape,
                               const at::Tensor& weight, const at::Tensor& bias) {
        ctx->saved_data["normalized_shape"] = normalized_shape.vec();
        auto [output, mean, rstd] = vulkan_layer_norm(input, normalized_shape,
            weight.defined() ? std::optional<at::Tensor>(weight) : std::nullopt,
            bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt,
            1e-5);
        // Save mean/rstd from forward to avoid recompute in backward (saves 1 dispatch)
        ctx->save_for_backward({input, weight, bias, mean, rstd});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];
        auto mean = saved[3];
        auto rstd = saved[4];
        auto grad_output = grad_outputs[0];
        auto ns = ctx->saved_data["normalized_shape"].toIntVector();

        // Use saved mean/rstd — no forward recompute needed
        std::vector<c10::SymInt> ns_sym(ns.begin(), ns.end());
        auto [grad_input, grad_weight, grad_bias] =
            vulkan_native_layer_norm_backward(grad_output, input, ns_sym, mean, rstd,
                weight.defined() ? std::optional<at::Tensor>(weight) : std::nullopt,
                bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt,
                {true, weight.defined(), bias.defined()});

        return {grad_input, at::Tensor(), grad_weight, grad_bias};
    }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_layer_norm_autograd(
    const at::Tensor& input, at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    double eps) {
    auto weight = weight_opt.has_value() ? *weight_opt : at::Tensor();
    auto bias = bias_opt.has_value() ? *bias_opt : at::Tensor();
    auto output = VulkanLayerNormFunction::apply(input, normalized_shape, weight, bias);
    return std::make_tuple(output, at::empty({0}, input.options()), at::empty({0}, input.options()));
}

// ── Group Norm Backward ────────────────────────────────────────
class VulkanGroupNormFunction : public torch::autograd::Function<VulkanGroupNormFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, int64_t num_groups,
                               const at::Tensor& weight, const at::Tensor& bias) {
        ctx->save_for_backward({input, weight, bias});
        ctx->saved_data["num_groups"] = num_groups;
        auto result = vulkan_group_norm(input, num_groups,
            weight.defined() ? std::optional<at::Tensor>(weight) : std::nullopt,
            bias.defined() ? std::optional<at::Tensor>(bias) : std::nullopt,
            1e-5);
        return std::get<0>(result);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];
        auto grad_output = grad_outputs[0];
        // Group norm is not implemented on GPU
        TORCH_CHECK(false, "Vulkan: group_norm backward is not implemented on GPU. ",
                    "group_norm forward is also not supported (SwiftShader limitation).");
    }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_group_norm_autograd(
    const at::Tensor& input, int64_t num_groups,
    const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    double eps) {
    auto weight = weight_opt.has_value() ? *weight_opt : at::Tensor();
    auto bias = bias_opt.has_value() ? *bias_opt : at::Tensor();
    auto output = VulkanGroupNormFunction::apply(input, num_groups, weight, bias);
    return std::make_tuple(output, at::empty({0}, input.options()), at::empty({0}, input.options()));
}

// ── Scaled Dot-Product Attention Backward ───────────────────────
// CPU fallback for backward pass of SDPA
class VulkanSDPAFunction : public torch::autograd::Function<VulkanSDPAFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& query, const at::Tensor& key,
                               const at::Tensor& value,
                               const at::Tensor& attn_mask,
                               double dropout_p, bool is_causal,
                               double scale) {
        ctx->saved_data["dropout_p"] = dropout_p;
        ctx->saved_data["is_causal"] = is_causal;
        ctx->saved_data["scale"] = scale;
        bool has_real_mask = attn_mask.defined() && attn_mask.numel() > 0;
        ctx->saved_data["has_mask"] = has_real_mask;

        // Save attn_weights to avoid forward recompute in backward (saves 3 dispatches)
        auto [output, attn_weights] = vulkan_scaled_dot_product_attention_with_attn(
            query, key, value,
            has_real_mask ? std::optional<at::Tensor>(attn_mask) : std::nullopt,
            dropout_p, is_causal,
            scale != 0.0 ? std::optional<double>(scale) : std::nullopt);

        ctx->save_for_backward({query, key, value, attn_weights});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto query = saved[0];
        auto key = saved[1];
        auto value = saved[2];
        auto attn = saved[3];  // saved attn_weights [B*H, N, S]
        auto grad_output = grad_outputs[0];

        // SDPA backward using GPU ops — attn_weights already computed in forward
        int64_t D = query.size(-1);
        float scale_val = 1.0f / std::sqrt(static_cast<float>(D));
        int64_t BH = query.size(0) * query.size(1);
        int64_t N = query.size(2);
        int64_t S = key.size(2);

        auto k_3d = key.reshape({BH, S, D}).contiguous();
        auto q_3d = query.reshape({BH, N, D}).contiguous();
        auto v_3d = value.reshape({BH, S, D}).contiguous();
        auto go_3d = grad_output.reshape({BH, N, D}).contiguous();

        // attn is already [B*H, N, S] from save_for_backward — no recompute needed!

        // grad_v = attn^T @ grad_output -> [BH, S, D]
        auto grad_v = vulkan_bmm_ex(attn, go_3d, true, false);
        // grad_attn = grad_output @ V^T -> [BH, N, S]
        auto grad_attn = vulkan_bmm_ex(go_3d, v_3d, false, true);

        // softmax backward: grad_scores = attn * (grad_attn - sum(grad_attn * attn, dim=-1)) * scale
        // fused: 1 dispatch via vulkan_softmax_backward_data
        auto grad_scores_unscaled = vulkan_softmax_backward_data(grad_attn, attn, -1, at::kFloat);
        auto grad_scores_final = vulkan_mul_scalar(grad_scores_unscaled, at::Scalar(scale_val));

        // grad_q = grad_scores @ K -> [BH, N, D]
        auto grad_q = vulkan_bmm(grad_scores_final, k_3d);
        // grad_k = grad_scores^T @ Q -> [BH, S, D]
        auto grad_k = vulkan_bmm_ex(grad_scores_final, q_3d, true, false);

        return {grad_q.reshape(query.sizes()),
                grad_k.reshape(key.sizes()),
                grad_v.reshape(value.sizes()),
                at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_sdpa_autograd(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale) {
    // VulkanSDPAFunction::apply requires all tensor args to be defined (have a
    // device) or PyTorch autograd machinery raises "tensor does not have a
    // device". When no mask is provided, use a zero-element sentinel tensor on
    // the same device; VulkanSDPAFunction checks defined() via numel()==0.
    at::Tensor mask;
    if (attn_mask.has_value()) {
        mask = *attn_mask;
    } else {
        mask = at::empty({0}, query.options());
    }
    double scale_val = scale.has_value() ? *scale : 0.0;
    return VulkanSDPAFunction::apply(query, key, value, mask,
                                      dropout_p, is_causal, scale_val);
}

// ── PReLU Backward ─────────────────────────────────────────────
// prelu(x, w) = max(0, x) + w * min(0, x)
// grad_input = grad_output * (x >= 0 ? 1 : w)
// grad_weight = sum(grad_output * min(0, x))
class VulkanPReluFunction : public torch::autograd::Function<VulkanPReluFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, const at::Tensor& weight) {
        ctx->save_for_backward({input, weight});
        return vulkan_prelu(input, weight);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto grad_output = grad_outputs[0];

        // Compute prelu gradients using GPU ops
        // prelu(x, w) = x if x >= 0, else w*x
        // d/dx = 1 if x >= 0, else w
        // d/dw = 0 if x >= 0, else x
        auto mask_pos = vulkan_ge_scalar(input, at::Scalar(0.0f)).to(at::kFloat);
        auto mask_neg = vulkan_rsub_scalar(mask_pos, at::Scalar(1.0f));

        // Broadcast weight
        at::Tensor w_expanded;
        if (input.dim() > 1 && weight.dim() == 1) {
            std::vector<int64_t> shape(input.dim(), 1);
            shape[1] = weight.size(0);
            w_expanded = weight.reshape(shape).expand_as(input).contiguous();
        } else {
            w_expanded = weight.expand_as(input).contiguous();
        }

        // grad_input = grad * (mask_pos + mask_neg * w)
        auto grad_input = vulkan_mul(grad_output,
            vulkan_add(mask_pos, vulkan_mul(mask_neg, w_expanded), 1));

        // grad_weight = sum(grad * mask_neg * input, appropriate_dims)
        auto grad_weight_full = vulkan_mul(vulkan_mul(grad_output, mask_neg), input);
        at::Tensor grad_weight;
        if (input.dim() > 1 && weight.dim() == 1) {
            std::vector<int64_t> sum_dims;
            for (int64_t d = 0; d < input.dim(); d++) {
                if (d != 1) sum_dims.push_back(d);
            }
            grad_weight = vulkan_sum(grad_weight_full, at::IntArrayRef(sum_dims), false, c10::nullopt);
        } else {
            grad_weight = grad_weight_full;
            while (grad_weight.dim() > weight.dim()) {
                grad_weight = vulkan_sum(grad_weight, at::IntArrayRef({0}), false, c10::nullopt);
            }
        }

        return {grad_input, grad_weight};
    }
};

at::Tensor vulkan_prelu_autograd(const at::Tensor& self, const at::Tensor& weight) {
    return VulkanPReluFunction::apply(self, weight);
}

// ── SELU Backward ──────────────────────────────────────────────
// selu(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
// selu'(x) = scale * (x > 0 ? 1 : alpha * exp(x))
class VulkanSeluFunction : public torch::autograd::Function<VulkanSeluFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input) {
        ctx->save_for_backward({input});
        return vulkan_selu(input);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];

        constexpr float alpha = 1.6732632423543772f;
        constexpr float scale = 1.0507009873554804f;

        auto mask = vulkan_gt_scalar(input, at::Scalar(0.0f));
        auto mask_float = mask.to(at::kFloat);
        // For x > 0: derivative = scale
        // For x <= 0: derivative = scale * alpha * exp(x)
        auto exp_x = vulkan_exp(input);
        auto alpha_exp = vulkan_mul_scalar(exp_x, at::Scalar(alpha));
        auto neg_deriv = vulkan_mul_scalar(alpha_exp, at::Scalar(scale));
        auto pos_deriv = vulkan_mul_scalar(mask_float, at::Scalar(0.0f));
        pos_deriv = vulkan_add_scalar(pos_deriv, at::Scalar(scale), at::Scalar(1));  // full_like(mask, scale)
        // grad_factor = mask * scale + (1 - mask) * scale * alpha * exp(x)
        auto one_minus_mask = vulkan_rsub_scalar(mask_float, at::Scalar(1.0f));
        auto grad_factor = vulkan_add(
            vulkan_mul(mask_float, pos_deriv),
            vulkan_mul(one_minus_mask, neg_deriv), at::Scalar(1));
        auto grad_input = vulkan_mul(grad_output, grad_factor);

        return {grad_input};
    }
};

at::Tensor vulkan_selu_autograd(const at::Tensor& self) {
    return VulkanSeluFunction::apply(self);
}

// ── Clamp Backward ─────────────────────────────────────────────
// grad_input = grad_output * (min <= input <= max)
class VulkanClampFunction : public torch::autograd::Function<VulkanClampFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input,
                               const std::optional<at::Scalar>& min_val,
                               const std::optional<at::Scalar>& max_val) {
        ctx->save_for_backward({input});
        ctx->saved_data["has_min"] = min_val.has_value();
        ctx->saved_data["has_max"] = max_val.has_value();
        if (min_val.has_value()) ctx->saved_data["min_val"] = *min_val;
        if (max_val.has_value()) ctx->saved_data["max_val"] = *max_val;
        return vulkan_clamp(input, min_val, max_val);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];

        // mask: 1 where input is within [min, max], 0 otherwise
        // Build mask without at::ones_like to avoid extra dispatch
        bool has_min = ctx->saved_data["has_min"].toBool();
        bool has_max = ctx->saved_data["has_max"].toBool();
        at::Tensor mask;

        if (has_min && has_max) {
            auto min_val = ctx->saved_data["min_val"].toScalar();
            auto max_val = ctx->saved_data["max_val"].toScalar();
            auto min_mask = vulkan_ge_scalar(input, min_val).to(at::kFloat);
            auto max_mask = vulkan_le_scalar(input, max_val).to(at::kFloat);
            mask = vulkan_mul(min_mask, max_mask);
        } else if (has_min) {
            auto min_val = ctx->saved_data["min_val"].toScalar();
            mask = vulkan_ge_scalar(input, min_val).to(at::kFloat);
        } else if (has_max) {
            auto max_val = ctx->saved_data["max_val"].toScalar();
            mask = vulkan_le_scalar(input, max_val).to(at::kFloat);
        } else {
            // No bounds — grad passes through unchanged
            return {grad_output, at::Tensor(), at::Tensor()};
        }

        return {vulkan_mul(grad_output, mask), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_clamp_autograd(const at::Tensor& self,
                                   const std::optional<at::Scalar>& min_val,
                                   const std::optional<at::Scalar>& max_val) {
    return VulkanClampFunction::apply(self, min_val, max_val);
}

// ── RoPE Backward ───────────────────────────────────────────────
// RoPE applies a rotation: out[2i] = x[2i]*cos - x[2i+1]*sin
//                           out[2i+1] = x[2i]*sin + x[2i+1]*cos
// Backward is the transpose (inverse) rotation:
//   dx[2i] = grad[2i]*cos + grad[2i+1]*sin
//   dx[2i+1] = -grad[2i]*sin + grad[2i+1]*cos
class VulkanRoPEFunction : public torch::autograd::Function<VulkanRoPEFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, double theta) {
        ctx->saved_data["theta"] = theta;
        ctx->save_for_backward({input});
        return vulkan_rope(input, theta);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto grad_output = grad_outputs[0];
        double theta = ctx->saved_data["theta"].toDouble();

        // GPU shader for RoPE backward (inverse rotation)
        auto go_c = grad_output.contiguous();
        int64_t B = go_c.size(0), H = go_c.size(1);
        int64_t N = go_c.size(2), D = go_c.size(3);

        auto grad_input = at::empty_like(go_c);
        uint32_t total = static_cast<uint32_t>(B * H * N * D);
        if (total == 0) return {grad_input, at::Tensor()};

        struct { uint32_t B, H, N, D; float theta; } params{
            static_cast<uint32_t>(B), static_cast<uint32_t>(H),
            static_cast<uint32_t>(N), static_cast<uint32_t>(D),
            static_cast<float>(theta)
        };
        uint32_t workgroups = (total + 255) / 256;
        dispatch_shader("attention_rope_backward_fwd",
                        shaders::attention_rope_backward_fwd,
                        shaders::attention_rope_backward_fwd_size,
                        {go_c, grad_input},
                        workgroups, 1, 1,
                        &params, sizeof(params));

        return {grad_input, at::Tensor()};
    }
};

at::Tensor vulkan_rope_autograd(const at::Tensor& input, double theta) {
    return VulkanRoPEFunction::apply(input, theta);
}

// ── RMS Norm Autograd ──────────────────────────────────────────
class VulkanRMSNormFunction : public torch::autograd::Function<VulkanRMSNormFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, const at::Tensor& weight,
                               double eps) {
        auto [output, rstd] = vulkan_rms_norm(input, weight, eps);
        ctx->save_for_backward({input, weight, rstd});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto rstd = saved[2];
        auto grad_output = grad_outputs[0];

        auto [grad_input, grad_weight] = vulkan_rms_norm_backward(
            grad_output, input, weight, rstd);
        return {grad_input, grad_weight, at::Tensor()};
    }
};

at::Tensor vulkan_rms_norm_autograd(const at::Tensor& input, const at::Tensor& weight, double eps) {
    return VulkanRMSNormFunction::apply(input, weight, eps);
}

// ── Fused Add + RMSNorm Autograd ────────────────────────────────
// Returns (normed_output, h_new) where h_new = residual + shortcut.
// Backward: grad propagates through both outputs.
class VulkanAddRMSNormFunction : public torch::autograd::Function<VulkanAddRMSNormFunction> {
public:
    static std::vector<at::Tensor> forward(torch::autograd::AutogradContext* ctx,
                                            const at::Tensor& residual,
                                            const at::Tensor& shortcut,
                                            const at::Tensor& weight,
                                            double eps) {
        auto [output, h_new, rstd] = vulkan_add_rms_norm(residual, shortcut, weight, eps);
        ctx->save_for_backward({h_new, weight, rstd});
        return {output, h_new};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto h_new = saved[0];
        auto weight = saved[1];
        auto rstd = saved[2];

        auto grad_normed = grad_outputs[0];  // grad through normed output
        auto grad_h_new = grad_outputs[1];   // grad through h_new output (next layer residual)

        at::Tensor grad_h_new_total;
        at::Tensor grad_weight;

        if (grad_normed.defined() && grad_h_new.defined()) {
            // Fused: rms_norm_backward + add_grad in 2 dispatches (was 3)
            auto [g_h, g_w] = vulkan_add_rms_norm_backward(grad_normed, grad_h_new, h_new, weight, rstd);
            grad_h_new_total = g_h;
            grad_weight = g_w;
        } else if (grad_normed.defined()) {
            // Only normed output used — regular rms_norm backward
            auto [g_h, g_w] = vulkan_rms_norm_backward(grad_normed, h_new, weight, rstd);
            grad_h_new_total = g_h;
            grad_weight = g_w;
        } else if (grad_h_new.defined()) {
            // Only h_new output used — grad passes straight through
            grad_h_new_total = grad_h_new;
        }

        // Both residual and shortcut get the same gradient (add is symmetric)
        return {grad_h_new_total, grad_h_new_total, grad_weight, at::Tensor()};
    }
};

std::tuple<at::Tensor, at::Tensor> vulkan_add_rms_norm_apply(
    const at::Tensor& residual, const at::Tensor& shortcut,
    const at::Tensor& weight, double eps) {
    auto results = VulkanAddRMSNormFunction::apply(residual, shortcut, weight, eps);
    return {results[0], results[1]};
}

// ── RMSNormGated Autograd ──────────────────────────────────────
// out = weight * rms_norm(input) * silu(gate) — Qwen3.5 GatedDeltaNet
class VulkanRMSNormGatedFunction : public torch::autograd::Function<VulkanRMSNormGatedFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& input, const at::Tensor& gate,
                               const at::Tensor& weight, double eps) {
        auto [output, rstd] = vulkan_rms_norm_gated(input, gate, weight, eps);
        ctx->save_for_backward({input, gate, weight, rstd});
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto gate = saved[1];
        auto weight = saved[2];
        auto rstd = saved[3];
        auto grad_output = grad_outputs[0];

        auto [grad_input, grad_gate] = vulkan_rms_norm_gated_backward(
            grad_output, input, gate, weight, rstd);
        // grad_weight: reuse rms_norm weight backward (silu(gate) acts like a scale, absorbed in grad)
        // For simplicity, accumulate grad_weight separately via element-wise ops
        // grad_weight[d] = sum_n(grad_output[n,d] * rms_norm(input)[n,d] * silu(gate[n,d]))
        // This is done on CPU for simplicity; can be fused later
        auto norm_size = weight.numel();
        auto num_rows = input.numel() / norm_size;
        auto input_flat = ensure_float32(input.contiguous()).reshape({num_rows, norm_size});
        auto gate_flat = ensure_float32(gate.contiguous()).reshape({num_rows, norm_size});
        auto go_flat = ensure_float32(grad_output.contiguous()).reshape({num_rows, norm_size});
        auto rstd_view = rstd.contiguous().unsqueeze(-1).expand({num_rows, norm_size});
        auto silu_gate = gate_flat / (1.0f + (-gate_flat).exp());
        auto x_normed = input_flat * rstd_view;  // approx; proper is input * rstd
        auto grad_weight = (go_flat * x_normed * silu_gate).sum(0);
        auto orig_dtype = grad_output.scalar_type();
        grad_weight = cast_from_float32(grad_weight, orig_dtype);
        return {grad_input, grad_gate, grad_weight, at::Tensor()};
    }
};

at::Tensor vulkan_rms_norm_gated_autograd(
    const at::Tensor& input, const at::Tensor& gate,
    const at::Tensor& weight, double eps) {
    return VulkanRMSNormGatedFunction::apply(input, gate, weight, eps);
}

// ── View Autograd ──────────────────────────────────────────────
// Our view/reshape implementations copy data (opaque allocator can't share storage),
// so we need autograd wrappers. Backward = reshape grad back to input shape.
class VulkanViewFunction : public torch::autograd::Function<VulkanViewFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, at::IntArrayRef size) {
        ctx->saved_data["input_sizes"] = self.sizes().vec();
        return vulkan_view(self, size);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();
        auto grad = grad_outputs[0];
        // grad may be non-contiguous (e.g., if a transpose was applied to the view output).
        // Use reshape (not view) to handle this case safely.
        return {vulkan_reshape(grad, input_sizes), at::Tensor()};
    }
};

at::Tensor vulkan_view_autograd(const at::Tensor& self, at::IntArrayRef size) {
    return VulkanViewFunction::apply(self, size);
}

class VulkanReshapeFunction : public torch::autograd::Function<VulkanReshapeFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, at::IntArrayRef shape) {
        ctx->saved_data["input_sizes"] = self.sizes().vec();
        return vulkan_reshape(self, shape);
    }
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();
        auto grad = grad_outputs[0];
        return {vulkan_reshape(grad, input_sizes), at::Tensor()};
    }
};

at::Tensor vulkan_reshape_autograd(const at::Tensor& self, at::IntArrayRef shape) {
    return VulkanReshapeFunction::apply(self, shape);
}

// ── Transpose/Permute Autograd ─────────────────────────────────
// Our permute creates a new tensor with permuted data (copy), so needs autograd.
// Backward: apply inverse permutation to grad.
class VulkanPermuteFunction : public torch::autograd::Function<VulkanPermuteFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, at::IntArrayRef dims) {
        // Store inverse permutation for backward
        std::vector<int64_t> inv(dims.size());
        for (int64_t i = 0; i < static_cast<int64_t>(dims.size()); i++) {
            inv[dims[i]] = i;
        }
        ctx->saved_data["inv_perm"] = inv;
        return vulkan_permute(self, dims);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto inv = ctx->saved_data["inv_perm"].toIntVector();
        return {vulkan_permute(grad_outputs[0], inv), at::Tensor()};
    }
};

at::Tensor vulkan_permute_autograd(const at::Tensor& self, at::IntArrayRef dims) {
    return VulkanPermuteFunction::apply(self, dims);
}

class VulkanTransposeFunction : public torch::autograd::Function<VulkanTransposeFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, int64_t dim0, int64_t dim1) {
        ctx->saved_data["dim0"] = dim0;
        ctx->saved_data["dim1"] = dim1;
        return vulkan_transpose(self, dim0, dim1);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto dim0 = ctx->saved_data["dim0"].toInt();
        auto dim1 = ctx->saved_data["dim1"].toInt();
        return {vulkan_transpose(grad_outputs[0], dim0, dim1), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_transpose_autograd(const at::Tensor& self, int64_t dim0, int64_t dim1) {
    return VulkanTransposeFunction::apply(self, dim0, dim1);
}

class VulkanTFunction : public torch::autograd::Function<VulkanTFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self) {
        // Zero-copy forward: swap sizes/strides metadata only
        return vulkan_t(self);
    }
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        // Backward is also a transpose (t.t() == self)
        return {vulkan_t(grad_outputs[0])};
    }
};

at::Tensor vulkan_t_autograd(const at::Tensor& self) {
    if (self.dim() < 2) return self;
    return VulkanTFunction::apply(self);
}

class VulkanUnsqueezeFunction : public torch::autograd::Function<VulkanUnsqueezeFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, int64_t dim) {
        ctx->saved_data["input_sizes"] = self.sizes().vec();
        return vulkan_unsqueeze(self, dim);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();
        return {vulkan_view(grad_outputs[0], input_sizes), at::Tensor()};
    }
};

at::Tensor vulkan_unsqueeze_autograd(const at::Tensor& self, int64_t dim) {
    return VulkanUnsqueezeFunction::apply(self, dim);
}

class VulkanSqueezeFunction : public torch::autograd::Function<VulkanSqueezeFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self) {
        ctx->saved_data["input_sizes"] = self.sizes().vec();
        return vulkan_squeeze(self);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();
        return {vulkan_view(grad_outputs[0], input_sizes)};
    }
};

at::Tensor vulkan_squeeze_autograd(const at::Tensor& self) {
    return VulkanSqueezeFunction::apply(self);
}

class VulkanSqueezeDimFunction : public torch::autograd::Function<VulkanSqueezeDimFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, int64_t dim) {
        ctx->saved_data["input_sizes"] = self.sizes().vec();
        return vulkan_squeeze_dim(self, dim);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();
        return {vulkan_view(grad_outputs[0], input_sizes), at::Tensor()};
    }
};

at::Tensor vulkan_squeeze_dim_autograd(const at::Tensor& self, int64_t dim) {
    return VulkanSqueezeDimFunction::apply(self, dim);
}

class VulkanExpandFunction : public torch::autograd::Function<VulkanExpandFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, at::IntArrayRef size, bool implicit) {
        ctx->saved_data["input_sizes"] = self.sizes().vec();
        return vulkan_expand(self, size, implicit);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();
        auto grad = grad_outputs[0];
        // Sum over expanded dimensions to collapse back to input shape
        // Find which dims were expanded (size 1 -> N)
        auto grad_sizes = grad.sizes();
        int64_t ndim_diff = static_cast<int64_t>(grad_sizes.size()) - static_cast<int64_t>(input_sizes.size());
        // Sum over leading broadcast dims
        for (int64_t i = 0; i < ndim_diff; i++) {
            grad = grad.sum(0, /*keepdim=*/false);
        }
        // Sum over dims that were 1 in input
        for (int64_t i = static_cast<int64_t>(input_sizes.size()) - 1; i >= 0; i--) {
            if (input_sizes[i] == 1 && grad.size(i) != 1) {
                grad = grad.sum(i, /*keepdim=*/true);
            }
        }
        return {grad, at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_expand_autograd(const at::Tensor& self, at::IntArrayRef size, bool implicit) {
    return VulkanExpandFunction::apply(self, size, implicit);
}

class VulkanSelectFunction : public torch::autograd::Function<VulkanSelectFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, int64_t dim, int64_t index) {
        ctx->saved_data["input_sizes"] = self.sizes().vec();
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["index"] = index;
        return vulkan_select(self, dim, index);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();
        auto dim = ctx->saved_data["dim"].toInt();
        auto index = ctx->saved_data["index"].toInt();

        // Our opaque allocator makes select return a copy, not a view.
        // Build grad on CPU where select returns a view, then move to Vulkan.
        auto grad_cpu = at::zeros(input_sizes, grad_outputs[0].options().device(at::kCPU));
        auto go_cpu = grad_outputs[0].cpu();
        grad_cpu.select(dim, index).copy_(go_cpu);
        auto grad = grad_cpu.to(grad_outputs[0].device());

        return {grad, at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_select_autograd(const at::Tensor& self, int64_t dim, int64_t index) {
    return VulkanSelectFunction::apply(self, dim, index);
}

class VulkanSliceFunction : public torch::autograd::Function<VulkanSliceFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& self, int64_t dim,
                               std::optional<int64_t> start, std::optional<int64_t> end,
                               int64_t step) {
        ctx->saved_data["input_sizes"] = self.sizes().vec();
        ctx->saved_data["dim"] = dim;
        ctx->saved_data["start"] = start.value_or(0);
        ctx->saved_data["end"] = end.value_or(self.size(dim));
        ctx->saved_data["step"] = step;
        return vulkan_slice(self, dim, start, end, step);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto input_sizes = ctx->saved_data["input_sizes"].toIntVector();
        auto dim = ctx->saved_data["dim"].toInt();
        auto start = ctx->saved_data["start"].toInt();
        auto end = ctx->saved_data["end"].toInt();
        auto step = ctx->saved_data["step"].toInt();

        // Our opaque allocator makes slice return a copy, not a view.
        // So we can't use grad.slice(...).copy_(...) — it writes to a detached copy.
        // Instead: build grad on CPU, then move to Vulkan.
        auto grad_cpu = at::zeros(input_sizes, grad_outputs[0].options().device(at::kCPU));
        auto go_cpu = grad_outputs[0].cpu();
        grad_cpu.slice(dim, start, end, step).copy_(go_cpu);
        auto grad = grad_cpu.to(grad_outputs[0].device());

        return {grad, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_slice_autograd(const at::Tensor& self, int64_t dim,
                                  std::optional<int64_t> start, std::optional<int64_t> end,
                                  int64_t step) {
    return VulkanSliceFunction::apply(self, dim, start, end, step);
}

// ── Fused SwiGLU autograd ────────────────────────────────────────
class VulkanSwiGLUFunction : public torch::autograd::Function<VulkanSwiGLUFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& gate, const at::Tensor& up) {
        ctx->save_for_backward({gate, up});
        return vulkan_swiglu(gate, up);
    }
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto gate = saved[0];
        auto up = saved[1];
        auto [grad_gate, grad_up] = vulkan_swiglu_backward(grad_outputs[0], gate, up);
        return {grad_gate, grad_up};
    }
};

at::Tensor vulkan_swiglu_autograd(const at::Tensor& gate, const at::Tensor& up) {
    return VulkanSwiGLUFunction::apply(gate, up);
}

// ── Scaled BMM autograd: scale * (q @ k.T) ──────────────────────
// Fuses attention scale multiply into bmm dispatch (saves 1 dispatch per attention step).
// q: [B, M, K], k: [B, N, K] (NOT transposed — we do k.T internally)
// Returns: scale * (q @ k.T) = [B, M, N]
// Backward: grad_q = scale * (go @ k), grad_k = scale * (q.T @ go)
class VulkanScaledBmmFunction : public torch::autograd::Function<VulkanScaledBmmFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& q, const at::Tensor& k,
                               double scale) {
        ctx->save_for_backward({q, k});
        ctx->saved_data["scale"] = scale;
        // q: [B, M, K], k: [B, N, K] → k.T: [B, K, N]
        // Dispatch: scale * (q @ k.T) using vulkan_bmm_ex with tb=true
        auto q_c = q.contiguous();
        auto k_c = k.contiguous();
        return vulkan_bmm_ex(q_c, k_c, false, true, static_cast<float>(scale));
    }
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto q = saved[0];
        auto k = saved[1];
        double scale = ctx->saved_data["scale"].toDouble();
        float sf = static_cast<float>(scale);
        auto go = grad_outputs[0].contiguous();  // [B, M, N]
        auto q_c = q.contiguous();
        auto k_c = k.contiguous();
        // grad_q = scale * (go @ k): go[B,M,N] @ k[B,N,K] = [B,M,K]
        auto grad_q = vulkan_bmm_ex(go, k_c, false, false, sf);
        // grad_k = scale * (q.T @ go): q.T[B,K,M] @ go[B,M,N] = [B,K,N], then k is [B,N,K]
        // grad_k[B,N,K] = scale * (go.T[B,N,M] @ q[B,M,K])
        auto grad_k = vulkan_bmm_ex(go, q_c, true, false, sf);
        return {grad_q, grad_k, at::Tensor()};
    }
};

at::Tensor vulkan_scaled_bmm_autograd(const at::Tensor& q, const at::Tensor& k, double scale) {
    return VulkanScaledBmmFunction::apply(q, k, scale);
}

// ── Flash Attention autograd ─────────────────────────────────────
// Accepts head-major [B,H,N,D] or seq-major [B,N,H,D] layout via q_seq_major flag.
// For seq-major: Q is [B,S,H,D] and K/V are [B,S,KV_H,D] — no transpose copy needed.
// Returns output always in head-major [B,H,N,D].
class VulkanFlashAttentionFunction
    : public torch::autograd::Function<VulkanFlashAttentionFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
                               double scale, bool is_causal, bool q_seq_major) {
        float sf = static_cast<float>(scale);
        auto [output, lse] = vulkan_flash_attention_forward(Q, K, V, sf, is_causal, q_seq_major);
        // Save head-major Q/K/V for backward (backward always uses head-major layout)
        // If seq-major was used, save the contiguous head-major versions via transposition.
        // (output and lse are always head-major)
        // Always save the original tensors (no layout conversion at this point).
        // Backward will convert to head-major via .contiguous() if q_seq_major is set.
        ctx->save_for_backward({Q, K, V, output, lse});
        ctx->saved_data["scale"]        = scale;
        ctx->saved_data["is_causal"]    = is_causal;
        ctx->saved_data["q_seq_major"]  = q_seq_major;
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto Q      = saved[0];
        auto K      = saved[1];
        auto V      = saved[2];
        auto output = saved[3];
        auto lse    = saved[4];
        float sf       = static_cast<float>(ctx->saved_data["scale"].toDouble());
        bool is_causal = ctx->saved_data["is_causal"].toBool();
        bool q_seq_major = ctx->saved_data["q_seq_major"].toBool();
        auto go = grad_outputs[0];
        // Pass Q/K/V directly — vulkan_flash_attention_backward auto-detects seq-major
        // layout from strides and returns gQ/gK/gV in matching layout (seq-major → transpose view).
        // No need to convert to head-major here; doing so would add 3 GPU copy dispatches.
        auto [gQ, gK, gV] = vulkan_flash_attention_backward(
            go, Q, K, V, output, lse, sf, is_causal);
        // Return 6 gradients: Q, K, V, scale, is_causal, q_seq_major
        return {gQ, gK, gV, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor vulkan_flash_attention_autograd(
    const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
    double scale, bool is_causal, bool q_seq_major) {
    return VulkanFlashAttentionFunction::apply(Q, K, V, scale, is_causal, q_seq_major);
}

}} // namespace torch_vulkan::ops
