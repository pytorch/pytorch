#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// ── Layer Norm ──────────────────────────────────────────────────
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    double eps) {

    auto input_c = input.contiguous();
    check_supported_float(input_c, "layer_norm");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);

    // normalized_shape is the trailing dimensions to normalize over
    int64_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;
    int64_t num_rows = input_c.numel() / norm_size;

    // Weight and bias (default to ones/zeros if not provided)
    at::Tensor weight = weight_opt.has_value()
        ? ensure_float32(weight_opt->contiguous().to(input_c.device()))
        : at::ones({norm_size}, input_c.options());
    at::Tensor bias = bias_opt.has_value()
        ? ensure_float32(bias_opt->contiguous().to(input_c.device()))
        : at::zeros({norm_size}, input_c.options());

    auto input_reshaped = input_c.reshape({num_rows, norm_size});
    at::Tensor mean, rstd, output;

    // Fused shader: one workgroup per row, strided access for any norm_size.
    output = at::empty_like(input_c);
    mean = at::empty({num_rows}, input_c.options());
    rstd = at::empty({num_rows}, input_c.options());

    struct { uint32_t norm_size; uint32_t num_rows; float eps; } params{
        static_cast<uint32_t>(norm_size),
        static_cast<uint32_t>(num_rows),
        static_cast<float>(eps)
    };

    dispatch_shader("normalization_layer_norm_fwd",
                    shaders::normalization_layer_norm_fwd,
                    shaders::normalization_layer_norm_fwd_size,
                    {input_c, weight, bias, output, mean, rstd},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params), 3);

    return std::make_tuple(cast_from_float32(output, orig_dtype), mean, rstd);
}

// ── Batch Norm (inference only) ─────────────────────────────────
at::Tensor vulkan_batch_norm(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::Tensor>& running_mean_opt,
    const std::optional<at::Tensor>& running_var_opt,
    bool training, double momentum, double eps, bool cudnn_enabled) {

    TORCH_CHECK(!training,
                "Vulkan batch_norm currently only supports inference (eval) mode");
    TORCH_CHECK(running_mean_opt.has_value() && running_var_opt.has_value(),
                "Vulkan batch_norm requires running_mean and running_var in eval mode");

    auto input_c = input.contiguous();
    check_supported_float(input_c, "batch_norm");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);
    TORCH_CHECK(input_c.dim() >= 2,
                "batch_norm requires at least 2D input");

    int64_t C = input_c.size(1);
    int64_t spatial = 1;
    for (int64_t i = 2; i < input_c.dim(); i++) spatial *= input_c.size(i);

    auto output = at::empty_like(input_c);

    at::Tensor running_mean = ensure_float32(running_mean_opt->contiguous().to(input_c.device()));
    at::Tensor running_var = ensure_float32(running_var_opt->contiguous().to(input_c.device()));
    at::Tensor weight = weight_opt.has_value()
        ? ensure_float32(weight_opt->contiguous().to(input_c.device()))
        : at::ones({C}, input_c.options());
    at::Tensor bias = bias_opt.has_value()
        ? ensure_float32(bias_opt->contiguous().to(input_c.device()))
        : at::zeros({C}, input_c.options());

    uint32_t total = static_cast<uint32_t>(input_c.numel());

    struct { uint32_t num_channels; uint32_t spatial_size; uint32_t total; float eps; } params{
        static_cast<uint32_t>(C),
        static_cast<uint32_t>(spatial),
        total,
        static_cast<float>(eps)
    };

    uint32_t workgroups = (total + 255) / 256;

    dispatch_shader("normalization_batch_norm_fwd",
                    shaders::normalization_batch_norm_fwd,
                    shaders::normalization_batch_norm_fwd_size,
                    {input_c, running_mean, running_var, weight, bias, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    return cast_from_float32(output, orig_dtype);
}

// ── Group Norm ──────────────────────────────────────────────────
// Multi-pass approach: compute mean/var using existing GPU ops (no shared memory),
// then apply normalization with an elementwise shader.
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_group_norm(
    const at::Tensor& input,
    int64_t num_groups,
    const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    double eps) {

    auto input_c = input.contiguous();
    check_supported_float(input_c, "group_norm");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);
    TORCH_CHECK(input_c.dim() >= 2,
                "group_norm requires at least 2D input");

    int64_t N = input_c.size(0);
    int64_t C = input_c.size(1);
    TORCH_CHECK(C % num_groups == 0,
                "group_norm: num_channels (", C, ") must be divisible by num_groups (", num_groups, ")");

    int64_t channels_per_group = C / num_groups;
    int64_t spatial = 1;
    for (int64_t i = 2; i < input_c.dim(); i++) spatial *= input_c.size(i);
    int64_t group_size = channels_per_group * spatial;

    // Reshape to [N * G, group_size] for per-group reduction
    auto x = input_c.reshape({N * num_groups, group_size});
    int64_t num_rows = N * num_groups;

    // Weight and bias (default to ones/zeros if not provided)
    at::Tensor weight = weight_opt.has_value()
        ? ensure_float32(weight_opt->contiguous().to(input_c.device()))
        : at::ones({C}, input_c.options());
    at::Tensor bias = bias_opt.has_value()
        ? ensure_float32(bias_opt->contiguous().to(input_c.device()))
        : at::zeros({C}, input_c.options());

    at::Tensor mean, rstd, output;

    // Fused shader: one workgroup per (batch, group), strided access for any group_size.
    output = at::empty_like(x);
    mean = at::empty({num_rows}, input_c.options());
    rstd = at::empty({num_rows}, input_c.options());

    struct {
        uint32_t num_groups;
        uint32_t group_size;
        uint32_t num_rows;
        uint32_t channels_per_group;
        uint32_t spatial_size;
        float eps;
    } params{
        static_cast<uint32_t>(num_groups),
        static_cast<uint32_t>(group_size),
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(channels_per_group),
        static_cast<uint32_t>(spatial),
        static_cast<float>(eps)
    };

    dispatch_shader("normalization_group_norm_fwd",
                    shaders::normalization_group_norm_fwd,
                    shaders::normalization_group_norm_fwd_size,
                    {x, weight, bias, output, mean, rstd},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params), 3);

    return std::make_tuple(cast_from_float32(output.reshape(input_c.sizes()), orig_dtype), mean, rstd);
}

// ── RMS Norm ────────────────────────────────────────────────────
// Fused single-pass: compute variance, normalize, scale by weight
// Returns (output, rstd) where rstd is saved for backward
std::tuple<at::Tensor, at::Tensor> vulkan_rms_norm(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps) {

    auto input_c = input.contiguous();
    check_supported_float(input_c, "rms_norm");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);

    int64_t norm_size = weight.numel();
    int64_t num_rows = input_c.numel() / norm_size;
    TORCH_CHECK(input_c.numel() % norm_size == 0,
                "rms_norm: input size not divisible by norm_size");

    auto input_flat = input_c.reshape({num_rows, norm_size});
    auto output = at::empty_like(input_flat);
    auto rstd = at::empty({num_rows}, input_c.options());

    auto weight_c = ensure_float32(weight.contiguous().to(input_c.device()));

    if (num_rows == 0) return {cast_from_float32(output.reshape(input_c.sizes()), orig_dtype), rstd};

    struct { uint32_t num_rows; uint32_t norm_size; float eps; } params{
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(norm_size),
        static_cast<float>(eps)
    };

    dispatch_shader("normalization_rms_norm_fwd",
                    shaders::normalization_rms_norm_fwd,
                    shaders::normalization_rms_norm_fwd_size,
                    {input_flat, weight_c, output, rstd},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params), 2);

    return {cast_from_float32(output.reshape(input_c.sizes()), orig_dtype), rstd};
}

// ── Fused Add + RMS Norm ─────────────────────────────────────────
// h_new = residual + shortcut; out = weight * (h_new / rms(h_new))
// Returns (normed_output, h_new, rstd).
// Saves 1 dispatch vs separate add + rms_norm (critical path in every transformer layer).
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_add_rms_norm(
    const at::Tensor& residual,
    const at::Tensor& shortcut,
    const at::Tensor& weight,
    double eps) {

    auto res_c = residual.contiguous();
    auto sc_c = shortcut.contiguous();
    check_supported_float(res_c, "add_rms_norm");
    TORCH_CHECK(res_c.sizes() == sc_c.sizes(),
                "add_rms_norm: residual and shortcut must have the same shape");
    auto orig_dtype = res_c.scalar_type();
    res_c = ensure_float32(res_c);
    sc_c = ensure_float32(sc_c);

    int64_t norm_size = weight.numel();
    int64_t num_rows = res_c.numel() / norm_size;
    TORCH_CHECK(res_c.numel() % norm_size == 0,
                "add_rms_norm: input size not divisible by norm_size");

    auto res_flat = res_c.reshape({num_rows, norm_size});
    auto sc_flat = sc_c.reshape({num_rows, norm_size});
    auto output = at::empty_like(res_flat);
    auto h_new = at::empty_like(res_flat);
    auto rstd = at::empty({num_rows}, res_c.options());

    auto weight_c = ensure_float32(weight.contiguous().to(res_c.device()));

    if (num_rows == 0) {
        return {cast_from_float32(output.reshape(residual.sizes()), orig_dtype),
                cast_from_float32(h_new.reshape(residual.sizes()), orig_dtype),
                rstd};
    }

    struct { uint32_t num_rows; uint32_t norm_size; float eps; } params{
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(norm_size),
        static_cast<float>(eps)
    };

    dispatch_shader("normalization_add_rms_norm_fwd",
                    shaders::normalization_add_rms_norm_fwd,
                    shaders::normalization_add_rms_norm_fwd_size,
                    {res_flat, sc_flat, weight_c, output, h_new, rstd},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params), 3);

    return {cast_from_float32(output.reshape(residual.sizes()), orig_dtype),
            cast_from_float32(h_new.reshape(residual.sizes()), orig_dtype),
            rstd};
}

// ── Fused Add + RMSNorm Backward ────────────────────────────────
// Given: grad_normed (grad through normed output), grad_h_new (grad through h_new output)
// Returns (grad_h, grad_weight) where grad_h = grad_residual = grad_shortcut
// Fuses rms_norm_backward + add_grad into 1 dispatch (was 3 separate ops).
std::tuple<at::Tensor, at::Tensor> vulkan_add_rms_norm_backward(
    const at::Tensor& grad_normed,
    const at::Tensor& grad_h_new,
    const at::Tensor& h_new,
    const at::Tensor& weight,
    const at::Tensor& rstd) {

    auto go_c = ensure_float32(grad_normed.contiguous());
    auto gh_c = ensure_float32(grad_h_new.contiguous());
    auto h_c = ensure_float32(h_new.contiguous());
    auto weight_c = ensure_float32(weight.contiguous().to(h_new.device()));
    auto rstd_c = rstd.contiguous();

    int64_t norm_size = weight_c.numel();
    int64_t num_rows = h_c.numel() / norm_size;

    auto go_flat = go_c.reshape({num_rows, norm_size});
    auto gh_flat = gh_c.reshape({num_rows, norm_size});
    auto h_flat = h_c.reshape({num_rows, norm_size});
    auto grad_h = at::empty_like(h_flat);
    auto grad_weight = at::empty({norm_size}, h_c.options());

    if (num_rows == 0) return {grad_h.reshape(h_new.sizes()), grad_weight};

    // Pass 1: fused grad_h = rms_norm_backward_grad + grad_h_new (single workgroup per row)
    struct { uint32_t num_rows; uint32_t norm_size; } params{
        static_cast<uint32_t>(num_rows), static_cast<uint32_t>(norm_size)
    };
    dispatch_shader("normalization_add_rms_norm_backward_fwd",
                    shaders::normalization_add_rms_norm_backward_fwd,
                    shaders::normalization_add_rms_norm_backward_fwd_size,
                    {go_flat, gh_flat, h_flat, weight_c, rstd_c, grad_h},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params));

    // Pass 2: grad_weight = sum_n(grad_normed * h_new * rstd) — reuse rms_norm weight backward shader
    uint32_t wg = (static_cast<uint32_t>(norm_size) + 255) / 256;
    dispatch_shader("normalization_rms_norm_weight_backward_fwd",
                    shaders::normalization_rms_norm_weight_backward_fwd,
                    shaders::normalization_rms_norm_weight_backward_fwd_size,
                    {go_flat, h_flat, rstd_c, grad_weight},
                    wg, 1, 1,
                    &params, sizeof(params));

    auto orig_dtype = grad_normed.scalar_type();
    return {cast_from_float32(grad_h.reshape(h_new.sizes()), orig_dtype),
            cast_from_float32(grad_weight, orig_dtype)};
}

// RMS Norm backward: returns (grad_input, grad_weight)
std::tuple<at::Tensor, at::Tensor> vulkan_rms_norm_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& rstd) {

    auto go_c = ensure_float32(grad_output.contiguous());
    auto input_c = ensure_float32(input.contiguous());
    auto weight_c = ensure_float32(weight.contiguous().to(input.device()));
    auto rstd_c = rstd.contiguous();  // rstd is always f32 from forward

    int64_t norm_size = weight_c.numel();
    int64_t num_rows = input_c.numel() / norm_size;

    auto go_flat = go_c.reshape({num_rows, norm_size});
    auto input_flat = input_c.reshape({num_rows, norm_size});
    auto grad_input = at::empty_like(input_flat);
    auto grad_weight = at::empty({norm_size}, input_c.options());  // shader writes all elements

    if (num_rows == 0) return {grad_input.reshape(input_c.sizes()), grad_weight};

    // Pass 1: compute grad_input (one workgroup per row)
    struct { uint32_t num_rows; uint32_t norm_size; } params{
        static_cast<uint32_t>(num_rows), static_cast<uint32_t>(norm_size)
    };
    dispatch_shader("normalization_rms_norm_backward_fwd",
                    shaders::normalization_rms_norm_backward_fwd,
                    shaders::normalization_rms_norm_backward_fwd_size,
                    {go_flat, input_flat, weight_c, rstd_c, grad_input},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params));

    // Pass 2: accumulate grad_weight across all rows
    uint32_t wg = (static_cast<uint32_t>(norm_size) + 255) / 256;
    dispatch_shader("normalization_rms_norm_weight_backward_fwd",
                    shaders::normalization_rms_norm_weight_backward_fwd,
                    shaders::normalization_rms_norm_weight_backward_fwd_size,
                    {go_flat, input_flat, rstd_c, grad_weight},
                    wg, 1, 1,
                    &params, sizeof(params));

    auto orig_dtype = grad_output.scalar_type();
    return {cast_from_float32(grad_input.reshape(input_c.sizes()), orig_dtype),
            cast_from_float32(grad_weight, orig_dtype)};
}

// ── RMSNormGated ───────────────────────────────────────────────
// Fused: out = weight * rms_norm(input) * silu(gate)
// Used by Qwen3.5-0.8B GatedDeltaNet layers (Qwen3_5RMSNormGated).
// Returns (output, rstd) for backward.
std::tuple<at::Tensor, at::Tensor> vulkan_rms_norm_gated(
    const at::Tensor& input,
    const at::Tensor& gate,
    const at::Tensor& weight,
    double eps) {

    auto input_c = input.contiguous();
    auto gate_c = gate.contiguous();
    check_supported_float(input_c, "rms_norm_gated");
    TORCH_CHECK(input_c.sizes() == gate_c.sizes(),
                "rms_norm_gated: input and gate must have same shape");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);
    gate_c = ensure_float32(gate_c);

    int64_t norm_size = weight.numel();
    int64_t num_rows = input_c.numel() / norm_size;
    TORCH_CHECK(input_c.numel() % norm_size == 0,
                "rms_norm_gated: input size not divisible by norm_size");

    auto input_flat = input_c.reshape({num_rows, norm_size});
    auto gate_flat = gate_c.reshape({num_rows, norm_size});
    auto output = at::empty_like(input_flat);
    auto rstd = at::empty({num_rows}, input_c.options());

    auto weight_c = ensure_float32(weight.contiguous().to(input_c.device()));

    if (num_rows == 0) return {cast_from_float32(output.reshape(input_c.sizes()), orig_dtype), rstd};

    struct { uint32_t num_rows; uint32_t norm_size; float eps; } params{
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(norm_size),
        static_cast<float>(eps)
    };

    dispatch_shader("normalization_rms_norm_gated_fwd",
                    shaders::normalization_rms_norm_gated_fwd,
                    shaders::normalization_rms_norm_gated_fwd_size,
                    {input_flat, gate_flat, weight_c, output, rstd},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params), 2);

    return {cast_from_float32(output.reshape(input_c.sizes()), orig_dtype), rstd};
}

// RMSNormGated backward: returns (grad_input, grad_gate)
// grad_weight is accumulated separately (same pattern as rms_norm_weight_backward)
std::tuple<at::Tensor, at::Tensor> vulkan_rms_norm_gated_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& gate,
    const at::Tensor& weight,
    const at::Tensor& rstd) {

    auto go_c = ensure_float32(grad_output.contiguous());
    auto input_c = ensure_float32(input.contiguous());
    auto gate_c = ensure_float32(gate.contiguous());
    auto weight_c = ensure_float32(weight.contiguous().to(input.device()));
    auto rstd_c = rstd.contiguous();

    int64_t norm_size = weight_c.numel();
    int64_t num_rows = input_c.numel() / norm_size;

    auto go_flat = go_c.reshape({num_rows, norm_size});
    auto input_flat = input_c.reshape({num_rows, norm_size});
    auto gate_flat = gate_c.reshape({num_rows, norm_size});
    auto grad_input = at::empty_like(input_flat);
    auto grad_gate = at::empty_like(gate_flat);

    if (num_rows == 0) {
        return {grad_input.reshape(input_c.sizes()), grad_gate.reshape(gate_c.sizes())};
    }

    struct { uint32_t num_rows; uint32_t norm_size; } params{
        static_cast<uint32_t>(num_rows), static_cast<uint32_t>(norm_size)
    };
    dispatch_shader("normalization_rms_norm_gated_backward_fwd",
                    shaders::normalization_rms_norm_gated_backward_fwd,
                    shaders::normalization_rms_norm_gated_backward_fwd_size,
                    {go_flat, input_flat, gate_flat, weight_c, rstd_c, grad_input, grad_gate},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params), 2);

    auto orig_dtype = grad_output.scalar_type();
    return {cast_from_float32(grad_input.reshape(input_c.sizes()), orig_dtype),
            cast_from_float32(grad_gate.reshape(gate_c.sizes()), orig_dtype)};
}

}} // namespace torch_vulkan::ops
