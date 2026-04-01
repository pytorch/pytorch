// Backward helper ops for PyTorch's built-in autograd decompositions.
// These are called when AutogradPrivateUse1 is NOT registered for an op,
// allowing PyTorch's default autograd formulas (derivatives.yaml) to work.
// This enables torch.compile/AOT Autograd tracing through standard ops.

#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <cstring>

namespace torch_vulkan { namespace ops {

// ── threshold_backward (fused GPU shader) ───────────────────────
// grad_input = grad_output * (self > threshold) — single pass
at::Tensor vulkan_threshold_backward(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Scalar& threshold) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);
    auto grad_input = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);
    struct { uint32_t numel; float threshold; } params{numel, threshold.toFloat()};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("activation_threshold_backward_fwd",
                    shaders::activation_threshold_backward_fwd,
                    shaders::activation_threshold_backward_fwd_size,
                    {go_c, self_c, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

// ── sigmoid_backward (fused GPU shader) ─────────────────────────
// grad_input = grad_output * output * (1 - output) — single pass
at::Tensor vulkan_sigmoid_backward(
    const at::Tensor& grad_output, const at::Tensor& output) {
    auto go_c = grad_output.contiguous();
    auto out_c = output.contiguous();
    auto orig_dtype = out_c.scalar_type();
    go_c = ensure_float32(go_c);
    out_c = ensure_float32(out_c);
    auto grad_input = at::empty_like(out_c);
    uint32_t numel = static_cast<uint32_t>(out_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);
    dispatch_elementwise("activation_sigmoid_backward_fwd",
                         shaders::activation_sigmoid_backward_fwd,
                         shaders::activation_sigmoid_backward_fwd_size,
                         {go_c, out_c, grad_input}, numel);
    return cast_from_float32(grad_input, orig_dtype);
}

// ── tanh_backward (fused GPU shader) ────────────────────────────
// grad_input = grad_output * (1 - output * output) — single pass
at::Tensor vulkan_tanh_backward(
    const at::Tensor& grad_output, const at::Tensor& output) {
    auto go_c = grad_output.contiguous();
    auto out_c = output.contiguous();
    auto orig_dtype = out_c.scalar_type();
    go_c = ensure_float32(go_c);
    out_c = ensure_float32(out_c);
    auto grad_input = at::empty_like(out_c);
    uint32_t numel = static_cast<uint32_t>(out_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);
    dispatch_elementwise("activation_tanh_backward_fwd",
                         shaders::activation_tanh_backward_fwd,
                         shaders::activation_tanh_backward_fwd_size,
                         {go_c, out_c, grad_input}, numel);
    return cast_from_float32(grad_input, orig_dtype);
}

// ── gelu_backward ───────────────────────────────────────────────
// GPU shader for tanh approximation backward
at::Tensor vulkan_gelu_backward(
    const at::Tensor& grad_output, const at::Tensor& self, c10::string_view approximate) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    check_supported_float(self_c, "gelu_backward");
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);

    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    dispatch_elementwise("activation_gelu_backward_fwd",
                         shaders::activation_gelu_backward_fwd,
                         shaders::activation_gelu_backward_fwd_size,
                         {go_c, self_c, output}, numel);
    return cast_from_float32(output, orig_dtype);
}

// ── silu_backward (fused GPU shader) ────────────────────────────
// grad_input = grad_output * sigmoid(x) * (1 + x - x*sigmoid(x)) — single pass
at::Tensor vulkan_silu_backward(
    const at::Tensor& grad_output, const at::Tensor& self) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);
    auto grad_input = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);
    dispatch_elementwise("activation_silu_backward_fwd",
                         shaders::activation_silu_backward_fwd,
                         shaders::activation_silu_backward_fwd_size,
                         {go_c, self_c, grad_input}, numel);
    return cast_from_float32(grad_input, orig_dtype);
}

// ── leaky_relu_backward ─────────────────────────────────────────
// GPU shader: grad * (x > 0 ? 1 : negative_slope)
at::Tensor vulkan_leaky_relu_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Scalar& negative_slope, bool /*self_is_result*/) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    check_supported_float(self_c, "leaky_relu_backward");
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);

    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct { uint32_t numel; float negative_slope; } params{
        numel, negative_slope.toFloat()};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("activation_leaky_relu_backward_fwd",
                    shaders::activation_leaky_relu_backward_fwd,
                    shaders::activation_leaky_relu_backward_fwd_size,
                    {go_c, self_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── elu_backward ────────────────────────────────────────────────
// GPU shader: grad * (x > 0 ? 1 : alpha * exp(x))
at::Tensor vulkan_elu_backward(
    const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale,
    const at::Scalar& input_scale, bool is_result, const at::Tensor& self_or_result) {
    auto go_c = grad_output.contiguous();
    auto sor_c = self_or_result.contiguous();

    // Simple case: scale=1, input_scale=1, not is_result (common)
    if (!is_result && scale.toFloat() == 1.0f && input_scale.toFloat() == 1.0f) {
        check_supported_float(sor_c, "elu_backward");
        auto orig_dtype = sor_c.scalar_type();
        go_c = ensure_float32(go_c);
        sor_c = ensure_float32(sor_c);

        auto output = at::empty_like(sor_c);
        uint32_t numel = static_cast<uint32_t>(sor_c.numel());
        if (numel == 0) return cast_from_float32(output, orig_dtype);

        struct { uint32_t numel; float alpha; } params{numel, alpha.toFloat()};
        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("activation_elu_backward_fwd",
                        shaders::activation_elu_backward_fwd,
                        shaders::activation_elu_backward_fwd_size,
                        {go_c, sor_c, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(output, orig_dtype);
    }
    // Complex case (non-default scale/input_scale or is_result): not implemented
    TORCH_CHECK(false, "Vulkan elu_backward: only supports scale=1, input_scale=1, is_result=false. ",
                "Got scale=", scale.toFloat(), ", input_scale=", input_scale.toFloat(),
                ", is_result=", is_result, ", dtype=", sor_c.scalar_type());
}

// ── _softmax_backward_data (fused GPU shader for last-dim) ──────
// grad_input = output * (grad_output - sum(grad_output * output, dim))
at::Tensor vulkan_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output,
    int64_t dim, at::ScalarType /*input_dtype*/) {
    auto go_c = grad_output.contiguous();
    auto out_c = output.contiguous();

    dim = at::maybe_wrap_dim(dim, go_c.dim());
    int64_t row_size = go_c.size(dim);
    int64_t num_rows = go_c.numel() / row_size;

    // Use fused shader when reducing over last dim and row fits in one workgroup
    if (dim == go_c.dim() - 1 && row_size <= 256 && is_supported_float(go_c.scalar_type())) {
        auto orig_dtype = go_c.scalar_type();
        auto go_f32 = ensure_float32(go_c);
        auto out_f32 = ensure_float32(out_c);
        auto grad_input = at::empty_like(out_f32);
        struct { uint32_t row_size; uint32_t num_rows; } params{
            static_cast<uint32_t>(row_size), static_cast<uint32_t>(num_rows)};
        uint32_t workgroups = static_cast<uint32_t>(num_rows);
        dispatch_shader("activation_softmax_backward_fwd",
                        shaders::activation_softmax_backward_fwd,
                        shaders::activation_softmax_backward_fwd_size,
                        {go_f32, out_f32, grad_input},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(grad_input, orig_dtype);
    }

    // Fused large-row shader: one workgroup per row with strided access
    if (dim == go_c.dim() - 1 && is_supported_float(go_c.scalar_type())) {
        auto orig_dtype = go_c.scalar_type();
        auto go_f32 = ensure_float32(go_c);
        auto out_f32 = ensure_float32(out_c);
        auto grad_input = at::empty_like(out_f32);
        struct { uint32_t row_size; uint32_t num_rows; } params{
            static_cast<uint32_t>(row_size), static_cast<uint32_t>(num_rows)};
        uint32_t workgroups = static_cast<uint32_t>(num_rows);
        dispatch_shader("activation_softmax_backward_large_fwd",
                        shaders::activation_softmax_backward_large_fwd,
                        shaders::activation_softmax_backward_large_fwd_size,
                        {go_f32, out_f32, grad_input},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(grad_input, orig_dtype);
    }

    // Fallback: op-composition approach (non-last-dim)
    auto mul_result = vulkan_mul(grad_output, output);
    auto sum_result = vulkan_sum(mul_result, at::IntArrayRef({dim}), /*keepdim=*/true, c10::nullopt);
    auto shifted = vulkan_sub(grad_output, sum_result, /*alpha=*/1);
    return vulkan_mul(shifted, output);
}

// ── _log_softmax_backward_data ──────────────────────────────────
// grad_input = grad_output - exp(output) * sum(grad_output, dim)
at::Tensor vulkan_log_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output,
    int64_t dim, at::ScalarType /*input_dtype*/) {
    auto go_c = grad_output.contiguous();
    auto out_c = output.contiguous();

    dim = at::maybe_wrap_dim(dim, go_c.dim());
    int64_t row_size = go_c.size(dim);
    int64_t num_rows = go_c.numel() / row_size;

    // Fused shader when reducing over last dim and row fits in one workgroup
    if (dim == go_c.dim() - 1 && row_size <= 256 && is_supported_float(go_c.scalar_type())) {
        auto orig_dtype = go_c.scalar_type();
        auto go_f32 = ensure_float32(go_c);
        auto out_f32 = ensure_float32(out_c);
        auto grad_input = at::empty_like(out_f32);
        struct { uint32_t row_size; uint32_t num_rows; } params{
            static_cast<uint32_t>(row_size), static_cast<uint32_t>(num_rows)};
        uint32_t workgroups = static_cast<uint32_t>(num_rows);
        dispatch_shader("activation_log_softmax_backward_fwd",
                        shaders::activation_log_softmax_backward_fwd,
                        shaders::activation_log_softmax_backward_fwd_size,
                        {go_f32, out_f32, grad_input},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(grad_input, orig_dtype);
    }

    // Fused large-row shader: one workgroup per row with strided access
    if (dim == go_c.dim() - 1 && is_supported_float(go_c.scalar_type())) {
        auto orig_dtype = go_c.scalar_type();
        auto go_f32 = ensure_float32(go_c);
        auto out_f32 = ensure_float32(out_c);
        auto grad_input = at::empty_like(out_f32);
        struct { uint32_t row_size; uint32_t num_rows; } params{
            static_cast<uint32_t>(row_size), static_cast<uint32_t>(num_rows)};
        uint32_t workgroups = static_cast<uint32_t>(num_rows);
        dispatch_shader("activation_log_softmax_backward_large_fwd",
                        shaders::activation_log_softmax_backward_large_fwd,
                        shaders::activation_log_softmax_backward_large_fwd_size,
                        {go_f32, out_f32, grad_input},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(grad_input, orig_dtype);
    }

    // Fallback: op-composition approach (non-last-dim)
    auto sum_grad = vulkan_sum(go_c, at::IntArrayRef({dim}), /*keepdim=*/true, c10::nullopt);
    auto exp_output = vulkan_exp(out_c);
    auto correction = vulkan_mul(exp_output, sum_grad);
    return vulkan_sub(go_c, correction, /*alpha=*/1);
}

// ── avg_pool2d_backward (GPU shader) ────────────────────────────
at::Tensor vulkan_avg_pool2d_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
    bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    check_supported_float(self_c, "avg_pool2d_backward");
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);

    int64_t N = self_c.size(0);
    int64_t C = self_c.size(1);
    int64_t iH = self_c.size(2), iW = self_c.size(3);
    int64_t oH = go_c.size(2), oW = go_c.size(3);
    int64_t kH = kernel_size[0], kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride.empty() ? kH : stride[0], sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
    int64_t pH = padding.empty() ? 0 : padding[0], pW = padding.size() > 1 ? padding[1] : pH;

    auto grad_input = at::empty_like(self_c);
    uint32_t total = static_cast<uint32_t>(N * C * iH * iW);
    if (total == 0) return grad_input;

    struct {
        uint32_t total, N, C;
        uint32_t iH, iW, oH, oW;
        uint32_t kH, kW, sH, sW, pH, pW;
        uint32_t count_include_pad;
    } params{
        total, static_cast<uint32_t>(N), static_cast<uint32_t>(C),
        static_cast<uint32_t>(iH), static_cast<uint32_t>(iW),
        static_cast<uint32_t>(oH), static_cast<uint32_t>(oW),
        static_cast<uint32_t>(kH), static_cast<uint32_t>(kW),
        static_cast<uint32_t>(sH), static_cast<uint32_t>(sW),
        static_cast<uint32_t>(pH), static_cast<uint32_t>(pW),
        count_include_pad ? 1u : 0u
    };

    uint32_t workgroups = (total + 255) / 256;
    dispatch_shader("pooling_avg_pool2d_backward_fwd",
                    shaders::pooling_avg_pool2d_backward_fwd,
                    shaders::pooling_avg_pool2d_backward_fwd_size,
                    {go_c, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

// ── max_pool2d_with_indices (GPU shader) ────────────────────────
std::tuple<at::Tensor, at::Tensor> vulkan_max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool ceil_mode) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "max_pool2d_with_indices");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    TORCH_CHECK(self_c.dim() == 4, "max_pool2d requires 4D input [N,C,H,W]");

    int64_t N = self_c.size(0), C = self_c.size(1);
    int64_t iH = self_c.size(2), iW = self_c.size(3);
    int64_t kH = kernel_size[0], kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride.empty() ? kH : stride[0], sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
    int64_t pH = padding.empty() ? 0 : padding[0], pW = padding.size() > 1 ? padding[1] : pH;

    int64_t oH = (iH + 2 * pH - kH) / sH + 1;
    int64_t oW = (iW + 2 * pW - kW) / sW + 1;

    auto values = at::empty({N, C, oH, oW}, self_c.options());
    auto indices_float = at::empty({N, C, oH, oW}, self_c.options());  // uint stored as float

    uint32_t total = static_cast<uint32_t>(N * C * oH * oW);
    if (total == 0) {
        auto indices_long = at::empty({N, C, oH, oW}, self_c.options().dtype(at::kLong));
        return {values, indices_long};
    }

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
                    {self_c, values, indices_float},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Read uint indices back and convert to Long
    auto& alloc = VulkanAllocator::instance();
    std::vector<uint32_t> idx_buf(total);
    alloc.get_buffer(indices_float.data_ptr())->read(idx_buf.data(), total * sizeof(uint32_t));
    auto indices_cpu = at::empty({N, C, oH, oW}, at::TensorOptions().dtype(at::kLong));
    auto* lptr = indices_cpu.data_ptr<int64_t>();
    for (uint32_t i = 0; i < total; i++) lptr[i] = static_cast<int64_t>(idx_buf[i]);
    auto indices_long = indices_cpu.to(self.device());

    return {cast_from_float32(values, orig_dtype), indices_long};
}

// ── max_pool2d_with_indices_backward (GPU shader) ───────────────
at::Tensor vulkan_max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntArrayRef kernel_size, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation,
    bool ceil_mode, const at::Tensor& indices) {
    auto go_c = grad_output.contiguous();
    check_supported_float(go_c, "max_pool2d_backward");
    auto orig_dtype = go_c.scalar_type();
    go_c = ensure_float32(go_c);

    // Zero-initialize grad_input
    auto grad_input = at::zeros_like(self);
    uint32_t output_numel = static_cast<uint32_t>(go_c.numel());
    uint32_t input_numel = static_cast<uint32_t>(self.numel());
    if (output_numel == 0) return grad_input;

    // Extract spatial dimensions
    int64_t N = self.size(0), C = self.size(1);
    int64_t iH = self.size(2), iW = self.size(3);
    int64_t oH = go_c.size(2), oW = go_c.size(3);
    int64_t kH = kernel_size[0], kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride[0], sW = stride.size() > 1 ? stride[1] : sH;
    int64_t pH = padding[0], pW = padding.size() > 1 ? padding[1] : pH;

    // Convert Long indices to uint stored as float on Vulkan
    auto indices_cpu = indices.cpu().contiguous();
    auto indices_float = at::empty({static_cast<int64_t>(output_numel)}, go_c.options());
    {
        auto& alloc = VulkanAllocator::instance();
        std::vector<uint32_t> idx_buf(output_numel);
        auto* lptr = indices_cpu.data_ptr<int64_t>();
        for (uint32_t i = 0; i < output_numel; i++)
            idx_buf[i] = static_cast<uint32_t>(lptr[i]);
        alloc.get_buffer(indices_float.data_ptr())->write(idx_buf.data(), output_numel * sizeof(uint32_t));
    }

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
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(iH), static_cast<uint32_t>(iW),
        static_cast<uint32_t>(oH), static_cast<uint32_t>(oW),
        static_cast<uint32_t>(kH), static_cast<uint32_t>(kW),
        static_cast<uint32_t>(sH), static_cast<uint32_t>(sW),
        static_cast<uint32_t>(pH), static_cast<uint32_t>(pW)
    };
    uint32_t workgroups = (input_numel + 255) / 256;
    dispatch_shader("pooling_max_pool2d_backward_fwd",
                    shaders::pooling_max_pool2d_backward_fwd,
                    shaders::pooling_max_pool2d_backward_fwd_size,
                    {go_c, indices_float, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

// ── embedding_dense_backward (GPU shader with atomic accumulation) ───
// Uses CAS-based float atomics to handle duplicate indices correctly on GPU.
// No CPU readback needed — indices stay on device throughout.
at::Tensor vulkan_embedding_dense_backward(
    const at::Tensor& grad_output, const at::Tensor& indices,
    c10::SymInt num_weights, c10::SymInt padding_idx, bool scale_grad_by_freq) {
    auto go_c = grad_output.contiguous();
    check_supported_float(go_c, "embedding_dense_backward");
    auto orig_dtype = go_c.scalar_type();
    go_c = ensure_float32(go_c);
    int64_t nw = num_weights.expect_int();
    int64_t emb_dim = go_c.size(-1);
    int64_t num_indices = indices.numel();
    int64_t pad_idx = padding_idx.expect_int();

    // Initialize grad_weight to zeros
    auto grad_weight = at::zeros({nw, emb_dim}, go_c.options());
    if (num_indices == 0) return cast_from_float32(grad_weight, orig_dtype);

    // Convert/reinterpret indices to uint32 on GPU (same as forward embedding).
    // For int32 on Vulkan: reinterpret buffer as float (asint() in shader, no copy).
    at::Tensor indices_uint;
    if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
        indices.scalar_type() == at::kLong) {
        auto indices_c = indices.contiguous();
        indices_uint = at::empty({num_indices}, go_c.options());
        struct { uint32_t numel; } i2i_params{static_cast<uint32_t>(num_indices)};
        dispatch_shader("indexing_i64_to_i32_fwd",
                        shaders::indexing_i64_to_i32_fwd,
                        shaders::indexing_i64_to_i32_fwd_size,
                        {indices_c, indices_uint},
                        (static_cast<uint32_t>(num_indices) + 255) / 256, 1, 1,
                        &i2i_params, sizeof(i2i_params));
    } else if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
               indices.scalar_type() == at::kInt) {
        // Reinterpret int32 buffer as float — same 4-byte layout, no copy needed.
        auto indices_c = indices.contiguous();
        auto impl = c10::make_intrusive<at::TensorImpl>(
            c10::Storage(indices_c.storage()),
            indices_c.key_set(),
            at::scalarTypeToTypeMeta(at::kFloat));
        std::vector<int64_t> sz = {num_indices}, st = {1};
        impl->set_sizes_and_strides(sz, st);
        impl->set_storage_offset(indices_c.storage_offset());
        indices_uint = at::Tensor(std::move(impl));
    } else {
        // CPU indices — upload as uint
        indices_uint = at::empty({num_indices}, go_c.options());
        auto indices_cpu = indices.cpu().to(at::kInt).contiguous();
        auto* idx_ptr = indices_cpu.data_ptr<int32_t>();
        auto& alloc = VulkanAllocator::instance();
        std::vector<uint32_t> idx_buf(num_indices);
        for (int64_t i = 0; i < num_indices; i++)
            idx_buf[i] = static_cast<uint32_t>(idx_ptr[i]);
        alloc.get_buffer(indices_uint.data_ptr())->write(idx_buf.data(), num_indices * sizeof(uint32_t));
    }

    // Use atomic shader for correctness with duplicate indices
    struct { uint32_t num_indices; uint32_t embedding_dim; int32_t padding_idx; } params{
        static_cast<uint32_t>(num_indices), static_cast<uint32_t>(emb_dim),
        static_cast<int32_t>(pad_idx)};
    uint32_t total = static_cast<uint32_t>(num_indices * emb_dim);
    uint32_t workgroups = (total + 255) / 256;
    dispatch_shader("indexing_embedding_backward_atomic_fwd",
                    shaders::indexing_embedding_backward_atomic_fwd,
                    shaders::indexing_embedding_backward_atomic_fwd_size,
                    {go_c, indices_uint, grad_weight},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_weight, orig_dtype);
}

// ── native_layer_norm_backward (GPU using existing ops) ─────────
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_native_layer_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const at::Tensor& mean, const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    std::array<bool, 3> output_mask) {
    // Fused GPU shaders for layer norm backward:
    //   Pass 1: grad_input via shared-memory reduction (1 dispatch, 1 workgroup/row)
    //   Pass 2: grad_weight + grad_bias via batch reduction (1 dispatch)

    int64_t norm_size = 1;
    for (const auto& s : normalized_shape) norm_size *= s.expect_int();
    int64_t num_rows = input.numel() / norm_size;

    auto go_c = ensure_float32(grad_out.contiguous());
    auto input_c = ensure_float32(input.contiguous());
    auto orig_dtype = grad_out.scalar_type();

    auto go_2d = go_c.reshape({num_rows, norm_size});
    auto in_2d = input_c.reshape({num_rows, norm_size});
    auto mean_c = mean.contiguous();
    auto rstd_c = rstd.contiguous();

    at::Tensor grad_input, grad_weight, grad_bias;

    // Pass 1: Fused grad_input (one workgroup per row)
    if (output_mask[0]) {
        grad_input = at::empty_like(in_2d);
        at::Tensor weight_c;
        uint32_t has_weight = 0;
        if (weight.has_value()) {
            weight_c = ensure_float32(weight->contiguous().to(input.device()));
            has_weight = 1;
        } else {
            weight_c = at::ones({norm_size}, in_2d.options());
        }
        struct { uint32_t num_rows; uint32_t norm_size; uint32_t has_weight; } params{
            static_cast<uint32_t>(num_rows), static_cast<uint32_t>(norm_size), has_weight
        };
        dispatch_shader("normalization_layer_norm_backward_fwd",
                        shaders::normalization_layer_norm_backward_fwd,
                        shaders::normalization_layer_norm_backward_fwd_size,
                        {go_2d, in_2d, mean_c, rstd_c, weight_c, grad_input},
                        static_cast<uint32_t>(num_rows), 1, 1,
                        &params, sizeof(params));
        grad_input = cast_from_float32(grad_input.reshape(input.sizes()), orig_dtype);
    }

    // Pass 2: Fused grad_weight + grad_bias (one thread per norm_size element)
    if (output_mask[1] || output_mask[2]) {
        // Use empty (not zeros) — the shader initializes every element via atomic add from zero
        grad_weight = at::empty({norm_size}, in_2d.options());
        grad_bias = at::empty({norm_size}, in_2d.options());
        struct { uint32_t num_rows; uint32_t norm_size; uint32_t compute_weight; uint32_t compute_bias; } params2{
            static_cast<uint32_t>(num_rows), static_cast<uint32_t>(norm_size),
            output_mask[1] ? 1u : 0u, output_mask[2] ? 1u : 0u
        };
        uint32_t wg = (static_cast<uint32_t>(norm_size) + 255) / 256;
        dispatch_shader("normalization_layer_norm_backward_weight_fwd",
                        shaders::normalization_layer_norm_backward_weight_fwd,
                        shaders::normalization_layer_norm_backward_weight_fwd_size,
                        {go_2d, in_2d, mean_c, rstd_c, grad_weight, grad_bias},
                        wg, 1, 1,
                        &params2, sizeof(params2), 2);
        if (output_mask[1]) grad_weight = cast_from_float32(grad_weight, orig_dtype);
        if (output_mask[2]) grad_bias = cast_from_float32(grad_bias, orig_dtype);
    }

    return std::make_tuple(
        output_mask[0] ? grad_input : at::Tensor(),
        output_mask[1] ? grad_weight : at::Tensor(),
        output_mask[2] ? grad_bias : at::Tensor());
}

// ── native_group_norm_backward (fused GPU shaders) ──────────────
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_native_group_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& mean, const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight,
    c10::SymInt N_sym, c10::SymInt C_sym, c10::SymInt HxW_sym, int64_t group,
    std::array<bool, 3> output_mask) {

    int64_t N_val = N_sym.expect_int();
    int64_t C_val = C_sym.expect_int();
    int64_t HxW_val = HxW_sym.expect_int();
    int64_t cpg = C_val / group;
    int64_t group_size = cpg * HxW_val;
    int64_t num_rows = N_val * group;

    auto go_c = ensure_float32(grad_out.contiguous());
    auto input_c = ensure_float32(input.contiguous());
    auto orig_dtype = grad_out.scalar_type();
    auto mean_c = mean.contiguous();
    auto rstd_c = rstd.contiguous();

    auto go_2d = go_c.reshape({num_rows, group_size});
    auto x_2d = input_c.reshape({num_rows, group_size});

    at::Tensor grad_input, grad_weight, grad_bias;

    // Pass 1: Fused grad_input (one workgroup per row = N*G workgroups)
    if (output_mask[0]) {
        grad_input = at::empty_like(x_2d);
        at::Tensor weight_c;
        uint32_t has_weight = 0;
        if (weight.has_value()) {
            weight_c = ensure_float32(weight->contiguous().to(input.device()));
            has_weight = 1;
        } else {
            weight_c = at::ones({C_val}, x_2d.options());
        }
        struct { uint32_t num_rows; uint32_t group_size; uint32_t num_groups;
                 uint32_t cpg; uint32_t HxW; uint32_t has_weight; } params{
            static_cast<uint32_t>(num_rows), static_cast<uint32_t>(group_size),
            static_cast<uint32_t>(group), static_cast<uint32_t>(cpg),
            static_cast<uint32_t>(HxW_val), has_weight
        };
        dispatch_shader("normalization_group_norm_backward_fwd",
                        shaders::normalization_group_norm_backward_fwd,
                        shaders::normalization_group_norm_backward_fwd_size,
                        {go_2d, x_2d, mean_c, rstd_c, weight_c, grad_input},
                        static_cast<uint32_t>(num_rows), 1, 1,
                        &params, sizeof(params));
        grad_input = cast_from_float32(grad_input.reshape(input_c.sizes()), orig_dtype);
    }

    // Pass 2: Fused grad_weight + grad_bias (one thread per channel)
    if (output_mask[1] || output_mask[2]) {
        // Use empty (not zeros) — shader fully initializes via sum_gw/sum_gb = 0 + accumulate + assign
        grad_weight = at::empty({C_val}, x_2d.options());
        grad_bias = at::empty({C_val}, x_2d.options());
        // Reshape to [N, C, HxW] for weight backward
        auto go_nchw = go_c.reshape({N_val, C_val, HxW_val});
        auto x_nchw = input_c.reshape({N_val, C_val, HxW_val});
        struct { uint32_t N; uint32_t C; uint32_t HxW; uint32_t G;
                 uint32_t cpg; uint32_t compute_weight; uint32_t compute_bias; } params2{
            static_cast<uint32_t>(N_val), static_cast<uint32_t>(C_val),
            static_cast<uint32_t>(HxW_val), static_cast<uint32_t>(group),
            static_cast<uint32_t>(cpg),
            output_mask[1] ? 1u : 0u, output_mask[2] ? 1u : 0u
        };
        uint32_t wg = (static_cast<uint32_t>(C_val) + 255) / 256;
        dispatch_shader("normalization_group_norm_backward_weight_fwd",
                        shaders::normalization_group_norm_backward_weight_fwd,
                        shaders::normalization_group_norm_backward_weight_fwd_size,
                        {go_nchw, x_nchw, mean_c, rstd_c, grad_weight, grad_bias},
                        wg, 1, 1,
                        &params2, sizeof(params2), 2);
        if (output_mask[1]) grad_weight = cast_from_float32(grad_weight, orig_dtype);
        if (output_mask[2]) grad_bias = cast_from_float32(grad_bias, orig_dtype);
    }

    return std::make_tuple(
        output_mask[0] ? grad_input : at::Tensor(),
        output_mask[1] ? grad_weight : at::Tensor(),
        output_mask[2] ? grad_bias : at::Tensor());
}

// ── native_batch_norm_backward (GPU using existing ops) ─────────
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& running_mean,
    const std::optional<at::Tensor>& running_var,
    const std::optional<at::Tensor>& save_mean,
    const std::optional<at::Tensor>& save_invstd,
    bool train, double eps, std::array<bool, 3> output_mask) {
    // Fused GPU shaders for batch norm backward
    auto go_c = ensure_float32(grad_out.contiguous());
    auto input_c = ensure_float32(input.contiguous());
    auto orig_dtype = grad_out.scalar_type();
    int64_t N_batch = input_c.size(0);
    int64_t C = input_c.size(1);
    int64_t spatial = 1;
    for (int64_t i = 2; i < input_c.dim(); i++) spatial *= input_c.size(i);
    int64_t M = N_batch * spatial;

    at::Tensor mean_t, invstd_t;
    if (save_mean.has_value() && save_invstd.has_value()) {
        mean_t = save_mean->contiguous();
        invstd_t = save_invstd->contiguous();
    } else if (running_mean.has_value() && running_var.has_value()) {
        mean_t = running_mean->contiguous();
        auto var_eps = vulkan_add_scalar(running_var->contiguous(), at::Scalar(static_cast<float>(eps)), 1);
        invstd_t = at::rsqrt(var_eps);
    } else {
        TORCH_CHECK(false, "batch_norm_backward requires either save_mean/save_invstd or running_mean/running_var");
    }

    auto go_3d = go_c.reshape({N_batch, C, spatial});
    auto in_3d = input_c.reshape({N_batch, C, spatial});

    at::Tensor grad_input, grad_weight, grad_bias;

    // Pass 1: Fused grad_input (one workgroup per channel)
    if (output_mask[0]) {
        grad_input = at::empty_like(in_3d);
        at::Tensor weight_c;
        uint32_t has_weight = 0;
        if (weight.has_value()) {
            weight_c = ensure_float32(weight->contiguous().to(input.device()));
            has_weight = 1;
        } else {
            weight_c = at::ones({C}, in_3d.options());
        }

        if (train) {
            struct { uint32_t C; uint32_t M; uint32_t N; uint32_t HxW; uint32_t has_weight; } params{
                static_cast<uint32_t>(C), static_cast<uint32_t>(M),
                static_cast<uint32_t>(N_batch), static_cast<uint32_t>(spatial), has_weight
            };
            dispatch_shader("normalization_batch_norm_backward_fwd",
                            shaders::normalization_batch_norm_backward_fwd,
                            shaders::normalization_batch_norm_backward_fwd_size,
                            {go_3d, in_3d, mean_t, invstd_t, weight_c, grad_input},
                            static_cast<uint32_t>(C), 1, 1,
                            &params, sizeof(params));
        } else {
            // Eval backward: grad_input = invstd * weight * grad_output
            auto invstd_ex = invstd_t.reshape({1, C, 1}).expand({N_batch, C, spatial}).contiguous();
            if (weight.has_value()) {
                auto w_ex = weight_c.reshape({1, C, 1}).expand({N_batch, C, spatial}).contiguous();
                grad_input = vulkan_mul(vulkan_mul(go_3d, w_ex), invstd_ex);
            } else {
                grad_input = vulkan_mul(go_3d, invstd_ex);
            }
        }
        grad_input = cast_from_float32(grad_input.reshape(input.sizes()), orig_dtype);
    }

    // Pass 2: Fused grad_weight + grad_bias
    if (output_mask[1] || output_mask[2]) {
        // Use empty (not zeros) — shader fully initializes via assignment (not atomics)
        grad_weight = at::empty({C}, in_3d.options());
        grad_bias = at::empty({C}, in_3d.options());
        struct { uint32_t C; uint32_t N; uint32_t HxW; uint32_t compute_weight; uint32_t compute_bias; } params2{
            static_cast<uint32_t>(C), static_cast<uint32_t>(N_batch),
            static_cast<uint32_t>(spatial),
            output_mask[1] ? 1u : 0u, output_mask[2] ? 1u : 0u
        };
        uint32_t wg = (static_cast<uint32_t>(C) + 255) / 256;
        dispatch_shader("normalization_batch_norm_backward_weight_fwd",
                        shaders::normalization_batch_norm_backward_weight_fwd,
                        shaders::normalization_batch_norm_backward_weight_fwd_size,
                        {go_3d, in_3d, mean_t, invstd_t, grad_weight, grad_bias},
                        wg, 1, 1,
                        &params2, sizeof(params2), 2);
        if (output_mask[1]) grad_weight = cast_from_float32(grad_weight, orig_dtype);
        if (output_mask[2]) grad_bias = cast_from_float32(grad_bias, orig_dtype);
    }

    return std::make_tuple(
        output_mask[0] ? grad_input : at::Tensor(),
        output_mask[1] ? grad_weight : at::Tensor(),
        output_mask[2] ? grad_bias : at::Tensor());
}

// ── linear_backward ─────────────────────────────────────────────
// PyTorch calls this directly for linear's autograd decomposition
// Handles batched inputs (3D+) by flattening to 2D for mm.
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_linear_backward(
    const at::Tensor& self, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool, 3> output_mask) {
    at::Tensor grad_input, grad_weight, grad_bias;

    // Flatten batched tensors to 2D for mm
    auto go_shape = grad_output.sizes().vec();
    int64_t out_features = grad_output.size(-1);
    int64_t batch = grad_output.numel() / out_features;
    auto grad_2d = grad_output.reshape({batch, out_features}).contiguous();

    auto self_shape = self.sizes().vec();
    int64_t in_features = self.size(-1);
    int64_t self_batch = self.numel() / in_features;
    auto self_2d = self.reshape({self_batch, in_features}).contiguous();

    if (output_mask[0]) {
        // grad_input = grad_output @ weight  (weight is [out, in])
        auto gi = vulkan_mm(grad_2d, weight);
        // Reshape back to original input shape
        grad_input = gi.reshape(self_shape);
    }
    if (output_mask[1]) {
        // grad_weight = grad_output.T @ self
        // Use vulkan_mm_ex to avoid the GPU permute copy from .t()
        grad_weight = vulkan_mm_ex(grad_2d, self_2d, /*transpose_a=*/true, /*transpose_b=*/false);
    }
    if (output_mask[2]) {
        // Sum over all batch dims
        grad_bias = vulkan_sum(grad_2d, at::IntArrayRef({0}), /*keepdim=*/false, c10::nullopt);
    }
    return std::make_tuple(
        output_mask[0] ? grad_input : at::Tensor(),
        output_mask[1] ? grad_weight : at::Tensor(),
        output_mask[2] ? grad_bias : at::Tensor());
}

}} // namespace torch_vulkan::ops
