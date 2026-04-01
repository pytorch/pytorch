#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// Helper for simple unary activations (2 buffers: input, output; numel push constant)
static at::Tensor activation_unary(
    const at::Tensor& self,
    const std::string& key,
    const uint32_t* spirv,
    size_t spirv_size) {

    auto self_c = self.contiguous();
    check_supported_float(self_c, "activation");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);

    auto output = at::empty(self_f32.sizes(), self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());

    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct { uint32_t numel; } params{numel};
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader(key, spirv, spirv_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// Helper: copy result buffer into self (for in-place ops)
static void copy_buffer_inplace(const at::Tensor& src, const at::Tensor& dst) {
    dispatch_copy_buffer(src, dst);
}

// ── ReLU ────────────────────────────────────────────────────────
at::Tensor vulkan_relu(const at::Tensor& self) {
    return activation_unary(self, "activation_relu_fwd",
                            shaders::activation_relu_fwd, shaders::activation_relu_fwd_size);
}

at::Tensor& vulkan_relu_(at::Tensor& self) {
    auto result = vulkan_relu(self);
    copy_buffer_inplace(result, self);
    return self;
}

// ── Sigmoid ─────────────────────────────────────────────────────
at::Tensor vulkan_sigmoid(const at::Tensor& self) {
    return activation_unary(self, "activation_sigmoid_fwd",
                            shaders::activation_sigmoid_fwd, shaders::activation_sigmoid_fwd_size);
}

// ── Tanh ────────────────────────────────────────────────────────
at::Tensor vulkan_tanh(const at::Tensor& self) {
    return activation_unary(self, "activation_tanh_fwd",
                            shaders::activation_tanh_fwd, shaders::activation_tanh_fwd_size);
}

// ── GELU ────────────────────────────────────────────────────────
at::Tensor vulkan_gelu(const at::Tensor& self, c10::string_view approximate) {
    return activation_unary(self, "activation_gelu_fwd",
                            shaders::activation_gelu_fwd, shaders::activation_gelu_fwd_size);
}

// ── SiLU ────────────────────────────────────────────────────────
at::Tensor vulkan_silu(const at::Tensor& self) {
    return activation_unary(self, "activation_silu_fwd",
                            shaders::activation_silu_fwd, shaders::activation_silu_fwd_size);
}

// ── Leaky ReLU ──────────────────────────────────────────────────
at::Tensor vulkan_leaky_relu(const at::Tensor& self, const at::Scalar& negative_slope) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "leaky_relu");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);

    auto output = at::empty(self_f32.sizes(), self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());

    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct { float negative_slope; uint32_t numel; } params{
        negative_slope.toFloat(), numel
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("activation_leaky_relu_fwd",
                    shaders::activation_leaky_relu_fwd, shaders::activation_leaky_relu_fwd_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── ELU ─────────────────────────────────────────────────────────
at::Tensor vulkan_elu(const at::Tensor& self, const at::Scalar& alpha,
                      const at::Scalar& scale, const at::Scalar& input_scale) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "elu");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);

    auto output = at::empty(self_f32.sizes(), self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());

    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct { float alpha; uint32_t numel; } params{
        alpha.toFloat(), numel
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("activation_elu_fwd",
                    shaders::activation_elu_fwd, shaders::activation_elu_fwd_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Apply scale and input_scale if non-default using GPU mul
    if (scale.toDouble() != 1.0 || input_scale.toDouble() != 1.0) {
        output = vulkan_mul_scalar(output, scale);
    }
    return cast_from_float32(output, orig_dtype);
}

// ── Clamp ───────────────────────────────────────────────────────
at::Tensor vulkan_clamp(const at::Tensor& self,
                        const std::optional<at::Scalar>& min_val,
                        const std::optional<at::Scalar>& max_val) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "clamp");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);

    auto output = at::empty(self_f32.sizes(), self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());

    if (numel == 0) return cast_from_float32(output, orig_dtype);

    float min_f = min_val.has_value() ? min_val->toFloat() : -3.402823466e+38f;
    float max_f = max_val.has_value() ? max_val->toFloat() :  3.402823466e+38f;

    struct { float min_val; float max_val; uint32_t numel; } params{min_f, max_f, numel};
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("unary_clamp_fwd",
                    shaders::unary_clamp_fwd, shaders::unary_clamp_fwd_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_clamp_min(const at::Tensor& self, const at::Scalar& min_val) {
    return vulkan_clamp(self, min_val, std::nullopt);
}

at::Tensor vulkan_clamp_min_tensor(const at::Tensor& self, const at::Tensor& min_val) {
    // Tensor variant: use .item() for scalar tensors, else element-wise max
    if (min_val.numel() == 1) {
        return vulkan_clamp(self, min_val.item(), std::nullopt);
    }
    // Element-wise: clamp_min(self, min) = where(self > min, self, min)
    auto cond = vulkan_gt(self, min_val);
    return vulkan_where(cond, self, min_val);
}

at::Tensor& vulkan_clamp_min_tensor_out(const at::Tensor& self, const at::Tensor& min_val, at::Tensor& out) {
    auto result = vulkan_clamp_min_tensor(self, min_val);
    copy_buffer_inplace(result, out);
    return out;
}

at::Tensor& vulkan_clamp_min_(at::Tensor& self, const at::Scalar& min_val) {
    auto result = vulkan_clamp(self, min_val, std::nullopt);
    copy_buffer_inplace(result, self);
    return self;
}

at::Tensor vulkan_clamp_max(const at::Tensor& self, const at::Scalar& max_val) {
    return vulkan_clamp(self, std::nullopt, max_val);
}

at::Tensor& vulkan_clamp_max_(at::Tensor& self, const at::Scalar& max_val) {
    auto result = vulkan_clamp(self, std::nullopt, max_val);
    copy_buffer_inplace(result, self);
    return self;
}

at::Tensor& vulkan_clamp_min_out(const at::Tensor& self, const at::Scalar& min_val, at::Tensor& out) {
    auto result = vulkan_clamp(self, min_val, std::nullopt);
    copy_buffer_inplace(result, out);
    return out;
}

at::Tensor& vulkan_clamp_max_out(const at::Tensor& self, const at::Scalar& max_val, at::Tensor& out) {
    auto result = vulkan_clamp(self, std::nullopt, max_val);
    copy_buffer_inplace(result, out);
    return out;
}

// ── Softmax ─────────────────────────────────────────────────────
at::Tensor vulkan_softmax(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "softmax");
    auto orig_dtype = self_c.scalar_type();
    // Softmax always computes in f32 for numerical stability
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());

    // Move target dim to last position (skip if already last — avoids 2 GPU copy dispatches)
    bool need_perm = (dim != self_c.dim() - 1);
    auto perm = need_perm ? self_c.movedim(dim, -1).contiguous() : self_c;
    int64_t inner_size = perm.size(-1);
    int64_t num_rows = perm.numel() / inner_size;

    auto output = at::empty_like(perm);

    if (num_rows == 0 || inner_size == 0) {
        return need_perm ? output.movedim(-1, dim) : output;
    }

    if (inner_size <= 256) {
        // Fast path: single-workgroup fused shader
        struct { uint32_t inner_size; uint32_t num_rows; } params{
            static_cast<uint32_t>(inner_size),
            static_cast<uint32_t>(num_rows)
        };

        dispatch_shader("activation_softmax_fwd",
                        shaders::activation_softmax_fwd, shaders::activation_softmax_fwd_size,
                        {perm, output},
                        static_cast<uint32_t>(num_rows), 1, 1,
                        &params, sizeof(params));
    } else {
        // Fused large-row shader: one workgroup per row with strided access
        struct { uint32_t inner_size; uint32_t num_rows; } params{
            static_cast<uint32_t>(inner_size),
            static_cast<uint32_t>(num_rows)
        };
        dispatch_shader("activation_softmax_large_fwd",
                        shaders::activation_softmax_large_fwd, shaders::activation_softmax_large_fwd_size,
                        {perm, output},
                        static_cast<uint32_t>(num_rows), 1, 1,
                        &params, sizeof(params));
    }

    return cast_from_float32(need_perm ? output.movedim(-1, dim) : output, orig_dtype);
}

// ── Log Softmax ─────────────────────────────────────────────────
at::Tensor vulkan_log_softmax(const at::Tensor& self, int64_t dim,
                              std::optional<at::ScalarType> dtype) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "log_softmax");
    auto orig_dtype = self_c.scalar_type();
    // Log softmax always computes in f32 for numerical stability
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());

    // Skip movedim when dim is already last (avoids 2 GPU copy dispatches)
    bool need_perm = (dim != self_c.dim() - 1);
    auto perm = need_perm ? self_c.movedim(dim, -1).contiguous() : self_c;
    int64_t inner_size = perm.size(-1);
    int64_t num_rows = perm.numel() / inner_size;

    auto output = at::empty_like(perm);

    if (num_rows == 0 || inner_size == 0) {
        return need_perm ? output.movedim(-1, dim) : output;
    }

    if (inner_size <= 256) {
        // Fast path: single-workgroup fused shader
        struct { uint32_t inner_size; uint32_t num_rows; } params{
            static_cast<uint32_t>(inner_size),
            static_cast<uint32_t>(num_rows)
        };

        dispatch_shader("activation_log_softmax_fwd",
                        shaders::activation_log_softmax_fwd, shaders::activation_log_softmax_fwd_size,
                        {perm, output},
                        static_cast<uint32_t>(num_rows), 1, 1,
                        &params, sizeof(params));
    } else {
        // Fused large-row shader: one workgroup per row with strided access
        struct { uint32_t inner_size; uint32_t num_rows; } params{
            static_cast<uint32_t>(inner_size),
            static_cast<uint32_t>(num_rows)
        };
        dispatch_shader("activation_log_softmax_large_fwd",
                        shaders::activation_log_softmax_large_fwd, shaders::activation_log_softmax_large_fwd_size,
                        {perm, output},
                        static_cast<uint32_t>(num_rows), 1, 1,
                        &params, sizeof(params));
    }

    return cast_from_float32(need_perm ? output.movedim(-1, dim) : output, orig_dtype);
}

// ── SELU ────────────────────────────────────────────────────────
at::Tensor vulkan_selu(const at::Tensor& self) {
    return activation_unary(self, "activation_selu_fwd",
                            shaders::activation_selu_fwd, shaders::activation_selu_fwd_size);
}

// ── PReLU ───────────────────────────────────────────────────────
at::Tensor vulkan_prelu(const at::Tensor& self, const at::Tensor& weight) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "prelu");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);

    auto output = at::empty(self_f32.sizes(), self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());

    if (numel == 0) return cast_from_float32(output, orig_dtype);

    // Broadcast weight to match input shape
    auto weight_f32 = ensure_float32(weight.contiguous());
    at::Tensor weight_expanded;
    if (weight_f32.numel() == 1) {
        weight_expanded = weight_f32.expand_as(self_f32).contiguous();
    } else {
        std::vector<int64_t> shape(self_f32.dim(), 1);
        if (self_f32.dim() > 1) shape[1] = weight_f32.numel();
        weight_expanded = weight_f32.reshape(shape).expand_as(self_f32).contiguous();
    }

    struct { uint32_t numel; } params{numel};
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("activation_prelu_fwd",
                    shaders::activation_prelu_fwd, shaders::activation_prelu_fwd_size,
                    {self_f32, weight_expanded, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── Hardtanh ────────────────────────────────────────────────────
at::Tensor vulkan_hardtanh(const at::Tensor& self,
                           const at::Scalar& min_val,
                           const at::Scalar& max_val) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "hardtanh");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);

    auto output = at::empty(self_f32.sizes(), self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct { uint32_t numel; float min_val; float max_val; } params{
        numel, min_val.toFloat(), max_val.toFloat()
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("activation_hardtanh_fwd",
                    shaders::activation_hardtanh_fwd, shaders::activation_hardtanh_fwd_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor& vulkan_hardtanh_(at::Tensor& self,
                              const at::Scalar& min_val,
                              const at::Scalar& max_val) {
    auto result = vulkan_hardtanh(self, min_val, max_val);
    copy_buffer_inplace(result, self);
    return self;
}

at::Tensor vulkan_hardtanh_backward(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Scalar& min_val,
                                     const at::Scalar& max_val) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);
    auto grad_input = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    struct { uint32_t numel; float min_val; float max_val; } params{
        numel, min_val.toFloat(), max_val.toFloat()
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("activation_hardtanh_backward_fwd",
                    shaders::activation_hardtanh_backward_fwd, shaders::activation_hardtanh_backward_fwd_size,
                    {go_c, self_c, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

// ── Hardswish ───────────────────────────────────────────────────
at::Tensor vulkan_hardswish(const at::Tensor& self) {
    return activation_unary(self, "activation_hardswish_fwd",
                            shaders::activation_hardswish_fwd, shaders::activation_hardswish_fwd_size);
}

at::Tensor& vulkan_hardswish_(at::Tensor& self) {
    auto result = vulkan_hardswish(self);
    copy_buffer_inplace(result, self);
    return self;
}

at::Tensor vulkan_hardswish_backward(const at::Tensor& grad_output, const at::Tensor& self) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);
    auto grad_input = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    dispatch_elementwise("activation_hardswish_backward_fwd",
                         shaders::activation_hardswish_backward_fwd,
                         shaders::activation_hardswish_backward_fwd_size,
                         {go_c, self_c, grad_input}, numel);
    return cast_from_float32(grad_input, orig_dtype);
}

// ── Hardsigmoid ─────────────────────────────────────────────────
at::Tensor vulkan_hardsigmoid(const at::Tensor& self) {
    return activation_unary(self, "activation_hardsigmoid_fwd",
                            shaders::activation_hardsigmoid_fwd, shaders::activation_hardsigmoid_fwd_size);
}

at::Tensor& vulkan_hardsigmoid_(at::Tensor& self) {
    auto result = vulkan_hardsigmoid(self);
    copy_buffer_inplace(result, self);
    return self;
}

at::Tensor vulkan_hardsigmoid_backward(const at::Tensor& grad_output, const at::Tensor& self) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);
    auto grad_input = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    dispatch_elementwise("activation_hardsigmoid_backward_fwd",
                         shaders::activation_hardsigmoid_backward_fwd,
                         shaders::activation_hardsigmoid_backward_fwd_size,
                         {go_c, self_c, grad_input}, numel);
    return cast_from_float32(grad_input, orig_dtype);
}

// ── Softplus ────────────────────────────────────────────────────
at::Tensor vulkan_softplus(const at::Tensor& self,
                           const at::Scalar& beta,
                           const at::Scalar& threshold) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "softplus");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);

    auto output = at::empty(self_f32.sizes(), self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct { uint32_t numel; float beta; float threshold; } params{
        numel, beta.toFloat(), threshold.toFloat()
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("activation_softplus_fwd",
                    shaders::activation_softplus_fwd, shaders::activation_softplus_fwd_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_softplus_backward(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Scalar& beta,
                                     const at::Scalar& threshold) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);
    auto grad_input = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    struct { uint32_t numel; float beta; float threshold; } params{
        numel, beta.toFloat(), threshold.toFloat()
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("activation_softplus_backward_fwd",
                    shaders::activation_softplus_backward_fwd,
                    shaders::activation_softplus_backward_fwd_size,
                    {go_c, self_c, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

// ── Mish ─────────────────────────────────────────────────────────
at::Tensor vulkan_mish(const at::Tensor& self) {
    return activation_unary(self, "activation_mish_fwd",
                            shaders::activation_mish_fwd,
                            shaders::activation_mish_fwd_size);
}

at::Tensor vulkan_mish_backward(const at::Tensor& grad_output, const at::Tensor& self) {
    auto go_c = grad_output.contiguous();
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    go_c = ensure_float32(go_c);
    self_c = ensure_float32(self_c);
    auto grad_input = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);
    dispatch_elementwise("activation_mish_backward_fwd",
                         shaders::activation_mish_backward_fwd,
                         shaders::activation_mish_backward_fwd_size,
                         {go_c, self_c, grad_input}, numel);
    return cast_from_float32(grad_input, orig_dtype);
}

// ── Fused SwiGLU: out = silu(gate) * up ─────────────────────────
at::Tensor vulkan_swiglu(const at::Tensor& gate, const at::Tensor& up) {
    auto gate_c = gate.contiguous();
    auto up_c = up.contiguous();
    check_supported_float(gate_c, "swiglu");
    TORCH_CHECK(gate_c.sizes() == up_c.sizes(),
                "Vulkan swiglu: gate and up must have same shape");
    auto orig_dtype = gate_c.scalar_type();
    auto gate_f32 = ensure_float32(gate_c);
    auto up_f32 = ensure_float32(up_c);

    auto output = at::empty_like(gate_f32);
    uint32_t numel = static_cast<uint32_t>(gate_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct { uint32_t numel; } params{numel};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("activation_swiglu_fwd",
                    shaders::activation_swiglu_fwd,
                    shaders::activation_swiglu_fwd_size,
                    {gate_f32, up_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

std::tuple<at::Tensor, at::Tensor> vulkan_swiglu_backward(
    const at::Tensor& grad_output, const at::Tensor& gate, const at::Tensor& up) {
    auto go_c = grad_output.contiguous();
    auto gate_c = gate.contiguous();
    auto up_c = up.contiguous();
    auto orig_dtype = gate_c.scalar_type();

    // Widen to f32 for compute
    go_c = ensure_float32(go_c);
    gate_c = ensure_float32(gate_c);
    up_c = ensure_float32(up_c);

    auto grad_gate = at::empty_like(gate_c);
    auto grad_up = at::empty_like(up_c);
    uint32_t numel = static_cast<uint32_t>(gate_c.numel());
    if (numel == 0) return {cast_from_float32(grad_gate, orig_dtype), cast_from_float32(grad_up, orig_dtype)};

    struct { uint32_t numel; } params{numel};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("activation_swiglu_backward_fwd",
                    shaders::activation_swiglu_backward_fwd,
                    shaders::activation_swiglu_backward_fwd_size,
                    {go_c, gate_c, up_c, grad_gate, grad_up},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return {cast_from_float32(grad_gate, orig_dtype), cast_from_float32(grad_up, orig_dtype)};
}

}} // namespace torch_vulkan::ops
