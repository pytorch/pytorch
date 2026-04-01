#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// Push constant layout for element-wise binary ops (must match shader Params)
struct ElementwiseParams {
    uint32_t numel;
};

static at::Tensor binary_op(
    const at::Tensor& self,
    const at::Tensor& other,
    const std::string& key,
    const uint32_t* spirv,
    size_t spirv_size,
    const uint32_t* bcast_spirv = nullptr,
    size_t bcast_spirv_size = 0) {

    check_supported_float(self, "binary op");

    // Determine output dtype (promote: bf16+f16->f32, same->same, mixed with f32->f32)
    auto orig_dtype = self.scalar_type();

    // Ensure both tensors are on the same device
    auto other_dev = other;
    if (other_dev.device() != self.device())
        other_dev = other_dev.to(self.device());

    // Widen both to f32 BEFORE broadcast/expand/contiguous
    // (8-byte dtypes like Double corrupt data during expand on Vulkan's 4-byte buffers)
    auto self_f32 = ensure_float32(self);
    auto other_f32 = ensure_float32(other_dev);

    // Broadcast to common shape — skip expand when shapes already match
    at::Tensor self_c, other_c;
    bool same_shape = self_f32.sizes().equals(other_f32.sizes());
    if (same_shape) {
        self_c = self_f32;
        other_c = other_f32;
    } else if (bcast_spirv != nullptr) {
        // Fast path: use broadcast shader to avoid explicit expand copy.
        // Handles the case where one tensor's elements are a suffix-repetition
        // of the output: e.g. [B,H,S,S] + [S,S] or [B,S,D] + [D].
        // Requires: smaller_numel divides evenly into larger_numel.
        auto bcast_shape = at::infer_size(self_f32.sizes(), other_f32.sizes());
        int64_t numel_output = 1;
        for (auto s : bcast_shape) numel_output *= s;
        int64_t numel_self = self_f32.numel();
        int64_t numel_other = other_f32.numel();

        // Fast path: use broadcast shader for the common "leading-dim broadcast" pattern.
        // Valid only when the small tensor's shape is a suffix of the output shape with
        // NO expansion in the trailing dims. That means:
        //   out[k] = big[k] + small[k % numel_small]
        // Examples that work:
        //   [B,H,S,S] + [S,S]: small=[S,S], last 2 dims of output match exactly ✓
        //   [B,S] + [S]: small=[S], last dim of output matches ✓
        //   [B,D]+[1,D] also works since b's effective size = D
        // Examples that do NOT work:
        //   [8,16] + [8,1]: last dim of small is 1 (broadcast to 16) — modulo gives wrong index ✗
        //   [8,16] + [8]: small doesn't align with trailing dims ✗
        //
        // Check: is the small tensor's shape a contiguous suffix of the output shape?
        // After left-padding with 1s, small dims align with the output trailing dims.
        // We need: for every aligned dim i, small_size[i] == out_size[i] (no trailing expansion).
        auto is_suffix_broadcast = [&](const at::Tensor& big_t, const at::Tensor& small_t) -> bool {
            if (!big_t.is_contiguous() || !small_t.is_contiguous()) return false;
            if (big_t.numel() != numel_output) return false;
            // Effective small shape (after left-padding 1s to match output ndim)
            int64_t big_ndim = static_cast<int64_t>(bcast_shape.size());
            int64_t small_ndim = small_t.dim();
            int64_t pad = big_ndim - small_ndim;
            for (int64_t i = 0; i < small_ndim; i++) {
                int64_t small_size = small_t.size(i);
                int64_t out_size = bcast_shape[pad + i];
                if (small_size != 1 && small_size != out_size) return false;
                // If the small dim is 1 but bcast_shape is > 1, this is leading-1 broadcast
                // which is invalid for the suffix modulo pattern (e.g. [8,1] col broadcast)
                if (small_size == 1 && out_size != 1) return false;
            }
            return true;
        };

        bool self_is_big = is_suffix_broadcast(self_f32, other_f32);
        bool other_is_big = is_suffix_broadcast(other_f32, self_f32);

        if (self_is_big) {
            auto output = at::empty({numel_output}, self_f32.options());
            struct { uint32_t numel_a; uint32_t numel_b; } bcast_params{
                static_cast<uint32_t>(numel_output), static_cast<uint32_t>(numel_other)};
            uint32_t workgroups = (static_cast<uint32_t>(numel_output) + 255) / 256;
            dispatch_shader("binary_add_broadcast_fwd", bcast_spirv, bcast_spirv_size,
                            {self_f32, other_f32, output},
                            workgroups, 1, 1,
                            &bcast_params, sizeof(bcast_params));
            return cast_from_float32(output.reshape(bcast_shape), orig_dtype);
        } else if (other_is_big) {
            auto output = at::empty({numel_output}, other_f32.options());
            struct { uint32_t numel_a; uint32_t numel_b; } bcast_params{
                static_cast<uint32_t>(numel_output), static_cast<uint32_t>(numel_self)};
            uint32_t workgroups = (static_cast<uint32_t>(numel_output) + 255) / 256;
            dispatch_shader("binary_add_broadcast_fwd", bcast_spirv, bcast_spirv_size,
                            {other_f32, self_f32, output},
                            workgroups, 1, 1,
                            &bcast_params, sizeof(bcast_params));
            return cast_from_float32(output.reshape(bcast_shape), orig_dtype);
        }
        // Fallback if broadcast conditions not met
        self_c = self_f32.expand(bcast_shape).contiguous();
        other_c = other_f32.expand(bcast_shape).contiguous();
    } else {
        auto bcast_shape = at::infer_size(self_f32.sizes(), other_f32.sizes());
        self_c = self_f32.expand(bcast_shape).contiguous();
        other_c = other_f32.expand(bcast_shape).contiguous();
    }

    auto output = at::empty(self_c.sizes(), self_c.options());
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    if (numel == 0) return cast_from_float32(output, orig_dtype);

    ElementwiseParams params{numel};
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader(key, spirv, spirv_size,
                    {self_c, other_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    // Shortcut: CPU scalar tensor → use scalar add
    if (other.dim() == 0 && other.device().type() == c10::DeviceType::CPU && alpha.toDouble() == 1.0) {
        return vulkan_add_scalar(self, other.item(), alpha);
    }
    if (self.dim() == 0 && self.device().type() == c10::DeviceType::CPU && alpha.toDouble() == 1.0) {
        return vulkan_add_scalar(other, self.item(), alpha);
    }
    if (alpha.toDouble() != 1.0) {
        // add(self, other, alpha) = self + alpha * other
        auto scaled = vulkan_mul_scalar(other, alpha);
        return binary_op(self, scaled,
                         "binary_add_fwd", shaders::binary_add_fwd, shaders::binary_add_fwd_size,
                         shaders::binary_add_broadcast_fwd, shaders::binary_add_broadcast_fwd_size);
    }
    return binary_op(self, other,
                     "binary_add_fwd", shaders::binary_add_fwd, shaders::binary_add_fwd_size,
                     shaders::binary_add_broadcast_fwd, shaders::binary_add_broadcast_fwd_size);
}

at::Tensor vulkan_sub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    // Shortcut: CPU scalar tensor → use scalar sub
    if (other.dim() == 0 && other.device().type() == c10::DeviceType::CPU && alpha.toDouble() == 1.0) {
        return vulkan_sub_scalar(self, other.item(), alpha);
    }
    if (alpha.toDouble() != 1.0) {
        auto scaled = vulkan_mul_scalar(other, alpha);
        return binary_op(self, scaled,
                         "binary_sub_fwd", shaders::binary_sub_fwd, shaders::binary_sub_fwd_size);
    }
    return binary_op(self, other,
                     "binary_sub_fwd", shaders::binary_sub_fwd, shaders::binary_sub_fwd_size);
}

at::Tensor vulkan_mul(const at::Tensor& self, const at::Tensor& other) {
    // Shortcut: if either operand is a CPU scalar tensor, use the scalar op
    // to avoid device-copy + expand + broadcast overhead (3 dispatches → 1).
    if (other.dim() == 0 && other.device().type() == c10::DeviceType::CPU) {
        return vulkan_mul_scalar(self, other.item());
    }
    if (self.dim() == 0 && self.device().type() == c10::DeviceType::CPU) {
        return vulkan_mul_scalar(other, self.item());
    }
    return binary_op(self, other,
                     "binary_mul_fwd", shaders::binary_mul_fwd, shaders::binary_mul_fwd_size);
}

at::Tensor vulkan_div(const at::Tensor& self, const at::Tensor& other) {
    // Shortcut: CPU scalar tensor → use scalar div
    if (other.dim() == 0 && other.device().type() == c10::DeviceType::CPU) {
        return vulkan_div_scalar(self, other.item());
    }
    return binary_op(self, other,
                     "binary_div_fwd", shaders::binary_div_fwd, shaders::binary_div_fwd_size);
}

at::Tensor vulkan_pow(const at::Tensor& self, const at::Tensor& exponent) {
    return binary_op(self, exponent,
                     "binary_pow_fwd", shaders::binary_pow_fwd, shaders::binary_pow_fwd_size);
}

at::Tensor vulkan_pow_scalar(const at::Tensor& self, const at::Scalar& exponent) {
    // For common exponents, use simpler operations
    float exp_f = exponent.toFloat();
    if (exp_f == 2.0f) return vulkan_mul(self, self);
    if (exp_f == 0.5f) return at::sqrt(self);
    // General case: create scalar tensor for pow shader
    auto exp_tensor = at::full_like(self, exponent);
    return vulkan_pow(self, exp_tensor);
}

// ── Scalar variants (dedicated shaders — no temp tensor allocation) ──
at::Tensor vulkan_add_scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "add.Scalar");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);
    auto output = at::empty_like(self_f32);
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    struct { uint32_t numel; float scalar; float alpha; } params{
        numel, other.toFloat(), alpha.toFloat()};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("binary_add_scalar_fwd",
                    shaders::binary_add_scalar_fwd, shaders::binary_add_scalar_fwd_size,
                    {self_f32, output}, workgroups, 1, 1, &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_sub_scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "sub.Scalar");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);
    auto output = at::empty_like(self_f32);
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    struct { uint32_t numel; float scalar; float alpha; } params{
        numel, other.toFloat(), alpha.toFloat()};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("binary_sub_scalar_fwd",
                    shaders::binary_sub_scalar_fwd, shaders::binary_sub_scalar_fwd_size,
                    {self_f32, output}, workgroups, 1, 1, &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// scalar - tensor[i]  (reverse subtract: e.g., 1 - sigmoid)
at::Tensor vulkan_rsub_scalar(const at::Tensor& self, const at::Scalar& scalar) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "rsub_scalar");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);
    auto output = at::empty_like(self_f32);
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    struct { uint32_t numel; float scalar; } params{numel, scalar.toFloat()};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("binary_rsub_scalar_fwd",
                    shaders::binary_rsub_scalar_fwd, shaders::binary_rsub_scalar_fwd_size,
                    {self_f32, output}, workgroups, 1, 1, &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_mul_scalar(const at::Tensor& self, const at::Scalar& other) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "mul.Scalar");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);
    auto output = at::empty_like(self_f32);
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    struct { uint32_t numel; float scalar; } params{numel, other.toFloat()};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("binary_mul_scalar_fwd",
                    shaders::binary_mul_scalar_fwd, shaders::binary_mul_scalar_fwd_size,
                    {self_f32, output}, workgroups, 1, 1, &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_div_scalar(const at::Tensor& self, const at::Scalar& other) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "div.Scalar");
    auto orig_dtype = self_c.scalar_type();
    auto self_f32 = ensure_float32(self_c);
    auto output = at::empty_like(self_f32);
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    struct { uint32_t numel; float scalar; } params{numel, other.toFloat()};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("binary_div_scalar_fwd",
                    shaders::binary_div_scalar_fwd, shaders::binary_div_scalar_fwd_size,
                    {self_f32, output}, workgroups, 1, 1, &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_fmod(const at::Tensor& self, const at::Tensor& other) {
    return binary_op(self, other,
                     "binary_fmod_fwd", shaders::binary_fmod_fwd, shaders::binary_fmod_fwd_size);
}

at::Tensor vulkan_remainder(const at::Tensor& self, const at::Tensor& other) {
    return binary_op(self, other,
                     "binary_remainder_fwd", shaders::binary_remainder_fwd, shaders::binary_remainder_fwd_size);
}

at::Tensor vulkan_atan2(const at::Tensor& self, const at::Tensor& other) {
    return binary_op(self, other,
                     "binary_atan2_fwd", shaders::binary_atan2_fwd, shaders::binary_atan2_fwd_size);
}

}} // namespace torch_vulkan::ops
