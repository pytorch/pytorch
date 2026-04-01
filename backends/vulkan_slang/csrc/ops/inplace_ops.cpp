#include "ops.h"
#include "dispatch.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

at::Tensor& vulkan_add_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    // Fast path: true in-place add for same-shape f32 tensors — 1 dispatch instead of 2.
    // Skips intermediate allocation and dispatch_copy_buffer.
    if (self.scalar_type() == at::kFloat &&
        other.scalar_type() == at::kFloat &&
        self.is_contiguous() && other.is_contiguous() &&
        self.sizes() == other.sizes()) {
        float alpha_f = static_cast<float>(alpha.toFloat());
        uint32_t numel = static_cast<uint32_t>(self.numel());
        if (numel > 0) {
            struct { uint32_t numel; float alpha; } params{numel, alpha_f};
            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("binary_add_inplace_fwd",
                            shaders::binary_add_inplace_fwd, shaders::binary_add_inplace_fwd_size,
                            {self, other},
                            workgroups, 1, 1,
                            &params, sizeof(params), 2);
        }
        return self;
    }
    // Fallback: compute into new tensor then copy back (handles dtype conversion, broadcast, etc.)
    auto result = vulkan_add(self, other, alpha);
    dispatch_copy_buffer(result, self);
    return self;
}

at::Tensor& vulkan_sub_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    if (self.scalar_type() == at::kFloat &&
        other.scalar_type() == at::kFloat &&
        self.is_contiguous() && other.is_contiguous() &&
        self.sizes() == other.sizes()) {
        float alpha_f = static_cast<float>(alpha.toFloat());
        uint32_t numel = static_cast<uint32_t>(self.numel());
        if (numel > 0) {
            struct { uint32_t numel; float alpha; } params{numel, alpha_f};
            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("binary_sub_inplace_fwd",
                            shaders::binary_sub_inplace_fwd, shaders::binary_sub_inplace_fwd_size,
                            {self, other},
                            workgroups, 1, 1,
                            &params, sizeof(params), 2);
        }
        return self;
    }
    auto result = vulkan_sub(self, other, alpha);
    dispatch_copy_buffer(result, self);
    return self;
}

at::Tensor& vulkan_mul_(at::Tensor& self, const at::Tensor& other) {
    if (self.scalar_type() == at::kFloat &&
        other.scalar_type() == at::kFloat &&
        self.is_contiguous() && other.is_contiguous() &&
        self.sizes() == other.sizes()) {
        uint32_t numel = static_cast<uint32_t>(self.numel());
        if (numel > 0) {
            struct { uint32_t numel; } params{numel};
            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("binary_mul_inplace_fwd",
                            shaders::binary_mul_inplace_fwd, shaders::binary_mul_inplace_fwd_size,
                            {self, other},
                            workgroups, 1, 1,
                            &params, sizeof(params), 2);
        }
        return self;
    }
    auto result = vulkan_mul(self, other);
    dispatch_copy_buffer(result, self);
    return self;
}

at::Tensor& vulkan_div_(at::Tensor& self, const at::Tensor& other) {
    if (self.scalar_type() == at::kFloat &&
        other.scalar_type() == at::kFloat &&
        self.is_contiguous() && other.is_contiguous() &&
        self.sizes() == other.sizes()) {
        uint32_t numel = static_cast<uint32_t>(self.numel());
        if (numel > 0) {
            struct { uint32_t numel; } params{numel};
            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("binary_div_inplace_fwd",
                            shaders::binary_div_inplace_fwd, shaders::binary_div_inplace_fwd_size,
                            {self, other},
                            workgroups, 1, 1,
                            &params, sizeof(params), 2);
        }
        return self;
    }
    auto result = vulkan_div(self, other);
    dispatch_copy_buffer(result, self);
    return self;
}

// ── Scalar in-place variants ─────────────────────────────────────
at::Tensor& vulkan_add_scalar_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    if (self.scalar_type() == at::kFloat && self.is_contiguous()) {
        float scalar_f = static_cast<float>(other.toFloat());
        float alpha_f  = static_cast<float>(alpha.toFloat());
        uint32_t numel = static_cast<uint32_t>(self.numel());
        if (numel > 0) {
            struct { uint32_t numel; float scalar; float alpha; } params{numel, scalar_f, alpha_f};
            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("binary_add_scalar_inplace_fwd",
                            shaders::binary_add_scalar_inplace_fwd, shaders::binary_add_scalar_inplace_fwd_size,
                            {self},
                            workgroups, 1, 1,
                            &params, sizeof(params));
        }
        return self;
    }
    auto result = vulkan_add_scalar(self, other, alpha);
    dispatch_copy_buffer(result, self);
    return self;
}

at::Tensor& vulkan_sub_scalar_(at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
    if (self.scalar_type() == at::kFloat && self.is_contiguous()) {
        float scalar_f = static_cast<float>(other.toFloat());
        float alpha_f  = static_cast<float>(alpha.toFloat());
        uint32_t numel = static_cast<uint32_t>(self.numel());
        if (numel > 0) {
            struct { uint32_t numel; float scalar; float alpha; } params{numel, scalar_f, alpha_f};
            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("binary_sub_scalar_inplace_fwd",
                            shaders::binary_sub_scalar_inplace_fwd, shaders::binary_sub_scalar_inplace_fwd_size,
                            {self},
                            workgroups, 1, 1,
                            &params, sizeof(params));
        }
        return self;
    }
    auto result = vulkan_sub_scalar(self, other, alpha);
    dispatch_copy_buffer(result, self);
    return self;
}

at::Tensor& vulkan_mul_scalar_(at::Tensor& self, const at::Scalar& other) {
    if (self.scalar_type() == at::kFloat && self.is_contiguous()) {
        float scalar_f = static_cast<float>(other.toFloat());
        uint32_t numel = static_cast<uint32_t>(self.numel());
        if (numel > 0) {
            struct { uint32_t numel; float scalar; } params{numel, scalar_f};
            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("binary_mul_scalar_inplace_fwd",
                            shaders::binary_mul_scalar_inplace_fwd, shaders::binary_mul_scalar_inplace_fwd_size,
                            {self},
                            workgroups, 1, 1,
                            &params, sizeof(params));
        }
        return self;
    }
    auto result = vulkan_mul_scalar(self, other);
    dispatch_copy_buffer(result, self);
    return self;
}

at::Tensor& vulkan_div_scalar_(at::Tensor& self, const at::Scalar& other) {
    if (self.scalar_type() == at::kFloat && self.is_contiguous()) {
        float scalar_f = static_cast<float>(other.toFloat());
        uint32_t numel = static_cast<uint32_t>(self.numel());
        if (numel > 0) {
            struct { uint32_t numel; float scalar; } params{numel, scalar_f};
            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("binary_div_scalar_inplace_fwd",
                            shaders::binary_div_scalar_inplace_fwd, shaders::binary_div_scalar_inplace_fwd_size,
                            {self},
                            workgroups, 1, 1,
                            &params, sizeof(params));
        }
        return self;
    }
    auto result = vulkan_div_scalar(self, other);
    dispatch_copy_buffer(result, self);
    return self;
}

}} // namespace torch_vulkan::ops
