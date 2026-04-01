#include "dtype_utils.h"
#include "dispatch.h"
#include "../generated/shaders.h"

namespace torch_vulkan { namespace ops {

// Max workgroups in X dimension (Vulkan spec minimum guarantee)
static constexpr uint32_t kMaxWorkgroupsX = 65535;

at::Tensor ensure_float32(const at::Tensor& t) {
    if (t.scalar_type() == at::kFloat) return t;

    // Double/Long/Int: convert via CPU staging (no GPU shader needed)
    if (t.scalar_type() == at::kDouble || t.scalar_type() == at::kLong ||
        t.scalar_type() == at::kInt || t.scalar_type() == at::kBool) {
        auto cpu_f32 = t.cpu().to(at::kFloat);
        auto output = at::empty(t.sizes(), t.options().dtype(at::kFloat));
        output.copy_(cpu_f32);
        return output;
    }

    auto numel = t.numel();
    auto output = at::empty(t.sizes(),
        t.options().dtype(at::kFloat));
    if (numel == 0) return output;

    uint32_t n = static_cast<uint32_t>(numel);

    // Workgroup count for element-wise cast shaders (1 element per thread).
    // For large tensors (> 65535 × 256 elements), use 2D dispatch (wg_x=65535, wg_y=ceil(...))
    // so we can handle up to 65535 × 65535 × 256 ≈ 1T elements without CPU chunking.
    auto workgroups_for = [](uint32_t count, uint32_t& wg_x, uint32_t& wg_y) {
        uint32_t wg = (count + 255) / 256;
        if (wg <= kMaxWorkgroupsX) {
            wg_x = wg; wg_y = 1;
        } else {
            wg_x = kMaxWorkgroupsX;
            wg_y = (wg + kMaxWorkgroupsX - 1) / kMaxWorkgroupsX;
        }
    };

    uint32_t wg_x, wg_y;
    if (t.scalar_type() == at::kHalf) {
        struct { uint32_t numel; } params{n};
        workgroups_for(n, wg_x, wg_y);
        dispatch_shader("cast_f16_to_f32_fwd",
                        shaders::cast_f16_to_f32_fwd,
                        shaders::cast_f16_to_f32_fwd_size,
                        {t, output}, wg_x, wg_y, 1,
                        &params, sizeof(params));
    } else if (t.scalar_type() == at::kBFloat16) {
        struct { uint32_t numel; } params{n};
        workgroups_for(n, wg_x, wg_y);
        dispatch_shader("cast_bf16_to_f32_fwd",
                        shaders::cast_bf16_to_f32_fwd,
                        shaders::cast_bf16_to_f32_fwd_size,
                        {t, output}, wg_x, wg_y, 1,
                        &params, sizeof(params));
    } else if (t.scalar_type() == at::kFloat8_e4m3fn) {
        // e4m3fn -> f32: each thread reads one element from packed uint8 quads
        struct { uint32_t numel; } params{n};
        workgroups_for(n, wg_x, wg_y);
        dispatch_shader("cast_e4m3fn_to_f32_fwd",
                        shaders::cast_e4m3fn_to_f32_fwd,
                        shaders::cast_e4m3fn_to_f32_fwd_size,
                        {t, output}, wg_x, wg_y, 1,
                        &params, sizeof(params));
    } else if (t.scalar_type() == at::kFloat8_e5m2) {
        // e5m2 -> f32: each thread reads one element from packed uint8 quads
        struct { uint32_t numel; } params{n};
        workgroups_for(n, wg_x, wg_y);
        dispatch_shader("cast_e5m2_to_f32_fwd",
                        shaders::cast_e5m2_to_f32_fwd,
                        shaders::cast_e5m2_to_f32_fwd_size,
                        {t, output}, wg_x, wg_y, 1,
                        &params, sizeof(params));
    } else {
        // Fallback for other dtypes (Long, Int, Bool, etc.): convert via CPU
        auto cpu_f32 = t.cpu().to(at::kFloat);
        output.copy_(cpu_f32);
    }
    return output;
}

at::Tensor cast_from_float32(const at::Tensor& t, at::ScalarType target_dtype) {
    TORCH_INTERNAL_ASSERT(t.scalar_type() == at::kFloat);
    if (target_dtype == at::kFloat) return t;

    // For non-float types (Bool, Int, Long, etc.), convert via CPU staging
    if (target_dtype == at::kBool || target_dtype == at::kLong ||
        target_dtype == at::kInt || target_dtype == at::kShort ||
        target_dtype == at::kByte || target_dtype == at::kDouble) {
        auto cpu_cast = t.cpu().to(target_dtype);
        auto output = at::empty(t.sizes(), t.options().dtype(target_dtype));
        output.copy_(cpu_cast);
        return output;
    }

    auto numel = t.numel();
    auto output = at::empty(t.sizes(),
        t.options().dtype(target_dtype));
    if (numel == 0) return output;

    uint32_t n = static_cast<uint32_t>(numel);

    // Workgroup count with 2D dispatch support for large tensors.
    // For f32->f16/bf16: each thread handles a PAIR, so use num_pairs for workgroup count.
    // For f32->fp8: each thread handles a QUAD, so use num_quads for workgroup count.
    // params.numel is always the full element count for bounds-checking.
    auto workgroups_for = [](uint32_t count, uint32_t& wg_x, uint32_t& wg_y) {
        uint32_t wg = (count + 255) / 256;
        if (wg <= kMaxWorkgroupsX) {
            wg_x = wg; wg_y = 1;
        } else {
            wg_x = kMaxWorkgroupsX;
            wg_y = (wg + kMaxWorkgroupsX - 1) / kMaxWorkgroupsX;
        }
    };

    uint32_t wg_x, wg_y;
    if (target_dtype == at::kHalf) {
        // f32 -> f16: each thread processes a PAIR of elements
        struct { uint32_t numel; } params{n};
        uint32_t num_pairs = (n + 1) / 2;
        workgroups_for(num_pairs, wg_x, wg_y);
        dispatch_shader("cast_f32_to_f16_fwd",
                        shaders::cast_f32_to_f16_fwd,
                        shaders::cast_f32_to_f16_fwd_size,
                        {t, output}, wg_x, wg_y, 1,
                        &params, sizeof(params));
    } else if (target_dtype == at::kBFloat16) {
        // f32 -> bf16: each thread processes a PAIR of elements
        struct { uint32_t numel; } params{n};
        uint32_t num_pairs = (n + 1) / 2;
        workgroups_for(num_pairs, wg_x, wg_y);
        dispatch_shader("cast_f32_to_bf16_fwd",
                        shaders::cast_f32_to_bf16_fwd,
                        shaders::cast_f32_to_bf16_fwd_size,
                        {t, output}, wg_x, wg_y, 1,
                        &params, sizeof(params));
    } else if (target_dtype == at::kFloat8_e4m3fn) {
        // f32 -> e4m3fn: each thread processes a QUAD of elements
        struct { uint32_t numel; } params{n};
        uint32_t num_quads = (n + 3) / 4;
        workgroups_for(num_quads, wg_x, wg_y);
        dispatch_shader("cast_f32_to_e4m3fn_fwd",
                        shaders::cast_f32_to_e4m3fn_fwd,
                        shaders::cast_f32_to_e4m3fn_fwd_size,
                        {t, output}, wg_x, wg_y, 1,
                        &params, sizeof(params));
    } else if (target_dtype == at::kFloat8_e5m2) {
        // f32 -> e5m2: each thread processes a QUAD of elements
        struct { uint32_t numel; } params{n};
        uint32_t num_quads = (n + 3) / 4;
        workgroups_for(num_quads, wg_x, wg_y);
        dispatch_shader("cast_f32_to_e5m2_fwd",
                        shaders::cast_f32_to_e5m2_fwd,
                        shaders::cast_f32_to_e5m2_fwd_size,
                        {t, output}, wg_x, wg_y, 1,
                        &params, sizeof(params));
    } else {
        TORCH_CHECK(false, "cast_from_float32: unsupported target dtype ", target_dtype);
    }
    return output;
}

}} // namespace torch_vulkan::ops
