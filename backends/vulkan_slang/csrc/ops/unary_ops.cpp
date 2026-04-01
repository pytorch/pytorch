#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

struct ElementwiseParams {
    uint32_t numel;
};

static at::Tensor unary_op(
    const at::Tensor& self,
    const std::string& key,
    const uint32_t* spirv,
    size_t spirv_size) {

    auto self_c = self.contiguous();
    check_supported_float(self_c, "unary op");
    auto orig_dtype = self_c.scalar_type();

    // Widen to f32 if needed (f16/bf16 -> f32)
    auto self_f32 = ensure_float32(self_c);
    auto output = at::empty(self_f32.sizes(), self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());

    if (numel == 0) return cast_from_float32(output, orig_dtype);

    ElementwiseParams params{numel};
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader(key, spirv, spirv_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    // Narrow back to original dtype
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_neg(const at::Tensor& self) {
    return unary_op(self, "unary_neg_fwd", shaders::unary_neg_fwd, shaders::unary_neg_fwd_size);
}

at::Tensor vulkan_abs(const at::Tensor& self) {
    return unary_op(self, "unary_abs_fwd", shaders::unary_abs_fwd, shaders::unary_abs_fwd_size);
}

at::Tensor vulkan_exp(const at::Tensor& self) {
    return unary_op(self, "unary_exp_fwd", shaders::unary_exp_fwd, shaders::unary_exp_fwd_size);
}

at::Tensor vulkan_log(const at::Tensor& self) {
    return unary_op(self, "unary_log_fwd", shaders::unary_log_fwd, shaders::unary_log_fwd_size);
}

at::Tensor vulkan_sqrt(const at::Tensor& self) {
    return unary_op(self, "unary_sqrt_fwd", shaders::unary_sqrt_fwd, shaders::unary_sqrt_fwd_size);
}

at::Tensor vulkan_rsqrt(const at::Tensor& self) {
    return unary_op(self, "unary_rsqrt_fwd", shaders::unary_rsqrt_fwd, shaders::unary_rsqrt_fwd_size);
}

at::Tensor vulkan_ceil(const at::Tensor& self) {
    return unary_op(self, "unary_ceil_fwd", shaders::unary_ceil_fwd, shaders::unary_ceil_fwd_size);
}

at::Tensor vulkan_floor(const at::Tensor& self) {
    return unary_op(self, "unary_floor_fwd", shaders::unary_floor_fwd, shaders::unary_floor_fwd_size);
}

at::Tensor vulkan_round(const at::Tensor& self) {
    return unary_op(self, "unary_round_fwd", shaders::unary_round_fwd, shaders::unary_round_fwd_size);
}

at::Tensor vulkan_sign(const at::Tensor& self) {
    return unary_op(self, "unary_sign_fwd", shaders::unary_sign_fwd, shaders::unary_sign_fwd_size);
}

// Additional math ops
at::Tensor vulkan_tan(const at::Tensor& self) {
    return unary_op(self, "unary_tan_fwd", shaders::unary_tan_fwd, shaders::unary_tan_fwd_size);
}

at::Tensor vulkan_atan(const at::Tensor& self) {
    return unary_op(self, "unary_atan_fwd", shaders::unary_atan_fwd, shaders::unary_atan_fwd_size);
}

at::Tensor vulkan_log2(const at::Tensor& self) {
    return unary_op(self, "unary_log2_fwd", shaders::unary_log2_fwd, shaders::unary_log2_fwd_size);
}

at::Tensor vulkan_log10(const at::Tensor& self) {
    return unary_op(self, "unary_log10_fwd", shaders::unary_log10_fwd, shaders::unary_log10_fwd_size);
}

at::Tensor vulkan_log1p(const at::Tensor& self) {
    return unary_op(self, "unary_log1p_fwd", shaders::unary_log1p_fwd, shaders::unary_log1p_fwd_size);
}

// Check ops — output bool tensors
at::Tensor vulkan_isnan(const at::Tensor& self) {
    auto float_result = unary_op(self, "unary_isnan_fwd", shaders::unary_isnan_fwd, shaders::unary_isnan_fwd_size);
    return float_result.to(at::kBool);
}

at::Tensor vulkan_isinf(const at::Tensor& self) {
    auto float_result = unary_op(self, "unary_isinf_fwd", shaders::unary_isinf_fwd, shaders::unary_isinf_fwd_size);
    return float_result.to(at::kBool);
}

at::Tensor vulkan_isfinite(const at::Tensor& self) {
    auto float_result = unary_op(self, "unary_isfinite_fwd", shaders::unary_isfinite_fwd, shaders::unary_isfinite_fwd_size);
    return float_result.to(at::kBool);
}

}} // namespace torch_vulkan::ops
