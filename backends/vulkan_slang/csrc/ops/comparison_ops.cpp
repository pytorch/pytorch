#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

struct ElementwiseParams {
    uint32_t numel;
};

// Comparison ops output bool tensors on CPU, but we compute in float
// and convert. PyTorch comparison ops return Bool dtype.
static at::Tensor comparison_op(
    const at::Tensor& self,
    const at::Tensor& other,
    const std::string& key,
    const uint32_t* spirv,
    size_t spirv_size) {

    // Ensure both tensors are on the same device
    auto other_dev = other;
    if (other_dev.device() != self.device())
        other_dev = other_dev.to(self.device());

    // Broadcast to common shape
    auto output_shape = at::infer_size(self.sizes(), other_dev.sizes());
    auto self_c = self.expand(output_shape).contiguous().to(at::kFloat);
    auto other_c = other_dev.expand(output_shape).contiguous().to(at::kFloat);

    // We compute in float, then interpret as bool
    auto float_output = at::empty(output_shape, self_c.options());
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    if (numel == 0) return float_output.to(at::kBool);

    ElementwiseParams params{numel};
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader(key, spirv, spirv_size,
                    {self_c, other_c, float_output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Convert 0.0/1.0 float output to bool on GPU (no CPU roundtrip)
    auto bool_output = at::empty(output_shape, self.options().dtype(at::kBool));
    uint32_t num_packed = (numel + 3) / 4;
    struct { uint32_t numel; uint32_t num_packed; } f2b_params{numel, num_packed};
    dispatch_shader("copy_float_to_bool_fwd",
                    shaders::copy_float_to_bool_fwd, shaders::copy_float_to_bool_fwd_size,
                    {float_output, bool_output},
                    (num_packed + 255) / 256, 1, 1,
                    &f2b_params, sizeof(f2b_params));
    return bool_output;
}

at::Tensor vulkan_eq(const at::Tensor& self, const at::Tensor& other) {
    return comparison_op(self, other,
        "comparison_eq_fwd", shaders::comparison_eq_fwd, shaders::comparison_eq_fwd_size);
}

at::Tensor vulkan_ne(const at::Tensor& self, const at::Tensor& other) {
    return comparison_op(self, other,
        "comparison_ne_fwd", shaders::comparison_ne_fwd, shaders::comparison_ne_fwd_size);
}

at::Tensor vulkan_lt(const at::Tensor& self, const at::Tensor& other) {
    return comparison_op(self, other,
        "comparison_lt_fwd", shaders::comparison_lt_fwd, shaders::comparison_lt_fwd_size);
}

at::Tensor vulkan_gt(const at::Tensor& self, const at::Tensor& other) {
    return comparison_op(self, other,
        "comparison_gt_fwd", shaders::comparison_gt_fwd, shaders::comparison_gt_fwd_size);
}

at::Tensor vulkan_le(const at::Tensor& self, const at::Tensor& other) {
    return comparison_op(self, other,
        "comparison_le_fwd", shaders::comparison_le_fwd, shaders::comparison_le_fwd_size);
}

at::Tensor vulkan_ge(const at::Tensor& self, const at::Tensor& other) {
    return comparison_op(self, other,
        "comparison_ge_fwd", shaders::comparison_ge_fwd, shaders::comparison_ge_fwd_size);
}

at::Tensor vulkan_where(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {
    check_supported_float(self, "where");
    auto orig_dtype = self.scalar_type();

    // Broadcast all three to common shape
    auto output_shape = at::infer_size(condition.sizes(), at::infer_size(self.sizes(), other.sizes()));
    auto cond_float = condition.expand(output_shape).contiguous().to(at::kFloat);
    auto self_c = ensure_float32(self.expand(output_shape).contiguous());
    auto other_c = ensure_float32(other.expand(output_shape).contiguous());

    auto output = at::empty(output_shape, self_c.options());
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    if (numel == 0) return cast_from_float32(output, orig_dtype);

    ElementwiseParams params{numel};
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("comparison_where_fwd",
                    shaders::comparison_where_fwd, shaders::comparison_where_fwd_size,
                    {cond_float, self_c, other_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── Scalar comparison variants (dedicated shaders — no temp tensor) ──
static at::Tensor scalar_comparison_op(
    const at::Tensor& self,
    float scalar_val,
    const std::string& key,
    const uint32_t* spirv,
    size_t spirv_size) {

    auto self_c = self.contiguous().to(at::kFloat);
    auto float_output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return float_output.to(at::kBool);

    struct { uint32_t numel; float scalar; } params{numel, scalar_val};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader(key, spirv, spirv_size,
                    {self_c, float_output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Convert float 0.0/1.0 to bool on GPU (no CPU roundtrip)
    auto bool_output = at::empty(self_c.sizes(), self.options().dtype(at::kBool));
    uint32_t num_packed = (numel + 3) / 4;
    struct { uint32_t numel; uint32_t num_packed; } f2b_params{numel, num_packed};
    dispatch_shader("copy_float_to_bool_fwd",
                    shaders::copy_float_to_bool_fwd, shaders::copy_float_to_bool_fwd_size,
                    {float_output, bool_output},
                    (num_packed + 255) / 256, 1, 1,
                    &f2b_params, sizeof(f2b_params));
    return bool_output;
}

at::Tensor vulkan_eq_scalar(const at::Tensor& self, const at::Scalar& other) {
    return scalar_comparison_op(self, other.toFloat(),
        "comparison_eq_scalar_fwd", shaders::comparison_eq_scalar_fwd, shaders::comparison_eq_scalar_fwd_size);
}
at::Tensor vulkan_ne_scalar(const at::Tensor& self, const at::Scalar& other) {
    return scalar_comparison_op(self, other.toFloat(),
        "comparison_ne_scalar_fwd", shaders::comparison_ne_scalar_fwd, shaders::comparison_ne_scalar_fwd_size);
}
at::Tensor vulkan_lt_scalar(const at::Tensor& self, const at::Scalar& other) {
    return scalar_comparison_op(self, other.toFloat(),
        "comparison_lt_scalar_fwd", shaders::comparison_lt_scalar_fwd, shaders::comparison_lt_scalar_fwd_size);
}
at::Tensor vulkan_gt_scalar(const at::Tensor& self, const at::Scalar& other) {
    return scalar_comparison_op(self, other.toFloat(),
        "comparison_gt_scalar_fwd", shaders::comparison_gt_scalar_fwd, shaders::comparison_gt_scalar_fwd_size);
}
at::Tensor vulkan_le_scalar(const at::Tensor& self, const at::Scalar& other) {
    return scalar_comparison_op(self, other.toFloat(),
        "comparison_le_scalar_fwd", shaders::comparison_le_scalar_fwd, shaders::comparison_le_scalar_fwd_size);
}
at::Tensor vulkan_ge_scalar(const at::Tensor& self, const at::Scalar& other) {
    return scalar_comparison_op(self, other.toFloat(),
        "comparison_ge_scalar_fwd", shaders::comparison_ge_scalar_fwd, shaders::comparison_ge_scalar_fwd_size);
}

}} // namespace torch_vulkan::ops
