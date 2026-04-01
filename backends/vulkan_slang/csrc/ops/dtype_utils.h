#pragma once

#include <torch/torch.h>

namespace torch_vulkan { namespace ops {

// ── Dtype support utilities ──────────────────────────────────────
//
// All Slang shaders operate on StructuredBuffer<float> (float32).
// To support float16 and bfloat16, we use a "widen → compute → narrow"
// pattern: cast input tensors to float32 before shader dispatch, then
// cast the output back to the original dtype.
//
// The cast shaders use uint32 bit manipulation (no VK_KHR_shader_float16_int8
// needed), so they work on SwiftShader and all Vulkan devices.

// Returns true if dtype is a floating point type we can handle (f32, f16, bf16, f64, fp8).
inline bool is_supported_float(at::ScalarType dtype) {
    return dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16
        || dtype == at::kDouble
        || dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2;
}

// Returns true if dtype can be converted to float for computation.
inline bool is_numeric_type(at::ScalarType dtype) {
    return is_supported_float(dtype) || dtype == at::kLong || dtype == at::kInt
        || dtype == at::kShort || dtype == at::kByte || dtype == at::kBool;
}

// Returns true if dtype is an FP8 type.
inline bool is_fp8(at::ScalarType dtype) {
    return dtype == at::kFloat8_e4m3fn || dtype == at::kFloat8_e5m2;
}

// TORCH_CHECK that a tensor has a numeric dtype that can be processed.
inline void check_supported_float(const at::Tensor& t, const char* op_name) {
    TORCH_CHECK(is_numeric_type(t.scalar_type()),
                "Vulkan ", op_name, ": unsupported dtype=", t.scalar_type());
}

// Cast a Vulkan tensor to float32 if it's f16 or bf16.
// If already float32, returns the input tensor as-is (zero cost).
// Uses GPU cast shaders for the conversion.
at::Tensor ensure_float32(const at::Tensor& t);

// Cast a float32 Vulkan tensor to the target dtype.
// If target is already float32, returns the input tensor as-is (zero cost).
// Uses GPU cast shaders for the conversion.
at::Tensor cast_from_float32(const at::Tensor& t, at::ScalarType target_dtype);

}} // namespace torch_vulkan::ops
