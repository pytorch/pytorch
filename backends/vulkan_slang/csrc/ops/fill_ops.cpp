#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// Push constant layout matching fill.slang Params struct
struct FillParams {
    float value;
    uint32_t numel;
};

at::Tensor& vulkan_fill_scalar_gpu(at::Tensor& self, const at::Scalar& value) {
    check_supported_float(self, "fill_");
    auto orig_dtype = self.scalar_type();

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    if (orig_dtype == at::kFloat) {
        // Fill directly into the f32 buffer
        FillParams params{value.toFloat(), numel};
        uint32_t workgroups = (numel + 255) / 256;

        dispatch_shader("copy_fill_fwd",
                        shaders::copy_fill_fwd, shaders::copy_fill_fwd_size,
                        {self},
                        workgroups, 1, 1,
                        &params, sizeof(params));
    } else {
        // f16/bf16: fill a temporary f32 buffer, cast, then copy back
        auto tmp = at::empty(self.sizes(), self.options().dtype(at::kFloat));

        FillParams params{value.toFloat(), numel};
        uint32_t workgroups = (numel + 255) / 256;

        dispatch_shader("copy_fill_fwd",
                        shaders::copy_fill_fwd, shaders::copy_fill_fwd_size,
                        {tmp},
                        workgroups, 1, 1,
                        &params, sizeof(params));

        auto casted = cast_from_float32(tmp, orig_dtype);
        self.copy_(casted);
    }
    return self;
}

at::Tensor vulkan_clone(const at::Tensor& self, std::optional<at::MemoryFormat> memory_format) {
    // For non-contiguous tensors (e.g., zero-copy transposed views from vulkan_t),
    // use dispatch_strided_copy directly to avoid infinite recursion:
    //   contiguous() → clone() → contiguous() → ...
    auto output = at::empty(self.sizes(), self.options());

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return output;

    if (!self.is_contiguous()) {
        // Non-contiguous: use strided copy shader for float32 (handles arbitrary strides, up to 5D).
        if (self.scalar_type() == at::kFloat) {
            dispatch_strided_copy(self, output);
            return output;
        }
        // Non-float32 non-contiguous: should not occur in practice (vulkan_t returns
        // contiguous GPU copy for non-float32 types). Raise TORCH_CHECK to surface bugs.
        TORCH_CHECK(false, "vulkan_clone: non-contiguous non-float32 tensor not supported "
                    "(dtype=", self.scalar_type(), ")");
    }

    if (self.scalar_type() == at::kFloat) {
        // Use GPU copy shader for float32
        struct { uint32_t numel; } params{numel};
        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_copy_fwd",
                        shaders::copy_copy_fwd, shaders::copy_copy_fwd_size,
                        {self, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
    } else {
        // For f16/bf16/other dtypes, use copy_ (handles raw byte transfer)
        output.copy_(self);
    }
    return output;
}

}} // namespace torch_vulkan::ops
