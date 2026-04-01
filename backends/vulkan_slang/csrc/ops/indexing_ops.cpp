#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// ── Embedding ───────────────────────────────────────────────────
at::Tensor vulkan_embedding(const at::Tensor& weight, const at::Tensor& indices,
                            int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    auto weight_c = weight.contiguous();
    check_supported_float(weight_c, "embedding");
    auto orig_dtype = weight_c.scalar_type();

    int64_t num_indices = indices.numel();
    int64_t embedding_dim = weight.size(1);

    // Create output in the original dtype (no upcast needed for lookup)
    std::vector<int64_t> out_shape(indices.sizes().begin(), indices.sizes().end());
    out_shape.push_back(embedding_dim);

    // For f16/bf16 weights: use raw uint32 copy shader to avoid 2x memory upcast.
    // f16/bf16 each pack 2 values per uint32. f32 packs 1 per uint32.
    const bool is_packed16 = (orig_dtype == at::kHalf || orig_dtype == at::kBFloat16);
    if (is_packed16) {
        // Use embedding_raw shader: treat weight buffer as uint32 (2 f16/bf16 per uint32).
        // embedding_dim must be even (always true for transformer hidden sizes).
        TORCH_CHECK(embedding_dim % 2 == 0,
                    "embedding: f16/bf16 weight requires even embedding_dim, got ", embedding_dim);
        uint32_t dim_u32 = static_cast<uint32_t>(embedding_dim / 2);

        auto output = at::empty(out_shape, weight_c.options());  // bf16/f16 output
        if (num_indices == 0) return output;

        // Get indices as int32 on Vulkan (reinterpret as float buffer for dispatch).
        // The embedding_raw shader uses asint() on StructuredBuffer<float>, so int32
        // data in a float-typed buffer works correctly with no copy overhead.
        at::Tensor indices_vulkan;

        if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
            indices.scalar_type() == at::kLong) {
            auto indices_c = indices.contiguous();
            indices_vulkan = at::empty({num_indices}, weight_c.options());
            struct { uint32_t numel; } i2i_params{static_cast<uint32_t>(num_indices)};
            dispatch_shader("indexing_i64_to_i32_fwd",
                            shaders::indexing_i64_to_i32_fwd,
                            shaders::indexing_i64_to_i32_fwd_size,
                            {indices_c, indices_vulkan},
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
            indices_vulkan = at::Tensor(std::move(impl));
        } else {
            indices_vulkan = at::empty({num_indices}, weight_c.options());
            auto indices_cpu = indices.cpu().to(at::kInt).contiguous();
            auto& alloc = VulkanAllocator::instance();
            auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
            TORCH_CHECK(buf, "Failed to get Vulkan buffer for indices");
            buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));
        }

        struct { uint32_t num_indices; uint32_t embedding_dim_u32; } params{
            static_cast<uint32_t>(num_indices), dim_u32
        };
        uint32_t total_u32 = static_cast<uint32_t>(num_indices) * dim_u32;
        uint32_t workgroups = (total_u32 + 255) / 256;
        dispatch_shader("indexing_embedding_raw_fwd",
                        shaders::indexing_embedding_raw_fwd,
                        shaders::indexing_embedding_raw_fwd_size,
                        {weight_c, indices_vulkan, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return output;
    }

    // f32 path: standard upcast-lookup-downcast (original behavior)
    weight_c = ensure_float32(weight_c);

    auto output = at::empty(out_shape, weight_c.options());
    if (num_indices == 0) return output;

    // Get indices as int32 on Vulkan. Avoid CPU roundtrip when possible.
    // The embedding shader uses StructuredBuffer<float> for indices and reads them
    // via asint() — so we can pass int32 data in a float-typed buffer with no copy.
    at::Tensor indices_vulkan;

    if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
        indices.scalar_type() == at::kLong) {
        // Int64 on Vulkan: convert to int32 on GPU via shader
        // Int64 is stored as pairs of uint32 (little-endian), we extract low bits
        auto indices_c = indices.contiguous();
        indices_vulkan = at::empty({num_indices}, weight_c.options().dtype(at::kFloat));
        struct { uint32_t numel; } i2i_params{static_cast<uint32_t>(num_indices)};
        dispatch_shader("indexing_i64_to_i32_fwd",
                        shaders::indexing_i64_to_i32_fwd,
                        shaders::indexing_i64_to_i32_fwd_size,
                        {indices_c, indices_vulkan},
                        (static_cast<uint32_t>(num_indices) + 255) / 256, 1, 1,
                        &i2i_params, sizeof(i2i_params));
    } else if (indices.device().type() == c10::DeviceType::PrivateUse1 &&
               indices.scalar_type() == at::kInt) {
        // Int32 on Vulkan: reinterpret the buffer as float (same 4-byte layout, no copy).
        // The embedding shader uses asint(buffer[i]) to read indices, so passing int32
        // data in a float-typed binding works correctly.
        auto indices_c = indices.contiguous();
        auto impl = c10::make_intrusive<at::TensorImpl>(
            c10::Storage(indices_c.storage()),
            indices_c.key_set(),
            at::scalarTypeToTypeMeta(at::kFloat));
        std::vector<int64_t> sz2 = {num_indices}, st2 = {1};
        impl->set_sizes_and_strides(sz2, st2);
        impl->set_storage_offset(indices_c.storage_offset());
        indices_vulkan = at::Tensor(std::move(impl));
    } else {
        // CPU path: convert and upload
        indices_vulkan = at::empty({num_indices}, weight_c.options().dtype(at::kFloat));
        auto indices_cpu = indices.cpu().to(at::kInt).contiguous();
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
        TORCH_CHECK(buf, "Failed to get Vulkan buffer for indices");
        buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));
    }

    struct { uint32_t num_indices; uint32_t embedding_dim; } params{
        static_cast<uint32_t>(num_indices),
        static_cast<uint32_t>(embedding_dim)
    };

    uint32_t total = static_cast<uint32_t>(num_indices * embedding_dim);
    uint32_t workgroups = (total + 255) / 256;

    dispatch_shader("indexing_embedding_fwd",
                    shaders::indexing_embedding_fwd, shaders::indexing_embedding_fwd_size,
                    {weight_c, indices_vulkan, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return output;  // already f32 — no cast needed
}

// ── Index Select ────────────────────────────────────────────────
at::Tensor vulkan_index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "index_select");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());

    // For non-zero dims, transpose to make dim=0, then use GPU shader
    if (dim != 0) {
        auto transposed = self_c.movedim(dim, 0).contiguous();
        auto result = vulkan_index_select(transposed, 0, index);
        return cast_from_float32(result.movedim(0, dim).contiguous(), orig_dtype);
    }

    auto indices_cpu = index.cpu().to(at::kInt).contiguous();
    int64_t num_indices = indices_cpu.numel();
    int64_t slice_size = self_c.numel() / self_c.size(0);

    std::vector<int64_t> out_shape = {num_indices};
    for (int64_t i = 1; i < self_c.dim(); i++) out_shape.push_back(self_c.size(i));
    auto output = at::empty(out_shape, self_c.options());

    if (num_indices == 0) return output;

    // Transfer indices
    auto indices_vulkan = at::empty({num_indices}, self_c.options().dtype(at::kFloat));
    {
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
        buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));
    }

    struct { uint32_t num_indices; uint32_t slice_size; } params{
        static_cast<uint32_t>(num_indices),
        static_cast<uint32_t>(slice_size)
    };

    uint32_t total = static_cast<uint32_t>(num_indices * slice_size);
    uint32_t workgroups = (total + 255) / 256;

    dispatch_shader("indexing_index_select_fwd",
                    shaders::indexing_index_select_fwd, shaders::indexing_index_select_fwd_size,
                    {self_c, indices_vulkan, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── Masked Fill ─────────────────────────────────────────────────
at::Tensor& vulkan_masked_fill(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "masked_fill_");

    auto mask_float = mask.to(at::kFloat).contiguous().to(self.device());
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    if (numel == 0) return self;

    auto output = at::empty_like(self_c);

    struct { float fill_value; uint32_t numel; } params{
        value.toFloat(), numel
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("indexing_masked_fill_fwd",
                    shaders::indexing_masked_fill_fwd, shaders::indexing_masked_fill_fwd_size,
                    {self_c, mask_float, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Copy result back to self via GPU
    dispatch_copy_buffer(output, self);

    return self;
}

// ── Masked Scatter ──────────────────────────────────────────────
at::Tensor vulkan_masked_scatter(const at::Tensor& self, const at::Tensor& mask, const at::Tensor& source) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "masked_scatter");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    // Flatten everything for element-wise operation
    auto self_flat = self_c.reshape({-1});
    uint32_t numel = static_cast<uint32_t>(self_flat.numel());
    if (numel == 0) return self_c.reshape(self.sizes());

    // Get mask as bool on CPU to build index mapping
    auto mask_flat = mask.reshape({-1}).to(at::kBool).cpu();
    auto mask_accessor = mask_flat.accessor<bool, 1>();

    // Build index buffer on CPU: for each position, store the source index
    // if mask is true, or 0xFFFFFFFF if mask is false
    auto indices_cpu = at::empty({numel}, at::kInt);
    auto* indices_ptr = indices_cpu.data_ptr<int32_t>();
    int32_t src_idx = 0;
    for (uint32_t i = 0; i < numel; i++) {
        if (mask_accessor[i]) {
            indices_ptr[i] = src_idx++;
            TORCH_CHECK(src_idx <= static_cast<int32_t>(source.numel()),
                        "masked_scatter: source doesn't have enough values");
        } else {
            indices_ptr[i] = static_cast<int32_t>(0xFFFFFFFF);
        }
    }

    // Upload index buffer to Vulkan (pack as float — shader reads as uint)
    auto indices_vulkan = at::empty({static_cast<int64_t>(numel)}, self_c.options());
    {
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
        TORCH_CHECK(buf, "Failed to get Vulkan buffer for indices");
        buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(numel * sizeof(int32_t)));
    }

    // Ensure source is contiguous float32 on Vulkan
    auto src_c = source.contiguous();
    src_c = ensure_float32(src_c);
    if (src_c.device() != self_c.device()) {
        src_c = src_c.to(self_c.device());
    }

    auto output = at::empty_like(self_flat);

    struct { uint32_t numel; } params{ numel };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("indexing_masked_scatter_fwd",
                    shaders::indexing_masked_scatter_fwd, shaders::indexing_masked_scatter_fwd_size,
                    {self_flat, indices_vulkan, src_c.reshape({-1}), output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    return cast_from_float32(output.reshape(self.sizes()), orig_dtype);
}

// In-place version
at::Tensor& vulkan_masked_scatter_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source) {
    auto result = vulkan_masked_scatter(self, mask, source);
    dispatch_copy_buffer(result, self);
    return self;
}

}} // namespace torch_vulkan::ops
