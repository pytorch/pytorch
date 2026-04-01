#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// ── cumsum ──────────────────────────────────────────────────────
at::Tensor vulkan_cumsum(const at::Tensor& self, int64_t dim,
                          std::optional<at::ScalarType> dtype) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "cumsum");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());

    // Move target dim to last, flatten
    auto perm = self_c.movedim(dim, -1).contiguous();
    int64_t row_size = perm.size(-1);
    int64_t num_rows = perm.numel() / row_size;

    TORCH_CHECK(row_size <= 256,
                "Vulkan cumsum: dimension size ", row_size, " exceeds max 256");

    auto output = at::empty_like(perm);

    if (num_rows == 0 || row_size == 0) {
        return cast_from_float32(output.movedim(-1, dim), orig_dtype);
    }

    struct { uint32_t row_size; uint32_t num_rows; } params{
        static_cast<uint32_t>(row_size),
        static_cast<uint32_t>(num_rows)
    };

    dispatch_shader("reduction_cumsum_fwd",
                    shaders::reduction_cumsum_fwd, shaders::reduction_cumsum_fwd_size,
                    {perm, output},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params));

    return cast_from_float32(output.movedim(-1, dim), orig_dtype);
}

// ── sort (GPU shader) ───────────────────────────────────────────
std::tuple<at::Tensor, at::Tensor> vulkan_sort(
    const at::Tensor& self, int64_t dim, bool descending) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "sort");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());
    int64_t dim_size = self_c.size(dim);

    // Move sort dim to last, flatten to (num_rows, row_size)
    auto perm = self_c.movedim(dim, -1).contiguous();
    int64_t num_rows = perm.numel() / dim_size;
    auto flat = perm.reshape({num_rows, dim_size});

    auto out_values = at::empty_like(flat);
    auto out_indices_float = at::empty_like(flat);  // uint stored as float

    struct { uint32_t num_rows; uint32_t row_size; uint32_t descending; } params{
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(dim_size),
        descending ? 1u : 0u
    };
    uint32_t workgroups = (static_cast<uint32_t>(num_rows) + 255) / 256;
    dispatch_shader("reduction_sort_fwd",
                    shaders::reduction_sort_fwd, shaders::reduction_sort_fwd_size,
                    {flat, out_values, out_indices_float},
                    workgroups, 1, 1, &params, sizeof(params), 2);

    // Read indices back as Long tensor
    std::vector<uint32_t> idx_buf(num_rows * dim_size);
    auto& alloc = VulkanAllocator::instance();
    alloc.get_buffer(out_indices_float.data_ptr())->read(
        idx_buf.data(), num_rows * dim_size * sizeof(uint32_t));
    auto indices_cpu = at::empty({num_rows, dim_size}, at::TensorOptions().dtype(at::kLong));
    auto* lptr = indices_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_rows * dim_size; i++)
        lptr[i] = static_cast<int64_t>(idx_buf[i]);

    // Reshape back to original shape with sorted dim
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < self_c.dim(); i++) out_shape.push_back(self_c.size(i));

    // Values: movedim back
    auto result_values = out_values.reshape(perm.sizes()).movedim(-1, dim).contiguous();
    // Copy via GPU shader instead of host roundtrip
    auto final_values = at::empty(out_shape, self_c.options());
    dispatch_copy_buffer(result_values, final_values);

    auto final_indices = indices_cpu.reshape(perm.sizes().vec()).movedim(-1, dim).contiguous().to(self.device());
    return {cast_from_float32(final_values, orig_dtype), final_indices};
}

// ── topk (GPU shader) ──────────────────────────────────────────
std::tuple<at::Tensor, at::Tensor> vulkan_topk(
    const at::Tensor& self, int64_t k, int64_t dim,
    bool largest, bool sorted) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "topk");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());
    int64_t dim_size = self_c.size(dim);
    TORCH_CHECK(k <= dim_size, "topk k=", k, " exceeds dim size=", dim_size);

    // Move topk dim to last, flatten to (num_rows, row_size)
    auto perm = self_c.movedim(dim, -1).contiguous();
    int64_t num_rows = perm.numel() / dim_size;
    auto flat = perm.reshape({num_rows, dim_size});

    auto out_values = at::empty({num_rows, k}, self_c.options());
    auto out_indices_float = at::empty({num_rows, k}, self_c.options());

    struct { uint32_t num_rows; uint32_t row_size; uint32_t k; uint32_t largest; } params{
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(dim_size),
        static_cast<uint32_t>(k),
        largest ? 1u : 0u
    };
    uint32_t workgroups = (static_cast<uint32_t>(num_rows) + 255) / 256;
    dispatch_shader("reduction_topk_fwd",
                    shaders::reduction_topk_fwd, shaders::reduction_topk_fwd_size,
                    {flat, out_values, out_indices_float},
                    workgroups, 1, 1, &params, sizeof(params), 2);

    // Read indices back as Long tensor
    std::vector<uint32_t> idx_buf(num_rows * k);
    auto& alloc = VulkanAllocator::instance();
    alloc.get_buffer(out_indices_float.data_ptr())->read(
        idx_buf.data(), num_rows * k * sizeof(uint32_t));
    auto indices_cpu = at::empty({num_rows, k}, at::TensorOptions().dtype(at::kLong));
    auto* lptr = indices_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_rows * k; i++)
        lptr[i] = static_cast<int64_t>(idx_buf[i]);

    // Reshape: output shape is same as input but with dim replaced by k
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < self_c.dim(); i++) {
        if (i == dim) out_shape.push_back(k);
        else out_shape.push_back(self_c.size(i));
    }

    // For multi-dim tensors, movedim back
    std::vector<int64_t> perm_out_shape;
    for (int64_t i = 0; i < self_c.dim(); i++) {
        if (i == self_c.dim() - 1) perm_out_shape.push_back(k);
        else if (i < dim) perm_out_shape.push_back(self_c.size(i));
        else perm_out_shape.push_back(self_c.size(i));  // Will be fixed by movedim
    }

    auto val_shaped = out_values.reshape({num_rows, k});
    // Build proper output: we need shape with k in dim position
    // Since we moved dim to last, reshape and movedim back
    auto full_perm_shape = perm.sizes().vec();
    full_perm_shape.back() = k;
    auto val_reshaped = out_values.reshape(full_perm_shape).movedim(-1, dim).contiguous();

    // Copy values to vulkan tensor with correct shape via GPU
    auto final_values = at::empty(out_shape, self_c.options());
    dispatch_copy_buffer(val_reshaped, final_values);

    auto idx_reshaped = indices_cpu.reshape(full_perm_shape).movedim(-1, dim).contiguous();
    auto final_indices = idx_reshaped.to(self.device());

    return {cast_from_float32(final_values, orig_dtype), final_indices};
}

// ── gather (GPU shader) ─────────────────────────────────────────
at::Tensor vulkan_gather(const at::Tensor& self, int64_t dim,
                          const at::Tensor& index, bool sparse_grad) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "gather");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());
    auto index_c = index.contiguous();

    // Compute outer/inner sizes
    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; i++) outer_size *= self_c.size(i);
    for (int64_t i = dim + 1; i < self_c.dim(); i++) inner_size *= self_c.size(i);

    // Output has same shape as index
    auto output = at::empty(index_c.sizes(), self_c.options());
    uint32_t numel = static_cast<uint32_t>(index_c.numel());
    if (numel == 0) return output;

    // Transfer indices to Vulkan as uint buffer
    auto indices_cpu = index_c.cpu().to(at::kInt).contiguous();
    auto indices_vulkan = at::empty({static_cast<int64_t>(numel)}, self_c.options());
    {
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
        buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(numel * sizeof(int32_t)));
    }

    struct {
        uint32_t numel;
        uint32_t outer_size;
        uint32_t gather_dim;
        uint32_t input_dim;
        uint32_t inner_size;
    } params{
        numel,
        static_cast<uint32_t>(outer_size),
        static_cast<uint32_t>(index_c.size(dim)),
        static_cast<uint32_t>(self_c.size(dim)),
        static_cast<uint32_t>(inner_size)
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("indexing_gather_fwd",
                    shaders::indexing_gather_fwd, shaders::indexing_gather_fwd_size,
                    {self_c, indices_vulkan, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── scatter_ (GPU shader) ───────────────────────────────────────
at::Tensor& vulkan_scatter_(at::Tensor& self, int64_t dim,
                              const at::Tensor& index, const at::Tensor& src) {
    auto self_c = self.contiguous();
    auto src_c = src.contiguous();
    check_supported_float(self_c, "scatter_");
    // scatter_ is in-place, ensure_float32 inputs for dispatch, copy back
    self_c = ensure_float32(self_c);
    src_c = ensure_float32(src_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());
    auto index_c = index.contiguous();

    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; i++) outer_size *= self_c.size(i);
    for (int64_t i = dim + 1; i < self_c.dim(); i++) inner_size *= self_c.size(i);

    uint32_t src_numel = static_cast<uint32_t>(src_c.numel());
    if (src_numel == 0) return self;

    // First copy self into output buffer (scatter overwrites specific positions)
    auto output = self_c.clone();

    // Transfer indices to Vulkan as uint buffer
    auto indices_cpu = index_c.cpu().to(at::kInt).contiguous();
    auto indices_vulkan = at::empty({static_cast<int64_t>(src_numel)}, self_c.options());
    {
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(indices_vulkan.data_ptr());
        buf->write(indices_cpu.data_ptr(), static_cast<VkDeviceSize>(src_numel * sizeof(int32_t)));
    }

    struct {
        uint32_t numel;
        uint32_t outer_size;
        uint32_t scatter_dim;
        uint32_t self_dim;
        uint32_t inner_size;
    } params{
        src_numel,
        static_cast<uint32_t>(outer_size),
        static_cast<uint32_t>(index_c.size(dim)),
        static_cast<uint32_t>(self_c.size(dim)),
        static_cast<uint32_t>(inner_size)
    };

    uint32_t workgroups = (src_numel + 255) / 256;
    dispatch_shader("indexing_scatter_fwd",
                    shaders::indexing_scatter_fwd, shaders::indexing_scatter_fwd_size,
                    {src_c, indices_vulkan, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Copy result back to self via GPU
    dispatch_copy_buffer(output, self);
    return self;
}

// ── upsample_nearest2d (GPU shader) ─────────────────────────────
at::Tensor vulkan_upsample_nearest2d(
    const at::Tensor& self, at::IntArrayRef output_size,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "upsample_nearest2d");
    TORCH_CHECK(self_c.dim() == 4,
                "Vulkan upsample_nearest2d: only 4D tensors are supported. Got dim=", self_c.dim());
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    int64_t N = self_c.size(0);
    int64_t C = self_c.size(1);
    int64_t in_h = self_c.size(2);
    int64_t in_w = self_c.size(3);
    int64_t out_h = output_size[0];
    int64_t out_w = output_size[1];

    auto output = at::empty({N, C, out_h, out_w}, self_c.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct {
        uint32_t numel;
        uint32_t batch_channels;
        uint32_t in_h;
        uint32_t in_w;
        uint32_t out_h;
        uint32_t out_w;
    } params{
        numel,
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(in_h),
        static_cast<uint32_t>(in_w),
        static_cast<uint32_t>(out_h),
        static_cast<uint32_t>(out_w)
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_upsample_nearest2d_fwd",
                    shaders::copy_upsample_nearest2d_fwd, shaders::copy_upsample_nearest2d_fwd_size,
                    {self_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── upsample_nearest2d_backward (GPU shader) ───────────────────
at::Tensor vulkan_upsample_nearest2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    auto go_c = grad_output.contiguous();
    check_supported_float(go_c, "upsample_nearest2d_backward");
    auto orig_dtype = go_c.scalar_type();
    go_c = ensure_float32(go_c);

    int64_t N = input_size[0];
    int64_t C = input_size[1];
    int64_t in_h = input_size[2];
    int64_t in_w = input_size[3];
    int64_t out_h = output_size[0];
    int64_t out_w = output_size[1];

    auto grad_input = at::empty({N, C, in_h, in_w}, go_c.options());
    uint32_t numel = static_cast<uint32_t>(grad_input.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    struct {
        uint32_t numel;
        uint32_t batch_channels;
        uint32_t in_h;
        uint32_t in_w;
        uint32_t out_h;
        uint32_t out_w;
    } params{
        numel,
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(in_h),
        static_cast<uint32_t>(in_w),
        static_cast<uint32_t>(out_h),
        static_cast<uint32_t>(out_w)
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_upsample_nearest2d_backward_fwd",
                    shaders::copy_upsample_nearest2d_backward_fwd,
                    shaders::copy_upsample_nearest2d_backward_fwd_size,
                    {go_c, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

// ── upsample_bilinear2d (GPU shader) ────────────────────────────
at::Tensor vulkan_upsample_bilinear2d(
    const at::Tensor& self, at::IntArrayRef output_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "upsample_bilinear2d");
    TORCH_CHECK(self_c.dim() == 4,
                "Vulkan upsample_bilinear2d: only 4D tensors are supported. Got dim=", self_c.dim());
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    int64_t N = self_c.size(0);
    int64_t C = self_c.size(1);
    int64_t in_h = self_c.size(2);
    int64_t in_w = self_c.size(3);
    int64_t out_h = output_size[0];
    int64_t out_w = output_size[1];

    auto output = at::empty({N, C, out_h, out_w}, self_c.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    struct {
        uint32_t numel;
        uint32_t batch_channels;
        uint32_t in_h;
        uint32_t in_w;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t align_corners;
    } params{
        numel,
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(in_h),
        static_cast<uint32_t>(in_w),
        static_cast<uint32_t>(out_h),
        static_cast<uint32_t>(out_w),
        align_corners ? 1u : 0u
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_upsample_bilinear2d_fwd",
                    shaders::copy_upsample_bilinear2d_fwd, shaders::copy_upsample_bilinear2d_fwd_size,
                    {self_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── upsample_bilinear2d_backward (GPU shader) ───────────────────
at::Tensor vulkan_upsample_bilinear2d_backward(
    const at::Tensor& grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    auto go_c = grad_output.contiguous();
    check_supported_float(go_c, "upsample_bilinear2d_backward");
    auto orig_dtype = go_c.scalar_type();
    go_c = ensure_float32(go_c);

    int64_t N = input_size[0];
    int64_t C = input_size[1];
    int64_t in_h = input_size[2];
    int64_t in_w = input_size[3];
    int64_t out_h = output_size[0];
    int64_t out_w = output_size[1];

    auto grad_input = at::empty({N, C, in_h, in_w}, go_c.options());
    uint32_t numel = static_cast<uint32_t>(grad_input.numel());
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    struct {
        uint32_t numel;
        uint32_t batch_channels;
        uint32_t in_h;
        uint32_t in_w;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t align_corners;
    } params{
        numel,
        static_cast<uint32_t>(N * C),
        static_cast<uint32_t>(in_h),
        static_cast<uint32_t>(in_w),
        static_cast<uint32_t>(out_h),
        static_cast<uint32_t>(out_w),
        align_corners ? 1u : 0u
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_upsample_bilinear2d_backward_fwd",
                    shaders::copy_upsample_bilinear2d_backward_fwd,
                    shaders::copy_upsample_bilinear2d_backward_fwd_size,
                    {go_c, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

// ── grid_sample (GPU shader) ────────────────────────────────────
// Bilinear grid sampling for spatial transformer networks.
at::Tensor vulkan_grid_sampler_2d(
    const at::Tensor& input, const at::Tensor& grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {

    auto input_c = input.contiguous();
    auto grid_c = grid.contiguous();
    check_supported_float(input_c, "grid_sampler_2d");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);
    grid_c = ensure_float32(grid_c);
    TORCH_CHECK(input_c.dim() == 4, "Vulkan grid_sampler_2d: input must be 4D [N,C,H,W]");
    TORCH_CHECK(grid_c.dim() == 4, "Vulkan grid_sampler_2d: grid must be 4D [N,H_out,W_out,2]");
    TORCH_CHECK(grid_c.size(3) == 2, "Vulkan grid_sampler_2d: grid last dim must be 2");
    TORCH_CHECK(interpolation_mode == 0,
                "Vulkan grid_sampler_2d: only bilinear interpolation (mode=0) is supported");

    int64_t N = input_c.size(0);
    int64_t C = input_c.size(1);
    int64_t in_h = input_c.size(2);
    int64_t in_w = input_c.size(3);
    int64_t out_h = grid_c.size(1);
    int64_t out_w = grid_c.size(2);

    auto output = at::empty({N, C, out_h, out_w}, input_c.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());
    if (numel == 0) return output;

    struct {
        uint32_t numel;
        uint32_t N;
        uint32_t C;
        uint32_t in_h;
        uint32_t in_w;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t padding_mode;
        uint32_t align_corners;
    } params{
        numel,
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(C),
        static_cast<uint32_t>(in_h),
        static_cast<uint32_t>(in_w),
        static_cast<uint32_t>(out_h),
        static_cast<uint32_t>(out_w),
        static_cast<uint32_t>(padding_mode),
        align_corners ? 1u : 0u
    };

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("indexing_grid_sample_fwd",
                    shaders::indexing_grid_sample_fwd,
                    shaders::indexing_grid_sample_fwd_size,
                    {input_c, grid_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── index_put_ (GPU shader) ─────────────────────────────────────
// Supports 1D indexing: self[indices[0]] = values
at::Tensor& vulkan_index_put_(at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
                                const at::Tensor& values, bool accumulate) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "index_put_");

    // Collect defined index tensors
    std::vector<at::Tensor> idx_tensors;
    for (size_t i = 0; i < indices.size(); i++) {
        auto idx_opt = indices.get(i);
        if (idx_opt.has_value() && idx_opt->defined()) {
            idx_tensors.push_back(idx_opt->contiguous());
        }
    }
    TORCH_CHECK(idx_tensors.size() == 1,
                "Vulkan index_put_: only single-index (1D) mode is supported. Got ",
                idx_tensors.size(), " index tensors.");

    auto idx = idx_tensors[0].cpu().to(at::kInt).contiguous();
    auto vals = values.contiguous();
    int64_t num_indices = idx.numel();
    if (num_indices == 0) return self;

    // Upload indices to Vulkan
    auto& alloc = VulkanAllocator::instance();
    auto idx_vk = at::empty({num_indices}, self_c.options());
    alloc.get_buffer(idx_vk.data_ptr())->write(
        idx.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));

    struct {
        uint32_t num_indices;
        uint32_t accumulate;
    } params{
        static_cast<uint32_t>(num_indices),
        accumulate ? 1u : 0u
    };

    uint32_t workgroups = (params.num_indices + 255) / 256;
    dispatch_shader("indexing_index_put_fwd",
                    shaders::indexing_index_put_fwd,
                    shaders::indexing_index_put_fwd_size,
                    {idx_vk, vals, self},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return self;
}

// ── cumprod ─────────────────────────────────────────────────────
at::Tensor vulkan_cumprod(const at::Tensor& self, int64_t dim,
                           std::optional<at::ScalarType> dtype) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "cumprod");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());

    // Move target dim to last, flatten
    auto perm = self_c.movedim(dim, -1).contiguous();
    int64_t row_size = perm.size(-1);
    int64_t num_rows = perm.numel() / row_size;

    auto output = at::empty_like(perm);

    if (num_rows == 0 || row_size == 0) {
        return cast_from_float32(output.movedim(-1, dim), orig_dtype);
    }

    struct { uint32_t num_rows; uint32_t row_size; } params{
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(row_size)
    };

    dispatch_shader("reduction_cumprod_fwd",
                    shaders::reduction_cumprod_fwd, shaders::reduction_cumprod_fwd_size,
                    {perm, output},
                    static_cast<uint32_t>(num_rows), 1, 1,
                    &params, sizeof(params));

    return cast_from_float32(output.movedim(-1, dim), orig_dtype);
}

}} // namespace torch_vulkan::ops
