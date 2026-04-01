#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"
#include "../backend/Allocator.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// ── triu / tril ────────────────────────────────────────────────
// GPU shader — treats the flat buffer as repeating (rows x cols) matrices.
// The shader uses tid.x % (rows*cols) to find row/col within each matrix.
at::Tensor vulkan_triu(const at::Tensor& self, int64_t diagonal) {
    auto self_c = self.contiguous();
    TORCH_CHECK(self_c.dim() >= 2, "triu: input must be at least 2D");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    int64_t rows = self_c.size(-2);
    int64_t cols = self_c.size(-1);

    // Flatten to single buffer, dispatch over all elements
    auto flat_in = self_c.reshape({-1});
    auto output = at::empty_like(flat_in);
    uint32_t numel = static_cast<uint32_t>(flat_in.numel());
    if (numel == 0) return cast_from_float32(output.reshape(self_c.sizes()), orig_dtype);

    struct { uint32_t rows; uint32_t cols; int32_t diagonal; uint32_t numel_val; } params{
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(cols),
        static_cast<int32_t>(diagonal),
        numel};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_triu_fwd",
                    shaders::copy_triu_fwd, shaders::copy_triu_fwd_size,
                    {flat_in, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output.reshape(self_c.sizes()), orig_dtype);
}

at::Tensor vulkan_tril(const at::Tensor& self, int64_t diagonal) {
    auto self_c = self.contiguous();
    TORCH_CHECK(self_c.dim() >= 2, "tril: input must be at least 2D");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    int64_t rows = self_c.size(-2);
    int64_t cols = self_c.size(-1);

    auto flat_in = self_c.reshape({-1});
    auto output = at::empty_like(flat_in);
    uint32_t numel = static_cast<uint32_t>(flat_in.numel());
    if (numel == 0) return cast_from_float32(output.reshape(self_c.sizes()), orig_dtype);

    struct { uint32_t rows; uint32_t cols; int32_t diagonal; uint32_t numel_val; } params{
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(cols),
        static_cast<int32_t>(diagonal),
        numel};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_tril_fwd",
                    shaders::copy_tril_fwd, shaders::copy_tril_fwd_size,
                    {flat_in, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output.reshape(self_c.sizes()), orig_dtype);
}

// ── constant_pad_nd ────────────────────────────────────────────
// GPU shader for 2D padding (last 2 dims), CPU fallback for other cases
at::Tensor vulkan_constant_pad_nd(const at::Tensor& self,
                                   c10::IntArrayRef pad,
                                   const at::Scalar& value) {
    auto self_c = self.contiguous();

    // For 2D padding (pad has 4 elements: left, right, top, bottom)
    if (pad.size() == 4 && self_c.dim() >= 2 && is_supported_float(self_c.scalar_type())) {
        auto orig_dtype = self_c.scalar_type();
        auto self_f32 = ensure_float32(self_c);
        int64_t pad_left = pad[0], pad_right = pad[1];
        int64_t pad_top = pad[2], pad_bottom = pad[3];
        int64_t in_h = self_f32.size(-2), in_w = self_f32.size(-1);
        int64_t out_h = in_h + pad_top + pad_bottom;
        int64_t out_w = in_w + pad_left + pad_right;
        int64_t batch = self_f32.numel() / (in_h * in_w);

        std::vector<int64_t> out_shape(self_f32.sizes().begin(), self_f32.sizes().end());
        out_shape[self_f32.dim() - 2] = out_h;
        out_shape[self_f32.dim() - 1] = out_w;

        auto output = at::empty(out_shape, self_f32.options());
        uint32_t numel = static_cast<uint32_t>(output.numel());
        if (numel == 0) return cast_from_float32(output, orig_dtype);

        struct {
            uint32_t numel;
            uint32_t batch_size;
            uint32_t in_h, in_w;
            uint32_t out_h, out_w;
            uint32_t pad_left, pad_top;
            float pad_value;
        } params{numel,
                 static_cast<uint32_t>(batch),
                 static_cast<uint32_t>(in_h), static_cast<uint32_t>(in_w),
                 static_cast<uint32_t>(out_h), static_cast<uint32_t>(out_w),
                 static_cast<uint32_t>(pad_left), static_cast<uint32_t>(pad_top),
                 value.toFloat()};

        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_pad_fwd",
                        shaders::copy_pad_fwd, shaders::copy_pad_fwd_size,
                        {self_f32, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(output, orig_dtype);
    }

    // 1D padding (pad has 2 elements: left, right)
    if (pad.size() == 2 && self_c.dim() >= 1 && is_supported_float(self_c.scalar_type())) {
        auto orig_dtype = self_c.scalar_type();
        auto self_f32 = ensure_float32(self_c);
        int64_t pad_left = pad[0], pad_right = pad[1];
        int64_t in_w = self_f32.size(-1);
        int64_t out_w = in_w + pad_left + pad_right;
        int64_t batch = self_f32.numel() / in_w;

        std::vector<int64_t> out_shape(self_f32.sizes().begin(), self_f32.sizes().end());
        out_shape[self_f32.dim() - 1] = out_w;

        auto output = at::empty(out_shape, self_f32.options());
        uint32_t numel = static_cast<uint32_t>(output.numel());
        if (numel == 0) return cast_from_float32(output, orig_dtype);

        struct {
            uint32_t numel;
            uint32_t batch_size;
            uint32_t in_w;
            uint32_t out_w;
            uint32_t pad_left;
            float pad_value;
        } params{numel,
                 static_cast<uint32_t>(batch),
                 static_cast<uint32_t>(in_w),
                 static_cast<uint32_t>(out_w),
                 static_cast<uint32_t>(pad_left),
                 value.toFloat()};

        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_pad1d_fwd",
                        shaders::copy_pad1d_fwd, shaders::copy_pad1d_fwd_size,
                        {self_f32, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(output, orig_dtype);
    }

    // Unsupported padding configuration
    TORCH_CHECK(false, "Vulkan constant_pad_nd: only 1D (2-element) and 2D (4-element) padding on supported float dtypes is supported on GPU. ",
                "Got pad size=", pad.size(), ", dim=", self_c.dim(), ", dtype=", self_c.scalar_type());
}

// ── index.Tensor (advanced indexing) ───────────────────────────
// GPU shader for 1-index (row selection) and 2-index (element selection) modes.
at::Tensor vulkan_index_tensor(const at::Tensor& self,
                                const c10::List<std::optional<at::Tensor>>& indices) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "index.Tensor");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    // Count non-None index tensors
    std::vector<at::Tensor> idx_tensors;
    for (size_t i = 0; i < indices.size(); i++) {
        auto idx_opt = indices.get(i);
        if (idx_opt.has_value() && idx_opt->defined()) {
            idx_tensors.push_back(idx_opt->contiguous());
        }
    }
    TORCH_CHECK(!idx_tensors.empty(), "index.Tensor: at least one index tensor required");

    auto& alloc = VulkanAllocator::instance();

    if (idx_tensors.size() == 1) {
        // Single index tensor: select rows along dim 0
        // out[i, ...] = self[idx[i], ...]
        auto idx = idx_tensors[0].cpu().to(at::kInt).contiguous();
        int64_t num_indices = idx.numel();
        int64_t slice_size = (self_c.dim() > 1) ? self_c.numel() / self_c.size(0) : 1;
        int64_t out_numel = num_indices * slice_size;

        // Build output shape
        std::vector<int64_t> out_shape;
        for (int64_t i = 0; i < idx.dim(); i++) out_shape.push_back(idx.size(i));
        for (int64_t i = 1; i < self_c.dim(); i++) out_shape.push_back(self_c.size(i));
        if (out_shape.empty()) out_shape.push_back(1);

        auto output = at::empty(out_shape, self_c.options());
        if (out_numel == 0) return output;

        // Upload indices to Vulkan
        auto idx_vk = at::empty({num_indices}, self_c.options());
        alloc.get_buffer(idx_vk.data_ptr())->write(
            idx.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));

        // Dummy second index buffer (unused in mode 0)
        auto idx1_vk = at::empty({1}, self_c.options());

        struct {
            uint32_t numel;
            uint32_t num_indices;
            uint32_t slice_size;
            uint32_t num_cols;
            uint32_t mode;
        } params{
            static_cast<uint32_t>(out_numel),
            static_cast<uint32_t>(num_indices),
            static_cast<uint32_t>(slice_size),
            0u,
            0u  // mode 0: single index
        };

        uint32_t workgroups = (params.numel + 255) / 256;
        dispatch_shader("indexing_index_tensor_fwd",
                        shaders::indexing_index_tensor_fwd,
                        shaders::indexing_index_tensor_fwd_size,
                        {self_c, idx_vk, idx1_vk, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(output, orig_dtype);

    } else if (idx_tensors.size() == 2) {
        // Two index tensors: element selection
        // out[i] = self[idx0[i], idx1[i]]
        TORCH_CHECK(self_c.dim() == 2,
                    "Vulkan index.Tensor: two-index mode requires 2D input tensor");
        auto idx0 = idx_tensors[0].cpu().to(at::kInt).contiguous();
        auto idx1 = idx_tensors[1].cpu().to(at::kInt).contiguous();
        TORCH_CHECK(idx0.numel() == idx1.numel(),
                    "Vulkan index.Tensor: index tensors must have the same number of elements");

        int64_t num_indices = idx0.numel();
        auto output = at::empty(idx0.sizes(), self_c.options());
        if (num_indices == 0) return cast_from_float32(output, orig_dtype);

        // Upload indices to Vulkan
        auto idx0_vk = at::empty({num_indices}, self_c.options());
        auto idx1_vk = at::empty({num_indices}, self_c.options());
        alloc.get_buffer(idx0_vk.data_ptr())->write(
            idx0.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));
        alloc.get_buffer(idx1_vk.data_ptr())->write(
            idx1.data_ptr(), static_cast<VkDeviceSize>(num_indices * sizeof(int32_t)));

        struct {
            uint32_t numel;
            uint32_t num_indices;
            uint32_t slice_size;
            uint32_t num_cols;
            uint32_t mode;
        } params{
            static_cast<uint32_t>(num_indices),
            static_cast<uint32_t>(num_indices),
            1u,
            static_cast<uint32_t>(self_c.size(1)),
            1u  // mode 1: two indices
        };

        uint32_t workgroups = (params.numel + 255) / 256;
        dispatch_shader("indexing_index_tensor_fwd",
                        shaders::indexing_index_tensor_fwd,
                        shaders::indexing_index_tensor_fwd_size,
                        {self_c, idx0_vk, idx1_vk, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(output, orig_dtype);

    } else {
        TORCH_CHECK(false, "Vulkan index.Tensor: only 1 or 2 index tensors are supported. Got ", idx_tensors.size());
    }
}

// ── repeat ─────────────────────────────────────────────────────
// GPU shader using modulo index mapping
at::Tensor vulkan_repeat(const at::Tensor& self, c10::IntArrayRef repeats) {
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    int64_t in_ndim = self_c.dim();
    int64_t rep_ndim = static_cast<int64_t>(repeats.size());
    int64_t ndim = std::max(in_ndim, rep_ndim);
    TORCH_CHECK(ndim <= 8, "repeat: max 8 dimensions");

    // Pad input shape and repeats to same ndim
    std::vector<int64_t> out_shape(ndim);
    struct {
        uint32_t numel;
        uint32_t ndim;
        uint32_t in_shape[8];
        uint32_t out_shape[8];
    } params{};
    params.ndim = static_cast<uint32_t>(ndim);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < ndim; i++) {
        int64_t in_s = (i < ndim - in_ndim) ? 1 : self_c.size(i - (ndim - in_ndim));
        int64_t rep = (i < ndim - rep_ndim) ? 1 : repeats[i - (ndim - rep_ndim)];
        out_shape[i] = in_s * rep;
        params.in_shape[i] = static_cast<uint32_t>(in_s);
        params.out_shape[i] = static_cast<uint32_t>(out_shape[i]);
        out_numel *= out_shape[i];
    }

    auto output = at::empty(out_shape, self_c.options());
    uint32_t numel = static_cast<uint32_t>(out_numel);
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    params.numel = numel;
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_repeat_fwd",
                    shaders::copy_repeat_fwd, shaders::copy_repeat_fwd_size,
                    {self_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── repeat_interleave (self_int) ───────────────────────────────
// GPU shader: each output element maps back to input via division
at::Tensor vulkan_repeat_interleave_self_int(const at::Tensor& self,
                                              int64_t repeats,
                                              std::optional<int64_t> dim,
                                              std::optional<int64_t> output_size) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "repeat_interleave");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    if (!dim.has_value()) {
        // Flatten and repeat_interleave
        auto flat = self_c.reshape({-1});
        int64_t n = flat.numel();
        int64_t out_n = n * repeats;
        auto output = at::empty({out_n}, flat.options());
        if (out_n == 0) return output;

        struct { uint32_t numel; uint32_t repeats; } params{
            static_cast<uint32_t>(out_n), static_cast<uint32_t>(repeats)};
        uint32_t workgroups = (static_cast<uint32_t>(out_n) + 255) / 256;
        dispatch_shader("copy_repeat_interleave_fwd",
                        shaders::copy_repeat_interleave_fwd, shaders::copy_repeat_interleave_fwd_size,
                        {flat, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(output, orig_dtype);
    }

    int64_t d = at::maybe_wrap_dim(dim.value(), self_c.dim());
    int64_t dim_size = self_c.size(d);

    // Move target dim to last, flatten to (num_rows, dim_size)
    auto perm = self_c.movedim(d, -1).contiguous();
    int64_t num_rows = perm.numel() / dim_size;
    auto flat = perm.reshape({num_rows, dim_size});

    int64_t out_dim = dim_size * repeats;
    auto out_flat = at::empty({num_rows, out_dim}, flat.options());

    // Each output element out[row, j] = flat[row, j / repeats]
    uint32_t total = static_cast<uint32_t>(num_rows * out_dim);
    if (total == 0) {
        std::vector<int64_t> out_shape = self_c.sizes().vec();
        out_shape[d] = out_dim;
        return at::empty(out_shape, self_c.options());
    }

    struct { uint32_t numel; uint32_t repeats; } params{
        total, static_cast<uint32_t>(repeats)};
    uint32_t workgroups = (total + 255) / 256;
    dispatch_shader("copy_repeat_interleave_fwd",
                    shaders::copy_repeat_interleave_fwd, shaders::copy_repeat_interleave_fwd_size,
                    {flat, out_flat},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    // Reshape back: movedim last to dim d
    auto perm_shape = perm.sizes().vec();
    perm_shape.back() = out_dim;
    auto result = out_flat.reshape(perm_shape).movedim(-1, d).contiguous();
    return cast_from_float32(result, orig_dtype);
}

// ── stack ──────────────────────────────────────────────────────
// GPU: unsqueeze each tensor, then cat along dim
at::Tensor vulkan_stack(at::TensorList tensors, int64_t dim) {
    TORCH_CHECK(!tensors.empty(), "stack: empty tensor list");
    dim = at::maybe_wrap_dim(dim, tensors[0].dim() + 1);

    // Unsqueeze each tensor at dim, then cat
    std::vector<at::Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }
    return at::cat(unsqueezed, dim);
}

// ── chunk ──────────────────────────────────────────────────────
// Uses slice, no CPU transfer needed
std::vector<at::Tensor> vulkan_chunk(const at::Tensor& self, int64_t chunks, int64_t dim) {
    // Normalize dim
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "chunk dim out of range");

    int64_t size = self.size(dim);
    int64_t chunk_size = (size + chunks - 1) / chunks;

    std::vector<at::Tensor> result;
    int64_t offset = 0;
    for (int64_t i = 0; i < chunks && offset < size; i++) {
        int64_t len = std::min(chunk_size, size - offset);
        result.push_back(self.narrow(dim, offset, len));
        offset += len;
    }
    return result;
}

// ── erf ────────────────────────────────────────────────────────
// GPU shader — Abramowitz & Stegun approximation
at::Tensor vulkan_erf(const at::Tensor& self) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "erf");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    dispatch_elementwise("unary_erf_fwd",
                         shaders::unary_erf_fwd, shaders::unary_erf_fwd_size,
                         {self_c, output}, numel);
    return cast_from_float32(output, orig_dtype);
}

at::Tensor& vulkan_erf_(at::Tensor& self) {
    auto result = vulkan_erf(self);
    dispatch_copy_buffer(result, self);
    return self;
}

// ── narrow ─────────────────────────────────────────────────────
// Uses slice internally
at::Tensor vulkan_narrow(const at::Tensor& self, int64_t dim, int64_t start, int64_t length) {
    return self.slice(dim, start, start + length, 1);
}

// ── flip ───────────────────────────────────────────────────────
// GPU shader — flip one dim at a time
at::Tensor vulkan_flip(const at::Tensor& self, at::IntArrayRef dims) {
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    auto result = ensure_float32(self_c);
    for (auto d : dims) {
        d = at::maybe_wrap_dim(d, result.dim());
        int64_t outer = 1, inner = 1;
        for (int64_t i = 0; i < d; i++) outer *= result.size(i);
        for (int64_t i = d + 1; i < result.dim(); i++) inner *= result.size(i);

        auto output = at::empty_like(result);
        uint32_t numel = static_cast<uint32_t>(result.numel());
        if (numel == 0) return cast_from_float32(output, orig_dtype);

        struct {
            uint32_t numel;
            uint32_t outer_size;
            uint32_t dim_size;
            uint32_t inner_size;
        } params{numel,
                 static_cast<uint32_t>(outer),
                 static_cast<uint32_t>(result.size(d)),
                 static_cast<uint32_t>(inner)};

        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_flip_fwd",
                        shaders::copy_flip_fwd, shaders::copy_flip_fwd_size,
                        {result, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        result = output;
    }
    return cast_from_float32(result, orig_dtype);
}

// ── roll ───────────────────────────────────────────────────────
// GPU shader — roll one dim at a time
at::Tensor vulkan_roll(const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims) {
    auto orig_dtype = self.scalar_type();
    // Handle the case where dims is empty — roll over flattened tensor
    if (dims.empty()) {
        TORCH_CHECK(shifts.size() == 1, "roll: expected 1 shift when dims is empty");
        auto flat = ensure_float32(self.contiguous()).reshape({-1});
        int64_t n = flat.numel();
        int64_t shift = ((shifts[0] % n) + n) % n;
        if (shift == 0) return self.clone();

        auto output = at::empty_like(flat);
        uint32_t numel = static_cast<uint32_t>(n);
        struct {
            uint32_t numel;
            uint32_t outer_size;
            uint32_t dim_size;
            uint32_t inner_size;
            uint32_t shift;
        } params{numel, 1, numel, 1, static_cast<uint32_t>(shift)};
        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_roll_fwd",
                        shaders::copy_roll_fwd, shaders::copy_roll_fwd_size,
                        {flat, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(output.reshape(self.sizes()), orig_dtype);
    }
    TORCH_CHECK(shifts.size() == dims.size(), "roll: shifts and dims must have same size");
    auto result = ensure_float32(self.contiguous());

    for (size_t i = 0; i < shifts.size(); i++) {
        int64_t d = at::maybe_wrap_dim(dims[i], result.dim());
        int64_t dim_size = result.size(d);
        int64_t shift = ((shifts[i] % dim_size) + dim_size) % dim_size;  // normalize to [0, dim_size)
        if (shift == 0) continue;

        int64_t outer = 1, inner = 1;
        for (int64_t j = 0; j < d; j++) outer *= result.size(j);
        for (int64_t j = d + 1; j < result.dim(); j++) inner *= result.size(j);

        auto output = at::empty_like(result);
        uint32_t numel = static_cast<uint32_t>(result.numel());

        struct {
            uint32_t numel;
            uint32_t outer_size;
            uint32_t dim_size;
            uint32_t inner_size;
            uint32_t shift;
        } params{numel,
                 static_cast<uint32_t>(outer),
                 static_cast<uint32_t>(dim_size),
                 static_cast<uint32_t>(inner),
                 static_cast<uint32_t>(shift)};

        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_roll_fwd",
                        shaders::copy_roll_fwd, shaders::copy_roll_fwd_size,
                        {result, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        result = output;
    }
    return cast_from_float32(result, orig_dtype);
}

// ── _unsafe_view ───────────────────────────────────────────────
// Alias for reshape/view
at::Tensor vulkan_unsafe_view(const at::Tensor& self, at::IntArrayRef size) {
    return vulkan_reshape(self, size);
}

// ── is_contiguous ──────────────────────────────────────────────
// All our tensors are contiguous (we always create contiguous storage)

// ── contiguous ─────────────────────────────────────────────────
// Most Vulkan tensors are contiguous (opaque allocator). Return self
// when already contiguous to avoid wasteful GPU copy. For non-contiguous
// tensors (e.g., from zero-copy transpose), use the strided copy shader.
at::Tensor vulkan_contiguous(const at::Tensor& self,
                              at::MemoryFormat memory_format) {
    // Fast path: most Vulkan tensors from the opaque allocator are already contiguous.
    if (self.is_contiguous(memory_format)) return self;
    // Non-contiguous (e.g., zero-copy transposed view from vulkan_t for float32).
    // For float32: use strided copy shader (handles arbitrary strides, up to 5D).
    // For other dtypes: should not occur (vulkan_t returns contiguous GPU copy for non-float32).
    auto output = at::empty(self.sizes(), self.options());
    if (self.scalar_type() == at::kFloat) {
        dispatch_strided_copy(self, output);
    } else {
        TORCH_CHECK(false, "vulkan_contiguous: non-contiguous non-float32 tensor not supported "
                    "(dtype=", self.scalar_type(), ")");
    }
    return output;
}

// ── _to_copy ───────────────────────────────────────────────────
// Handles dtype/device conversion. Must not call self.cpu() or self.to()
// to avoid infinite recursion (those dispatch through _to_copy).
at::Tensor vulkan_to_copy(const at::Tensor& self,
                           std::optional<at::ScalarType> dtype,
                           std::optional<at::Layout> layout,
                           std::optional<at::Device> device,
                           std::optional<bool> pin_memory,
                           bool non_blocking,
                           std::optional<at::MemoryFormat> memory_format) {
    auto target_dtype = dtype.value_or(self.scalar_type());
    auto target_device = device.value_or(self.device());

    // Allocate output tensor on target device
    auto out = at::empty(self.sizes(), self.options()
        .dtype(target_dtype)
        .device(target_device));

    // Use copy_ which handles CPU<->Vulkan transfers and dtype conversion
    out.copy_(self, non_blocking);
    return out;
}

// ── as_strided ─────────────────────────────────────────────────
// GPU shader: copy elements using arbitrary strides
at::Tensor vulkan_as_strided(const at::Tensor& self, at::IntArrayRef size,
                              at::IntArrayRef stride,
                              std::optional<int64_t> storage_offset) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "as_strided");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    TORCH_CHECK(size.size() <= 8, "Vulkan as_strided: max 8 dimensions supported");

    int64_t numel = 1;
    for (auto s : size) numel *= s;
    auto output = at::empty(size, self_c.options());
    if (numel == 0) return output;

    struct {
        uint32_t numel;
        uint32_t ndim;
        uint32_t sizes[8];
        uint32_t strides[8];
        uint32_t storage_offset;
    } params{};
    params.numel = static_cast<uint32_t>(numel);
    params.ndim = static_cast<uint32_t>(size.size());
    params.storage_offset = static_cast<uint32_t>(storage_offset.value_or(0));
    for (size_t i = 0; i < size.size(); i++) {
        params.sizes[i] = static_cast<uint32_t>(size[i]);
        params.strides[i] = static_cast<uint32_t>(stride[i]);
    }

    uint32_t workgroups = (params.numel + 255) / 256;
    dispatch_shader("copy_as_strided_fwd",
                    shaders::copy_as_strided_fwd, shaders::copy_as_strided_fwd_size,
                    {self_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── resize_ ────────────────────────────────────────────────────
const at::Tensor& vulkan_resize_(const at::Tensor& self, at::IntArrayRef size,
                                  std::optional<at::MemoryFormat> memory_format) {
    // If size matches, nothing to do
    if (self.sizes() == size) return self;

    // Allocate new storage via the allocator
    auto dtype = self.scalar_type();
    size_t nbytes = c10::elementSize(dtype);
    for (auto s : size) nbytes *= s;
    if (nbytes == 0) nbytes = 1; // minimum allocation

    auto& alloc = VulkanAllocator::instance();
    auto data_ptr = alloc.allocate(nbytes);

    // Update the tensor's storage and sizes
    auto* impl = self.unsafeGetTensorImpl();
    auto new_storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(nbytes),
        std::move(data_ptr),
        &alloc,
        /*resizable=*/false);
    impl->set_storage_and_dtype(std::move(new_storage), caffe2::TypeMeta::fromScalarType(dtype));
    impl->set_sizes_contiguous(size);

    return self;
}

// ── reciprocal ─────────────────────────────────────────────────
// GPU shader: 1.0 / self
at::Tensor vulkan_reciprocal(const at::Tensor& self) {
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    // Convert non-float32 to float32, compute, convert back
    if (orig_dtype != at::kFloat) {
        self_c = self_c.to(at::kFloat);
    }
    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) {
        return orig_dtype != at::kFloat ? output.to(orig_dtype) : output;
    }
    dispatch_elementwise("unary_reciprocal_fwd",
                         shaders::unary_reciprocal_fwd, shaders::unary_reciprocal_fwd_size,
                         {self_c, output}, numel);
    return orig_dtype != at::kFloat ? output.to(orig_dtype) : output;
}

// ── sin / cos ──────────────────────────────────────────────────
// GPU shaders
at::Tensor vulkan_sin(const at::Tensor& self) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "sin");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    dispatch_elementwise("unary_sin_fwd",
                         shaders::unary_sin_fwd, shaders::unary_sin_fwd_size,
                         {self_c, output}, numel);
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_cos(const at::Tensor& self) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "cos");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);

    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);
    dispatch_elementwise("unary_cos_fwd",
                         shaders::unary_cos_fwd, shaders::unary_cos_fwd_size,
                         {self_c, output}, numel);
    return cast_from_float32(output, orig_dtype);
}

// ── logical_not ────────────────────────────────────────────────
// GPU shader: out = (input == 0) ? 1 : 0
at::Tensor vulkan_logical_not(const at::Tensor& self) {
    auto self_c = self.contiguous().to(at::kFloat);
    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return output;
    dispatch_elementwise("unary_logical_not_fwd",
                         shaders::unary_logical_not_fwd, shaders::unary_logical_not_fwd_size,
                         {self_c, output}, numel);
    return output;
}

// ── bitwise_not ────────────────────────────────────────────────
// GPU shader: flip 0↔1 for bool tensors
at::Tensor vulkan_bitwise_not(const at::Tensor& self) {
    auto self_c = self.contiguous().to(at::kFloat);
    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return output;
    dispatch_elementwise("unary_bitwise_not_fwd",
                         shaders::unary_bitwise_not_fwd, shaders::unary_bitwise_not_fwd_size,
                         {self_c, output}, numel);
    return output;
}

// ── bitwise_and.Tensor_out ──────────────────────────────────────
// For bool tensors: a AND b = a * b (both stored as 0.0/1.0)
at::Tensor& vulkan_bitwise_and_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
    auto a = self.contiguous().to(at::kFloat);
    auto b = other.contiguous().to(at::kFloat);
    auto result = vulkan_mul(a, b);
    out.resize_(result.sizes());
    out.copy_(result);
    return out;
}

// ── random_.from ────────────────────────────────────────────────
// Fill tensor with random integers in [from, to)
at::Tensor& vulkan_random_from(at::Tensor& self, int64_t from, std::optional<int64_t> to,
                                std::optional<at::Generator> generator) {
    TORCH_CHECK(to.has_value(), "random_.from requires 'to' argument");
    int64_t range = to.value() - from;
    // Use uniform_() to generate [0, 1), scale to range, add offset, floor
    auto float_self = self.to(at::kFloat);
    vulkan_uniform_(float_self, 0.0, 1.0, generator);
    // scale to [from, to) and floor
    auto scaled = vulkan_add_scalar(
        vulkan_mul_scalar(float_self, at::Scalar(static_cast<float>(range))),
        at::Scalar(static_cast<float>(from)), 1);
    auto floored = at::floor(scaled);
    self.copy_(floored);
    return self;
}

// ── _amp_foreach_non_finite_check_and_unscale_ ─────────────────
// GradScaler uses this to check for inf/nan and unscale grads.
// All-GPU: fused shader checks inf/nan + unscales in single dispatch per grad.
void vulkan_amp_non_finite_check_and_unscale_(
    at::TensorList scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {

    // Read inv_scale scalar (single float read — unavoidable for push constant)
    float inv_scale_val = inv_scale.cpu().item<float>();

    for (const auto& grad : scaled_grads) {
        if (!grad.defined()) continue;

        auto grad_c = grad.contiguous();
        auto orig_dtype = grad_c.scalar_type();
        auto grad_f32 = ensure_float32(grad_c);
        uint32_t numel = static_cast<uint32_t>(grad_f32.numel());
        if (numel == 0) continue;

        struct { uint32_t numel; float inv_scale; } params{numel, inv_scale_val};
        uint32_t workgroups = (numel + 255) / 256;

        // Fused: checks inf/nan, writes 1.0 to found_inf if found, unscales in-place
        dispatch_shader("training_amp_unscale_fwd",
                        shaders::training_amp_unscale_fwd,
                        shaders::training_amp_unscale_fwd_size,
                        {grad_f32, found_inf},
                        workgroups, 1, 1,
                        &params, sizeof(params), 2);

        // Copy unscaled f32 result back to original grad tensor
        if (orig_dtype != at::kFloat) {
            auto result = cast_from_float32(grad_f32, orig_dtype);
            const_cast<at::Tensor&>(grad).copy_(result);
        } else {
            const_cast<at::Tensor&>(grad).copy_(grad_f32);
        }
    }
}

// ── _amp_update_scale_ ─────────────────────────────────────────
// GradScaler uses this to update the scale factor based on inf/nan detection.
at::Tensor& vulkan_amp_update_scale_(
    at::Tensor& current_scale,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double scale_growth_factor,
    double scale_backoff_factor,
    int64_t growth_interval) {

    float found_inf_val = found_inf.cpu().item<float>();
    float current_scale_val = current_scale.cpu().item<float>();
    int32_t growth_tracker_val = growth_tracker.cpu().item<int32_t>();

    if (found_inf_val > 0) {
        // Inf/nan found: reduce scale
        current_scale_val *= static_cast<float>(scale_backoff_factor);
        growth_tracker_val = 0;
    } else {
        // No inf/nan: track consecutive successful steps
        growth_tracker_val++;
        if (growth_tracker_val >= growth_interval) {
            current_scale_val *= static_cast<float>(scale_growth_factor);
            growth_tracker_val = 0;
        }
    }

    // Write back
    current_scale.copy_(at::tensor({current_scale_val}, current_scale.options().device(c10::kCPU)));
    growth_tracker.copy_(at::tensor({growth_tracker_val}, growth_tracker.options().device(c10::kCPU)));

    return current_scale;
}

// ── mse_loss ────────────────────────────────────────────────────
// GPU shader: per-element (input - target)^2, then reduce
at::Tensor vulkan_mse_loss(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    check_supported_float(self_c, "mse_loss");
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);

    // Compute per-element squared differences
    auto diff_sq = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) {
        if (reduction == 0) return diff_sq;  // none
        return at::zeros({}, self.options());
    }

    dispatch_elementwise("loss_mse_fwd",
                         shaders::loss_mse_fwd, shaders::loss_mse_fwd_size,
                         {self_c, target_c, diff_sq}, numel);

    // Apply reduction
    if (reduction == 0) return diff_sq;                    // none
    auto s = diff_sq.sum();
    if (reduction == 1) return s / static_cast<float>(numel);  // mean
    return s;                                               // sum
}

// ── mse_loss_backward ──────────────────────────────────────────
// GPU shader: grad_input = grad_output * 2 * (input - target) / divisor
at::Tensor vulkan_mse_loss_backward(const at::Tensor& grad_output, const at::Tensor& self,
                                     const at::Tensor& target, int64_t reduction) {
    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    check_supported_float(self_c, "mse_loss_backward");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);

    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    auto grad_input = at::empty_like(self_c);
    if (numel == 0) return grad_input;

    // Expand scalar grad_output to match input shape
    auto grad_expanded = ensure_float32(grad_output.expand_as(self_c).contiguous());

    float divisor = (reduction == 1) ? static_cast<float>(numel) : 1.0f;
    struct { uint32_t numel; float divisor; } params{numel, divisor};
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("loss_mse_backward_fwd",
                    shaders::loss_mse_backward_fwd, shaders::loss_mse_backward_fwd_size,
                    {grad_expanded, self_c, target_c, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

}} // namespace torch_vulkan::ops
