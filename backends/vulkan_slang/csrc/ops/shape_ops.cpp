#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>
#include <ATen/TensorUtils.h>

namespace torch_vulkan { namespace ops {

// ── view / reshape ──────────────────────────────────────────────
at::Tensor vulkan_view(const at::Tensor& self, at::IntArrayRef size) {
    // Infer -1 dimension
    int64_t numel = self.numel();
    int64_t inferred = -1;
    int64_t product = 1;

    for (int64_t i = 0; i < static_cast<int64_t>(size.size()); i++) {
        if (size[i] == -1) {
            TORCH_CHECK(inferred == -1, "Only one -1 allowed in view");
            inferred = i;
        } else {
            product *= size[i];
        }
    }

    std::vector<int64_t> new_size(size.begin(), size.end());
    if (inferred >= 0) {
        TORCH_CHECK(product != 0, "Cannot infer dimension with zero-size other dims");
        new_size[inferred] = numel / product;
    }

    // Verify numel matches
    int64_t new_numel = 1;
    for (auto s : new_size) new_numel *= s;
    TORCH_CHECK(new_numel == numel,
                "shape '", new_size, "' is invalid for input of size ", numel);

    // Use PyTorch's computeStride to get the correct strides for this view.
    // Returns nullopt if the view is not possible without a copy.
    auto maybe_strides = at::detail::computeStride(self.sizes(), self.strides(), new_size);
    TORCH_CHECK(maybe_strides.has_value(),
                "view size is not compatible with input tensor's size and stride "
                "(at least one dimension spans across two contiguous subspaces). "
                "Use .reshape(...) instead.");

    // Zero-copy view: create a new TensorImpl sharing the same storage.
    auto impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
    impl->set_sizes_and_strides(new_size, *maybe_strides);
    impl->set_storage_offset(self.storage_offset());
    return at::Tensor(std::move(impl));
}

at::Tensor vulkan_reshape(const at::Tensor& self, at::IntArrayRef shape) {
    // Infer -1 dimension and compute new sizes
    int64_t numel = self.numel();
    int64_t inferred = -1;
    int64_t product = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); i++) {
        if (shape[i] == -1) { inferred = i; }
        else { product *= shape[i]; }
    }
    std::vector<int64_t> new_size(shape.begin(), shape.end());
    if (inferred >= 0) new_size[inferred] = numel / product;

    // Try zero-copy view first using PyTorch's computeStride.
    auto maybe_strides = at::detail::computeStride(self.sizes(), self.strides(), new_size);
    if (maybe_strides.has_value()) {
        auto impl = c10::make_intrusive<at::TensorImpl>(
            c10::Storage(self.storage()),
            self.key_set(),
            self.dtype());
        impl->set_sizes_and_strides(new_size, *maybe_strides);
        impl->set_storage_offset(self.storage_offset());
        return at::Tensor(std::move(impl));
    }

    // Fall back: make contiguous copy then view (same as CPU reshape behavior).
    auto contig = self.contiguous();
    return vulkan_view(contig, new_size);
}

// ── unsqueeze ───────────────────────────────────────────────────
at::Tensor vulkan_unsqueeze(const at::Tensor& self, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, self.dim() + 1, /*wrap_scalar=*/true);
    auto new_sizes = self.sizes().vec();
    auto new_strides = self.strides().vec();
    new_sizes.insert(new_sizes.begin() + dim, 1);
    // Stride for a size-1 dim: use the stride of the dim to the right (or 1 if at the end).
    // This correctly handles non-contiguous inputs (e.g., transposed tensors).
    int64_t new_stride = (dim < static_cast<int64_t>(new_strides.size()))
                         ? new_strides[dim]
                         : 1;
    new_strides.insert(new_strides.begin() + dim, new_stride);
    auto impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
    impl->set_sizes_and_strides(new_sizes, new_strides);
    impl->set_storage_offset(self.storage_offset());
    return at::Tensor(std::move(impl));
}

// ── squeeze ─────────────────────────────────────────────────────
at::Tensor vulkan_squeeze(const at::Tensor& self) {
    auto new_sizes = std::vector<int64_t>{};
    auto new_strides = std::vector<int64_t>{};
    for (int64_t i = 0; i < self.dim(); i++) {
        if (self.size(i) != 1) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }
    if (new_sizes.empty()) { new_sizes.push_back(1); new_strides.push_back(1); }
    auto impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(self.storage()), self.key_set(), self.dtype());
    impl->set_sizes_and_strides(new_sizes, new_strides);
    impl->set_storage_offset(self.storage_offset());
    return at::Tensor(std::move(impl));
}

at::Tensor vulkan_squeeze_dim(const at::Tensor& self, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, self.dim());
    if (self.size(dim) != 1) {
        return vulkan_clone(self, c10::nullopt);
    }
    auto new_sizes = self.sizes().vec();
    auto new_strides = self.strides().vec();
    new_sizes.erase(new_sizes.begin() + dim);
    new_strides.erase(new_strides.begin() + dim);
    if (new_sizes.empty()) { new_sizes.push_back(1); new_strides.push_back(1); }
    auto impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(self.storage()), self.key_set(), self.dtype());
    impl->set_sizes_and_strides(new_sizes, new_strides);
    impl->set_storage_offset(self.storage_offset());
    return at::Tensor(std::move(impl));
}

// ── permute / transpose ─────────────────────────────────────────
// These require actual data movement since our buffers are always contiguous.
at::Tensor vulkan_permute(const at::Tensor& self, at::IntArrayRef dims) {
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    int64_t ndim = self_c.dim();
    TORCH_CHECK(static_cast<int64_t>(dims.size()) == ndim,
                "permute: number of dims must match tensor dim");
    TORCH_CHECK(ndim <= 8, "permute: max 8 dimensions supported");

    // Validate dims
    std::vector<bool> seen(ndim, false);
    for (auto d : dims) {
        int64_t dd = at::maybe_wrap_dim(d, ndim);
        TORCH_CHECK(!seen[dd], "permute: repeated dim");
        seen[dd] = true;
    }

    // Compute output shape and input strides for the permuted layout
    std::vector<int64_t> out_shape(ndim);
    for (int64_t i = 0; i < ndim; i++) {
        out_shape[i] = self_c.size(at::maybe_wrap_dim(dims[i], ndim));
    }

    // Widen to f32 for GPU shader (StructuredBuffer<float>)
    auto self_f32 = ensure_float32(self);
    auto output = at::empty(out_shape, self_f32.options());
    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    // Build input strides as seen through the permutation
    // For each output dim d, the input stride is the stride of dims[d] in the input
    struct {
        uint32_t numel;
        uint32_t ndim;
        uint32_t in_strides[8];
        uint32_t out_shape[8];
    } params{};
    params.numel = numel;
    params.ndim = static_cast<uint32_t>(ndim);

    // Compute contiguous strides for input
    std::vector<int64_t> in_strides(ndim);
    in_strides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i + 1] * self_c.size(i + 1);
    }

    for (int64_t i = 0; i < ndim; i++) {
        int64_t src_dim = at::maybe_wrap_dim(dims[i], ndim);
        params.in_strides[i] = static_cast<uint32_t>(in_strides[src_dim]);
        params.out_shape[i] = static_cast<uint32_t>(out_shape[i]);
    }

    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_permute_fwd",
                    shaders::copy_permute_fwd, shaders::copy_permute_fwd_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
    auto ndim = self.dim();
    dim0 = at::maybe_wrap_dim(dim0, ndim);
    dim1 = at::maybe_wrap_dim(dim1, ndim);

    if (dim0 == dim1) return self;

    if (self.scalar_type() == at::kFloat) {
        // Zero-copy: swap sizes + strides (metadata only, no GPU copy).
        // This works for both contiguous and non-contiguous float32 tensors.
        // The resulting tensor may be non-contiguous, but subsequent ops that need
        // contiguous data will trigger a strided copy via vulkan_contiguous.
        // Importantly, if the INPUT is non-contiguous but the RESULT IS contiguous
        // (common case: transposing back a previously-transposed tensor), this is free.
        auto sizes = self.sizes().vec();
        auto strides = self.strides().vec();
        std::swap(sizes[dim0], sizes[dim1]);
        std::swap(strides[dim0], strides[dim1]);
        auto impl = c10::make_intrusive<at::TensorImpl>(
            c10::Storage(self.storage()),
            self.key_set(),
            self.dtype());
        impl->set_sizes_and_strides(sizes, strides);
        impl->set_storage_offset(self.storage_offset());
        return at::Tensor(std::move(impl));
    }

    std::vector<int64_t> perm(ndim);
    for (int64_t i = 0; i < ndim; i++) perm[i] = i;
    std::swap(perm[dim0], perm[dim1]);

    return vulkan_permute(self, perm);
}

at::Tensor vulkan_t(const at::Tensor& self) {
    TORCH_CHECK(self.dim() <= 2, "t() expects a 0, 1, or 2-D tensor");
    if (self.dim() < 2) return vulkan_clone(self, c10::nullopt);
    // Zero-copy transpose for float32: swap sizes/strides (metadata only, no GPU copy).
    // vulkan_mm / vulkan_linear detect this stride pattern and use mm_ex transpose flags.
    // For other dtypes (fp16, bf16, etc.), fall back to a GPU copy via permute since
    // the strided copy shader only handles 4-byte float32 elements.
    if (self.scalar_type() != at::kFloat) {
        return vulkan_permute(self, {1, 0});
    }
    int64_t M = self.size(0), N = self.size(1);
    int64_t sM = self.stride(0), sN = self.stride(1);
    std::vector<int64_t> new_size = {N, M};
    std::vector<int64_t> new_strides = {sN, sM};
    auto impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype());
    impl->set_sizes_and_strides(new_size, new_strides);
    impl->set_storage_offset(self.storage_offset());
    return at::Tensor(std::move(impl));
}

// ── expand ──────────────────────────────────────────────────────
at::Tensor vulkan_expand(const at::Tensor& self, at::IntArrayRef size,
                         bool implicit) {
    int64_t ndim = static_cast<int64_t>(size.size());
    TORCH_CHECK(ndim <= 8, "expand: max 8 dimensions supported");

    // Fast path: if shapes already match (or -1 entries resolve to same), return self
    if (ndim == self.dim()) {
        bool same = true;
        for (int64_t i = 0; i < ndim; i++) {
            int64_t target = size[i];
            if (target == -1) target = self.size(i);
            if (target != self.size(i)) { same = false; break; }
        }
        if (same) return self;
    }

    auto orig_dtype = self.scalar_type();

    // Pad input shape with leading 1s if needed
    int64_t in_ndim = self.dim();
    int64_t pad = ndim - in_ndim;

    std::vector<int64_t> out_shape(ndim);
    struct {
        uint32_t numel;
        uint32_t ndim;
        uint32_t in_shape[8];
        uint32_t out_shape[8];
        uint32_t in_strides[8];
    } params{};
    params.ndim = static_cast<uint32_t>(ndim);

    int64_t out_numel = 1;
    for (int64_t i = 0; i < ndim; i++) {
        int64_t in_s = (i < pad) ? 1 : self.size(i - pad);
        int64_t out_s = size[i];
        if (out_s == -1) out_s = in_s;
        TORCH_CHECK(in_s == out_s || in_s == 1,
                    "expand: incompatible size at dim ", i);
        out_shape[i] = out_s;
        params.in_shape[i] = static_cast<uint32_t>(in_s);
        params.out_shape[i] = static_cast<uint32_t>(out_s);
        // Actual input stride: 0 for leading pad dims, real stride otherwise.
        // Padded (broadcast) dims have in_shape=1, so the stride value doesn't matter
        // (in_coord will be 0). Use 0 for padded dims.
        params.in_strides[i] = (i < pad) ? 0u : static_cast<uint32_t>(self.stride(i - pad));
        out_numel *= out_s;
    }

    // Widen to f32 for GPU shader. For f32 tensors the expand shader handles non-contiguous
    // inputs via in_strides. For other dtypes, make contiguous first so the cast shader
    // works correctly (cast shaders assume contiguous layout).
    at::Tensor self_c;
    if (self.scalar_type() != at::kFloat && !self.is_contiguous()) {
        self_c = self.contiguous();
        // Update strides to contiguous layout after copy
        for (int64_t i = 0; i < ndim; i++) {
            params.in_strides[i] = (i < pad) ? 0u : static_cast<uint32_t>(self_c.stride(i - pad));
        }
    } else {
        self_c = self;
    }
    auto self_f32 = ensure_float32(self_c);
    auto output = at::empty(out_shape, self_f32.options());
    uint32_t numel = static_cast<uint32_t>(out_numel);
    if (numel == 0) return cast_from_float32(output, orig_dtype);

    params.numel = numel;
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("copy_expand_fwd",
                    shaders::copy_expand_fwd, shaders::copy_expand_fwd_size,
                    {self_f32, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── cat ─────────────────────────────────────────────────────────
at::Tensor vulkan_cat(const at::ITensorListRef& tensors, int64_t dim) {
    // Collect all tensors, widen to f32 for GPU shader
    std::vector<at::Tensor> tvec;
    at::ScalarType orig_dtype = at::kFloat;
    for (const auto& t : tensors) {
        if (tvec.empty()) orig_dtype = t.scalar_type();
        tvec.push_back(ensure_float32(t.contiguous()));
    }
    TORCH_CHECK(!tvec.empty(), "cat: empty tensor list");
    if (tvec.size() == 1) return cast_from_float32(tvec[0], orig_dtype);

    dim = at::maybe_wrap_dim(dim, tvec[0].dim());

    // Multi-tensor fast path: for 2..8 inputs use a single dispatch via cat_n shader.
    // Falls back to pairwise for > 8 inputs.
    static const size_t kMaxCatN = 8;
    if (tvec.size() <= kMaxCatN) {
        int64_t ndim = tvec[0].dim();
        int64_t outer = 1, inner = 1;
        for (int64_t d = 0; d < dim; d++) outer *= tvec[0].size(d);
        for (int64_t d = dim + 1; d < ndim; d++) inner *= tvec[0].size(d);

        // Compute output shape
        std::vector<int64_t> out_shape(tvec[0].sizes().begin(), tvec[0].sizes().end());
        int64_t total_dim = 0;
        for (auto& t : tvec) total_dim += t.size(dim);
        out_shape[dim] = total_dim;

        auto output = at::empty(out_shape, tvec[0].options());
        uint32_t numel = static_cast<uint32_t>(output.numel());
        if (numel == 0) return cast_from_float32(output, orig_dtype);

        struct {
            uint32_t numel;
            uint32_t outer_size;
            uint32_t inner_size;
            uint32_t n_inputs;
            uint32_t dim_offsets[8];
            uint32_t dim_sizes[8];
        } params{};
        params.numel = numel;
        params.outer_size = static_cast<uint32_t>(outer);
        params.inner_size = static_cast<uint32_t>(inner);
        params.n_inputs = static_cast<uint32_t>(tvec.size());

        uint32_t offset = 0;
        for (size_t i = 0; i < tvec.size(); i++) {
            params.dim_offsets[i] = offset;
            params.dim_sizes[i] = static_cast<uint32_t>(tvec[i].size(dim));
            offset += params.dim_sizes[i];
        }

        // Pad bindings list to 8 inputs + 1 output
        std::vector<at::Tensor> bindings(tvec.begin(), tvec.end());
        // Pad unused bindings with the first tensor (won't be accessed by shader)
        while (bindings.size() < kMaxCatN) bindings.push_back(tvec[0]);
        bindings.push_back(output);

        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_cat_n_fwd",
                        shaders::copy_cat_n_fwd, shaders::copy_cat_n_fwd_size,
                        bindings,
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return cast_from_float32(output, orig_dtype);
    }

    // Fallback: pairwise for > 8 inputs
    at::Tensor result = tvec[0];
    for (size_t i = 1; i < tvec.size(); i++) {
        auto& b = tvec[i];
        int64_t ndim_r = result.dim();

        int64_t outer = 1, inner = 1;
        for (int64_t d = 0; d < dim; d++) outer *= result.size(d);
        for (int64_t d = dim + 1; d < ndim_r; d++) inner *= result.size(d);

        int64_t dim_a = result.size(dim);
        int64_t dim_b = b.size(dim);
        int64_t dim_out = dim_a + dim_b;

        std::vector<int64_t> out_shape(result.sizes().begin(), result.sizes().end());
        out_shape[dim] = dim_out;

        auto output = at::empty(out_shape, result.options());
        uint32_t numel = static_cast<uint32_t>(output.numel());

        if (numel > 0) {
            struct {
                uint32_t numel;
                uint32_t outer_size;
                uint32_t inner_size;
                uint32_t dim_size_a;
                uint32_t dim_size_b;
            } params{numel,
                     static_cast<uint32_t>(outer),
                     static_cast<uint32_t>(inner),
                     static_cast<uint32_t>(dim_a),
                     static_cast<uint32_t>(dim_b)};

            uint32_t workgroups = (numel + 255) / 256;
            dispatch_shader("copy_cat_fwd",
                            shaders::copy_cat_fwd, shaders::copy_cat_fwd_size,
                            {result, b, output},
                            workgroups, 1, 1,
                            &params, sizeof(params));
        }
        result = output;
    }
    return cast_from_float32(result, orig_dtype);
}

// ── select / slice / narrow ─────────────────────────────────────
at::Tensor vulkan_select(const at::Tensor& self, int64_t dim, int64_t index) {
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    dim = at::maybe_wrap_dim(dim, self_c.dim());
    if (index < 0) index += self_c.size(dim);
    TORCH_CHECK(index >= 0 && index < self_c.size(dim), "select: index out of range");

    // Compute outer/inner
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= self_c.size(d);
    for (int64_t d = dim + 1; d < self_c.dim(); d++) inner *= self_c.size(d);

    // Output shape: remove the selected dim
    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < self_c.dim(); d++) {
        if (d != dim) out_shape.push_back(self_c.size(d));
    }
    if (out_shape.empty()) out_shape.push_back(1);

    // Widen to f32 for GPU shader
    auto self_f32 = ensure_float32(self);
    auto output = at::empty(out_shape, self_f32.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());

    if (numel > 0) {
        struct {
            uint32_t numel;
            uint32_t outer_size;
            uint32_t inner_size;
            uint32_t dim_size;
            uint32_t index;
        } params{numel,
                 static_cast<uint32_t>(outer),
                 static_cast<uint32_t>(inner),
                 static_cast<uint32_t>(self_c.size(dim)),
                 static_cast<uint32_t>(index)};

        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_select_fwd",
                        shaders::copy_select_fwd, shaders::copy_select_fwd_size,
                        {self_f32, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
    }
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_slice(const at::Tensor& self, int64_t dim,
                        std::optional<int64_t> start, std::optional<int64_t> end,
                        int64_t step) {
    auto self_c = self.contiguous();
    auto orig_dtype = self_c.scalar_type();
    dim = at::maybe_wrap_dim(dim, self_c.dim());
    int64_t dim_size = self_c.size(dim);

    // Normalize start/end
    int64_t s = start.value_or(0);
    int64_t e = end.value_or(dim_size);
    if (s < 0) s += dim_size;
    if (e < 0) e += dim_size;
    s = std::max(int64_t(0), std::min(s, dim_size));
    e = std::max(int64_t(0), std::min(e, dim_size));

    int64_t out_dim_size = (e - s + step - 1) / step;
    if (out_dim_size < 0) out_dim_size = 0;

    // Compute outer/inner
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; d++) outer *= self_c.size(d);
    for (int64_t d = dim + 1; d < self_c.dim(); d++) inner *= self_c.size(d);

    std::vector<int64_t> out_shape(self_c.sizes().begin(), self_c.sizes().end());
    out_shape[dim] = out_dim_size;

    // Widen to f32 for GPU shader
    auto self_f32 = ensure_float32(self);
    auto output = at::empty(out_shape, self_f32.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());

    if (numel > 0) {
        struct {
            uint32_t numel;
            uint32_t outer_size;
            uint32_t inner_size;
            uint32_t dim_size;
            uint32_t out_dim_size;
            uint32_t start;
            uint32_t step;
        } params{numel,
                 static_cast<uint32_t>(outer),
                 static_cast<uint32_t>(inner),
                 static_cast<uint32_t>(dim_size),
                 static_cast<uint32_t>(out_dim_size),
                 static_cast<uint32_t>(s),
                 static_cast<uint32_t>(step)};

        uint32_t workgroups = (numel + 255) / 256;
        dispatch_shader("copy_slice_fwd",
                        shaders::copy_slice_fwd, shaders::copy_slice_fwd_size,
                        {self_f32, output},
                        workgroups, 1, 1,
                        &params, sizeof(params));
    }
    return cast_from_float32(output, orig_dtype);
}

// ── split ────────────────────────────────────────────────────────
std::vector<at::Tensor> vulkan_split(const at::Tensor& self, int64_t split_size, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, self.dim());
    int64_t dim_size = self.size(dim);
    int64_t num_splits = (dim_size + split_size - 1) / split_size;

    std::vector<at::Tensor> result;
    result.reserve(num_splits);

    for (int64_t i = 0; i < num_splits; i++) {
        int64_t start = i * split_size;
        int64_t length = std::min(split_size, dim_size - start);
        result.push_back(vulkan_slice(self, dim, start, start + length, 1));
    }
    return result;
}

}} // namespace torch_vulkan::ops
