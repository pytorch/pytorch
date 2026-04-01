#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>
#include <algorithm>
#include <functional>

namespace torch_vulkan { namespace ops {

struct ReductionParams {
    uint32_t numel;
};

// Multi-pass reduction: dispatch shader that outputs one partial per workgroup,
// then repeat until we have a single value.
static at::Tensor reduce_full(
    const at::Tensor& self,
    const std::string& key,
    const uint32_t* spirv,
    size_t spirv_size) {

    auto self_c = self.contiguous();
    check_supported_float(self_c, "reduction");
    auto self_f32 = ensure_float32(self_c);

    uint32_t numel = static_cast<uint32_t>(self_f32.numel());
    if (numel == 0) return at::empty({}, self_f32.options());

    // Iteratively reduce: each pass converts N elements → ceil(N/256) partial results
    at::Tensor current = self_f32;
    while (numel > 1) {
        uint32_t workgroups = (numel + 255) / 256;
        auto partial = at::empty({static_cast<int64_t>(workgroups)}, current.options());

        ReductionParams params{numel};
        dispatch_shader(key, spirv, spirv_size,
                        {current, partial},
                        workgroups, 1, 1,
                        &params, sizeof(params));

        current = partial;
        numel = workgroups;
    }

    // Reshape to scalar
    return current.reshape({});
}

// Dimensional reduction using per-row GPU shaders.
// Moves the target dim to last position, reshapes to [num_rows, row_size],
// dispatches one workgroup per row. Single GPU dispatch — no host loops.
static at::Tensor reduce_dim(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    const std::string& full_key,
    const uint32_t* full_spirv,
    size_t full_spirv_size,
    const std::string& dim_key = "",
    const uint32_t* dim_spirv = nullptr,
    size_t dim_spirv_size = 0,
    const std::string& strided_key = "",
    const uint32_t* strided_spirv = nullptr,
    size_t strided_spirv_size = 0) {

    auto self_c = self.contiguous();
    check_supported_float(self_c, "reduction");
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());

    int64_t dim_size = self_c.size(dim);
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; i++) outer *= self_c.size(i);
    for (int64_t i = dim + 1; i < self_c.dim(); i++) inner *= self_c.size(i);

    // Build output shape
    std::vector<int64_t> result_shape;
    for (int64_t i = 0; i < self_c.dim(); i++) {
        if (i == dim) {
            if (keepdim) result_shape.push_back(1);
        } else {
            result_shape.push_back(self_c.size(i));
        }
    }

    int64_t num_rows = outer * inner;

    // Fast path: use strided shader when dim is not last — avoids movedim+copy dispatch.
    // For contiguous tensor: outer_stride = dim_size * inner, dim_stride = inner, inner_stride = 1.
    bool need_perm = (dim != self_c.dim() - 1);
    if (strided_spirv != nullptr && need_perm) {
        // Strided reduction: each of (outer*inner) output elements reduces dim_size inputs
        // strided by `inner` in the flat buffer.
        struct { uint32_t outer; uint32_t dim_size; uint32_t inner;
                 uint32_t outer_stride; uint32_t dim_stride; uint32_t inner_stride; } strided_params{
            static_cast<uint32_t>(outer),
            static_cast<uint32_t>(dim_size),
            static_cast<uint32_t>(inner),
            static_cast<uint32_t>(dim_size * inner),  // outer_stride
            static_cast<uint32_t>(inner),              // dim_stride
            1u                                          // inner_stride (contiguous)
        };
        auto result = at::empty({num_rows}, self_c.options());
        uint32_t workgroups = static_cast<uint32_t>(num_rows);
        dispatch_shader(strided_key, strided_spirv, strided_spirv_size,
                        {self_c, result},
                        workgroups, 1, 1,
                        &strided_params, sizeof(strided_params));
        return result.reshape(result_shape);
    }

    // Standard path: dim is already last (or no per-dim shader available)
    auto perm = need_perm ? self_c.movedim(dim, -1).contiguous() : self_c;
    auto flat = perm.reshape({num_rows, dim_size});

    // Use per-dim shader if available (single GPU dispatch, one workgroup per row)
    if (dim_spirv != nullptr) {
        auto result = at::empty({num_rows}, self_c.options());
        struct { uint32_t num_rows; uint32_t row_size; } params{
            static_cast<uint32_t>(num_rows), static_cast<uint32_t>(dim_size)};
        uint32_t workgroups = static_cast<uint32_t>(num_rows);
        dispatch_shader(dim_key, dim_spirv, dim_spirv_size,
                        {flat, result},
                        workgroups, 1, 1,
                        &params, sizeof(params));
        return result.reshape(result_shape);
    }

    // Fallback: dispatch separate full reductions per row (legacy path)
    auto result = at::empty({num_rows}, self_c.options());
    for (int64_t i = 0; i < num_rows; i++) {
        auto slice = flat[i];
        auto reduced = reduce_full(slice, full_key, full_spirv, full_spirv_size);
        auto& alloc = VulkanAllocator::instance();
        auto* src_buf = alloc.get_buffer(reduced.data_ptr());
        auto* dst_buf = alloc.get_buffer(result.data_ptr());
        float val;
        src_buf->read(&val, sizeof(float));
        dst_buf->write(&val, sizeof(float), static_cast<VkDeviceSize>(i * sizeof(float)));
    }

    return result.reshape(result_shape);
}

// ── sum ─────────────────────────────────────────────────────────
at::Tensor vulkan_sum(const at::Tensor& self, at::OptionalIntArrayRef dim,
                      bool keepdim, std::optional<at::ScalarType> dtype) {
    if (!dim.has_value() || dim->empty()) {
        return reduce_full(self, "reduction_sum_fwd",
                           shaders::reduction_sum_fwd, shaders::reduction_sum_fwd_size);
    }

    if (dim->size() == 1) {
        return reduce_dim(self, (*dim)[0], keepdim,
                          "reduction_sum_fwd", shaders::reduction_sum_fwd, shaders::reduction_sum_fwd_size,
                          "reduction_sum_dim_fwd", shaders::reduction_sum_dim_fwd, shaders::reduction_sum_dim_fwd_size,
                          "reduction_sum_dim_strided_fwd", shaders::reduction_sum_dim_strided_fwd, shaders::reduction_sum_dim_strided_fwd_size);
    }

    // Multi-dim reduction: reduce one dim at a time (largest first to preserve indices)
    auto dims = dim->vec();
    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
    at::Tensor result = self;
    for (int64_t d : dims) {
        result = reduce_dim(result, d, keepdim,
                            "reduction_sum_fwd", shaders::reduction_sum_fwd, shaders::reduction_sum_fwd_size,
                            "reduction_sum_dim_fwd", shaders::reduction_sum_dim_fwd, shaders::reduction_sum_dim_fwd_size,
                            "reduction_sum_dim_strided_fwd", shaders::reduction_sum_dim_strided_fwd, shaders::reduction_sum_dim_strided_fwd_size);
    }
    return result;
}

// ── mean ────────────────────────────────────────────────────────
at::Tensor vulkan_mean(const at::Tensor& self, at::OptionalIntArrayRef dim,
                       bool keepdim, std::optional<at::ScalarType> dtype) {
    if (!dim.has_value() || dim->empty()) {
        auto s = reduce_full(self, "reduction_sum_fwd",
                             shaders::reduction_sum_fwd, shaders::reduction_sum_fwd_size);
        // Divide by numel on GPU using scalar division
        return vulkan_div_scalar(s, at::Scalar(static_cast<float>(self.numel())));
    }

    if (dim->size() == 1) {
        auto s = reduce_dim(self, (*dim)[0], keepdim,
                            "reduction_sum_fwd", shaders::reduction_sum_fwd, shaders::reduction_sum_fwd_size,
                            "reduction_sum_dim_fwd", shaders::reduction_sum_dim_fwd, shaders::reduction_sum_dim_fwd_size,
                            "reduction_sum_dim_strided_fwd", shaders::reduction_sum_dim_strided_fwd, shaders::reduction_sum_dim_strided_fwd_size);
        int64_t d = at::maybe_wrap_dim((*dim)[0], self.dim());
        return vulkan_div_scalar(s, at::Scalar(static_cast<float>(self.size(d))));
    }

    // Multi-dim mean: reduce one dim at a time (largest first), then divide by total count
    auto dims = dim->vec();
    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
    int64_t total_count = 1;
    for (int64_t d : dims) {
        int64_t dd = at::maybe_wrap_dim(d, self.dim());
        total_count *= self.size(dd);
    }
    at::Tensor result = self;
    for (int64_t d : dims) {
        result = reduce_dim(result, d, keepdim,
                            "reduction_sum_fwd", shaders::reduction_sum_fwd, shaders::reduction_sum_fwd_size,
                            "reduction_sum_dim_fwd", shaders::reduction_sum_dim_fwd, shaders::reduction_sum_dim_fwd_size,
                            "reduction_sum_dim_strided_fwd", shaders::reduction_sum_dim_strided_fwd, shaders::reduction_sum_dim_strided_fwd_size);
    }
    return vulkan_div_scalar(result, at::Scalar(static_cast<float>(total_count)));
}

// ── amax / amin ─────────────────────────────────────────────────
at::Tensor vulkan_amax(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    if (dim.empty()) {
        return reduce_full(self, "reduction_max_fwd",
                           shaders::reduction_max_fwd, shaders::reduction_max_fwd_size);
    }
    TORCH_CHECK(dim.size() == 1,
                "Vulkan amax currently supports reducing over a single dimension");
    return reduce_dim(self, dim[0], keepdim,
                      "reduction_max_fwd", shaders::reduction_max_fwd, shaders::reduction_max_fwd_size,
                      "reduction_max_dim_fwd", shaders::reduction_max_dim_fwd, shaders::reduction_max_dim_fwd_size);
}

at::Tensor vulkan_amin(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    if (dim.empty()) {
        return reduce_full(self, "reduction_min_fwd",
                           shaders::reduction_min_fwd, shaders::reduction_min_fwd_size);
    }
    TORCH_CHECK(dim.size() == 1,
                "Vulkan amin currently supports reducing over a single dimension");
    return reduce_dim(self, dim[0], keepdim,
                      "reduction_min_fwd", shaders::reduction_min_fwd, shaders::reduction_min_fwd_size,
                      "reduction_min_dim_fwd", shaders::reduction_min_dim_fwd, shaders::reduction_min_dim_fwd_size);
}

// ── max.dim / min.dim (values + indices) ────────────────────────
std::tuple<at::Tensor, at::Tensor> vulkan_max_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
    auto values = vulkan_amax(self, at::IntArrayRef({dim}), keepdim);
    auto indices = vulkan_argmax(self, dim, keepdim);
    return std::make_tuple(values, indices);
}

std::tuple<at::Tensor, at::Tensor> vulkan_max_dim_out(const at::Tensor& self, int64_t dim, bool keepdim,
                                                       at::Tensor& values_out, at::Tensor& indices_out) {
    auto [values, indices] = vulkan_max_dim(self, dim, keepdim);
    values_out.resize_(values.sizes());
    values_out.copy_(values);
    indices_out.resize_(indices.sizes());
    indices_out.copy_(indices);
    return std::make_tuple(values_out, indices_out);
}

std::tuple<at::Tensor, at::Tensor> vulkan_min_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
    auto values = vulkan_amin(self, at::IntArrayRef({dim}), keepdim);
    auto indices = vulkan_argmin(self, dim, keepdim);
    return std::make_tuple(values, indices);
}

std::tuple<at::Tensor, at::Tensor> vulkan_min_dim_out(const at::Tensor& self, int64_t dim, bool keepdim,
                                                       at::Tensor& values_out, at::Tensor& indices_out) {
    auto [values, indices] = vulkan_min_dim(self, dim, keepdim);
    values_out.resize_(values.sizes());
    values_out.copy_(values);
    indices_out.resize_(indices.sizes());
    indices_out.copy_(indices);
    return std::make_tuple(values_out, indices_out);
}

// ── prod ─────────────────────────────────────────────────────────
at::Tensor vulkan_prod(const at::Tensor& self, int64_t dim, bool keepdim,
                        std::optional<at::ScalarType> dtype) {
    // For full reduction (no dim specified), call reduce_full with prod shader
    // For dim reduction, use same approach as sum but with prod shader
    auto self_c = self.contiguous();
    check_supported_float(self_c, "prod");
    self_c = ensure_float32(self_c);

    dim = at::maybe_wrap_dim(dim, self_c.dim());
    return reduce_dim(self_c, dim, keepdim,
                      "reduction_prod_fwd", shaders::reduction_prod_fwd, shaders::reduction_prod_fwd_size,
                      "reduction_prod_dim_fwd", shaders::reduction_prod_dim_fwd, shaders::reduction_prod_dim_fwd_size);
}

// ── argmax (GPU shader) ─────────────────────────────────────────
at::Tensor vulkan_argmax(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "argmax");
    self_c = ensure_float32(self_c);

    if (!dim.has_value()) {
        // Full reduction argmax — use multi-pass reduction shader
        uint32_t numel = static_cast<uint32_t>(self_c.numel());
        if (numel == 0) return at::empty({}, self.options().dtype(at::kLong));

        // For full argmax, use per-row shader with single row
        auto flat = self_c.reshape({1, static_cast<int64_t>(numel)});
        auto result_uint = at::empty({1}, self_c.options());
        struct { uint32_t num_rows; uint32_t row_size; } params{1, numel};
        dispatch_shader("reduction_argmax_dim_fwd",
                        shaders::reduction_argmax_dim_fwd, shaders::reduction_argmax_dim_fwd_size,
                        {flat, result_uint}, 1, 1, 1, &params, sizeof(params));
        // Read index and return as Long scalar
        auto& alloc = VulkanAllocator::instance();
        uint32_t idx;
        alloc.get_buffer(result_uint.data_ptr())->read(&idx, sizeof(uint32_t));
        return at::tensor(static_cast<int64_t>(idx), self.options().dtype(at::kLong).device(self.device()));
    }

    // Per-dim argmax
    int64_t d = at::maybe_wrap_dim(dim.value(), self_c.dim());
    int64_t dim_size = self_c.size(d);

    // Move target dim to last, flatten to (num_rows, row_size)
    auto perm = self_c.movedim(d, -1).contiguous();
    int64_t num_rows = perm.numel() / dim_size;
    auto flat = perm.reshape({num_rows, dim_size});

    auto result_uint = at::empty({num_rows}, self_c.options());

    struct { uint32_t num_rows; uint32_t row_size; } params{
        static_cast<uint32_t>(num_rows), static_cast<uint32_t>(dim_size)
    };
    uint32_t workgroups = (static_cast<uint32_t>(num_rows) + 255) / 256;
    dispatch_shader("reduction_argmax_dim_fwd",
                    shaders::reduction_argmax_dim_fwd, shaders::reduction_argmax_dim_fwd_size,
                    {flat, result_uint}, workgroups, 1, 1, &params, sizeof(params));

    // Read indices and build Long tensor
    std::vector<uint32_t> indices(num_rows);
    auto& alloc = VulkanAllocator::instance();
    alloc.get_buffer(result_uint.data_ptr())->read(indices.data(), num_rows * sizeof(uint32_t));
    auto result_cpu = at::empty({num_rows}, at::TensorOptions().dtype(at::kLong));
    auto* lptr = result_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_rows; i++) lptr[i] = static_cast<int64_t>(indices[i]);

    // Reshape to output shape
    std::vector<int64_t> result_shape;
    for (int64_t i = 0; i < self_c.dim(); i++) {
        if (i == d) { if (keepdim) result_shape.push_back(1); }
        else result_shape.push_back(self_c.size(i));
    }
    auto result = result_cpu.reshape(result_shape).to(self.device());
    return result;
}

// ── argmin (GPU shader) ─────────────────────────────────────────
at::Tensor vulkan_argmin(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "argmin");
    self_c = ensure_float32(self_c);

    if (!dim.has_value()) {
        uint32_t numel = static_cast<uint32_t>(self_c.numel());
        if (numel == 0) return at::empty({}, self.options().dtype(at::kLong));

        auto flat = self_c.reshape({1, static_cast<int64_t>(numel)});
        auto result_uint = at::empty({1}, self_c.options());
        struct { uint32_t num_rows; uint32_t row_size; } params{1, numel};
        dispatch_shader("reduction_argmin_dim_fwd",
                        shaders::reduction_argmin_dim_fwd, shaders::reduction_argmin_dim_fwd_size,
                        {flat, result_uint}, 1, 1, 1, &params, sizeof(params));
        auto& alloc = VulkanAllocator::instance();
        uint32_t idx;
        alloc.get_buffer(result_uint.data_ptr())->read(&idx, sizeof(uint32_t));
        return at::tensor(static_cast<int64_t>(idx), self.options().dtype(at::kLong).device(self.device()));
    }

    int64_t d = at::maybe_wrap_dim(dim.value(), self_c.dim());
    int64_t dim_size = self_c.size(d);
    auto perm = self_c.movedim(d, -1).contiguous();
    int64_t num_rows = perm.numel() / dim_size;
    auto flat = perm.reshape({num_rows, dim_size});

    auto result_uint = at::empty({num_rows}, self_c.options());
    struct { uint32_t num_rows; uint32_t row_size; } params{
        static_cast<uint32_t>(num_rows), static_cast<uint32_t>(dim_size)
    };
    uint32_t workgroups = (static_cast<uint32_t>(num_rows) + 255) / 256;
    dispatch_shader("reduction_argmin_dim_fwd",
                    shaders::reduction_argmin_dim_fwd, shaders::reduction_argmin_dim_fwd_size,
                    {flat, result_uint}, workgroups, 1, 1, &params, sizeof(params));

    std::vector<uint32_t> indices(num_rows);
    auto& alloc = VulkanAllocator::instance();
    alloc.get_buffer(result_uint.data_ptr())->read(indices.data(), num_rows * sizeof(uint32_t));
    auto result_cpu = at::empty({num_rows}, at::TensorOptions().dtype(at::kLong));
    auto* lptr = result_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < num_rows; i++) lptr[i] = static_cast<int64_t>(indices[i]);

    std::vector<int64_t> result_shape;
    for (int64_t i = 0; i < self_c.dim(); i++) {
        if (i == d) { if (keepdim) result_shape.push_back(1); }
        else result_shape.push_back(self_c.size(i));
    }
    auto result = result_cpu.reshape(result_shape).to(self.device());
    return result;
}

// ── any (GPU shader) ────────────────────────────────────────────
at::Tensor vulkan_any(const at::Tensor& self) {
    auto self_c = self.contiguous().to(at::kFloat);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return at::zeros({}, self.options());

    // Multi-pass reduction with any shader
    at::Tensor current = self_c;
    while (numel > 1) {
        uint32_t workgroups = (numel + 255) / 256;
        auto partial = at::empty({static_cast<int64_t>(workgroups)}, current.options());
        ReductionParams params{numel};
        dispatch_shader("reduction_any_fwd",
                        shaders::reduction_any_fwd, shaders::reduction_any_fwd_size,
                        {current, partial}, workgroups, 1, 1, &params, sizeof(params));
        current = partial;
        numel = workgroups;
    }
    return current.reshape({});
}

at::Tensor vulkan_any_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
    auto self_c = self.contiguous().to(at::kFloat);
    dim = at::maybe_wrap_dim(dim, self_c.dim());
    int64_t dim_size = self_c.size(dim);

    auto perm = self_c.movedim(dim, -1).contiguous();
    int64_t num_rows = perm.numel() / dim_size;
    auto flat = perm.reshape({num_rows, dim_size});

    auto result = at::empty({num_rows}, self_c.options());
    struct { uint32_t num_rows; uint32_t row_size; } params{
        static_cast<uint32_t>(num_rows), static_cast<uint32_t>(dim_size)
    };
    uint32_t workgroups = (static_cast<uint32_t>(num_rows) + 255) / 256;
    dispatch_shader("reduction_any_dim_fwd",
                    shaders::reduction_any_dim_fwd, shaders::reduction_any_dim_fwd_size,
                    {flat, result}, workgroups, 1, 1, &params, sizeof(params));

    std::vector<int64_t> shape;
    for (int64_t i = 0; i < self_c.dim(); i++) {
        if (i == dim) { if (keepdim) shape.push_back(1); }
        else shape.push_back(self_c.size(i));
    }
    return result.reshape(shape);
}

// ── all (GPU shader) ────────────────────────────────────────────
at::Tensor vulkan_all(const at::Tensor& self) {
    auto self_c = self.contiguous().to(at::kFloat);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return at::ones({}, self.options());

    at::Tensor current = self_c;
    while (numel > 1) {
        uint32_t workgroups = (numel + 255) / 256;
        auto partial = at::empty({static_cast<int64_t>(workgroups)}, current.options());
        ReductionParams params{numel};
        dispatch_shader("reduction_all_fwd",
                        shaders::reduction_all_fwd, shaders::reduction_all_fwd_size,
                        {current, partial}, workgroups, 1, 1, &params, sizeof(params));
        current = partial;
        numel = workgroups;
    }
    return current.reshape({});
}

at::Tensor vulkan_all_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
    auto self_c = self.contiguous().to(at::kFloat);
    dim = at::maybe_wrap_dim(dim, self_c.dim());
    int64_t dim_size = self_c.size(dim);

    auto perm = self_c.movedim(dim, -1).contiguous();
    int64_t num_rows = perm.numel() / dim_size;
    auto flat = perm.reshape({num_rows, dim_size});

    auto result = at::empty({num_rows}, self_c.options());
    struct { uint32_t num_rows; uint32_t row_size; } params{
        static_cast<uint32_t>(num_rows), static_cast<uint32_t>(dim_size)
    };
    uint32_t workgroups = (static_cast<uint32_t>(num_rows) + 255) / 256;
    dispatch_shader("reduction_all_dim_fwd",
                    shaders::reduction_all_dim_fwd, shaders::reduction_all_dim_fwd_size,
                    {flat, result}, workgroups, 1, 1, &params, sizeof(params));

    std::vector<int64_t> shape;
    for (int64_t i = 0; i < self_c.dim(); i++) {
        if (i == dim) { if (keepdim) shape.push_back(1); }
        else shape.push_back(self_c.size(i));
    }
    return result.reshape(shape);
}

// ── norm (GPU-accelerated via existing ops) ─────────────────────
// L2 norm = sqrt(sum(x^2)), Lp = sum(|x|^p)^(1/p)
at::Tensor vulkan_norm(const at::Tensor& self, const at::Scalar& ord,
                        at::OptionalIntArrayRef dim, bool keepdim,
                        std::optional<at::ScalarType> dtype) {
    auto self_c = self.contiguous();
    check_supported_float(self_c, "norm");
    self_c = ensure_float32(self_c);

    double p = ord.toDouble();
    if (p == 2.0) {
        // L2: sqrt(sum(x^2))
        auto sq = self_c * self_c;
        auto s = dim.has_value() && !dim->empty()
            ? vulkan_sum(sq, dim, keepdim, dtype)
            : vulkan_sum(sq, c10::OptionalIntArrayRef{}, false, dtype);
        return at::sqrt(s);
    } else if (p == 1.0) {
        // L1: sum(|x|)
        auto abs_x = at::abs(self_c);
        return dim.has_value() && !dim->empty()
            ? vulkan_sum(abs_x, dim, keepdim, dtype)
            : vulkan_sum(abs_x, c10::OptionalIntArrayRef{}, false, dtype);
    } else {
        // General Lp: sum(|x|^p)^(1/p) — use existing GPU ops
        auto abs_x = at::abs(self_c);
        auto pow_x = at::pow(abs_x, p);
        auto s = dim.has_value() && !dim->empty()
            ? vulkan_sum(pow_x, dim, keepdim, dtype)
            : vulkan_sum(pow_x, c10::OptionalIntArrayRef{}, false, dtype);
        return at::pow(s, 1.0 / p);
    }
}

// ── norm.ScalarOpt_dim (used by F.normalize) ────────────────────
at::Tensor vulkan_norm_ScalarOpt_dim(const at::Tensor& self,
                                      const std::optional<at::Scalar>& p,
                                      at::IntArrayRef dim, bool keepdim) {
    at::Scalar ord = p.has_value() ? *p : at::Scalar(2.0);
    return vulkan_norm(self, ord, dim, keepdim, c10::nullopt);
}

}} // namespace torch_vulkan::ops
