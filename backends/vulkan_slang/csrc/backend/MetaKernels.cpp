// Meta (FakeTensor) kernel registrations for torch.compile support.
// These describe output shape/dtype/device without running actual computation.
// Required for AOT Autograd and Inductor tracing.

#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/ExpandUtils.h>

namespace torch_vulkan {

// Wrap at::empty to avoid ambiguity with std::empty in C++20
static at::Tensor meta_empty(std::vector<int64_t> sizes, at::TensorOptions opts) {
    return at::empty(c10::IntArrayRef(sizes), opts);
}

// Helper: make meta tensor with same dtype/device as input but given shape
static at::Tensor meta_like(const at::Tensor& input, at::IntArrayRef sizes) {
    return at::empty(sizes, input.options().device(at::kMeta));
}

static at::Tensor meta_like_self(const at::Tensor& self) {
    return at::empty(self.sizes().vec(), self.options().device(at::kMeta));
}

// ── Element-wise ops: same shape as input ──────────────────────

static at::Tensor meta_unary(const at::Tensor& self) {
    return meta_like_self(self);
}

static at::Tensor meta_binary(const at::Tensor& self, const at::Tensor& other, const at::Scalar&) {
    // Broadcasting: use inferred shapes from at::infer_size
    auto out_size = at::infer_size_dimvector(self.sizes(), other.sizes());
    return meta_like(self, out_size);
}

static at::Tensor meta_binary_no_alpha(const at::Tensor& self, const at::Tensor& other) {
    auto out_size = at::infer_size_dimvector(self.sizes(), other.sizes());
    return meta_like(self, out_size);
}

// ── Comparison ops: same shape, bool dtype ─────────────────────

static at::Tensor meta_comparison(const at::Tensor& self, const at::Tensor& other) {
    auto out_size = at::infer_size_dimvector(self.sizes(), other.sizes());
    return at::empty(out_size, self.options().dtype(at::kBool).device(at::kMeta));
}

// ── BLAS ────────────────────────────────────────────────────────

static at::Tensor meta_mm(const at::Tensor& self, const at::Tensor& mat2) {
    return meta_empty({self.size(0), mat2.size(1)}, self.options().device(at::kMeta));
}

static at::Tensor meta_bmm(const at::Tensor& self, const at::Tensor& mat2) {
    return meta_empty({self.size(0), self.size(1), mat2.size(2)},
                     self.options().device(at::kMeta));
}

static at::Tensor meta_addmm(const at::Tensor& bias, const at::Tensor& self,
                               const at::Tensor& mat2, const at::Scalar&, const at::Scalar&) {
    return meta_empty({self.size(0), mat2.size(1)}, self.options().device(at::kMeta));
}

static at::Tensor meta_linear(const at::Tensor& input, const at::Tensor& weight,
                                const std::optional<at::Tensor>&) {
    auto out_shape = input.sizes().vec();
    out_shape.back() = weight.size(0);
    return at::empty(out_shape, input.options().device(at::kMeta));
}

// ── Convolution ─────────────────────────────────────────────────

static int64_t conv_out_size(int64_t in, int64_t k, int64_t p, int64_t s, int64_t d) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

static at::Tensor meta_conv2d(const at::Tensor& input, const at::Tensor& weight,
                                const std::optional<at::Tensor>&, at::IntArrayRef stride,
                                at::IntArrayRef padding, at::IntArrayRef dilation, int64_t) {
    int64_t sH = stride[0], sW = stride.size() > 1 ? stride[1] : sH;
    int64_t pH = padding[0], pW = padding.size() > 1 ? padding[1] : pH;
    int64_t dH = dilation.size() > 0 ? dilation[0] : 1;
    int64_t dW = dilation.size() > 1 ? dilation[1] : dH;

    int64_t oH = conv_out_size(input.size(2), weight.size(2), pH, sH, dH);
    int64_t oW = conv_out_size(input.size(3), weight.size(3), pW, sW, dW);
    return meta_empty({input.size(0), weight.size(0), oH, oW},
                     input.options().device(at::kMeta));
}

// ── Pooling ─────────────────────────────────────────────────────

static at::Tensor meta_max_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef, bool) {
    int64_t kH = kernel_size[0], kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride.empty() ? kH : stride[0];
    int64_t sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
    int64_t pH = padding.empty() ? 0 : padding[0];
    int64_t pW = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pH);

    int64_t oH = (self.size(2) + 2*pH - kH) / sH + 1;
    int64_t oW = (self.size(3) + 2*pW - kW) / sW + 1;
    return meta_empty({self.size(0), self.size(1), oH, oW},
                     self.options().device(at::kMeta));
}

static at::Tensor meta_adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size) {
    return meta_empty({self.size(0), self.size(1), output_size[0], output_size[1]},
                     self.options().device(at::kMeta));
}

// ── Normalization ───────────────────────────────────────────────

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
meta_layer_norm(const at::Tensor& input, at::IntArrayRef normalized_shape,
                const std::optional<at::Tensor>&,
                const std::optional<at::Tensor>&, double) {
    int64_t norm_size = 1;
    for (auto s : normalized_shape) norm_size *= s;
    int64_t num_rows = input.numel() / norm_size;
    return std::make_tuple(
        meta_like_self(input),
        meta_empty({num_rows}, input.options().device(at::kMeta)),
        meta_empty({num_rows}, input.options().device(at::kMeta)));
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
meta_batch_norm(const at::Tensor& input, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, bool, double, double) {
    return std::make_tuple(
        meta_like_self(input),
        meta_empty({0}, input.options().device(at::kMeta)),
        meta_empty({0}, input.options().device(at::kMeta)));
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
meta_group_norm(const at::Tensor& input,
                const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
                int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
    return std::make_tuple(
        meta_like_self(input),
        meta_empty({0}, input.options().device(at::kMeta)),
        meta_empty({0}, input.options().device(at::kMeta)));
}

// ── Softmax ─────────────────────────────────────────────────────

static at::Tensor meta_softmax(const at::Tensor& self, int64_t, std::optional<at::ScalarType>) {
    return meta_like_self(self);
}

// ── Embedding ───────────────────────────────────────────────────

static at::Tensor meta_embedding(const at::Tensor& weight, const at::Tensor& indices,
    int64_t, bool, bool) {
    auto out_shape = indices.sizes().vec();
    out_shape.push_back(weight.size(1));
    return at::empty(out_shape, weight.options().device(at::kMeta));
}

// ── Attention ───────────────────────────────────────────────────

static at::Tensor meta_sdpa(const at::Tensor& query, const at::Tensor&,
    const at::Tensor& value, const std::optional<at::Tensor>&,
    double, bool, std::optional<double>) {
    return at::empty(query.sizes().vec(), query.options().device(at::kMeta));
}

// ── Scalar binary ops: same shape as self ────────────────────────

static at::Tensor meta_binary_scalar(const at::Tensor& self, const at::Scalar&, const at::Scalar&) {
    return meta_like_self(self);
}

static at::Tensor meta_binary_scalar_no_alpha(const at::Tensor& self, const at::Scalar&) {
    return meta_like_self(self);
}

static at::Tensor meta_comparison_scalar(const at::Tensor& self, const at::Scalar&) {
    return at::empty(self.sizes().vec(), self.options().dtype(at::kBool).device(at::kMeta));
}

static at::Tensor meta_pow_tensor_scalar(const at::Tensor& self, const at::Scalar&) {
    return meta_like_self(self);
}

// ── In-place ops: return self ────────────────────────────────────

static at::Tensor& meta_inplace_self(at::Tensor& self, const at::Tensor&, const at::Scalar&) {
    return self;
}

static at::Tensor& meta_inplace_self_no_alpha(at::Tensor& self, const at::Tensor&) {
    return self;
}

static at::Tensor& meta_inplace_unary(at::Tensor& self) {
    return self;
}

static at::Tensor& meta_inplace_scalar(at::Tensor& self, const at::Scalar&, const at::Scalar&) {
    return self;
}

static at::Tensor& meta_inplace_scalar_no_alpha(at::Tensor& self, const at::Scalar&) {
    return self;
}

// ── Additional activations ──────────────────────────────────────

static at::Tensor meta_gelu(const at::Tensor& self, c10::string_view) {
    return meta_like_self(self);
}

static at::Tensor meta_leaky_relu(const at::Tensor& self, const at::Scalar&) {
    return meta_like_self(self);
}

static at::Tensor meta_elu(const at::Tensor& self, const at::Scalar&,
                            const at::Scalar&, const at::Scalar&) {
    return meta_like_self(self);
}

static at::Tensor meta_clamp(const at::Tensor& self,
                               const std::optional<at::Scalar>&, const std::optional<at::Scalar>&) {
    return meta_like_self(self);
}

static at::Tensor meta_selu(const at::Tensor& self) {
    return meta_like_self(self);
}

static at::Tensor meta_prelu(const at::Tensor& self, const at::Tensor&) {
    return meta_like_self(self);
}

static at::Tensor meta_where(const at::Tensor& condition, const at::Tensor& self,
                               const at::Tensor& other) {
    auto out_size = at::infer_size_dimvector(self.sizes(), other.sizes());
    return meta_like(self, out_size);
}

// ── Reductions ──────────────────────────────────────────────────

static at::Tensor meta_sum(const at::Tensor& self, at::OptionalIntArrayRef dim,
                            bool keepdim, std::optional<at::ScalarType>) {
    if (!dim.has_value() || dim->empty()) {
        return meta_empty({}, self.options().device(at::kMeta));
    }
    auto sizes = self.sizes().vec();
    for (auto d : *dim) {
        d = at::maybe_wrap_dim(d, self.dim());
        if (keepdim) sizes[d] = 1;
        else sizes[d] = 0;  // mark for removal
    }
    if (!keepdim) {
        std::vector<int64_t> out;
        for (auto s : sizes) if (s != 0) out.push_back(s);
        if (out.empty()) return meta_empty({}, self.options().device(at::kMeta));
        return at::empty(out, self.options().device(at::kMeta));
    }
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_reduce_dim(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
    if (dim.empty()) {
        return meta_empty({}, self.options().device(at::kMeta));
    }
    auto sizes = self.sizes().vec();
    for (auto d : dim) {
        d = at::maybe_wrap_dim(d, self.dim());
        sizes[d] = keepdim ? 1 : 0;
    }
    if (!keepdim) {
        std::vector<int64_t> out;
        for (auto s : sizes) if (s != 0) out.push_back(s);
        if (out.empty()) return meta_empty({}, self.options().device(at::kMeta));
        return at::empty(out, self.options().device(at::kMeta));
    }
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_argmax(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        return meta_empty({}, self.options().dtype(at::kLong).device(at::kMeta));
    }
    auto d = at::maybe_wrap_dim(*dim, self.dim());
    auto sizes = self.sizes().vec();
    if (keepdim) sizes[d] = 1;
    else sizes.erase(sizes.begin() + d);
    return at::empty(sizes, self.options().dtype(at::kLong).device(at::kMeta));
}

static at::Tensor meta_prod(const at::Tensor& self, int64_t dim, bool keepdim,
                              std::optional<at::ScalarType>) {
    auto d = at::maybe_wrap_dim(dim, self.dim());
    auto sizes = self.sizes().vec();
    if (keepdim) sizes[d] = 1;
    else sizes.erase(sizes.begin() + d);
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_any_full(const at::Tensor& self) {
    return meta_empty({}, self.options().dtype(at::kBool).device(at::kMeta));
}

static at::Tensor meta_any_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
    auto d = at::maybe_wrap_dim(dim, self.dim());
    auto sizes = self.sizes().vec();
    if (keepdim) sizes[d] = 1;
    else sizes.erase(sizes.begin() + d);
    return at::empty(sizes, self.options().dtype(at::kBool).device(at::kMeta));
}

// ── Shape ops ───────────────────────────────────────────────────

static at::Tensor meta_view(const at::Tensor& self, at::IntArrayRef size) {
    return at::empty(size, self.options().device(at::kMeta));
}

static at::Tensor meta_reshape(const at::Tensor& self, at::IntArrayRef shape) {
    // Infer -1 dimension
    auto sizes = shape.vec();
    int64_t numel = self.numel();
    int64_t infer_idx = -1;
    int64_t product = 1;
    for (int64_t i = 0; i < (int64_t)sizes.size(); i++) {
        if (sizes[i] == -1) infer_idx = i;
        else product *= sizes[i];
    }
    if (infer_idx >= 0) sizes[infer_idx] = numel / product;
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_unsqueeze(const at::Tensor& self, int64_t dim) {
    auto sizes = self.sizes().vec();
    dim = at::maybe_wrap_dim(dim, self.dim() + 1);
    sizes.insert(sizes.begin() + dim, 1);
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_squeeze(const at::Tensor& self) {
    std::vector<int64_t> sizes;
    for (auto s : self.sizes()) if (s != 1) sizes.push_back(s);
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_squeeze_dim(const at::Tensor& self, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, self.dim());
    auto sizes = self.sizes().vec();
    if (sizes[dim] == 1) sizes.erase(sizes.begin() + dim);
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_permute(const at::Tensor& self, at::IntArrayRef dims) {
    std::vector<int64_t> sizes(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        sizes[i] = self.size(dims[i]);
    }
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
    dim0 = at::maybe_wrap_dim(dim0, self.dim());
    dim1 = at::maybe_wrap_dim(dim1, self.dim());
    auto sizes = self.sizes().vec();
    std::swap(sizes[dim0], sizes[dim1]);
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_t(const at::Tensor& self) {
    if (self.dim() <= 1) return meta_like_self(self);
    auto sizes = self.sizes().vec();
    std::swap(sizes[0], sizes[1]);
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_expand(const at::Tensor& self, at::IntArrayRef size, bool) {
    return at::empty(size, self.options().device(at::kMeta));
}

static at::Tensor meta_cat(const at::ITensorListRef& tensors, int64_t dim) {
    auto it = tensors.begin();
    auto first = *it;
    dim = at::maybe_wrap_dim(dim, first.dim());
    auto sizes = first.sizes().vec();
    int64_t total = 0;
    for (auto t_it = tensors.begin(); t_it != tensors.end(); ++t_it) {
        total += (*t_it).size(dim);
    }
    sizes[dim] = total;
    return at::empty(sizes, first.options().device(at::kMeta));
}

static at::Tensor meta_select(const at::Tensor& self, int64_t dim, int64_t) {
    dim = at::maybe_wrap_dim(dim, self.dim());
    auto sizes = self.sizes().vec();
    sizes.erase(sizes.begin() + dim);
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_slice(const at::Tensor& self, int64_t dim,
                               std::optional<int64_t> start, std::optional<int64_t> end,
                               int64_t step) {
    dim = at::maybe_wrap_dim(dim, self.dim());
    int64_t len = self.size(dim);
    int64_t s = start.value_or(0);
    int64_t e = end.value_or(len);
    if (s < 0) s += len;
    if (e < 0) e += len;
    s = std::max(int64_t(0), std::min(s, len));
    e = std::max(int64_t(0), std::min(e, len));
    auto sizes = self.sizes().vec();
    sizes[dim] = (e - s + step - 1) / step;
    return at::empty(sizes, self.options().device(at::kMeta));
}

static std::vector<at::Tensor> meta_split(const at::Tensor& self, int64_t split_size, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, self.dim());
    int64_t len = self.size(dim);
    std::vector<at::Tensor> result;
    for (int64_t i = 0; i < len; i += split_size) {
        auto sizes = self.sizes().vec();
        sizes[dim] = std::min(split_size, len - i);
        result.push_back(at::empty(sizes, self.options().device(at::kMeta)));
    }
    return result;
}

// ── Avg pool2d ──────────────────────────────────────────────────

static at::Tensor meta_avg_pool2d(const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, bool, bool, std::optional<int64_t>) {
    int64_t kH = kernel_size[0], kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride.empty() ? kH : stride[0];
    int64_t sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
    int64_t pH = padding.empty() ? 0 : padding[0];
    int64_t pW = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pH);

    int64_t oH = (self.size(2) + 2*pH - kH) / sH + 1;
    int64_t oW = (self.size(3) + 2*pW - kW) / sW + 1;
    return meta_empty({self.size(0), self.size(1), oH, oW},
                     self.options().device(at::kMeta));
}

// ── Indexing ops ────────────────────────────────────────────────

static at::Tensor meta_index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
    dim = at::maybe_wrap_dim(dim, self.dim());
    auto sizes = self.sizes().vec();
    sizes[dim] = index.numel();
    return at::empty(sizes, self.options().device(at::kMeta));
}

static at::Tensor meta_gather(const at::Tensor& self, int64_t, const at::Tensor& index, bool) {
    return at::empty(index.sizes().vec(), self.options().device(at::kMeta));
}

// ── Sort / TopK ─────────────────────────────────────────────────

static std::tuple<at::Tensor, at::Tensor> meta_sort(const at::Tensor& self, int64_t, bool) {
    return std::make_tuple(
        meta_like_self(self),
        at::empty(self.sizes().vec(), self.options().dtype(at::kLong).device(at::kMeta)));
}

static std::tuple<at::Tensor, at::Tensor> meta_topk(const at::Tensor& self,
    int64_t k, int64_t dim, bool, bool) {
    dim = at::maybe_wrap_dim(dim, self.dim());
    auto sizes = self.sizes().vec();
    sizes[dim] = k;
    return std::make_tuple(
        at::empty(sizes, self.options().device(at::kMeta)),
        at::empty(sizes, self.options().dtype(at::kLong).device(at::kMeta)));
}

// ── Cumsum ──────────────────────────────────────────────────────

static at::Tensor meta_cumsum(const at::Tensor& self, int64_t, std::optional<at::ScalarType>) {
    return meta_like_self(self);
}

// ── Loss functions ──────────────────────────────────────────────

static std::tuple<at::Tensor, at::Tensor> meta_nll_loss_forward(
    const at::Tensor& self, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t reduction, int64_t) {
    at::Tensor output;
    if (reduction == 0) {  // None
        output = meta_empty({self.size(0)}, self.options().device(at::kMeta));
    } else {
        output = meta_empty({}, self.options().device(at::kMeta));
    }
    auto total_weight = meta_empty({}, self.options().device(at::kMeta));
    return std::make_tuple(output, total_weight);
}

static at::Tensor meta_cross_entropy_loss(
    const at::Tensor& self, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t reduction, int64_t, double) {
    if (reduction == 0) {
        return meta_empty({self.size(0)}, self.options().device(at::kMeta));
    }
    return meta_empty({}, self.options().device(at::kMeta));
}

// ── Upsample / Grid sample ─────────────────────────────────────

static at::Tensor meta_upsample_nearest2d(const at::Tensor& self, at::IntArrayRef output_size,
    std::optional<double>, std::optional<double>) {
    return meta_empty({self.size(0), self.size(1), output_size[0], output_size[1]},
                     self.options().device(at::kMeta));
}

static at::Tensor meta_upsample_nearest2d_backward(const at::Tensor& grad_output,
    at::IntArrayRef output_size, at::IntArrayRef input_size,
    std::optional<double>, std::optional<double>) {
    return meta_empty({input_size[0], input_size[1], input_size[2], input_size[3]},
                     grad_output.options().device(at::kMeta));
}

static at::Tensor meta_upsample_bilinear2d(const at::Tensor& self, at::IntArrayRef output_size,
    bool, std::optional<double>, std::optional<double>) {
    return meta_empty({self.size(0), self.size(1), output_size[0], output_size[1]},
                     self.options().device(at::kMeta));
}

static at::Tensor meta_upsample_bilinear2d_backward(const at::Tensor& grad_output,
    at::IntArrayRef output_size, at::IntArrayRef input_size,
    bool, std::optional<double>, std::optional<double>) {
    return meta_empty({input_size[0], input_size[1], input_size[2], input_size[3]},
                     grad_output.options().device(at::kMeta));
}

static at::Tensor meta_grid_sampler_2d(const at::Tensor& input, const at::Tensor& grid,
    int64_t, int64_t, bool) {
    return meta_empty({input.size(0), input.size(1), grid.size(1), grid.size(2)},
                     input.options().device(at::kMeta));
}

// ── Conv transpose 2d ───────────────────────────────────────────

static at::Tensor meta_conv_transpose2d(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>&, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef output_padding,
    int64_t, at::IntArrayRef dilation) {
    int64_t sH = stride[0], sW = stride.size() > 1 ? stride[1] : sH;
    int64_t pH = padding[0], pW = padding.size() > 1 ? padding[1] : pH;
    int64_t opH = output_padding.size() > 0 ? output_padding[0] : 0;
    int64_t opW = output_padding.size() > 1 ? output_padding[1] : opH;
    int64_t dH = dilation.size() > 0 ? dilation[0] : 1;
    int64_t dW = dilation.size() > 1 ? dilation[1] : dH;

    int64_t oH = (input.size(2) - 1) * sH - 2 * pH + dH * (weight.size(2) - 1) + opH + 1;
    int64_t oW = (input.size(3) - 1) * sW - 2 * pW + dW * (weight.size(3) - 1) + opW + 1;
    return meta_empty({input.size(0), weight.size(1), oH, oW},
                     input.options().device(at::kMeta));
}

// ── Clone ───────────────────────────────────────────────────────

static at::Tensor meta_clone(const at::Tensor& self, std::optional<at::MemoryFormat>) {
    return meta_like_self(self);
}

// ── Dropout ─────────────────────────────────────────────────────

static std::tuple<at::Tensor, at::Tensor> meta_native_dropout(
    const at::Tensor& input, double, std::optional<bool>) {
    return std::make_tuple(
        meta_like_self(input),
        at::empty(input.sizes().vec(), input.options().dtype(at::kBool).device(at::kMeta)));
}

static at::Tensor meta_native_dropout_backward(
    const at::Tensor& grad_output, const at::Tensor&, double) {
    return meta_like_self(grad_output);
}

// ── Foreach ops ─────────────────────────────────────────────────

static void meta_foreach_inplace_scalar(at::TensorList, const at::Scalar&) {
    // In-place: no output tensors to create
}

static void meta_foreach_inplace_list(at::TensorList, at::TensorList, const at::Scalar&) {
}

static void meta_foreach_addcmul(at::TensorList, at::TensorList, at::TensorList, const at::Scalar&) {
}

static std::vector<at::Tensor> meta_foreach_unary(at::TensorList self) {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& t : self) {
        result.push_back(at::empty(t.sizes().vec(), t.options().device(at::kMeta)));
    }
    return result;
}

static std::vector<at::Tensor> meta_foreach_binary_list(at::TensorList self, at::TensorList) {
    std::vector<at::Tensor> result;
    result.reserve(self.size());
    for (const auto& t : self) {
        result.push_back(at::empty(t.sizes().vec(), t.options().device(at::kMeta)));
    }
    return result;
}

// ── Backward helper meta kernels ─────────────────────────────────

static at::Tensor meta_threshold_backward(const at::Tensor& grad_output, const at::Tensor&, const at::Scalar&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_sigmoid_backward(const at::Tensor& grad_output, const at::Tensor&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_tanh_backward(const at::Tensor& grad_output, const at::Tensor&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_gelu_backward(const at::Tensor& grad_output, const at::Tensor&, c10::string_view) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_silu_backward(const at::Tensor& grad_output, const at::Tensor&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_leaky_relu_backward(const at::Tensor& grad_output, const at::Tensor&, const at::Scalar&, bool) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_elu_backward(const at::Tensor& grad_output, const at::Scalar&, const at::Scalar&, const at::Scalar&, bool, const at::Tensor&) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_softmax_backward_data(const at::Tensor& grad_output, const at::Tensor&, int64_t, at::ScalarType) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_log_softmax_backward_data(const at::Tensor& grad_output, const at::Tensor&, int64_t, at::ScalarType) {
    return meta_like_self(grad_output);
}

static at::Tensor meta_avg_pool2d_backward(const at::Tensor&, const at::Tensor& self,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, bool, std::optional<int64_t>) {
    return meta_like_self(self);
}

static std::tuple<at::Tensor, at::Tensor> meta_max_pool2d_with_indices(
    const at::Tensor& self, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef, bool) {
    int64_t kH = kernel_size[0], kW = kernel_size.size() > 1 ? kernel_size[1] : kH;
    int64_t sH = stride.empty() ? kH : stride[0];
    int64_t sW = stride.empty() ? kW : (stride.size() > 1 ? stride[1] : sH);
    int64_t pH = padding.empty() ? 0 : padding[0];
    int64_t pW = padding.empty() ? 0 : (padding.size() > 1 ? padding[1] : pH);
    int64_t oH = (self.size(2) + 2*pH - kH) / sH + 1;
    int64_t oW = (self.size(3) + 2*pW - kW) / sW + 1;
    auto opts = self.options().device(at::kMeta);
    return std::make_tuple(
        meta_empty({self.size(0), self.size(1), oH, oW}, opts),
        meta_empty({self.size(0), self.size(1), oH, oW}, opts.dtype(at::kLong)));
}

static at::Tensor meta_max_pool2d_with_indices_backward(
    const at::Tensor&, const at::Tensor& self,
    at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, bool, const at::Tensor&) {
    return meta_like_self(self);
}

static at::Tensor meta_embedding_dense_backward(
    const at::Tensor& grad_output, const at::Tensor&,
    c10::SymInt num_weights, c10::SymInt, bool) {
    return at::empty({num_weights.expect_int(), grad_output.size(-1)},
                     grad_output.options().device(at::kMeta));
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_native_layer_norm_backward(
    const at::Tensor&, const at::Tensor& input,
    c10::SymIntArrayRef, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>&,
    std::array<bool, 3> output_mask) {
    auto opts = input.options().device(at::kMeta);
    return std::make_tuple(
        output_mask[0] ? at::empty(input.sizes().vec(), opts) : at::Tensor(),
        output_mask[1] && weight.has_value() ? at::empty(weight->sizes().vec(), opts) : at::Tensor(),
        output_mask[2] && weight.has_value() ? at::empty(weight->sizes().vec(), opts) : at::Tensor());
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_native_group_norm_backward(
    const at::Tensor&, const at::Tensor& input,
    const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>& weight,
    c10::SymInt, c10::SymInt C, c10::SymInt, int64_t,
    std::array<bool, 3> output_mask) {
    auto opts = input.options().device(at::kMeta);
    return std::make_tuple(
        output_mask[0] ? at::empty(input.sizes().vec(), opts) : at::Tensor(),
        output_mask[1] ? at::empty({C.expect_int()}, opts) : at::Tensor(),
        output_mask[2] ? at::empty({C.expect_int()}, opts) : at::Tensor());
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_native_batch_norm_backward(
    const at::Tensor&, const at::Tensor& input,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    const std::optional<at::Tensor>&, const std::optional<at::Tensor>&,
    bool, double, std::array<bool, 3> output_mask) {
    auto opts = input.options().device(at::kMeta);
    int64_t C = input.size(1);
    return std::make_tuple(
        output_mask[0] ? at::empty(input.sizes().vec(), opts) : at::Tensor(),
        output_mask[1] ? at::empty({C}, opts) : at::Tensor(),
        output_mask[2] ? at::empty({C}, opts) : at::Tensor());
}

// ── linear_backward meta kernel ──────────────────────────────────

static std::tuple<at::Tensor, at::Tensor, at::Tensor> meta_linear_backward(
    const at::Tensor& self, const at::Tensor& grad_output,
    const at::Tensor& weight, std::array<bool, 3> output_mask) {
    auto opts = self.options().device(at::kMeta);
    return std::make_tuple(
        output_mask[0] ? at::empty(self.sizes().vec(), opts) : at::Tensor(),
        output_mask[1] ? at::empty(weight.sizes().vec(), opts) : at::Tensor(),
        output_mask[2] ? at::empty({weight.size(0)}, opts) : at::Tensor());
}

// ── Registration ────────────────────────────────────────────────

// NOTE: PyTorch 2.10+ has built-in Meta kernels for most standard ATen ops.
// We only register Meta implementations for ops where PyTorch's built-in
// decompositions or meta kernels don't cover our custom dispatch.
// Registering meta kernels with wrong signatures causes fatal errors at import.
// If torch.compile needs a meta kernel for a specific op, add it here with
// the EXACT signature matching the op schema.
// ── Phase 3: Model coverage Meta kernels ───────────────────────

// triu/tril: same shape
static at::Tensor meta_triu(const at::Tensor& self, int64_t /*diagonal*/) {
    return meta_like_self(self);
}
static at::Tensor meta_tril(const at::Tensor& self, int64_t /*diagonal*/) {
    return meta_like_self(self);
}

// constant_pad_nd: compute padded shape
static at::Tensor meta_constant_pad_nd(const at::Tensor& self,
                                        c10::SymIntArrayRef pad,
                                        const at::Scalar& /*value*/) {
    auto sizes = self.sizes().vec();
    int64_t ndim = sizes.size();
    int64_t npairs = pad.size() / 2;
    for (int64_t i = 0; i < npairs; i++) {
        int64_t dim = ndim - 1 - i;
        sizes[dim] += pad[2 * i].expect_int() + pad[2 * i + 1].expect_int();
    }
    return meta_like(self, sizes);
}

// erf: same shape
static at::Tensor meta_erf(const at::Tensor& self) {
    return meta_like_self(self);
}

// flip: same shape
static at::Tensor meta_flip(const at::Tensor& self, at::IntArrayRef /*dims*/) {
    return meta_like_self(self);
}

// roll: same shape
static at::Tensor meta_roll(const at::Tensor& self, c10::SymIntArrayRef /*shifts*/,
                             at::IntArrayRef /*dims*/) {
    return meta_like_self(self);
}

// fmod/remainder: broadcast binary, same shape
static at::Tensor meta_fmod(const at::Tensor& self, const at::Tensor& other) {
    auto out_size = at::infer_size_dimvector(self.sizes(), other.sizes());
    return meta_like(self, out_size);
}

// cumprod: same shape as input
static at::Tensor meta_cumprod(const at::Tensor& self, int64_t, std::optional<at::ScalarType>) {
    return meta_like_self(self);
}

TORCH_LIBRARY_IMPL(aten, Meta, m) {
    // Scalar-promoted binary ops — these override PyTorch's default dispatch
    // on PrivateUse1 so torch.compile's FakeTensor tracing needs Meta kernels.
    m.impl("add.Scalar", meta_binary_scalar);
    m.impl("sub.Scalar", meta_binary_scalar);
    m.impl("mul.Scalar", meta_binary_scalar_no_alpha);
    m.impl("div.Scalar", meta_binary_scalar_no_alpha);
    m.impl("pow.Tensor_Scalar", meta_pow_tensor_scalar);

    // In-place scalar variants
    m.impl("add_.Scalar", meta_inplace_scalar);
    m.impl("sub_.Scalar", meta_inplace_scalar);
    m.impl("mul_.Scalar", meta_inplace_scalar_no_alpha);
    m.impl("div_.Scalar", meta_inplace_scalar_no_alpha);

    // Scalar comparison ops
    m.impl("eq.Scalar", meta_comparison_scalar);
    m.impl("ne.Scalar", meta_comparison_scalar);
    m.impl("lt.Scalar", meta_comparison_scalar);
    m.impl("gt.Scalar", meta_comparison_scalar);
    m.impl("le.Scalar", meta_comparison_scalar);
    m.impl("ge.Scalar", meta_comparison_scalar);

    // Backward helper ops (for torch.compile tracing)
    m.impl("threshold_backward", meta_threshold_backward);
    m.impl("sigmoid_backward", meta_sigmoid_backward);
    m.impl("tanh_backward", meta_tanh_backward);
    m.impl("gelu_backward", meta_gelu_backward);
    m.impl("silu_backward", meta_silu_backward);
    m.impl("leaky_relu_backward", meta_leaky_relu_backward);
    m.impl("elu_backward", meta_elu_backward);
    m.impl("_softmax_backward_data", meta_softmax_backward_data);
    m.impl("_log_softmax_backward_data", meta_log_softmax_backward_data);
    m.impl("avg_pool2d_backward", meta_avg_pool2d_backward);
    m.impl("max_pool2d_with_indices", meta_max_pool2d_with_indices);
    m.impl("max_pool2d_with_indices_backward", meta_max_pool2d_with_indices_backward);
    m.impl("embedding_dense_backward", meta_embedding_dense_backward);
    m.impl("native_layer_norm_backward", meta_native_layer_norm_backward);
    m.impl("native_group_norm_backward", meta_native_group_norm_backward);
    m.impl("native_batch_norm_backward", meta_native_batch_norm_backward);
    m.impl("linear_backward", meta_linear_backward);

    // Phase 3: Model coverage ops
    m.impl("triu", meta_triu);
    m.impl("tril", meta_tril);
    m.impl("constant_pad_nd", meta_constant_pad_nd);
    m.impl("erf", meta_erf);
    m.impl("flip", meta_flip);
    m.impl("roll", meta_roll);

    // Upsample backward
    m.impl("upsample_nearest2d_backward", meta_upsample_nearest2d_backward);
    m.impl("upsample_bilinear2d_backward", meta_upsample_bilinear2d_backward);

    // New ops
    m.impl("fmod.Tensor", meta_fmod);
    m.impl("remainder.Tensor", meta_fmod);  // same shape logic
    m.impl("cumprod", meta_cumprod);
}

} // namespace torch_vulkan
