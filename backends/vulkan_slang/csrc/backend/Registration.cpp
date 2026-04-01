#include "Allocator.h"
#include "../vulkan/Context.h"
#include "../ops/ops.h"
#include "../ops/dispatch.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace torch_vulkan {

// ── copy_ implementation (CPU ↔ Vulkan) ─────────────────────────
at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    auto& alloc = VulkanAllocator::instance();

    if (self.device().type() == c10::DeviceType::PrivateUse1 &&
        src.device().type() == c10::DeviceType::CPU) {
        // CPU → Vulkan (with dtype conversion if needed)
        auto src_contig = src.contiguous();
        if (src_contig.scalar_type() != self.scalar_type()) {
            src_contig = src_contig.to(self.scalar_type());
        }
        auto* buf = alloc.get_buffer(self.data_ptr());
        TORCH_CHECK(buf, "Vulkan tensor has no backing buffer");
        buf->write(src_contig.data_ptr(),
                   static_cast<VkDeviceSize>(src_contig.nbytes()));
        return self;
    }

    if (self.device().type() == c10::DeviceType::CPU &&
        src.device().type() == c10::DeviceType::PrivateUse1) {
        // Note: VulkanBuffer::read() auto-flushes pending GPU work via callback
        // Vulkan → CPU (read raw bytes, then convert dtype on CPU if needed)
        if (src.scalar_type() == self.scalar_type()) {
            auto* buf = alloc.get_buffer(src.data_ptr());
            TORCH_CHECK(buf, "Vulkan tensor has no backing buffer");
            buf->read(self.data_ptr(),
                      static_cast<VkDeviceSize>(self.nbytes()));
        } else {
            // Read into a temp CPU tensor with src's dtype, then convert
            auto tmp = at::empty(src.sizes(), src.options().device(c10::kCPU));
            auto* buf = alloc.get_buffer(src.data_ptr());
            TORCH_CHECK(buf, "Vulkan tensor has no backing buffer");
            buf->read(tmp.data_ptr(),
                      static_cast<VkDeviceSize>(tmp.nbytes()));
            self.copy_(tmp);
        }
        return self;
    }

    if (self.device().type() == c10::DeviceType::PrivateUse1 &&
        src.device().type() == c10::DeviceType::PrivateUse1) {
        // Vulkan → Vulkan: use GPU strided copy shader for float32 when possible.
        // The strided copy shader only handles 4-byte float elements; other dtypes
        // must go through CPU staging (non-float32 non-contiguous should not occur
        // in practice since vulkan_t returns contiguous copies for non-float32).
        if (src.scalar_type() == self.scalar_type() && self.is_contiguous() &&
            src.dim() <= 5 && src.scalar_type() == at::kFloat) {
            // GPU strided copy: reads float32 src with arbitrary strides, writes
            // contiguously to dst. Avoids host readback.
            ops::dispatch_strided_copy(src, self);
            return self;
        }
        if (src.is_contiguous() && src.scalar_type() == self.scalar_type()) {
            // Both contiguous same dtype: use GPU buffer copy (raw byte copy)
            ops::dispatch_copy_buffer(src, self);
            return self;
        }
        // Fallback: go through CPU staging
        auto cpu_tmp = src.cpu();
        return vulkan_copy_(self, cpu_tmp, non_blocking);
    }

    TORCH_CHECK(false, "Unsupported copy direction");
    return self;
}

// ── empty.memory_format ─────────────────────────────────────────
at::Tensor vulkan_empty(
    c10::IntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt) {

    auto dtype = dtype_opt.value_or(at::ScalarType::Float);
    auto device = device_opt.value_or(c10::Device(c10::DeviceType::PrivateUse1, 0));

    auto nbytes = c10::elementSize(dtype);
    for (auto s : size) nbytes *= s;

    auto allocator = &VulkanAllocator::instance();
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(nbytes),
        allocator,
        /*resizable=*/false);

    auto type_meta = caffe2::TypeMeta::fromScalarType(dtype);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        type_meta);

    // Set sizes and strides
    auto* impl = tensor.unsafeGetTensorImpl();
    impl->set_sizes_contiguous(size);

    return tensor;
}

// ── fill_.Scalar ────────────────────────────────────────────────
at::Tensor& vulkan_fill_scalar(at::Tensor& self, const at::Scalar& value) {
    // Use GPU shader for float32, CPU fallback for other dtypes
    if (self.scalar_type() == at::kFloat && self.numel() > 0) {
        return ops::vulkan_fill_scalar_gpu(self, value);
    }
    auto cpu_tensor = at::empty(self.sizes(), self.options().device(c10::kCPU));
    cpu_tensor.fill_(value);
    vulkan_copy_(self, cpu_tensor, false);
    return self;
}

// ── _copy_from_and_resize (needed for torch.zeros/ones/full on device) ──
at::Tensor vulkan_copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
    // self is CPU source, dst is Vulkan target
    // Create a fresh Vulkan tensor with the right size, then copy
    auto new_tensor = vulkan_empty(
        self.sizes(),
        self.scalar_type(),
        self.layout(),
        dst.device(),
        false,
        c10::MemoryFormat::Contiguous);
    vulkan_copy_(new_tensor, self, false);
    return new_tensor;
}

// ── _copy_from (needed for .cpu() operations) ───────────────────
at::Tensor vulkan_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
    auto dst_mut = dst;
    vulkan_copy_(dst_mut, self, non_blocking);
    return dst_mut;
}

// ── empty_strided ───────────────────────────────────────────────
at::Tensor vulkan_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    // For now, ignore strides and create contiguous
    return vulkan_empty(size, dtype_opt, layout_opt, device_opt,
                        pin_memory_opt, c10::MemoryFormat::Contiguous);
}

// ── zero_ ───────────────────────────────────────────────────────
at::Tensor& vulkan_zero_(at::Tensor& self) {
    return vulkan_fill_scalar(self, 0);
}

// ── _local_scalar_dense (.item()) ───────────────────────────────
at::Scalar vulkan_local_scalar_dense(const at::Tensor& self) {
    TORCH_CHECK(self.numel() == 1, "_local_scalar_dense requires a 1-element tensor");
    auto cpu_tensor = self.cpu();
    return cpu_tensor.item();
}

// ── CPU fallback for unimplemented ops ──────────────────────────
// NOTE: CPU fallback doesn't work with opaque pointer allocators.
// For now, throw a clear error for unimplemented ops.
void vulkan_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false,
        "Operation '", op.schema(), "' is not yet implemented for the Vulkan backend. ",
        "Please file a feature request.");
}

// ── Helper: SymInt → int64_t conversion ────────────────────────
static std::vector<int64_t> symint_to_int(c10::SymIntArrayRef arr) {
    std::vector<int64_t> result;
    result.reserve(arr.size());
    for (const auto& s : arr) result.push_back(s.expect_int());
    return result;
}

// ── Schema adapter wrappers ────────────────────────────────────
// PyTorch 2.10 _softmax schema takes `bool half_to_float`, not `optional<ScalarType>`
static at::Tensor vulkan_softmax_adapter(const at::Tensor& self, int64_t dim, bool /*half_to_float*/) {
    return ops::vulkan_softmax(self, dim, c10::nullopt);
}

static at::Tensor vulkan_log_softmax_adapter(const at::Tensor& self, int64_t dim, bool /*half_to_float*/) {
    return ops::vulkan_log_softmax(self, dim, c10::nullopt);
}

// native_batch_norm returns tuple but our impl returns single tensor
static std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_batch_norm_adapter(
    const at::Tensor& input, const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::Tensor>& running_mean_opt,
    const std::optional<at::Tensor>& running_var_opt,
    bool training, double momentum, double eps) {
    auto result = ops::vulkan_batch_norm(input, weight_opt, bias_opt,
                                          running_mean_opt, running_var_opt,
                                          training, momentum, eps, false);
    return std::make_tuple(result, at::Tensor(), at::Tensor());
}

// native_group_norm has different signature in 2.10
static std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_group_norm_adapter(
    const at::Tensor& input, const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
    return ops::vulkan_group_norm(input, group, weight_opt, bias_opt, eps);
}

// scaled_dot_product_attention: PyTorch 2.10 adds `bool enable_gqa`
static at::Tensor vulkan_sdpa_adapter(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale, bool /*enable_gqa*/) {
    return ops::vulkan_scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale);
}

// nll_loss_forward: ignore_index is SymInt in 2.10
static std::tuple<at::Tensor, at::Tensor> vulkan_nll_loss_forward_adapter(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction, c10::SymInt ignore_index) {
    return ops::vulkan_nll_loss_forward(self, target, weight, reduction, ignore_index.expect_int());
}

// nll_loss_backward: ignore_index is SymInt in 2.10
static at::Tensor vulkan_nll_loss_backward_adapter(
    const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    c10::SymInt ignore_index, const at::Tensor& total_weight) {
    return ops::vulkan_nll_loss_backward(grad_output, self, target, weight, reduction,
                                          ignore_index.expect_int(), total_weight);
}

// cross_entropy_loss: ignore_index is SymInt in 2.10
static at::Tensor vulkan_cross_entropy_loss_adapter(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    c10::SymInt ignore_index, double label_smoothing) {
    return ops::vulkan_cross_entropy_loss(self, target, weight, reduction, ignore_index.expect_int(), label_smoothing);
}

// convolution_overrideable: the actual dispatch point for conv1d/conv2d/conv_transpose2d
static at::Tensor vulkan_convolution_overrideable_adapter(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups) {
    if (transposed) {
        return ops::vulkan_conv_transpose2d(input, weight, bias_opt,
            symint_to_int(stride), symint_to_int(padding), symint_to_int(output_padding),
            groups.expect_int(), symint_to_int(dilation));
    }
    // Conv1d: unsqueeze 3D→4D, run conv2d, squeeze back
    if (input.dim() == 3 && weight.dim() == 3) {
        auto input_4d = input.unsqueeze(2);  // [N, C, L] → [N, C, 1, L]
        auto weight_4d = weight.unsqueeze(2); // [C_out, C_in/g, K] → [C_out, C_in/g, 1, K]
        auto s = symint_to_int(stride);
        auto p = symint_to_int(padding);
        auto d = symint_to_int(dilation);
        std::vector<int64_t> stride_2d = {1, s[0]};
        std::vector<int64_t> padding_2d = {0, p[0]};
        std::vector<int64_t> dilation_2d = {1, d[0]};
        auto result = ops::vulkan_conv2d(input_4d, weight_4d, bias_opt,
            stride_2d, padding_2d, dilation_2d, groups.expect_int());
        return result.squeeze(2); // [N, C_out, 1, L_out] → [N, C_out, L_out]
    }
    // Conv3d: decompose into Conv2d by processing temporal slices
    // Input: [N, C_in, D, H, W], Weight: [C_out, C_in/g, kD, kH, kW]
    if (input.dim() == 5 && weight.dim() == 5) {
        auto s = symint_to_int(stride);
        auto p = symint_to_int(padding);
        auto d = symint_to_int(dilation);
        TORCH_CHECK(s.size() == 3 && p.size() == 3 && d.size() == 3,
                    "conv3d: expected 3D stride/padding/dilation");

        int64_t N = input.size(0), C_in = input.size(1);
        int64_t D = input.size(2), H = input.size(3), W = input.size(4);
        int64_t C_out = weight.size(0);
        int64_t kD = weight.size(2);
        int64_t sD = s[0], sH = s[1], sW = s[2];
        int64_t pD = p[0], pH = p[1], pW = p[2];
        int64_t dD = d[0], dH = d[1], dW = d[2];

        // Compute output sizes
        int64_t D_out = (D + 2 * pD - dD * (kD - 1) - 1) / sD + 1;

        auto input_c = input.contiguous();
        auto weight_c = weight.contiguous();
        std::vector<int64_t> stride_2d = {sH, sW};
        std::vector<int64_t> padding_2d = {pH, pW};
        std::vector<int64_t> dilation_2d = {dH, dW};

        // Process each output temporal position, collecting 2D conv results
        std::vector<at::Tensor> temporal_results;
        for (int64_t t_out = 0; t_out < D_out; t_out++) {
            at::Tensor accum;
            for (int64_t kt = 0; kt < kD; kt++) {
                int64_t t_in = t_out * sD - pD + kt * dD;
                if (t_in < 0 || t_in >= D) continue;

                // input_slice: [N, C_in, H, W] — select returns copy on Vulkan
                auto input_slice = input_c.select(2, t_in).contiguous();
                // weight_slice: [C_out, C_in/g, kH, kW]
                auto weight_slice = weight_c.select(2, kt).contiguous();

                auto conv_result = ops::vulkan_conv2d(input_slice, weight_slice,
                    std::nullopt, stride_2d, padding_2d, dilation_2d,
                    groups.expect_int());

                if (!accum.defined()) {
                    accum = conv_result;
                } else {
                    accum = accum + conv_result;
                }
            }
            temporal_results.push_back(accum);  // [N, C_out, H_out, W_out]
        }

        // Stack along temporal dim: [N, C_out, D_out, H_out, W_out]
        auto output = at::stack(temporal_results, 2);

        // Add bias if present (broadcast over spatial dims)
        if (bias_opt.has_value() && bias_opt->defined()) {
            auto bias = bias_opt->contiguous();
            output = output + bias.reshape({1, C_out, 1, 1, 1});
        }

        return output;
    }
    return ops::vulkan_conv2d(input, weight, bias_opt,
        symint_to_int(stride), symint_to_int(padding), symint_to_int(dilation), groups.expect_int());
}

// convolution_backward_overrideable: CPU fallback for conv backward
static std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_convolution_backward_overrideable_adapter(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups,
    std::array<bool, 3> output_mask) {
    // CPU fallback for convolution backward
    // Must convert to f32 for CPU backward (f16/bf16 not well-supported on CPU)
    auto orig_dtype = grad_output.scalar_type();
    auto grad_cpu = grad_output.cpu().to(at::kFloat);
    auto input_cpu = input.cpu().to(at::kFloat);
    auto weight_cpu = weight.cpu().to(at::kFloat);
    auto result = at::convolution_backward(
        grad_cpu, input_cpu, weight_cpu,
        /*bias_sizes_opt=*/output_mask[2] ? std::optional<c10::IntArrayRef>(c10::IntArrayRef{weight.size(0)}) : std::nullopt,
        symint_to_int(stride), symint_to_int(padding), symint_to_int(dilation),
        transposed, symint_to_int(output_padding), groups.expect_int(), output_mask);
    auto dev = input.device();
    return std::make_tuple(
        output_mask[0] ? std::get<0>(result).to(orig_dtype).to(dev) : at::Tensor(),
        output_mask[1] ? std::get<1>(result).to(orig_dtype).to(dev) : at::Tensor(),
        output_mask[2] ? std::get<2>(result).to(orig_dtype).to(dev) : at::Tensor());
}

// convolution: intercept before _convolution decomposition to handle None bias
static at::Tensor vulkan_convolution_adapter(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups) {
    return vulkan_convolution_overrideable_adapter(
        input, weight, bias_opt, stride, padding, dilation, transposed, output_padding, groups);
}

// upsample_nearest2d: SymInt output_size in 2.10
static at::Tensor vulkan_upsample_nearest2d_adapter(
    const at::Tensor& self, c10::SymIntArrayRef output_size,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    return ops::vulkan_upsample_nearest2d(self, symint_to_int(output_size), scales_h, scales_w);
}

// upsample_nearest2d_backward: SymInt sizes
static at::Tensor vulkan_upsample_nearest2d_backward_adapter(
    const at::Tensor& grad_output, c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    return ops::vulkan_upsample_nearest2d_backward(
        grad_output, symint_to_int(output_size), symint_to_int(input_size), scales_h, scales_w);
}

static at::Tensor& vulkan_upsample_nearest2d_backward_grad_input_adapter(
    const at::Tensor& grad_output, c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size,
    std::optional<double> scales_h, std::optional<double> scales_w,
    at::Tensor& grad_input) {
    auto result = ops::vulkan_upsample_nearest2d_backward(
        grad_output, symint_to_int(output_size), symint_to_int(input_size), scales_h, scales_w);
    grad_input.copy_(result);
    return grad_input;
}

// upsample_bilinear2d: SymInt output_size in 2.10
static at::Tensor vulkan_upsample_bilinear2d_adapter(
    const at::Tensor& self, c10::SymIntArrayRef output_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    return ops::vulkan_upsample_bilinear2d(self, symint_to_int(output_size), align_corners, scales_h, scales_w);
}

// upsample_bilinear2d_backward: SymInt sizes
static at::Tensor vulkan_upsample_bilinear2d_backward_adapter(
    const at::Tensor& grad_output, c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w) {
    return ops::vulkan_upsample_bilinear2d_backward(
        grad_output, symint_to_int(output_size), symint_to_int(input_size),
        align_corners, scales_h, scales_w);
}

static at::Tensor& vulkan_upsample_bilinear2d_backward_grad_input_adapter(
    const at::Tensor& grad_output, c10::SymIntArrayRef output_size,
    c10::SymIntArrayRef input_size, bool align_corners,
    std::optional<double> scales_h, std::optional<double> scales_w, at::Tensor& grad_input) {
    auto result = ops::vulkan_upsample_bilinear2d_backward(
        grad_output, symint_to_int(output_size), symint_to_int(input_size),
        align_corners, scales_h, scales_w);
    grad_input.copy_(result);
    return grad_input;
}

// empty.memory_format: SymInt size in 2.10
static at::Tensor vulkan_empty_adapter(
    c10::SymIntArrayRef size,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<at::MemoryFormat> memory_format_opt) {
    return vulkan_empty(symint_to_int(size), dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
}

// empty_strided: SymInt size and stride in 2.10
static at::Tensor vulkan_empty_strided_adapter(
    c10::SymIntArrayRef size, c10::SymIntArrayRef stride,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    return vulkan_empty_strided(symint_to_int(size), symint_to_int(stride),
        dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// view: SymInt size in 2.10
static at::Tensor vulkan_view_adapter(const at::Tensor& self, c10::SymIntArrayRef size) {
    return ops::vulkan_view(self, symint_to_int(size));
}

// reshape: SymInt shape in 2.10
static at::Tensor vulkan_reshape_adapter(const at::Tensor& self, c10::SymIntArrayRef shape) {
    return ops::vulkan_reshape(self, symint_to_int(shape));
}

// Autograd-aware view/reshape adapters
static at::Tensor vulkan_view_autograd_adapter(const at::Tensor& self, c10::SymIntArrayRef size) {
    return ops::vulkan_view_autograd(self, symint_to_int(size));
}

static at::Tensor vulkan_reshape_autograd_adapter(const at::Tensor& self, c10::SymIntArrayRef shape) {
    return ops::vulkan_reshape_autograd(self, symint_to_int(shape));
}

// Autograd-aware shape op adapters
static at::Tensor vulkan_expand_autograd_adapter(const at::Tensor& self, c10::SymIntArrayRef size, bool implicit) {
    return ops::vulkan_expand_autograd(self, symint_to_int(size), implicit);
}

static at::Tensor vulkan_select_autograd_adapter(const at::Tensor& self, int64_t dim, c10::SymInt index) {
    return ops::vulkan_select_autograd(self, dim, index.expect_int());
}

static at::Tensor vulkan_slice_autograd_adapter(const at::Tensor& self, int64_t dim,
    std::optional<c10::SymInt> start, std::optional<c10::SymInt> end, c10::SymInt step) {
    std::optional<int64_t> start_int = start.has_value() ? std::optional<int64_t>(start->expect_int()) : std::nullopt;
    std::optional<int64_t> end_int = end.has_value() ? std::optional<int64_t>(end->expect_int()) : std::nullopt;
    return ops::vulkan_slice_autograd(self, dim, start_int, end_int, step.expect_int());
}

// expand: SymInt size in 2.10
static at::Tensor vulkan_expand_adapter(const at::Tensor& self, c10::SymIntArrayRef size, bool implicit) {
    return ops::vulkan_expand(self, symint_to_int(size), implicit);
}

// slice.Tensor: SymInt start/end/step in 2.10
static at::Tensor vulkan_slice_adapter(const at::Tensor& self, int64_t dim,
    std::optional<c10::SymInt> start, std::optional<c10::SymInt> end, c10::SymInt step) {
    std::optional<int64_t> start_int = start.has_value() ? std::optional<int64_t>(start->expect_int()) : std::nullopt;
    std::optional<int64_t> end_int = end.has_value() ? std::optional<int64_t>(end->expect_int()) : std::nullopt;
    return ops::vulkan_slice(self, dim, start_int, end_int, step.expect_int());
}

// split.Tensor: SymInt split_size in 2.10
static std::vector<at::Tensor> vulkan_split_adapter(const at::Tensor& self, c10::SymInt split_size, int64_t dim) {
    return ops::vulkan_split(self, split_size.expect_int(), dim);
}

// select.int: SymInt index in 2.10
static at::Tensor vulkan_select_adapter(const at::Tensor& self, int64_t dim, c10::SymInt index) {
    return ops::vulkan_select(self, dim, index.expect_int());
}

// embedding: SymInt padding_idx in 2.10
static at::Tensor vulkan_embedding_adapter(
    const at::Tensor& weight, const at::Tensor& indices,
    c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
    return ops::vulkan_embedding(weight, indices, padding_idx.expect_int(), scale_grad_by_freq, sparse);
}

// native_layer_norm: SymInt[] normalized_shape in 2.10
static std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_layer_norm_adapter(
    const at::Tensor& input, c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    double eps) {
    return ops::vulkan_layer_norm(input, symint_to_int(normalized_shape), weight_opt, bias_opt, eps);
}

// native_group_norm adapter already exists but needs SymInt N/C/HxW
static std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_group_norm_symint_adapter(
    const at::Tensor& input, const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, double eps) {
    return ops::vulkan_group_norm(input, group, weight_opt, bias_opt, eps);
}

// adaptive_avg_pool2d: SymInt output_size in 2.10
static at::Tensor vulkan_adaptive_avg_pool2d_adapter(
    const at::Tensor& self, c10::SymIntArrayRef output_size) {
    return ops::vulkan_adaptive_avg_pool2d(self, symint_to_int(output_size));
}
static at::Tensor vulkan_adaptive_avg_pool2d_autograd_adapter(
    const at::Tensor& self, c10::SymIntArrayRef output_size) {
    return ops::vulkan_adaptive_avg_pool2d_autograd(self, symint_to_int(output_size));
}

// full: SymInt[] size in 2.10
static at::Tensor vulkan_full_adapter(
    c10::SymIntArrayRef size, const at::Scalar& fill_value,
    std::optional<at::ScalarType> dtype_opt, std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt, std::optional<bool> pin_memory_opt) {
    return ops::vulkan_full(symint_to_int(size), fill_value, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// eye: SymInt n in 2.10
static at::Tensor vulkan_eye_adapter(
    c10::SymInt n, std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    return ops::vulkan_eye(n.expect_int(), dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// eye.m: SymInt n, m in 2.10
static at::Tensor vulkan_eye_m_adapter(
    c10::SymInt n, c10::SymInt m, std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
    return ops::vulkan_eye_m(n.expect_int(), m.expect_int(), dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// triu/tril: SymInt diagonal in 2.10
static at::Tensor vulkan_triu_adapter(const at::Tensor& self, c10::SymInt diagonal) {
    return ops::vulkan_triu(self, diagonal.expect_int());
}
static at::Tensor vulkan_tril_adapter(const at::Tensor& self, c10::SymInt diagonal) {
    return ops::vulkan_tril(self, diagonal.expect_int());
}

// repeat: SymInt repeats in 2.10
static at::Tensor vulkan_repeat_adapter(const at::Tensor& self, c10::SymIntArrayRef repeats) {
    return ops::vulkan_repeat(self, symint_to_int(repeats));
}

// repeat_interleave: SymInt repeats in 2.10
static at::Tensor vulkan_repeat_interleave_self_int_adapter(
    const at::Tensor& self, c10::SymInt repeats,
    std::optional<int64_t> dim, std::optional<c10::SymInt> output_size) {
    std::optional<int64_t> os;
    if (output_size.has_value()) os = output_size->expect_int();
    return ops::vulkan_repeat_interleave_self_int(self, repeats.expect_int(), dim, os);
}

// constant_pad_nd: SymInt pad in 2.10
static at::Tensor vulkan_constant_pad_nd_adapter(
    const at::Tensor& self, c10::SymIntArrayRef pad, const at::Scalar& value) {
    return ops::vulkan_constant_pad_nd(self, symint_to_int(pad), value);
}

// narrow: SymInt start/length in 2.10
static at::Tensor vulkan_narrow_adapter(const at::Tensor& self, int64_t dim,
    c10::SymInt start, c10::SymInt length) {
    return ops::vulkan_narrow(self, dim, start.expect_int(), length.expect_int());
}

// _unsafe_view: SymInt size in 2.10
static at::Tensor vulkan_unsafe_view_adapter(const at::Tensor& self, c10::SymIntArrayRef size) {
    return ops::vulkan_unsafe_view(self, symint_to_int(size));
}

// _to_copy: SymInt-aware in 2.10
static at::Tensor vulkan_to_copy_adapter(const at::Tensor& self,
    std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
    std::optional<at::Device> device, std::optional<bool> pin_memory,
    bool non_blocking, std::optional<at::MemoryFormat> memory_format) {
    return ops::vulkan_to_copy(self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

// as_strided: SymInt in 2.10
static at::Tensor vulkan_as_strided_adapter(const at::Tensor& self,
    c10::SymIntArrayRef size, c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
    std::optional<int64_t> so;
    if (storage_offset.has_value()) so = storage_offset->expect_int();
    return ops::vulkan_as_strided(self, symint_to_int(size), symint_to_int(stride), so);
}

// resize_: SymInt in 2.10
static const at::Tensor& vulkan_resize_adapter(const at::Tensor& self,
    c10::SymIntArrayRef size, std::optional<at::MemoryFormat> memory_format) {
    return ops::vulkan_resize_(self, symint_to_int(size), memory_format);
}

// roll: SymInt in 2.10
static at::Tensor vulkan_roll_adapter(const at::Tensor& self,
    c10::SymIntArrayRef shifts, at::IntArrayRef dims) {
    return ops::vulkan_roll(self, symint_to_int(shifts), dims);
}

// topk: SymInt k in 2.10
static std::tuple<at::Tensor, at::Tensor> vulkan_topk_adapter(
    const at::Tensor& self, c10::SymInt k, int64_t dim, bool largest, bool sorted) {
    return ops::vulkan_topk(self, k.expect_int(), dim, largest, sorted);
}

// embedding autograd adapter (SymInt padding_idx)
static at::Tensor vulkan_embedding_autograd_adapter(
    const at::Tensor& weight, const at::Tensor& indices,
    c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
    return ops::vulkan_embedding_autograd(weight, indices, padding_idx.expect_int(), scale_grad_by_freq, sparse);
}

// layer_norm autograd adapter (SymInt normalized_shape)
static std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_layer_norm_autograd_adapter(
    const at::Tensor& input, c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    double eps) {
    return ops::vulkan_layer_norm_autograd(input, symint_to_int(normalized_shape), weight_opt, bias_opt, eps);
}

// group_norm autograd adapter (SymInt N/C/HxW)
static std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_group_norm_autograd_symint_adapter(
    const at::Tensor& input, const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    c10::SymInt N, c10::SymInt C, c10::SymInt HxW, int64_t group, double eps) {
    return ops::vulkan_group_norm_autograd(input, group, weight_opt, bias_opt, eps);
}

// convolution_overrideable autograd adapter
static at::Tensor vulkan_convolution_overrideable_autograd_adapter(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias_opt,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups) {
    // For autograd, delegate to conv2d_autograd which handles forward+backward
    if (transposed) {
        // Fall back for transpose conv autograd
        return vulkan_convolution_overrideable_adapter(input, weight, bias_opt,
            stride, padding, dilation, transposed, output_padding, groups);
    }
    // Conv1d autograd: unsqueeze, run conv2d_autograd, squeeze
    if (input.dim() == 3 && weight.dim() == 3) {
        auto input_4d = input.unsqueeze(2);
        auto weight_4d = weight.unsqueeze(2);
        auto s = symint_to_int(stride);
        auto p = symint_to_int(padding);
        auto d = symint_to_int(dilation);
        std::vector<int64_t> stride_2d = {1, s[0]};
        std::vector<int64_t> padding_2d = {0, p[0]};
        std::vector<int64_t> dilation_2d = {1, d[0]};
        auto result = ops::vulkan_conv2d_autograd(input_4d, weight_4d, bias_opt,
            stride_2d, padding_2d, dilation_2d, groups.expect_int());
        return result.squeeze(2);
    }
    // Conv3d autograd: delegate to non-autograd adapter (uses decomposition
    // into conv2d + add which already have autograd support)
    if (input.dim() == 5 && weight.dim() == 5) {
        return vulkan_convolution_overrideable_adapter(input, weight, bias_opt,
            stride, padding, dilation, transposed, output_padding, groups);
    }
    return ops::vulkan_conv2d_autograd(input, weight, bias_opt,
        symint_to_int(stride), symint_to_int(padding), symint_to_int(dilation), groups.expect_int());
}

// SDPA autograd adapter (enable_gqa)
static at::Tensor vulkan_sdpa_autograd_adapter(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale, bool /*enable_gqa*/) {
    return ops::vulkan_sdpa_autograd(query, key, value, attn_mask, dropout_p, is_causal, scale);
}

// arange.default: only end param
static at::Tensor vulkan_arange_default_adapter(
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype_opt, std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt, std::optional<bool> pin_memory_opt) {
    return ops::vulkan_arange(at::Scalar(0), end, at::Scalar(1),
        dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// arange.start: start+end params
static at::Tensor vulkan_arange_start_adapter(
    const at::Scalar& start, const at::Scalar& end,
    std::optional<at::ScalarType> dtype_opt, std::optional<at::Layout> layout_opt,
    std::optional<at::Device> device_opt, std::optional<bool> pin_memory_opt) {
    return ops::vulkan_arange(start, end, at::Scalar(1),
        dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

// ── Register ops ────────────────────────────────────────────────
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Memory & tensor creation
    m.impl("empty.memory_format", vulkan_empty_adapter);
    m.impl("empty_strided", vulkan_empty_strided_adapter);
    m.impl("copy_", vulkan_copy_);
    m.impl("fill_.Scalar", vulkan_fill_scalar);
    m.impl("_copy_from_and_resize", vulkan_copy_from_and_resize);
    m.impl("_copy_from", vulkan_copy_from);
    m.impl("zero_", vulkan_zero_);
    m.impl("clone", ops::vulkan_clone);
    m.impl("_local_scalar_dense", vulkan_local_scalar_dense);

    // Binary ops
    m.impl("add.Tensor", ops::vulkan_add);
    m.impl("sub.Tensor", ops::vulkan_sub);
    m.impl("mul.Tensor", ops::vulkan_mul);
    m.impl("add.Scalar", ops::vulkan_add_scalar);
    m.impl("sub.Scalar", ops::vulkan_sub_scalar);
    m.impl("mul.Scalar", ops::vulkan_mul_scalar);
    m.impl("div.Scalar", ops::vulkan_div_scalar);
    m.impl("div.Tensor", ops::vulkan_div);
    m.impl("pow.Tensor_Tensor", ops::vulkan_pow);
    m.impl("pow.Tensor_Scalar", ops::vulkan_pow_scalar);
    m.impl("fmod.Tensor", ops::vulkan_fmod);
    m.impl("remainder.Tensor", ops::vulkan_remainder);

    // In-place binary ops
    m.impl("add_.Tensor", ops::vulkan_add_);
    m.impl("sub_.Tensor", ops::vulkan_sub_);
    m.impl("mul_.Tensor", ops::vulkan_mul_);
    m.impl("div_.Tensor", ops::vulkan_div_);
    m.impl("add_.Scalar", ops::vulkan_add_scalar_);
    m.impl("sub_.Scalar", ops::vulkan_sub_scalar_);
    m.impl("mul_.Scalar", ops::vulkan_mul_scalar_);
    m.impl("div_.Scalar", ops::vulkan_div_scalar_);

    // Unary ops
    m.impl("neg", ops::vulkan_neg);
    m.impl("abs", ops::vulkan_abs);
    m.impl("exp", ops::vulkan_exp);
    m.impl("log", ops::vulkan_log);
    m.impl("sqrt", ops::vulkan_sqrt);
    m.impl("rsqrt", ops::vulkan_rsqrt);
    m.impl("ceil", ops::vulkan_ceil);
    m.impl("floor", ops::vulkan_floor);
    m.impl("round", ops::vulkan_round);
    m.impl("sign", ops::vulkan_sign);
    m.impl("sgn", ops::vulkan_sign);

    // Comparison ops (tensor)
    m.impl("eq.Tensor", ops::vulkan_eq);
    m.impl("ne.Tensor", ops::vulkan_ne);
    m.impl("lt.Tensor", ops::vulkan_lt);
    m.impl("gt.Tensor", ops::vulkan_gt);
    m.impl("le.Tensor", ops::vulkan_le);
    m.impl("ge.Tensor", ops::vulkan_ge);
    m.impl("where.self", ops::vulkan_where);
    // Comparison ops (scalar)
    m.impl("eq.Scalar", ops::vulkan_eq_scalar);
    m.impl("ne.Scalar", ops::vulkan_ne_scalar);
    m.impl("lt.Scalar", ops::vulkan_lt_scalar);
    m.impl("gt.Scalar", ops::vulkan_gt_scalar);
    m.impl("le.Scalar", ops::vulkan_le_scalar);
    m.impl("ge.Scalar", ops::vulkan_ge_scalar);

    // Activations
    m.impl("relu", ops::vulkan_relu);
    m.impl("relu_", ops::vulkan_relu_);
    m.impl("sigmoid", ops::vulkan_sigmoid);
    m.impl("tanh", ops::vulkan_tanh);
    m.impl("gelu", ops::vulkan_gelu);
    m.impl("silu", ops::vulkan_silu);
    m.impl("leaky_relu", ops::vulkan_leaky_relu);
    m.impl("elu", ops::vulkan_elu);
    m.impl("clamp", ops::vulkan_clamp);
    m.impl("clamp_min", ops::vulkan_clamp_min);
    m.impl("clamp_min_", ops::vulkan_clamp_min_);
    m.impl("clamp_min.Tensor", ops::vulkan_clamp_min_tensor);
    m.impl("clamp_min.Tensor_out", ops::vulkan_clamp_min_tensor_out);
    m.impl("clamp_max", ops::vulkan_clamp_max);
    m.impl("clamp_max_", ops::vulkan_clamp_max_);
    m.impl("clamp_min.out", ops::vulkan_clamp_min_out);
    m.impl("clamp_max.out", ops::vulkan_clamp_max_out);
    m.impl("selu", ops::vulkan_selu);
    m.impl("prelu", ops::vulkan_prelu);
    m.impl("hardtanh", ops::vulkan_hardtanh);
    m.impl("hardtanh_", ops::vulkan_hardtanh_);
    m.impl("hardswish", ops::vulkan_hardswish);
    m.impl("hardswish_", ops::vulkan_hardswish_);
    m.impl("hardsigmoid", ops::vulkan_hardsigmoid);
    m.impl("hardsigmoid_", ops::vulkan_hardsigmoid_);
    m.impl("softplus", ops::vulkan_softplus);
    m.impl("mish", ops::vulkan_mish);

    // Reductions
    m.impl("sum.dim_IntList", ops::vulkan_sum);
    m.impl("mean.dim", ops::vulkan_mean);
    m.impl("amax", ops::vulkan_amax);
    m.impl("amin", ops::vulkan_amin);
    m.impl("max.dim", ops::vulkan_max_dim);
    m.impl("min.dim", ops::vulkan_min_dim);
    m.impl("prod.dim_int", ops::vulkan_prod);
    m.impl("argmax", ops::vulkan_argmax);
    m.impl("argmin", ops::vulkan_argmin);
    m.impl("any", ops::vulkan_any);
    m.impl("any.dim", ops::vulkan_any_dim);
    m.impl("all", ops::vulkan_all);
    m.impl("all.dim", ops::vulkan_all_dim);
    m.impl("linalg_vector_norm", ops::vulkan_norm);
    m.impl("norm.ScalarOpt_dim", ops::vulkan_norm_ScalarOpt_dim);

    // Shape ops — view/reshape need autograd wrapping (registered on AutogradPrivateUse1).
    // Other shape ops are registered here because as_strided only supports float32,
    // so we need direct implementations for non-float dtypes (bool, int, etc.).
    m.impl("view", vulkan_view_adapter);
    m.impl("reshape", vulkan_reshape_adapter);
    m.impl("unsqueeze", ops::vulkan_unsqueeze);
    m.impl("squeeze", ops::vulkan_squeeze);
    m.impl("squeeze.dim", ops::vulkan_squeeze_dim);
    m.impl("permute", ops::vulkan_permute);
    m.impl("transpose.int", ops::vulkan_transpose);
    m.impl("t", ops::vulkan_t);
    m.impl("expand", vulkan_expand_adapter);
    m.impl("cat", ops::vulkan_cat);
    m.impl("narrow", vulkan_narrow_adapter);
    m.impl("select.int", vulkan_select_adapter);
    m.impl("slice.Tensor", vulkan_slice_adapter);
    m.impl("split.Tensor", vulkan_split_adapter);

    // Softmax
    m.impl("_softmax", vulkan_softmax_adapter);
    m.impl("_log_softmax", vulkan_log_softmax_adapter);

    // Normalization
    m.impl("native_layer_norm", vulkan_layer_norm_adapter);
    m.impl("native_batch_norm", vulkan_batch_norm_adapter);
    m.impl("native_group_norm", vulkan_group_norm_symint_adapter);

    // Pooling
    m.impl("max_pool2d", ops::vulkan_max_pool2d);
    m.impl("avg_pool2d", ops::vulkan_avg_pool2d);
    m.impl("adaptive_avg_pool2d", vulkan_adaptive_avg_pool2d_adapter);

    // Indexing
    m.impl("embedding", vulkan_embedding_adapter);
    m.impl("index_select", ops::vulkan_index_select);
    m.impl("masked_fill_.Scalar", ops::vulkan_masked_fill);

    // Convolution — only register convolution_overrideable (not convolution).
    // Do NOT register convolution: its Tensor? bias arg causes dispatch key
    // computation to fail with "tensor does not have a device" when bias=None.
    m.impl("convolution_overrideable", vulkan_convolution_overrideable_adapter);
    m.impl("convolution_backward_overrideable", vulkan_convolution_backward_overrideable_adapter);

    // BLAS
    m.impl("mm", ops::vulkan_mm);
    m.impl("addmm", ops::vulkan_addmm);
    m.impl("bmm", ops::vulkan_bmm);
    m.impl("linear", ops::vulkan_linear);
    m.impl("_scaled_mm", ops::vulkan_scaled_mm);

    // Loss functions
    m.impl("nll_loss_forward", vulkan_nll_loss_forward_adapter);
    m.impl("nll_loss_backward", vulkan_nll_loss_backward_adapter);
    // NOTE: cross_entropy_loss NOT registered — let PyTorch decompose via
    // CompositeImplicitAutograd into log_softmax + nll_loss for proper autograd backward
    m.impl("mse_loss", ops::vulkan_mse_loss);
    m.impl("mse_loss_backward", ops::vulkan_mse_loss_backward);
    m.impl("binary_cross_entropy", ops::vulkan_binary_cross_entropy);
    m.impl("binary_cross_entropy_backward", ops::vulkan_binary_cross_entropy_backward);
    m.impl("binary_cross_entropy_with_logits", ops::vulkan_binary_cross_entropy_with_logits);
    m.impl("smooth_l1_loss", ops::vulkan_smooth_l1_loss);
    m.impl("smooth_l1_loss_backward", ops::vulkan_smooth_l1_loss_backward);
    m.impl("huber_loss", ops::vulkan_huber_loss);
    m.impl("huber_loss_backward", ops::vulkan_huber_loss_backward);
    m.impl("kl_div", ops::vulkan_kl_div);

    // Tensor factories
    m.impl("arange", vulkan_arange_default_adapter);
    m.impl("arange.start", vulkan_arange_start_adapter);
    m.impl("arange.start_step", ops::vulkan_arange);
    m.impl("linspace", ops::vulkan_linspace);
    m.impl("eye", vulkan_eye_adapter);
    m.impl("eye.m", vulkan_eye_m_adapter);
    m.impl("full", vulkan_full_adapter);
    m.impl("scalar_tensor", ops::vulkan_scalar_tensor);

    // RNG
    m.impl("uniform_", ops::vulkan_uniform_);
    m.impl("normal_", ops::vulkan_normal_);
    m.impl("native_dropout", ops::vulkan_native_dropout);
    m.impl("native_dropout_backward", ops::vulkan_native_dropout_backward);
    m.impl("bernoulli_.float", ops::vulkan_bernoulli_);
    m.impl("bernoulli_.Tensor", ops::vulkan_bernoulli_p);

    // Attention
    m.impl("scaled_dot_product_attention", vulkan_sdpa_adapter);

    // Advanced ops
    m.impl("cumsum", ops::vulkan_cumsum);
    m.impl("cumprod", ops::vulkan_cumprod);
    m.impl("sort", ops::vulkan_sort);
    m.impl("topk", vulkan_topk_adapter);
    m.impl("gather", ops::vulkan_gather);
    m.impl("scatter_.src", ops::vulkan_scatter_);
    m.impl("index_put_", ops::vulkan_index_put_);
    m.impl("upsample_nearest2d", vulkan_upsample_nearest2d_adapter);
    m.impl("upsample_nearest2d_backward", vulkan_upsample_nearest2d_backward_adapter);
    m.impl("upsample_nearest2d_backward.grad_input", vulkan_upsample_nearest2d_backward_grad_input_adapter);
    m.impl("upsample_bilinear2d", vulkan_upsample_bilinear2d_adapter);
    m.impl("upsample_bilinear2d_backward", vulkan_upsample_bilinear2d_backward_adapter);
    m.impl("upsample_bilinear2d_backward.grad_input", vulkan_upsample_bilinear2d_backward_grad_input_adapter);
    m.impl("grid_sampler_2d", ops::vulkan_grid_sampler_2d);

    // Optimizer ops
    m.impl("addcmul_", ops::vulkan_addcmul_);
    m.impl("addcdiv_", ops::vulkan_addcdiv_);
    m.impl("lerp_.Scalar", ops::vulkan_lerp_);
    m.impl("clamp_", ops::vulkan_clamp_);

    // Foreach ops (fused optimizer support)
    m.impl("_foreach_add_.Scalar", ops::vulkan_foreach_add_scalar_);
    m.impl("_foreach_add_.List", ops::vulkan_foreach_add_list_);
    m.impl("_foreach_mul_.Scalar", ops::vulkan_foreach_mul_scalar_);
    m.impl("_foreach_addcmul_.Scalar", ops::vulkan_foreach_addcmul_);
    m.impl("_foreach_addcdiv_.Scalar", ops::vulkan_foreach_addcdiv_);
    m.impl("_foreach_sqrt", ops::vulkan_foreach_sqrt);
    m.impl("_foreach_neg", ops::vulkan_foreach_neg);
    m.impl("_foreach_div_.Scalar", ops::vulkan_foreach_div_scalar_);
    m.impl("_foreach_lerp_.Scalar", ops::vulkan_foreach_lerp_);
    m.impl("_foreach_maximum.List", ops::vulkan_foreach_maximum);

    // AMP ops
    m.impl("_amp_foreach_non_finite_check_and_unscale_",
           ops::vulkan_amp_non_finite_check_and_unscale_);
    m.impl("_amp_update_scale_", ops::vulkan_amp_update_scale_);

    // Additional unary ops
    m.impl("reciprocal", ops::vulkan_reciprocal);
    m.impl("sin", ops::vulkan_sin);
    m.impl("cos", ops::vulkan_cos);
    m.impl("tan", ops::vulkan_tan);
    m.impl("atan", ops::vulkan_atan);
    m.impl("log2", ops::vulkan_log2);
    m.impl("log10", ops::vulkan_log10);
    m.impl("log1p", ops::vulkan_log1p);
    m.impl("logical_not", ops::vulkan_logical_not);
    m.impl("bitwise_not", ops::vulkan_bitwise_not);
    m.impl("bitwise_and.Tensor_out", ops::vulkan_bitwise_and_out);
    m.impl("random_.from", ops::vulkan_random_from);

    // Check ops
    m.impl("isnan", ops::vulkan_isnan);
    m.impl("isinf", ops::vulkan_isinf);

    // Additional binary ops
    m.impl("atan2", ops::vulkan_atan2);

    // Phase 3: Model coverage ops
    m.impl("triu", vulkan_triu_adapter);
    m.impl("tril", vulkan_tril_adapter);
    m.impl("constant_pad_nd", vulkan_constant_pad_nd_adapter);
    m.impl("index.Tensor", ops::vulkan_index_tensor);
    m.impl("repeat", vulkan_repeat_adapter);
    m.impl("repeat_interleave.self_int", vulkan_repeat_interleave_self_int_adapter);
    m.impl("stack", ops::vulkan_stack);
    m.impl("erf", ops::vulkan_erf);
    m.impl("erf_", ops::vulkan_erf_);
    m.impl("flip", ops::vulkan_flip);
    m.impl("roll", vulkan_roll_adapter);
    m.impl("_unsafe_view", vulkan_unsafe_view_adapter);
    m.impl("as_strided", vulkan_as_strided_adapter);
    m.impl("resize_", vulkan_resize_adapter);

    // Missing registrations for implemented ops
    m.impl("masked_scatter_", ops::vulkan_masked_scatter_);
    // NOTE: Do NOT register "chunk" — it's CompositeImplicitAutograd and decomposes
    // into slice ops. Registering it on PrivateUse1 breaks autograd ("derivative not implemented").
    m.impl("kl_div_backward", ops::vulkan_kl_div_backward);

    // Backward helper ops (for PyTorch's built-in autograd decompositions)
    // These enable removing custom AutogradPrivateUse1 registrations for standard ops,
    // which in turn enables torch.compile/AOT Autograd tracing.
    m.impl("threshold_backward", ops::vulkan_threshold_backward);
    m.impl("sigmoid_backward", ops::vulkan_sigmoid_backward);
    m.impl("tanh_backward", ops::vulkan_tanh_backward);
    m.impl("gelu_backward", ops::vulkan_gelu_backward);
    m.impl("silu_backward", ops::vulkan_silu_backward);
    m.impl("leaky_relu_backward", ops::vulkan_leaky_relu_backward);
    m.impl("elu_backward", ops::vulkan_elu_backward);
    m.impl("_softmax_backward_data", ops::vulkan_softmax_backward_data);
    m.impl("_log_softmax_backward_data", ops::vulkan_log_softmax_backward_data);
    m.impl("hardtanh_backward", ops::vulkan_hardtanh_backward);
    m.impl("hardswish_backward", ops::vulkan_hardswish_backward);
    m.impl("hardsigmoid_backward", ops::vulkan_hardsigmoid_backward);
    m.impl("softplus_backward", ops::vulkan_softplus_backward);
    m.impl("mish_backward", ops::vulkan_mish_backward);
    m.impl("avg_pool2d_backward", ops::vulkan_avg_pool2d_backward);
    m.impl("max_pool2d_with_indices", ops::vulkan_max_pool2d_with_indices);
    m.impl("max_pool2d_with_indices_backward", ops::vulkan_max_pool2d_with_indices_backward);
    m.impl("embedding_dense_backward", ops::vulkan_embedding_dense_backward);
    m.impl("native_layer_norm_backward", ops::vulkan_native_layer_norm_backward);
    m.impl("native_group_norm_backward", ops::vulkan_native_group_norm_backward);
    m.impl("native_batch_norm_backward", ops::vulkan_native_batch_norm_backward);
    m.impl("linear_backward", ops::vulkan_linear_backward);
}

// ── Autograd implementations ────────────────────────────────────
// Most standard ops (relu, sigmoid, tanh, gelu, silu, mm, bmm, addmm, linear,
// softmax, log_softmax, elu, leaky_relu, prelu, selu, clamp, avg_pool2d,
// embedding, layer_norm, group_norm, batch_norm) now use PyTorch's built-in
// autograd decompositions via backward helper ops registered above on PrivateUse1.
// This enables torch.compile/AOT Autograd tracing through these ops.
//
// Only ops needing custom autograd that can't decompose into standard backward
// helper ops are registered here.
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    // NOTE: BLAS ops (mm, bmm, addmm, linear) are NOT registered on
    // AutogradPrivateUse1 because it breaks torch.compile (FakeTensor dispatch
    // conflict). PyTorch's built-in autograd handles backward since all shape ops
    // (reshape, transpose, etc.) now have autograd wrappers.

    // Convolution: uses convolution_overrideable + convolution_backward_overrideable pattern
    m.impl("convolution_overrideable", vulkan_convolution_overrideable_autograd_adapter);

    // Max pool: needs indices-based backward (max_pool2d_with_indices pattern)
    m.impl("max_pool2d", ops::vulkan_max_pool2d_autograd);

    // SDPA: fully custom forward+backward (flash attention)
    m.impl("scaled_dot_product_attention", vulkan_sdpa_autograd_adapter);

    // Ops without standard backward helper ops in PyTorch — keep custom autograd
    m.impl("prelu", ops::vulkan_prelu_autograd);
    m.impl("selu", ops::vulkan_selu_autograd);
    m.impl("clamp", ops::vulkan_clamp_autograd);

    // Adaptive avg pool: needs custom autograd since PrivateUse1 registration bypasses built-in autograd
    m.impl("adaptive_avg_pool2d", vulkan_adaptive_avg_pool2d_autograd_adapter);

    // Shape ops: our backend copies data (opaque allocator can't share storage),
    // so all shape ops need autograd wrappers. Backward = inverse shape transform.
    m.impl("view", vulkan_view_autograd_adapter);
    m.impl("reshape", vulkan_reshape_autograd_adapter);
    m.impl("permute", ops::vulkan_permute_autograd);
    m.impl("transpose.int", ops::vulkan_transpose_autograd);
    m.impl("t", ops::vulkan_t_autograd);
    m.impl("unsqueeze", ops::vulkan_unsqueeze_autograd);
    m.impl("squeeze", ops::vulkan_squeeze_autograd);
    m.impl("squeeze.dim", ops::vulkan_squeeze_dim_autograd);
    m.impl("expand", vulkan_expand_autograd_adapter);
    m.impl("select.int", vulkan_select_autograd_adapter);
    m.impl("slice.Tensor", vulkan_slice_autograd_adapter);
}

// Fallback for everything else
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&vulkan_fallback>());
}

} // namespace torch_vulkan
