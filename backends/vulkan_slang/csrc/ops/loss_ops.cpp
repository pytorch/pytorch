#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"
#include "ops.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// Helper: get int32 target indices on Vulkan (avoids CPU roundtrip when possible)
static at::Tensor upload_targets(const at::Tensor& target, const at::Tensor& ref, int64_t N) {
    auto target_vulkan = at::empty({N}, ref.options().dtype(at::kFloat));

    if (target.device().type() == c10::DeviceType::PrivateUse1 &&
        target.scalar_type() == at::kLong) {
        // Int64 on Vulkan: convert to int32 on GPU via shader
        auto target_c = target.contiguous();
        struct { uint32_t numel; } params{static_cast<uint32_t>(N)};
        dispatch_shader("indexing_i64_to_i32_fwd",
                        shaders::indexing_i64_to_i32_fwd,
                        shaders::indexing_i64_to_i32_fwd_size,
                        {target_c, target_vulkan},
                        (static_cast<uint32_t>(N) + 255) / 256, 1, 1,
                        &params, sizeof(params));
    } else if (target.device().type() == c10::DeviceType::PrivateUse1 &&
               target.scalar_type() == at::kInt) {
        // Reinterpret int32 buffer as float — same 4-byte layout, no copy needed.
        // The NLL loss shader uses asint(StructuredBuffer<float>[i]) to read targets.
        auto target_c = target.contiguous();
        auto impl = c10::make_intrusive<at::TensorImpl>(
            c10::Storage(target_c.storage()),
            target_c.key_set(),
            at::scalarTypeToTypeMeta(at::kFloat));
        std::vector<int64_t> sz = {N}, st = {1};
        impl->set_sizes_and_strides(sz, st);
        impl->set_storage_offset(target_c.storage_offset());
        return at::Tensor(std::move(impl));
    } else {
        auto target_cpu = target.cpu().to(at::kInt).contiguous();
        auto& alloc = VulkanAllocator::instance();
        auto* buf = alloc.get_buffer(target_vulkan.data_ptr());
        TORCH_CHECK(buf, "Failed to get Vulkan buffer for target");
        buf->write(target_cpu.data_ptr(), static_cast<VkDeviceSize>(N * sizeof(int32_t)));
    }
    return target_vulkan;
}

// Count non-ignored samples, returning result as a GPU scalar tensor (no CPU roundtrip).
// Uses the already-uploaded target_vulkan (int32 packed as float) — must use asint()
// comparison in the shader since the data is bit-reinterpreted, not float-cast.
static at::Tensor count_valid_targets_gpu(const at::Tensor& target_vulkan, int64_t ignore_index, int64_t N) {
    auto mask = at::empty({N}, target_vulkan.options());

    struct { uint32_t N; int32_t ignore_index; } params{
        static_cast<uint32_t>(N), static_cast<int32_t>(ignore_index)
    };
    uint32_t workgroups = (static_cast<uint32_t>(N) + 255) / 256;

    dispatch_shader("loss_nll_loss_count_fwd",
                    shaders::loss_nll_loss_count_fwd, shaders::loss_nll_loss_count_fwd_size,
                    {target_vulkan, mask},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    return vulkan_sum(mask, c10::nullopt, false, c10::nullopt);
}

// ── NLL Loss ────────────────────────────────────────────────────
// nll_loss(log_probs, target) = -log_probs[target] averaged over batch
std::tuple<at::Tensor, at::Tensor> vulkan_nll_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {

    auto self_c = self.contiguous();
    check_supported_float(self_c, "nll_loss");
    self_c = ensure_float32(self_c);
    TORCH_CHECK(self_c.dim() == 2, "nll_loss expects 2D input [N, C]");

    int64_t N = self_c.size(0);
    int64_t C = self_c.size(1);

    auto target_vulkan = upload_targets(target, self_c, N);

    if (N == 0) {
        auto total_weight = at::zeros({}, self_c.options());
        if (reduction == 1) return {at::zeros({}, self_c.options()), total_weight};
        return {at::empty({N}, self_c.options()), total_weight};
    }

    struct { uint32_t N; uint32_t C; int32_t ignore_index; } params{
        static_cast<uint32_t>(N), static_cast<uint32_t>(C),
        static_cast<int32_t>(ignore_index)
    };

    uint32_t workgroups = (static_cast<uint32_t>(N) + 255) / 256;

    // Fast path: fused mean reduction for mean loss (reduction==1).
    // Single pass computes partial loss sums and counts per workgroup.
    // When N<=256 (1 workgroup), shader computes final mean directly (no div dispatch).
    if (reduction == 1) {
        bool single_wg = (workgroups == 1);
        struct { uint32_t N; uint32_t C; int32_t ignore_index; uint32_t single_workgroup; } fused_params{
            static_cast<uint32_t>(N), static_cast<uint32_t>(C),
            static_cast<int32_t>(ignore_index),
            single_wg ? 1u : 0u
        };

        auto partial_loss  = at::empty({static_cast<int64_t>(workgroups)}, self_c.options());
        auto partial_count = at::empty({static_cast<int64_t>(workgroups)}, self_c.options());
        auto scalar_out    = at::empty({1}, self_c.options());

        dispatch_shader("loss_nll_loss_fused_mean_fwd",
                        shaders::loss_nll_loss_fused_mean_fwd, shaders::loss_nll_loss_fused_mean_fwd_size,
                        {self_c, target_vulkan, partial_loss, partial_count, scalar_out},
                        workgroups, 1, 1,
                        &fused_params, sizeof(fused_params), 3);

        if (single_wg) {
            // Shader wrote final mean directly — no div dispatch needed.
            // total_weight = partial_count[0] (squeeze to scalar)
            auto total_weight = partial_count.reshape({});
            auto loss = scalar_out.reshape({});
            return {loss, total_weight};
        }

        // Multi-workgroup path: reduce partial results to scalars
        auto total_loss   = vulkan_sum(partial_loss,  c10::nullopt, false, c10::nullopt);
        auto total_weight = vulkan_sum(partial_count, c10::nullopt, false, c10::nullopt);
        auto loss = vulkan_div(total_loss, total_weight);
        return {loss, total_weight};
    }

    // Non-mean paths: compute per-sample losses first
    auto per_sample = at::empty({N}, self_c.options());

    dispatch_shader("loss_nll_loss_fwd",
                    shaders::loss_nll_loss_fwd, shaders::loss_nll_loss_fwd_size,
                    {self_c, target_vulkan, per_sample},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    if (reduction == 0) { // None
        // total_weight for non-mean: count valid targets on GPU
        auto total_weight = count_valid_targets_gpu(target_vulkan, ignore_index, N);
        return {per_sample, total_weight};
    } else { // Sum
        auto loss = vulkan_sum(per_sample, c10::nullopt, false, c10::nullopt);
        auto total_weight = count_valid_targets_gpu(target_vulkan, ignore_index, N);
        return {loss, total_weight};
    }
}

// ── NLL Loss Backward ──────────────────────────────────────────
at::Tensor vulkan_nll_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {

    auto self_c = self.contiguous();
    int64_t N = self_c.size(0);
    int64_t C = self_c.size(1);

    auto target_vulkan = upload_targets(target, self_c, N);
    auto grad_input = at::empty({N, C}, self_c.options());  // shader writes all elements

    if (N == 0) return grad_input;

    // total_weight is a GPU scalar tensor from forward. For sum/none, use a 1-element ones tensor.
    at::Tensor tw_gpu;
    if (reduction == 1) { // Mean
        tw_gpu = ensure_float32(total_weight.contiguous()).reshape({1});
    } else {
        tw_gpu = at::ones({1}, self_c.options().dtype(at::kFloat));
    }

    // grad_output passed as push constant float — avoids GPU allocation for scalar upstream grad.
    float go_val;
    if (grad_output.device().type() == c10::DeviceType::PrivateUse1) {
        // GPU tensor: flush and read scalar to pass as push constant.
        // Acceptable here — nll_loss_backward is called once per CE step, and grad_output
        // is always a scalar (loss.backward() passes 1.0 as CPU tensor, or a 1-elem GPU tensor).
        flush_stream();
        go_val = ensure_float32(grad_output.contiguous()).reshape({1}).item<float>();
    } else {
        // CPU scalar (e.g. 1.0 from loss.backward())
        go_val = grad_output.numel() == 1 ? grad_output.item<float>() : 1.0f;
    }

    struct { uint32_t N; uint32_t C; int32_t ignore_index; float grad_output_val; } params{
        static_cast<uint32_t>(N), static_cast<uint32_t>(C),
        static_cast<int32_t>(ignore_index), go_val
    };

    uint32_t total = static_cast<uint32_t>(N * C);
    uint32_t workgroups = (total + 255) / 256;

    dispatch_shader("loss_nll_loss_backward_fwd",
                    shaders::loss_nll_loss_backward_fwd, shaders::loss_nll_loss_backward_fwd_size,
                    {target_vulkan, tw_gpu, grad_input},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return grad_input;
}

// ── Cross Entropy Loss ──────────────────────────────────────────
// cross_entropy(input, target) = nll_loss(log_softmax(input), target)
at::Tensor vulkan_cross_entropy_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing) {

    auto log_probs = vulkan_log_softmax(self, /*dim=*/-1, c10::nullopt);
    auto [loss, total_weight] = vulkan_nll_loss_forward(
        log_probs, target, weight, reduction, ignore_index);
    return loss;
}

// ── BCE Loss ────────────────────────────────────────────────────
at::Tensor vulkan_binary_cross_entropy(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction) {

    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    check_supported_float(self_c, "binary_cross_entropy");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);

    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) {
        if (reduction == 0) return cast_from_float32(output, orig_dtype);
        return at::zeros({}, self_c.options());
    }

    dispatch_elementwise("loss_bce_fwd",
                         shaders::loss_bce_fwd, shaders::loss_bce_fwd_size,
                         {self_c, target_c, output}, numel);

    if (weight.has_value() && weight->defined()) {
        output = vulkan_mul(output, weight->expand_as(output).contiguous());
    }

    if (reduction == 0) return output;           // None
    auto loss_sum = vulkan_sum(output, c10::nullopt, false, c10::nullopt);
    if (reduction == 1) {                         // Mean
        return vulkan_div_scalar(loss_sum, at::Scalar(static_cast<float>(numel)));
    }
    return loss_sum;                              // Sum
}

at::Tensor vulkan_binary_cross_entropy_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction) {

    auto go = grad_output.contiguous();
    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    auto orig_dtype = self_c.scalar_type();
    go = ensure_float32(go);
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    // For mean reduction, scale grad_output
    at::Tensor go_scaled;
    if (reduction == 1) {
        go_scaled = vulkan_div_scalar(go.expand_as(self_c).contiguous(),
                                       at::Scalar(static_cast<float>(numel)));
    } else if (reduction == 2) {
        go_scaled = go.expand_as(self_c).contiguous();
    } else {
        go_scaled = go;
    }

    auto grad_input = at::empty_like(self_c);
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    dispatch_elementwise("loss_bce_backward_fwd",
                         shaders::loss_bce_backward_fwd, shaders::loss_bce_backward_fwd_size,
                         {go_scaled, self_c, target_c, grad_input}, numel);

    if (weight.has_value() && weight->defined()) {
        grad_input = vulkan_mul(grad_input, weight->expand_as(grad_input).contiguous());
    }
    return cast_from_float32(grad_input, orig_dtype);
}

// ── BCE with Logits Loss ────────────────────────────────────────
at::Tensor vulkan_binary_cross_entropy_with_logits(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& pos_weight,
    int64_t reduction) {

    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    check_supported_float(self_c, "bce_with_logits");
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);

    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) {
        if (reduction == 0) return cast_from_float32(output, orig_dtype);
        return at::zeros({}, self_c.options());
    }

    dispatch_elementwise("loss_bce_with_logits_fwd",
                         shaders::loss_bce_with_logits_fwd, shaders::loss_bce_with_logits_fwd_size,
                         {self_c, target_c, output}, numel);

    if (weight.has_value() && weight->defined()) {
        output = vulkan_mul(output, weight->expand_as(output).contiguous());
    }

    if (reduction == 0) return output;
    auto loss_sum = vulkan_sum(output, c10::nullopt, false, c10::nullopt);
    if (reduction == 1) {
        return vulkan_div_scalar(loss_sum, at::Scalar(static_cast<float>(numel)));
    }
    return loss_sum;
}

// ── Smooth L1 Loss ──────────────────────────────────────────────
at::Tensor vulkan_smooth_l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {

    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    check_supported_float(self_c, "smooth_l1_loss");
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);

    auto output = at::empty_like(self_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) {
        if (reduction == 0) return output;
        return at::zeros({}, self_c.options());
    }

    struct { uint32_t numel; float beta; } params{numel, static_cast<float>(beta)};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("loss_smooth_l1_fwd",
                    shaders::loss_smooth_l1_fwd, shaders::loss_smooth_l1_fwd_size,
                    {self_c, target_c, output}, workgroups, 1, 1,
                    &params, sizeof(params));

    if (reduction == 0) return output;
    auto loss_sum = vulkan_sum(output, c10::nullopt, false, c10::nullopt);
    if (reduction == 1) {
        return vulkan_div_scalar(loss_sum, at::Scalar(static_cast<float>(numel)));
    }
    return loss_sum;
}

at::Tensor vulkan_smooth_l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {

    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    at::Tensor go_scaled;
    if (reduction == 1) {
        go_scaled = vulkan_div_scalar(ensure_float32(grad_output.expand_as(self_c).contiguous()),
                                       at::Scalar(static_cast<float>(numel)));
    } else if (reduction == 2) {
        go_scaled = ensure_float32(grad_output.expand_as(self_c).contiguous());
    } else {
        go_scaled = ensure_float32(grad_output.contiguous());
    }

    auto grad_input = at::empty_like(self_c);
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    struct { uint32_t numel; float beta; } params{numel, static_cast<float>(beta)};
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("loss_smooth_l1_backward_fwd",
                    shaders::loss_smooth_l1_backward_fwd, shaders::loss_smooth_l1_backward_fwd_size,
                    {go_scaled, self_c, target_c, grad_input}, workgroups, 1, 1,
                    &params, sizeof(params));

    return cast_from_float32(grad_input, orig_dtype);
}

// ── Huber Loss (same as smooth_l1 with delta=beta) ──────────────
at::Tensor vulkan_huber_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double delta) {

    // Huber loss = delta * smooth_l1(x/delta, t/delta) * delta
    // But it's the same formula as smooth_l1 with beta=delta, just scaled by delta
    // Actually: huber(x,t,d) = 0.5*(x-t)^2 if |x-t|<d else d*(|x-t|-0.5*d)
    // smooth_l1(x,t,b) = 0.5*(x-t)^2/b if |x-t|<b else |x-t|-0.5*b
    // huber = delta * smooth_l1 when beta=delta
    auto loss = vulkan_smooth_l1_loss(self, target, reduction, delta);
    return vulkan_mul_scalar(loss, at::Scalar(static_cast<float>(delta)));
}

at::Tensor vulkan_huber_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double delta) {

    auto grad = vulkan_smooth_l1_loss_backward(grad_output, self, target, reduction, delta);
    return vulkan_mul_scalar(grad, at::Scalar(static_cast<float>(delta)));
}

// ── L1 Loss ─────────────────────────────────────────────────────
at::Tensor vulkan_l1_loss(const at::Tensor& self, const at::Tensor& target,
                           int64_t reduction) {
    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    auto output = at::empty_like(self_c);
    if (numel == 0) {
        if (reduction == 0) return cast_from_float32(output, orig_dtype);
        return at::zeros({}, self_c.options());
    }

    dispatch_elementwise("loss_l1_fwd", shaders::loss_l1_fwd, shaders::loss_l1_fwd_size,
                         {self_c, target_c, output}, numel);

    if (reduction == 0) return cast_from_float32(output, orig_dtype); // none
    auto loss_sum = vulkan_sum(output, c10::nullopt, false, c10::nullopt);
    if (reduction == 1) // mean
        return vulkan_div_scalar(loss_sum, at::Scalar(static_cast<float>(numel)));
    return loss_sum; // sum
}

at::Tensor vulkan_l1_loss_backward(const at::Tensor& grad_output,
                                     const at::Tensor& self, const at::Tensor& target,
                                     int64_t reduction) {
    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    at::Tensor grad_expanded;
    if (reduction == 1) { // mean
        grad_expanded = vulkan_div_scalar(ensure_float32(grad_output.contiguous()), at::Scalar(static_cast<float>(numel)));
        grad_expanded = grad_expanded.expand_as(self_c).contiguous();
    } else if (reduction == 2) { // sum
        grad_expanded = ensure_float32(grad_output.expand_as(self_c).contiguous());
    } else { // none
        grad_expanded = ensure_float32(grad_output.contiguous());
    }

    auto grad_input = at::empty_like(self_c);
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    dispatch_elementwise("loss_l1_backward_fwd", shaders::loss_l1_backward_fwd, shaders::loss_l1_backward_fwd_size,
                         {grad_expanded, self_c, target_c, grad_input}, numel);
    return cast_from_float32(grad_input, orig_dtype);
}

// ── KL Divergence Loss ──────────────────────────────────────────
at::Tensor vulkan_kl_div(const at::Tensor& self, const at::Tensor& target,
                          int64_t reduction, bool log_target) {
    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    auto output = at::empty_like(self_c);
    if (numel == 0) {
        if (reduction == 0) return cast_from_float32(output, orig_dtype);
        return at::zeros({}, self_c.options());
    }

    struct { uint32_t numel; uint32_t log_target; } params{
        numel, static_cast<uint32_t>(log_target ? 1 : 0)
    };
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("loss_kl_div_fwd", shaders::loss_kl_div_fwd, shaders::loss_kl_div_fwd_size,
                    {self_c, target_c, output}, workgroups, 1, 1,
                    &params, sizeof(params));

    if (reduction == 0) return output; // none
    auto loss_sum = vulkan_sum(output, c10::nullopt, false, c10::nullopt);
    if (reduction == 1) // mean
        return vulkan_div_scalar(loss_sum, at::Scalar(static_cast<float>(numel)));
    if (reduction == 2) // sum
        return loss_sum;
    // batchmean (reduction == 3 is not standard but PyTorch uses it internally for batchmean)
    return loss_sum;
}

at::Tensor vulkan_kl_div_backward(const at::Tensor& grad_output,
                                    const at::Tensor& self, const at::Tensor& target,
                                    int64_t reduction, bool log_target) {
    auto self_c = self.contiguous();
    auto target_c = target.contiguous();
    auto orig_dtype = self_c.scalar_type();
    self_c = ensure_float32(self_c);
    target_c = ensure_float32(target_c);
    uint32_t numel = static_cast<uint32_t>(self_c.numel());

    at::Tensor grad_expanded;
    if (reduction == 1) { // mean
        grad_expanded = vulkan_div_scalar(ensure_float32(grad_output.contiguous()), at::Scalar(static_cast<float>(numel)));
        grad_expanded = grad_expanded.expand_as(self_c).contiguous();
    } else if (reduction == 2) { // sum
        grad_expanded = ensure_float32(grad_output.expand_as(self_c).contiguous());
    } else { // none
        grad_expanded = ensure_float32(grad_output.contiguous());
    }

    auto grad_input = at::empty_like(self_c);
    if (numel == 0) return cast_from_float32(grad_input, orig_dtype);

    struct { uint32_t numel; uint32_t log_target; } params{
        numel, static_cast<uint32_t>(log_target ? 1 : 0)
    };
    uint32_t workgroups = (numel + 255) / 256;
    dispatch_shader("loss_kl_div_backward_fwd", shaders::loss_kl_div_backward_fwd, shaders::loss_kl_div_backward_fwd_size,
                    {grad_expanded, target_c, grad_input}, workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(grad_input, orig_dtype);
}

}} // namespace torch_vulkan::ops
