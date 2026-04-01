#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// Helper: copy Vulkan buffer data back into the in-place tensor
static void copy_result_inplace(const at::Tensor& result, at::Tensor& self) {
    dispatch_copy_buffer(result, self);
}

// ── addcmul_ ────────────────────────────────────────────────────
// self += value * tensor1 * tensor2
at::Tensor& vulkan_addcmul_(at::Tensor& self, const at::Tensor& tensor1,
                              const at::Tensor& tensor2, const at::Scalar& value) {
    auto self_c = self.contiguous();
    auto t1 = tensor1.contiguous();
    auto t2 = tensor2.contiguous();

    check_supported_float(self_c, "addcmul_");
    self_c = ensure_float32(self_c);
    t1 = ensure_float32(t1);
    t2 = ensure_float32(t2);

    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return self;

    auto output = at::empty_like(self_c);

    struct { float value; uint32_t numel; } params{
        value.toFloat(), numel
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("binary_addcmul_fwd",
                    shaders::binary_addcmul_fwd, shaders::binary_addcmul_fwd_size,
                    {self_c, t1, t2, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    copy_result_inplace(output, self);
    return self;
}

// ── addcdiv_ ────────────────────────────────────────────────────
// self += value * tensor1 / tensor2
at::Tensor& vulkan_addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                              const at::Tensor& tensor2, const at::Scalar& value) {
    auto self_c = self.contiguous();
    auto t1 = tensor1.contiguous();
    auto t2 = tensor2.contiguous();

    check_supported_float(self_c, "addcdiv_");
    self_c = ensure_float32(self_c);
    t1 = ensure_float32(t1);
    t2 = ensure_float32(t2);

    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return self;

    auto output = at::empty_like(self_c);

    struct { float value; uint32_t numel; } params{
        value.toFloat(), numel
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("binary_addcdiv_fwd",
                    shaders::binary_addcdiv_fwd, shaders::binary_addcdiv_fwd_size,
                    {self_c, t1, t2, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    copy_result_inplace(output, self);
    return self;
}

// ── lerp_ ───────────────────────────────────────────────────────
// self = self + weight * (end - self)
at::Tensor& vulkan_lerp_(at::Tensor& self, const at::Tensor& end,
                           const at::Scalar& weight) {
    auto self_c = self.contiguous();
    auto end_c = end.contiguous();

    check_supported_float(self_c, "lerp_");
    self_c = ensure_float32(self_c);
    end_c = ensure_float32(end_c);

    uint32_t numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return self;

    auto output = at::empty_like(self_c);

    struct { float weight; uint32_t numel; } params{
        weight.toFloat(), numel
    };
    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("binary_lerp_fwd",
                    shaders::binary_lerp_fwd, shaders::binary_lerp_fwd_size,
                    {self_c, end_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));

    copy_result_inplace(output, self);
    return self;
}

// ── clamp_ (in-place) ──────────────────────────────────────────
at::Tensor& vulkan_clamp_(at::Tensor& self,
                            const std::optional<at::Scalar>& min_val,
                            const std::optional<at::Scalar>& max_val) {
    auto result = vulkan_clamp(self, min_val, max_val);
    copy_result_inplace(result, self);
    return self;
}

// ── Fused SGD step ────────────────────────────────────────────
// Single-dispatch SGD with optional momentum, weight decay, dampening, nesterov.
// Updates param and momentum_buf in-place.
// Handles bf16/f16 params via widen-compute-narrow.
void vulkan_sgd_step(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& momentum_buf,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool has_momentum_buf) {

    uint32_t numel = static_cast<uint32_t>(param.numel());
    if (numel == 0) return;

    // Widen to float32 for computation
    const auto orig_dtype = param.scalar_type();
    const bool needs_cast = (orig_dtype != at::kFloat);

    at::Tensor param_f32 = needs_cast ? ensure_float32(param) : param;
    at::Tensor grad_f32  = needs_cast ? ensure_float32(grad)  : grad;
    // Momentum buffer is always float32 (allocated as float32 in Python SGD for stability)
    at::Tensor mbuf_f32  = momentum_buf;

    uint32_t workgroups = (numel + 255) / 256;

    if (has_momentum_buf && momentum != 0.0f) {
        // Fused SGD with momentum
        struct {
            uint32_t numel;
            float lr;
            float momentum;
            float dampening;
            float weight_decay;
            uint32_t nesterov;
        } params{
            numel, lr, momentum, dampening, weight_decay,
            nesterov ? 1u : 0u
        };

        dispatch_shader("training_sgd_momentum_fwd",
                        shaders::training_sgd_momentum_fwd,
                        shaders::training_sgd_momentum_fwd_size,
                        {param_f32, grad_f32, mbuf_f32},
                        workgroups, 1, 1,
                        &params, sizeof(params), 3);
    } else {
        // Simple SGD (no momentum)
        struct {
            uint32_t numel;
            float lr;
            float weight_decay;
        } params{numel, lr, weight_decay};

        dispatch_shader("training_sgd_fwd",
                        shaders::training_sgd_fwd,
                        shaders::training_sgd_fwd_size,
                        {param_f32, grad_f32},
                        workgroups, 1, 1,
                        &params, sizeof(params), 2);  // both param and grad bindings, mark param dirty
    }

    // Narrow back to original dtype if needed
    if (needs_cast) {
        auto param_narrow = cast_from_float32(param_f32, orig_dtype);
        dispatch_copy_buffer(param_narrow, param);
    }
}

// ── Batched SGD step (no momentum) ────────────────────────────
// Two variants:
//   sgd_batch15: up to 15 params/dispatch (30 bindings, 184-byte push constants).
//                Use on NVIDIA where maxPushConstantsSize >= 256.
//   sgd_batch:   up to 7 params/dispatch  (14 bindings, 88-byte push constants).
//                Portable fallback (Vulkan minimum is 128 bytes).
// PushConstants layout: {uint n_params, ParamConfig[N]} where ParamConfig={uint numel, float lr, float wd}
void vulkan_sgd_batch_step(
    const std::vector<at::Tensor*>& params,
    const std::vector<const at::Tensor*>& grads,
    float lr, float weight_decay) {

    constexpr int BATCH15 = 15;
    constexpr int BATCH7  = 7;
    const int n = static_cast<int>(params.size());
    TORCH_CHECK(n > 0 && n <= BATCH15, "vulkan_sgd_batch_step: n_params must be 1..15");

    struct ParamCfg { uint32_t numel; float lr; float wd; };

    uint32_t max_numel = 0;
    for (int i = 0; i < n; i++)
        max_numel = std::max(max_numel, static_cast<uint32_t>(params[i]->numel()));
    uint32_t wg_x = (max_numel + 255) / 256;

    if (n <= BATCH7) {
        // Small batch: use portable sgd_batch (88-byte push constants, 14 bindings)
        struct PushConstants {
            uint32_t n_params;
            ParamCfg cfg[7];
        } pc{};
        pc.n_params = static_cast<uint32_t>(n);
        for (int i = 0; i < n; i++)
            pc.cfg[i] = { static_cast<uint32_t>(params[i]->numel()), lr, weight_decay };

        std::vector<at::Tensor> bufs;
        bufs.reserve(2 * BATCH7);
        for (int i = 0; i < BATCH7; i++) {
            bufs.push_back(i < n ? *params[i] : *params[0]);
            bufs.push_back(i < n ? *grads[i]  : *grads[0]);
        }

        dispatch_shader("training_sgd_batch_fwd",
                        shaders::training_sgd_batch_fwd,
                        shaders::training_sgd_batch_fwd_size,
                        bufs, wg_x, static_cast<uint32_t>(n), 1,
                        &pc, sizeof(pc),
                        static_cast<uint32_t>(bufs.size()));
    } else {
        // Large batch: use sgd_batch15 (184-byte push constants, 30 bindings)
        struct PushConstants {
            uint32_t n_params;
            ParamCfg cfg[15];
        } pc{};
        pc.n_params = static_cast<uint32_t>(n);
        for (int i = 0; i < n; i++)
            pc.cfg[i] = { static_cast<uint32_t>(params[i]->numel()), lr, weight_decay };

        std::vector<at::Tensor> bufs;
        bufs.reserve(2 * BATCH15);
        for (int i = 0; i < BATCH15; i++) {
            bufs.push_back(i < n ? *params[i] : *params[0]);
            bufs.push_back(i < n ? *grads[i]  : *grads[0]);
        }

        dispatch_shader("training_sgd_batch15_fwd",
                        shaders::training_sgd_batch15_fwd,
                        shaders::training_sgd_batch15_fwd_size,
                        bufs, wg_x, static_cast<uint32_t>(n), 1,
                        &pc, sizeof(pc),
                        static_cast<uint32_t>(bufs.size()));
    }
}

// ── Batched AdamW step ────────────────────────────────────────
// Two variants:
//   adamw_batch7: up to 7 params/dispatch (28 bindings, 228-byte push constants).
//                 Use on NVIDIA where maxPushConstantsSize >= 256.
//   adamw_batch:  up to 3 params/dispatch (12 bindings, 100-byte push constants).
//                 Portable fallback (Vulkan minimum is 128 bytes).
// PushConstants layout: {uint n_params, ParamConfig[N]} where ParamConfig = 8 values (uint+7 floats)
void vulkan_adamw_batch_step(
    const std::vector<at::Tensor*>& params,
    const std::vector<const at::Tensor*>& grads,
    const std::vector<at::Tensor*>& m_bufs,
    const std::vector<at::Tensor*>& v_bufs,
    float lr, float beta1, float beta2, float eps,
    float weight_decay, float bc1, float bc2) {

    constexpr int BATCH7 = 7;
    constexpr int BATCH3 = 3;
    const int n = static_cast<int>(params.size());
    TORCH_CHECK(n > 0 && n <= BATCH7, "vulkan_adamw_batch_step: n_params must be 1..7");

    struct ParamCfg {
        uint32_t numel;
        float lr, beta1, beta2, eps, weight_decay, bc1, bc2;
    };

    uint32_t max_numel = 0;
    for (int i = 0; i < n; i++)
        max_numel = std::max(max_numel, static_cast<uint32_t>(params[i]->numel()));
    uint32_t wg_x = (max_numel + 255) / 256;

    if (n <= BATCH3) {
        // Small batch: use portable adamw_batch (100-byte push constants, 12 bindings)
        struct PushConstants {
            uint32_t n_params;
            ParamCfg cfg[3];
        } pc{};
        pc.n_params = static_cast<uint32_t>(n);
        for (int i = 0; i < n; i++)
            pc.cfg[i] = { static_cast<uint32_t>(params[i]->numel()),
                          lr, beta1, beta2, eps, weight_decay, bc1, bc2 };

        std::vector<at::Tensor> bufs;
        bufs.reserve(4 * BATCH3);
        for (int i = 0; i < BATCH3; i++) {
            bufs.push_back(i < n ? *params[i] : *params[0]);
            bufs.push_back(i < n ? *grads[i]  : *grads[0]);
            bufs.push_back(i < n ? *m_bufs[i] : *m_bufs[0]);
            bufs.push_back(i < n ? *v_bufs[i] : *v_bufs[0]);
        }

        dispatch_shader("training_adamw_batch_fwd",
                        shaders::training_adamw_batch_fwd,
                        shaders::training_adamw_batch_fwd_size,
                        bufs, wg_x, static_cast<uint32_t>(n), 1,
                        &pc, sizeof(pc),
                        static_cast<uint32_t>(bufs.size()));
    } else {
        // Large batch: use adamw_batch7 (228-byte push constants, 28 bindings)
        struct PushConstants {
            uint32_t n_params;
            ParamCfg cfg[7];
        } pc{};
        pc.n_params = static_cast<uint32_t>(n);
        for (int i = 0; i < n; i++)
            pc.cfg[i] = { static_cast<uint32_t>(params[i]->numel()),
                          lr, beta1, beta2, eps, weight_decay, bc1, bc2 };

        std::vector<at::Tensor> bufs;
        bufs.reserve(4 * BATCH7);
        for (int i = 0; i < BATCH7; i++) {
            bufs.push_back(i < n ? *params[i] : *params[0]);
            bufs.push_back(i < n ? *grads[i]  : *grads[0]);
            bufs.push_back(i < n ? *m_bufs[i] : *m_bufs[0]);
            bufs.push_back(i < n ? *v_bufs[i] : *v_bufs[0]);
        }

        dispatch_shader("training_adamw_batch7_fwd",
                        shaders::training_adamw_batch7_fwd,
                        shaders::training_adamw_batch7_fwd_size,
                        bufs, wg_x, static_cast<uint32_t>(n), 1,
                        &pc, sizeof(pc),
                        static_cast<uint32_t>(bufs.size()));
    }
}

// ── Fused AdamW step ──────────────────────────────────────────
// Decoupled weight decay (AdamW). Moment buffers always float32.
// Handles bf16/f16 params via widen-compute-narrow.
void vulkan_adamw_step(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& m,
    at::Tensor& v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int64_t step) {

    uint32_t numel = static_cast<uint32_t>(param.numel());
    if (numel == 0) return;

    const auto orig_dtype = param.scalar_type();
    const bool needs_cast = (orig_dtype != at::kFloat);

    at::Tensor param_f32 = needs_cast ? ensure_float32(param) : param;
    at::Tensor grad_f32  = needs_cast ? ensure_float32(grad)  : grad;
    // m and v are always float32

    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    struct {
        uint32_t numel;
        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;
        float bias_correction1;
        float bias_correction2;
    } params{numel, lr, beta1, beta2, eps, weight_decay, bc1, bc2};

    uint32_t workgroups = (numel + 255) / 256;

    dispatch_shader("training_adamw_fwd",
                    shaders::training_adamw_fwd,
                    shaders::training_adamw_fwd_size,
                    {param_f32, grad_f32, m, v},
                    workgroups, 1, 1,
                    &params, sizeof(params), 4);

    if (needs_cast) {
        auto param_narrow = cast_from_float32(param_f32, orig_dtype);
        dispatch_copy_buffer(param_narrow, param);
    }
}

}} // namespace torch_vulkan::ops
