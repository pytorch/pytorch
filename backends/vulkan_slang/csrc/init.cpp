#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <cstdlib>

#include "vulkan/Context.h"
#include "backend/Allocator.h"
#include "backend/Hooks.h"
#include "ops/ops.h"
#include "ops/dispatch.h"

namespace torch_vulkan {

// Defined in Profiler.cpp
void register_profiler_stubs();

bool is_available() {
    return vulkan::Context::instance().is_available();
}

int64_t device_count() {
    return static_cast<int64_t>(vulkan::Context::instance().device_count());
}

std::string get_device_name(int64_t device_index) {
    return vulkan::Context::instance().device_name(
        static_cast<uint32_t>(device_index));
}

void synchronize(int64_t device_index) {
    auto& ctx = vulkan::Context::instance();
    auto device = ctx.device(static_cast<uint32_t>(device_index));
    vkDeviceWaitIdle(device);
}

PYBIND11_MODULE(_C, m) {
    // Register PrivateUse1 backend as "vulkan"
    c10::register_privateuse1_backend("vulkan");

    // Register allocator
    auto* alloc = &VulkanAllocator::instance();
    c10::SetAllocator(c10::DeviceType::PrivateUse1, alloc);

    // Expose shutdown for Python atexit to call
    m.def("_shutdown", []() { vulkan::Context::instance().shutdown(); });

    // Register hooks
    static VulkanHooksInterface hooks;
    at::RegisterPrivateUse1HooksInterface(&hooks);

    // Register profiler stubs
    register_profiler_stubs();

    m.def("_is_available", &is_available);
    m.def("_device_count", &device_count);
    m.def("_get_device_name", &get_device_name);
    m.def("_synchronize", &synchronize);
    m.def("_manual_seed", [](uint64_t seed) { ops::vulkan_manual_seed(seed); });
    m.def("_empty_cache", []() { VulkanAllocator::instance().empty_cache(); });
    m.def("_memory_cached", []() -> int64_t {
        return static_cast<int64_t>(VulkanAllocator::instance().cached_bytes());
    });

    // Custom ops
    m.def("rope", [](const at::Tensor& input, double theta) {
        return ops::vulkan_rope_autograd(input, theta);
    }, py::arg("input"), py::arg("theta") = 10000.0);

    // Direct SDPA bypass — F.scaled_dot_product_attention dispatches through
    // CompositeImplicitAutograd which fails with None mask on PrivateUse1
    m.def("_sdpa", [](const at::Tensor& query, const at::Tensor& key,
                       const at::Tensor& value,
                       const std::optional<at::Tensor>& attn_mask,
                       double dropout_p, bool is_causal,
                       std::optional<double> scale) {
        return ops::vulkan_sdpa_autograd(query, key, value, attn_mask,
                                          dropout_p, is_causal, scale);
    }, py::arg("query"), py::arg("key"), py::arg("value"),
       py::arg("attn_mask") = py::none(), py::arg("dropout_p") = 0.0,
       py::arg("is_causal") = false, py::arg("scale") = py::none());

    m.def("rms_norm", [](const at::Tensor& input, const at::Tensor& weight, double eps) {
        return ops::vulkan_rms_norm_autograd(input, weight, eps);
    }, py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6);

    m.def("swiglu", [](const at::Tensor& gate, const at::Tensor& up) {
        return ops::vulkan_swiglu_autograd(gate, up);
    }, py::arg("gate"), py::arg("up"));

    // Fused attention score: scale * (q @ k.T) in 1 dispatch instead of 2
    // q: [B, M, K] contiguous, k: [B, N, K] contiguous (NOT pre-transposed)
    // Returns: scale * (q @ k.T) = [B, M, N]
    // Saves 1 dispatch vs (q @ k.T) then * scale
    m.def("scaled_bmm", [](const at::Tensor& q, const at::Tensor& k, double scale) {
        return ops::vulkan_scaled_bmm_autograd(q, k, scale);
    }, py::arg("q"), py::arg("k"), py::arg("scale"));


    // Fused Add + RMSNorm: h_new = residual + shortcut; normed = weight * (h_new / rms(h_new))
    // Returns (normed, h_new). Saves 1 dispatch vs separate add + rms_norm.
    m.def("add_rms_norm", [](const at::Tensor& residual, const at::Tensor& shortcut,
                              const at::Tensor& weight, double eps) {
        return ops::vulkan_add_rms_norm_apply(residual, shortcut, weight, eps);
    }, py::arg("residual"), py::arg("shortcut"), py::arg("weight"), py::arg("eps") = 1e-6);

    // Flash Attention: fused QK^T + softmax + @V (7 dispatches → 1)
    // Q: [B,H,N,D] or [B,S,H,D] (seq-major), K/V: matching layout
    // Returns output [B,H,N,D]. Fully differentiable via flash attention backward.
    // q_seq_major=True: Q/K/V in [B,S,H,D] layout — skips 3 contiguous() copies per call.
    m.def("flash_attention", [](const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
                                 double scale, bool is_causal, bool q_seq_major) {
        return ops::vulkan_flash_attention_autograd(Q, K, V, scale, is_causal, q_seq_major);
    }, py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"),
       py::arg("is_causal") = true, py::arg("q_seq_major") = false);

    // RMSNormGated: weight * rms_norm(input) * silu(gate) — Qwen3.5 GatedDeltaNet
    m.def("rms_norm_gated", [](const at::Tensor& input, const at::Tensor& gate,
                                const at::Tensor& weight, double eps) {
        return ops::vulkan_rms_norm_gated_autograd(input, gate, weight, eps);
    }, py::arg("input"), py::arg("gate"), py::arg("weight"), py::arg("eps") = 1e-6);

    // Fused SGD step
    m.def("_sgd_step", [](at::Tensor& param, const at::Tensor& grad,
                           at::Tensor& momentum_buf,
                           double lr, double momentum, double dampening,
                           double weight_decay, bool nesterov,
                           bool has_momentum_buf) {
        ops::vulkan_sgd_step(param, grad, momentum_buf,
                              static_cast<float>(lr),
                              static_cast<float>(momentum),
                              static_cast<float>(dampening),
                              static_cast<float>(weight_decay),
                              nesterov, has_momentum_buf);
    }, py::arg("param"), py::arg("grad"), py::arg("momentum_buf"),
       py::arg("lr"), py::arg("momentum"), py::arg("dampening"),
       py::arg("weight_decay"), py::arg("nesterov"),
       py::arg("has_momentum_buf"));

    // Batched SGD step (no momentum): up to 15 params per dispatch
    m.def("_sgd_batch_step", [](std::vector<at::Tensor> params,
                                 std::vector<at::Tensor> grads,
                                 double lr, double weight_decay) {
        std::vector<at::Tensor*> param_ptrs;
        std::vector<const at::Tensor*> grad_ptrs;
        param_ptrs.reserve(params.size());
        grad_ptrs.reserve(grads.size());
        for (auto& p : params) param_ptrs.push_back(&p);
        for (auto& g : grads)  grad_ptrs.push_back(&g);
        ops::vulkan_sgd_batch_step(param_ptrs, grad_ptrs,
                                    static_cast<float>(lr),
                                    static_cast<float>(weight_decay));
    }, py::arg("params"), py::arg("grads"), py::arg("lr"), py::arg("weight_decay"));

    // Batched AdamW step: up to 7 params per dispatch
    m.def("_adamw_batch_step", [](std::vector<at::Tensor> params,
                                   std::vector<at::Tensor> grads,
                                   std::vector<at::Tensor> m_bufs,
                                   std::vector<at::Tensor> v_bufs,
                                   double lr, double beta1, double beta2,
                                   double eps, double weight_decay,
                                   double bc1, double bc2) {
        std::vector<at::Tensor*> pp, mb, vb;
        std::vector<const at::Tensor*> gp;
        for (auto& p : params) pp.push_back(&p);
        for (auto& g : grads)  gp.push_back(&g);
        for (auto& m : m_bufs) mb.push_back(&m);
        for (auto& v : v_bufs) vb.push_back(&v);
        ops::vulkan_adamw_batch_step(pp, gp, mb, vb,
                                      static_cast<float>(lr),
                                      static_cast<float>(beta1),
                                      static_cast<float>(beta2),
                                      static_cast<float>(eps),
                                      static_cast<float>(weight_decay),
                                      static_cast<float>(bc1),
                                      static_cast<float>(bc2));
    }, py::arg("params"), py::arg("grads"), py::arg("m_bufs"), py::arg("v_bufs"),
       py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
       py::arg("weight_decay"), py::arg("bc1"), py::arg("bc2"));

    // Fused AdamW step
    m.def("_adamw_step", [](at::Tensor& param, const at::Tensor& grad,
                             at::Tensor& m_buf, at::Tensor& v_buf,
                             double lr, double beta1, double beta2, double eps,
                             double weight_decay, int64_t step) {
        ops::vulkan_adamw_step(param, grad, m_buf, v_buf,
                               static_cast<float>(lr),
                               static_cast<float>(beta1),
                               static_cast<float>(beta2),
                               static_cast<float>(eps),
                               static_cast<float>(weight_decay),
                               step);
    }, py::arg("param"), py::arg("grad"), py::arg("m"), py::arg("v"),
       py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
       py::arg("weight_decay"), py::arg("step"));

    // Flush pending GPU work (for benchmarking / synchronization)
    m.def("_flush", []() { ops::flush_stream(); });

    // Perf counters
    m.def("_get_dispatch_count", []() -> int64_t { return ops::get_dispatch_count(); });
    m.def("_get_flush_count", []() -> int64_t { return ops::get_flush_count(); });
    m.def("_get_war_flush_count", []() -> int64_t { return ops::get_war_flush_count(); });
    m.def("_get_preread_flush_count", []() -> int64_t { return ops::get_preread_flush_count(); });
    m.def("_get_capacity_flush_count", []() -> int64_t { return ops::get_capacity_flush_count(); });
    m.def("_get_descpool_flush_count", []() -> int64_t { return ops::get_descpool_flush_count(); });
    m.def("_get_barrier_count", []() -> int64_t { return ops::get_barrier_count(); });
    m.def("_get_barrier_skip_count", []() -> int64_t { return ops::get_barrier_skip_count(); });
    m.def("_reset_perf_counters", []() { ops::reset_perf_counters(); });
}

} // namespace torch_vulkan
