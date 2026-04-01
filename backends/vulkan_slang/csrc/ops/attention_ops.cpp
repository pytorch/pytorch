#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"

#include <torch/library.h>
#include <cmath>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace torch_vulkan { namespace ops {

// Cache for causal additive masks: key=(BH*65536^2 + N*65536 + S) -> mask [BH,N,S]
// Avoids re-creating triu mask every forward pass (~7 dispatches → 0).
static std::unordered_map<int64_t, at::Tensor> s_causal_mask_cache;
static std::mutex s_causal_mask_mutex;

at::Tensor get_causal_mask(int64_t BH, int64_t N, int64_t S,
                            const at::TensorOptions& opts) {
    int64_t key = BH * (65536LL * 65536LL) + N * 65536 + S;
    {
        std::lock_guard<std::mutex> lk(s_causal_mask_mutex);
        auto it = s_causal_mask_cache.find(key);
        if (it != s_causal_mask_cache.end()) {
            return it->second;
        }
    }
    // Build additive causal mask [BH, N, S]: 0 for attend, -1e9 for block
    auto upper = vulkan_triu(at::ones({N, S}, opts), 1);
    auto mask_2d = vulkan_mul_scalar(upper, at::Scalar(-1e9f));
    // Expand to [BH, N, S] and make contiguous once
    auto mask = mask_2d.unsqueeze(0).expand({BH, N, S}).contiguous();
    {
        std::lock_guard<std::mutex> lk(s_causal_mask_mutex);
        s_causal_mask_cache[key] = mask;
    }
    return mask;
}

// ── Scaled Dot-Product Attention ────────────────────────────────
// Implemented using existing GPU ops (bmm, softmax) for correctness.
// Q, K, V: [B, H, N, D]
// Output: [B, H, N, D]
at::Tensor vulkan_scaled_dot_product_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {

    auto q = query.contiguous();
    auto k = key.contiguous();
    auto v = value.contiguous();

    check_supported_float(q, "SDPA");
    auto orig_dtype = q.scalar_type();
    q = ensure_float32(q);
    k = ensure_float32(k);
    v = ensure_float32(v);
    TORCH_CHECK(q.dim() == 4, "SDPA expects 4D input [B, H, N, D]");

    int64_t B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    int64_t S = k.size(2);  // key sequence length
    float scale_val = scale.has_value()
        ? static_cast<float>(*scale)
        : 1.0f / std::sqrt(static_cast<float>(D));

    if (B * H * N * D == 0) return at::empty_like(q);

    // Reshape to [B*H, N, D] for bmm
    auto q_bh = q.reshape({B * H, N, D});
    auto k_bh = k.reshape({B * H, S, D});
    auto v_bh = v.reshape({B * H, S, D});

    // scores = Q @ K^T * scale => [B*H, N, S]
    // Use bmm_ex to avoid GPU permute copy from .transpose()
    auto scores = vulkan_bmm_ex(q_bh, k_bh, false, true);
    scores = vulkan_mul_scalar(scores, at::Scalar(scale_val));

    // Apply causal mask
    if (is_causal) {
        // Get cached additive causal mask [BH, N, S]: 0 for attend, -1e9 for block
        auto additive_mask = get_causal_mask(B * H, N, S, q.options());
        scores = vulkan_add(scores, additive_mask, 1);
    } else if (attn_mask.has_value()) {
        auto mask = attn_mask->contiguous();
        if (mask.scalar_type() == at::kBool) {
            // Bool mask: true=attend, false=block
            // Convert to float: 0 where true, -inf where false
            auto float_mask = mask.to(at::kFloat);
            // (1 - float_mask) * -inf
            auto ones_t = at::ones_like(float_mask);
            auto inv_mask = vulkan_sub(ones_t, float_mask, 1);
            auto neg_inf = at::full({1}, -std::numeric_limits<float>::infinity(), q.options());
            auto additive = vulkan_mul(inv_mask, neg_inf.expand_as(inv_mask).contiguous());
            if (additive.dim() == 2) {
                additive = additive.unsqueeze(0).expand({B * H, N, S}).contiguous();
            } else if (additive.dim() == 4) {
                additive = additive.reshape({B * H, N, S});
            }
            scores = vulkan_add(scores, additive, 1);
        } else {
            // Float additive mask
            auto additive = mask.contiguous();
            if (additive.dim() == 2) {
                additive = additive.unsqueeze(0).expand({B * H, N, S}).contiguous();
            } else if (additive.dim() == 4) {
                additive = additive.reshape({B * H, N, S});
            }
            scores = vulkan_add(scores, additive, 1);
        }
    }

    // attn_weights = softmax(scores, dim=-1) => [B*H, N, S]
    auto attn_weights = vulkan_softmax(scores, -1, std::nullopt);

    // output = attn_weights @ V => [B*H, N, D]
    auto out_bh = vulkan_bmm(attn_weights, v_bh);

    // Reshape back to [B, H, N, D]
    return cast_from_float32(out_bh.reshape({B, H, N, D}).contiguous(), orig_dtype);
}

// Returns (output, attn_weights_3d [B*H, N, S]) for saving in autograd context.
// Avoids recomputing softmax(scores) in backward.
std::tuple<at::Tensor, at::Tensor> vulkan_scaled_dot_product_attention_with_attn(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {

    auto q = query.contiguous();
    auto k = key.contiguous();
    auto v = value.contiguous();

    check_supported_float(q, "SDPA");
    auto orig_dtype = q.scalar_type();
    q = ensure_float32(q);
    k = ensure_float32(k);
    v = ensure_float32(v);
    TORCH_CHECK(q.dim() == 4, "SDPA expects 4D input [B, H, N, D]");

    int64_t B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    int64_t S = k.size(2);
    float scale_val = scale.has_value()
        ? static_cast<float>(*scale)
        : 1.0f / std::sqrt(static_cast<float>(D));

    if (B * H * N * D == 0) {
        auto empty = at::empty({B, H, N, D}, q.options());
        auto empty_attn = at::empty({B * H, N, S}, q.options());
        return std::make_tuple(empty, empty_attn);
    }

    auto q_bh = q.reshape({B * H, N, D});
    auto k_bh = k.reshape({B * H, S, D});
    auto v_bh = v.reshape({B * H, S, D});

    auto scores = vulkan_bmm_ex(q_bh, k_bh, false, true);
    scores = vulkan_mul_scalar(scores, at::Scalar(scale_val));

    if (is_causal) {
        auto additive_mask = get_causal_mask(B * H, N, S, q.options());
        scores = vulkan_add(scores, additive_mask, 1);
    } else if (attn_mask.has_value()) {
        auto mask = attn_mask->contiguous();
        if (mask.scalar_type() == at::kBool) {
            auto float_mask = mask.to(at::kFloat);
            auto ones_t = at::ones_like(float_mask);
            auto inv_mask = vulkan_sub(ones_t, float_mask, 1);
            auto neg_inf = at::full({1}, -std::numeric_limits<float>::infinity(), q.options());
            auto additive = vulkan_mul(inv_mask, neg_inf.expand_as(inv_mask).contiguous());
            if (additive.dim() == 2) {
                additive = additive.unsqueeze(0).expand({B * H, N, S}).contiguous();
            } else if (additive.dim() == 4) {
                additive = additive.reshape({B * H, N, S});
            }
            scores = vulkan_add(scores, additive, 1);
        } else {
            auto additive = mask.contiguous();
            if (additive.dim() == 2) {
                additive = additive.unsqueeze(0).expand({B * H, N, S}).contiguous();
            } else if (additive.dim() == 4) {
                additive = additive.reshape({B * H, N, S});
            }
            scores = vulkan_add(scores, additive, 1);
        }
    }

    // Save attn_weights (float32) for backward reuse
    auto attn_weights = vulkan_softmax(scores, -1, std::nullopt);
    auto out_bh = vulkan_bmm(attn_weights, v_bh);
    auto output = cast_from_float32(out_bh.reshape({B, H, N, D}).contiguous(), orig_dtype);

    return std::make_tuple(output, attn_weights);
}

// ── Flash Attention ─────────────────────────────────────────────
// Fused QK^T + causal_mask + softmax + @V in a single pass (one workgroup per (b,h,q)).
// Eliminates the intermediate [B*H, N, S] attention weight matrix entirely.
// Saves ~7 dispatches per layer vs composition (reshape+bmm+mul+add+softmax+bmm+reshape).
//
// Q: [B, H, N, D], K: [B, KV_H, S, D], V: [B, KV_H, S, D]
// Returns: output [B, H, N, D], lse [B, H, N]  (log-sum-exp saved for backward)
std::tuple<at::Tensor, at::Tensor> vulkan_flash_attention_forward(
    const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
    float scale, bool is_causal, bool q_seq_major) {

    TORCH_CHECK(Q.dim() == 4, "flash_attention: Q must be 4D");
    TORCH_CHECK(K.dim() == 4, "flash_attention: K must be 4D");
    TORCH_CHECK(V.dim() == 4, "flash_attention: V must be 4D");

    check_supported_float(Q, "flash_attention");
    auto orig_dtype = Q.scalar_type();

    // Detect layout: [B,H,N,D] (head-major, stride[1]>stride[2]) or [B,N,H,D] (seq-major)
    // seq-major: stride[1] = H*D, stride[2] = D → stride[1] > stride[2] and stride[1] < B*H*N*D
    // head-major: stride[1] = N*D, stride[2] = D → same pattern but different semantics
    // We detect by checking if Q.size(1) == H (head-major) vs Q.size(1) == N (seq-major).
    // The caller must indicate via the sizes. We accept both [B,H,N,D] and [B,N,H,D].
    // Detection: if Q is contiguous and stride[1] == D*N (head-major) or stride[1] == D*H (seq-major).
    // Simpler: accept that Q is passed as [B,H,N,D] OR [B,N,H,D], and detect from strides:
    //   head-major contiguous: strides = (H*N*D, N*D, D, 1)
    //   seq-major contiguous:  strides = (N*H*D, H*D, D, 1)
    // For head-major: size[1]=H, size[2]=N; for seq-major: size[1]=N, size[2]=H.
    // The user passes (B,N,H,D) as a seq-major tensor from view+no-transpose — size[1]=N, size[2]=H.
    // We distinguish: q_seq_major = (Q.size(2) == H && Q.size(1) == N) where H comes from context.
    // Actually, we get H from K's 2nd dimension (for non-GQA) or from input.
    // Simplest: use strides. If Q.stride(1) == Q.stride(2) * Q.size(2) (head-major) or not.

    // For contiguous tensors: both layouts have stride(3)=1 and stride(2)=D.
    // head-major: strides = (H*N*D, N*D, D, 1), so stride(1) = N*D = stride(2)*size(2)
    // seq-major:  strides = (N*H*D, H*D, D, 1), so stride(1) = H*D = stride(2)*size(2) as well!
    // Both satisfy stride(1) == stride(2)*size(2) for contiguous, but size(2) differs.
    // HEAD-MAJOR: size = [B, H, N, D], seq-major: size = [B, N, H, D].
    // We need to know H. The caller passes it. Since we can't tell from the tensor alone,
    // the forward function receives Q as the user passes it:
    // - Standard [B,H,N,D]: we make it contiguous and use head-major shader
    // - [B,N,H,D] (seq-major from view): we use seq-major shader, avoiding contiguous copy

    // Layout support: head-major [B,H,N,D] or seq-major [B,S,H,D].
    //
    // Auto-detection of transposed seq-major views (zero-copy optimization):
    //   linear(x).view(B,S,H,D).transpose(1,2) → size=[B,H,S,D], strides=(S*H*D, D, H*D, 1)
    //   Key: stride(1) == D (the head dim stride equals D, unlike contiguous where stride(1) = N*D).
    //   This pattern avoids 3 × .contiguous() copy dispatches per attention call (saving 3 dispatches
    //   per layer × 4 layers = 12 forward dispatches in a 4-layer model).
    //
    // For auto-detected: the tensor has size=[B,H,N,D] but memory layout is [B,S=N,H,D].
    //   params H = size[1] (num heads), N = size[2] (seq len) — same as head-major.
    // For q_seq_major=true: tensor has size=[B,N,H,D], so H = size[2], N = size[1].
    auto is_transposed_seq_major = [](const at::Tensor& t) -> bool {
        return t.dim() == 4 &&
               t.stride(3) == 1 &&
               t.stride(1) == t.size(3) &&              // stride(H-dim) == D
               t.stride(2) == t.size(1) * t.size(3);    // stride(S-dim) == H*D
    };

    bool auto_seq = !q_seq_major &&
                    is_transposed_seq_major(Q) &&
                    is_transposed_seq_major(K) &&
                    is_transposed_seq_major(V);
    bool use_seq_major = q_seq_major || auto_seq;

    at::Tensor q, k, v;
    int64_t B, H, N, D, KV_H, S;

    if (use_seq_major) {
        if (q_seq_major) {
            // Caller explicitly signaled [B, S, H, D] contiguous layout
            TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                        "flash_attention: q_seq_major=True requires contiguous tensors");
            B = Q.size(0); N = Q.size(1); H = Q.size(2); D = Q.size(3);
            S = K.size(1); KV_H = K.size(2);
        } else {
            // Auto-detected transposed view: size=[B,H,N,D] but memory=[B,S,H,D]
            B = Q.size(0); H = Q.size(1); N = Q.size(2); D = Q.size(3);
            KV_H = K.size(1); S = K.size(2);
        }
        // Pass tensors as-is — shader handles seq-major stride access
        q = ensure_float32(Q);
        k = ensure_float32(K);
        v = ensure_float32(V);
    } else {
        // head-major [B, H, N, D] — make contiguous if needed
        q = ensure_float32(Q.contiguous());
        k = ensure_float32(K.contiguous());
        v = ensure_float32(V.contiguous());
        B = q.size(0); H = q.size(1); N = q.size(2); D = q.size(3);
        KV_H = k.size(1); S = k.size(2);
    }
    TORCH_CHECK(H % KV_H == 0, "flash_attention: H must be divisible by KV_H");
    TORCH_CHECK(D <= 256, "flash_attention: head dim D must be <= 256");

    // When seq-major: output storage is [B,N,H,D]; returned as non-contiguous [B,H,N,D] view.
    // This lets the caller's .transpose(1,2).reshape(B,S,-1) be zero-copy.
    // When head-major: output is [B,H,N,D] contiguous.
    at::Tensor output = use_seq_major
        ? at::empty({B, N, H, D}, q.options())   // seq-major storage
        : at::empty({B, H, N, D}, q.options());  // head-major storage
    auto lse    = at::empty({B, H, N}, q.options());  // log-sum-exp always head-major

    if (B * H * N * D == 0) {
        at::Tensor out_view = use_seq_major ? output.transpose(1, 2) : output;
        return {cast_from_float32(out_view, orig_dtype), lse};
    }

    struct {
        uint32_t B, H, KV_H, N, S, D;
        float scale;
        uint32_t is_causal;
        uint32_t q_seq_major;  // 0 = [B,H,N,D], 1 = [B,N,H,D]
    } params{
        static_cast<uint32_t>(B), static_cast<uint32_t>(H),
        static_cast<uint32_t>(KV_H), static_cast<uint32_t>(N),
        static_cast<uint32_t>(S), static_cast<uint32_t>(D),
        scale, is_causal ? 1u : 0u, use_seq_major ? 1u : 0u
    };

    // One workgroup per (b, h, q) triplet
    // D<=32:  wave-intrinsic variant (32 threads, WaveActiveSum, zero barriers)
    // D<=64:  d64 variant (64 threads, 5 barrier syncs per key vs 8 for D>64)
    // D>64:   standard 256-thread variant
    uint32_t total_workgroups = static_cast<uint32_t>(B * H * N);
    if (D <= 32) {
        dispatch_shader("attention_flash_attention_fwd_wave_fwd",
                        shaders::attention_flash_attention_fwd_wave_fwd,
                        shaders::attention_flash_attention_fwd_wave_fwd_size,
                        {q, k, v, output, lse},
                        total_workgroups, 1, 1,
                        &params, sizeof(params), 2);
    } else if (D <= 64) {
        dispatch_shader("attention_flash_attention_fwd_d64_fwd",
                        shaders::attention_flash_attention_fwd_d64_fwd,
                        shaders::attention_flash_attention_fwd_d64_fwd_size,
                        {q, k, v, output, lse},
                        total_workgroups, 1, 1,
                        &params, sizeof(params), 2);
    } else {
        dispatch_shader("attention_flash_attention_fwd_fwd",
                        shaders::attention_flash_attention_fwd_fwd,
                        shaders::attention_flash_attention_fwd_fwd_size,
                        {q, k, v, output, lse},
                        total_workgroups, 1, 1,
                        &params, sizeof(params), 2);
    }

    // Return output as [B,H,N,D]: seq-major path returns non-contiguous view from [B,N,H,D] storage
    at::Tensor out_view = use_seq_major ? output.transpose(1, 2) : output;
    return {cast_from_float32(out_view, orig_dtype), lse};
}

// Flash Attention backward — 2 dispatches:
//   1. Compute grad_Q via bwd shader (workgroup per (b,h,q)); D_i computed inline
//   2. Compute grad_K, grad_V via bwd_kv shader (workgroup per (b,kv_h,k)); D_i fused with go_dot_v
// Q/K/V may be non-contiguous transposed seq-major views (auto-detected) to save 3 contiguous copies.
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_flash_attention_backward(
    const at::Tensor& grad_out, const at::Tensor& Q,
    const at::Tensor& K, const at::Tensor& V,
    const at::Tensor& out, const at::Tensor& lse,
    float scale, bool is_causal) {

    // Auto-detect seq-major layout (same criterion as forward)
    auto is_transposed_seq_major = [](const at::Tensor& t) -> bool {
        return t.dim() == 4 &&
               t.stride(3) == 1 &&
               t.stride(1) == t.size(3) &&
               t.stride(2) == t.size(1) * t.size(3);
    };
    bool use_seq_major = is_transposed_seq_major(Q) &&
                         is_transposed_seq_major(K) &&
                         is_transposed_seq_major(V);

    // grad_out and out layout depends on how the output was used downstream.
    // If output was consumed via .transpose(1,2).reshape() (seq-major path),
    // then grad_out flows back as seq-major non-contiguous [B,H,N,D] view.
    // We detect this the same way as Q/K/V: is_transposed_seq_major.
    bool go_seq_major = use_seq_major && is_transposed_seq_major(grad_out);
    bool out_seq_major = use_seq_major && is_transposed_seq_major(out);
    auto go = ensure_float32(go_seq_major  ? grad_out : grad_out.contiguous());
    auto o  = ensure_float32(out_seq_major ? out      : out.contiguous());

    // Q/K/V: use seq-major (no copy) if auto-detected, otherwise make contiguous head-major
    at::Tensor q, k, v;
    int64_t B, H, N, D, KV_H, S;
    if (use_seq_major) {
        q = ensure_float32(Q);  // non-contiguous seq-major, passed directly to shader
        k = ensure_float32(K);
        v = ensure_float32(V);
        B = Q.size(0); H = Q.size(1); N = Q.size(2); D = Q.size(3);
        KV_H = K.size(1); S = K.size(2);
    } else {
        q = ensure_float32(Q.contiguous());
        k = ensure_float32(K.contiguous());
        v = ensure_float32(V.contiguous());
        B = q.size(0); H = q.size(1); N = q.size(2); D = q.size(3);
        KV_H = k.size(1); S = k.size(2);
    }

    auto qo_opts = at::TensorOptions().dtype(at::kFloat).device(Q.device());
    // When seq-major, allocate grad_Q as [B,N,H,D] and write in seq-major layout.
    // This way, returning it as a [B,H,N,D] non-contiguous view means the
    // transpose_backward + view_backward chain can be zero-copy.
    at::Tensor grad_Q_storage = use_seq_major
        ? at::empty({B, N, H, D}, qo_opts)       // [B,N,H,D] seq-major storage
        : at::empty({B, H, N, D}, qo_opts);      // [B,H,N,D] head-major storage
    at::Tensor grad_K_storage = use_seq_major
        ? at::empty({B, S, KV_H, D}, qo_opts)    // [B,S,KV_H,D] seq-major storage
        : at::empty({B, KV_H, S, D}, qo_opts);   // [B,KV_H,S,D] head-major storage
    at::Tensor grad_V_storage = use_seq_major
        ? at::empty({B, S, KV_H, D}, qo_opts)
        : at::empty({B, KV_H, S, D}, qo_opts);
    struct BwdParams {
        uint32_t B, H, KV_H, N, S, D;
        float scale;
        uint32_t is_causal;
        uint32_t q_seq_major;   // Q/K/V + grad_Q/grad_K/grad_V layout
        uint32_t go_seq_major;  // grad_out layout (independent of q_seq_major)
    };
    BwdParams params{
        static_cast<uint32_t>(B), static_cast<uint32_t>(H),
        static_cast<uint32_t>(KV_H), static_cast<uint32_t>(N),
        static_cast<uint32_t>(S), static_cast<uint32_t>(D),
        scale, is_causal ? 1u : 0u,
        use_seq_major ? 1u : 0u,
        go_seq_major  ? 1u : 0u
    };

    // Dispatch routing: D<=32 → wave (0 barriers/reduction), D<=64 → d64 (5 barriers), D>64 → std (8 barriers)
    // Step 1: compute grad_Q (workgroup per (b,h,q))
    if (D <= 32) {
        dispatch_shader("attention_flash_attention_bwd_wave_fwd",
                        shaders::attention_flash_attention_bwd_wave_fwd,
                        shaders::attention_flash_attention_bwd_wave_fwd_size,
                        {go, q, k, v, lse, o, grad_Q_storage},
                        static_cast<uint32_t>(B * H * N), 1, 1,
                        &params, sizeof(params), 1);
    } else if (D <= 64) {
        dispatch_shader("attention_flash_attention_bwd_d64_fwd",
                        shaders::attention_flash_attention_bwd_d64_fwd,
                        shaders::attention_flash_attention_bwd_d64_fwd_size,
                        {go, q, k, v, lse, o, grad_Q_storage},
                        static_cast<uint32_t>(B * H * N), 1, 1,
                        &params, sizeof(params), 1);
    } else {
        dispatch_shader("attention_flash_attention_bwd_fwd",
                        shaders::attention_flash_attention_bwd_fwd,
                        shaders::attention_flash_attention_bwd_fwd_size,
                        {go, q, k, v, lse, o, grad_Q_storage},
                        static_cast<uint32_t>(B * H * N), 1, 1,
                        &params, sizeof(params), 1);
    }

    // Step 2: compute grad_K, grad_V (workgroup per (b,kv_h,k))
    if (D <= 32) {
        dispatch_shader("attention_flash_attention_bwd_kv_wave_fwd",
                        shaders::attention_flash_attention_bwd_kv_wave_fwd,
                        shaders::attention_flash_attention_bwd_kv_wave_fwd_size,
                        {go, q, k, v, lse, o, grad_K_storage, grad_V_storage},
                        static_cast<uint32_t>(B * KV_H * S), 1, 1,
                        &params, sizeof(params), 2);
    } else if (D <= 64) {
        dispatch_shader("attention_flash_attention_bwd_kv_d64_fwd",
                        shaders::attention_flash_attention_bwd_kv_d64_fwd,
                        shaders::attention_flash_attention_bwd_kv_d64_fwd_size,
                        {go, q, k, v, lse, o, grad_K_storage, grad_V_storage},
                        static_cast<uint32_t>(B * KV_H * S), 1, 1,
                        &params, sizeof(params), 2);
    } else {
        dispatch_shader("attention_flash_attention_bwd_kv_fwd",
                        shaders::attention_flash_attention_bwd_kv_fwd,
                        shaders::attention_flash_attention_bwd_kv_fwd_size,
                        {go, q, k, v, lse, o, grad_K_storage, grad_V_storage},
                        static_cast<uint32_t>(B * KV_H * S), 1, 1,
                        &params, sizeof(params), 2);
    }

    // Build output gradients.
    // When seq-major: storage is [B,N,H,D]/[B,S,KV_H,D]; return as transpose(1,2)
    // non-contiguous views. This means transpose_backward (and view_backward) in the
    // autograd chain can be zero-copy: transpose undoes via metadata, view uses computeStride
    // on the contiguous underlying storage.
    // When head-major: storage is already [B,H,N,D]/[B,KV_H,S,D], return directly.
    at::Tensor out_gQ, out_gK, out_gV;
    if (use_seq_major) {
        out_gQ = cast_from_float32(grad_Q_storage, Q.scalar_type()).transpose(1, 2);
        out_gK = cast_from_float32(grad_K_storage, K.scalar_type()).transpose(1, 2);
        out_gV = cast_from_float32(grad_V_storage, V.scalar_type()).transpose(1, 2);
    } else {
        out_gQ = cast_from_float32(grad_Q_storage, Q.scalar_type());
        out_gK = cast_from_float32(grad_K_storage, K.scalar_type());
        out_gV = cast_from_float32(grad_V_storage, V.scalar_type());
    }
    return {out_gQ, out_gK, out_gV};
}

// ── RoPE (Rotary Position Embedding) ────────────────────────────
at::Tensor vulkan_rope(const at::Tensor& input, double theta) {
    auto input_c = input.contiguous();
    check_supported_float(input_c, "RoPE");
    auto orig_dtype = input_c.scalar_type();
    input_c = ensure_float32(input_c);
    TORCH_CHECK(input_c.dim() == 4, "RoPE expects 4D input [B, H, N, D]");
    TORCH_CHECK(input_c.size(3) % 2 == 0, "RoPE: head dimension must be even");

    int64_t B = input_c.size(0), H = input_c.size(1);
    int64_t N = input_c.size(2), D = input_c.size(3);

    auto output = at::empty_like(input_c);
    uint32_t total = static_cast<uint32_t>(B * H * N * D);

    if (total == 0) return output;

    struct { uint32_t B, H, N, D; float theta; } params{
        static_cast<uint32_t>(B), static_cast<uint32_t>(H),
        static_cast<uint32_t>(N), static_cast<uint32_t>(D),
        static_cast<float>(theta)
    };

    uint32_t workgroups = (total + 255) / 256;

    dispatch_shader("attention_rope_fwd",
                    shaders::attention_rope_fwd, shaders::attention_rope_fwd_size,
                    {input_c, output},
                    workgroups, 1, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

}} // namespace torch_vulkan::ops
