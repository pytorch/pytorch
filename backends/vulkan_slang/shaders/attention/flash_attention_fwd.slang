// Flash Attention forward pass (fused QK^T + softmax + @V in one workgroup)
//
// Computes: out[b,h,q,:] = softmax(Q[b,h,q,:] @ K[b,h,:,:]^T * scale + causal_mask) @ V[b,h,:,:]
//
// Layout:
//   Q: [B, H, N, D]  — query
//   K: [B, KV_H, S, D]  — key (KV_H may be < H for GQA; n_rep = H / KV_H)
//   V: [B, KV_H, S, D]  — value
//   out: [B, H, N, D]
//
// Dispatch: one workgroup per (b, h, q) triplet — total B*H*N workgroups
// Threads: 256 per workgroup (strided over D or S)
//
// Algorithm (online softmax, no intermediate S-sized allocation):
//   1. For each key position k=0..S-1:
//        score[k] = dot(Q[q], K[k]) * scale  (+ -inf if causal and k > q)
//   2. Compute softmax weights online: running max m, running sum l
//   3. Accumulate out += softmax_weight[k] * V[k]
//
// All within one workgroup — the attn matrix [N,S] never exists in memory.
// Saves: 1 reshape copy (q) + 1 bmm (QK^T) + 1 mul_scalar + 1 broadcast_add
//        + 1 softmax + 1 bmm (PV) + 1 reshape copy (out) = 7 dispatches → 1

struct Params {
    uint B;
    uint H;
    uint KV_H;   // KV heads (for GQA: KV_H <= H, n_rep = H / KV_H)
    uint N;      // query sequence length
    uint S;      // key/value sequence length
    uint D;      // head dimension
    float scale;
    uint is_causal;  // 1 = apply causal mask (k > q positions get -inf)
    uint q_seq_major; // 1 = Q/K/V in [B,N,H,D] layout; 0 = [B,H,N,D] layout
};

[[vk::binding(0)]] StructuredBuffer<float> Q;     // [B, H, N, D]
[[vk::binding(1)]] StructuredBuffer<float> K;     // [B, KV_H, S, D]
[[vk::binding(2)]] StructuredBuffer<float> V;     // [B, KV_H, S, D]
[[vk::binding(3)]] RWStructuredBuffer<float> out; // [B, H, N, D]
[[vk::binding(4)]] RWStructuredBuffer<float> lse; // [B, H, N] log-sum-exp for backward
[[vk::push_constant]] Params params;

// Shared memory for D-dimensional dot products (max D=256)
groupshared float smem[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    // gid.x encodes (b, h, q): gid.x = b * H * N + h * N + q
    uint flat = gid.x;
    uint q_pos = flat % params.N;
    uint h     = (flat / params.N) % params.H;
    uint b     = flat / (params.N * params.H);

    if (b >= params.B) return;

    uint tid = lid.x;
    uint kv_h = h / (params.H / params.KV_H); // which KV head (GQA mapping)

    // Base offsets — supports [B,H,N,D] (head-major) and [B,N,H,D] (seq-major) layouts
    uint q_base, k_base0, o_base;
    if (params.q_seq_major != 0u) {
        // Q/K/V in [B, S, H, D] (seq-major): index = b*S*H*D + s*H*D + h*D
        q_base  = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
        k_base0 = b * params.S * params.KV_H * params.D + kv_h * params.D; // + k*KV_H*D
        // Output in seq-major [B, N, H, D]: index = b*N*H*D + q_pos*H*D + h*D
        // Returned to caller as non-contiguous [B,H,N,D] view — allows zero-copy transpose+reshape.
        o_base  = b * params.N * params.H * params.D + q_pos * params.H * params.D + h * params.D;
    } else {
        // [B, H, N, D]: index = b*H*N*D + h*N*D + q_pos*D
        q_base  = b * params.H   * params.N * params.D + h   * params.N * params.D + q_pos * params.D;
        k_base0 = b * params.KV_H * params.S * params.D + kv_h * params.S * params.D; // + k*D
        o_base  = b * params.H   * params.N * params.D + h   * params.N * params.D + q_pos * params.D;
    }

    // Phase 1: compute all dot products score[k] = dot(Q[q], K[k]) * scale
    // Then immediately compute online softmax + V accumulation.
    //
    // Two-pass approach (bounded S, fits in registers for S <= 4096):
    //   Pass 1: compute scores, find max, compute exp, find sum
    //   Pass 2: accumulate out = sum_k(softmax_k * V[k])
    //
    // For small S (<=256), we store scores in smem temporarily.
    // For larger S, we use a single-pass online algorithm (Milakov & Gimelshein 2018).
    //
    // We use the online approach always for generality.
    // Thread 0 drives the sequential key loop; other threads assist with dot products.

    // --- Online softmax accumulation ---
    // Each key position: compute score via parallel dot product reduction,
    // then thread 0 updates running max/sum and V accumulation.

    float m = -1e38f;  // running max
    float l = 0.0f;    // running sum of exp(score - m)
    float acc[256];    // output accumulator — but D may be > 256 threads...

    // We accumulate output in smem across iterations.
    // Since D <= 256 (typical head dim), each thread owns one output dim element.
    float o_acc = 0.0f; // thread tid's contribution to out[q_pos, tid]

    for (uint k = 0; k < params.S; k++) {
        // Causal mask: skip future keys
        float causal_bias = 0.0f;
        if (params.is_causal != 0 && k > q_pos) {
            causal_bias = -1e38f;
        }

        // Key/Value base for position k
        // [B,H,N,D]: k_base = k_base0 + k*D
        // [B,N,H,D]: k_base = k_base0 + k*KV_H*D  (k_base0 already accounts for kv_h*D)
        uint k_base = params.q_seq_major != 0u
            ? k_base0 + k * params.KV_H * params.D
            : k_base0 + k * params.D;

        // Parallel dot product: Q[q] . K[k]
        // Thread tid accumulates D/256 elements
        float dot_part = 0.0f;
        for (uint d = tid; d < params.D; d += 256) {
            dot_part += Q[q_base + d] * K[k_base + d];
        }
        smem[tid] = dot_part;
        GroupMemoryBarrierWithGroupSync();

        // Reduce to get full dot product in smem[0]
        for (uint s = 128; s > 0; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            GroupMemoryBarrierWithGroupSync();
        }
        float score = smem[0] * params.scale + causal_bias;

        // Online softmax update (all threads see the same score)
        float m_new = max(m, score);
        float exp_score = exp(score - m_new);
        float scale_old = exp(m - m_new); // rescale old accumulation

        // Rescale o_acc by exp(m_old - m_new)
        // V has same layout as K: use k_base for V[k] access
        float v_val = 0.0f;
        if (tid < params.D) {
            v_val = V[k_base + tid];
        }
        // o_acc = o_acc * scale_old + exp_score * V[k, tid]
        o_acc = o_acc * scale_old + exp_score * v_val;

        l = l * scale_old + exp_score;
        m = m_new;

        GroupMemoryBarrierWithGroupSync(); // ensure smem reuse is safe
    }

    // Normalize: out[q_pos, tid] = o_acc / l
    if (tid < params.D) {
        out[o_base + tid] = o_acc / l;
    }
    // Save log-sum-exp = m + log(l) for backward (thread 0 writes scalar per query)
    if (tid == 0) {
        uint lse_idx = b * params.H * params.N + h * params.N + q_pos;
        lse[lse_idx] = m + log(l);
    }
}
