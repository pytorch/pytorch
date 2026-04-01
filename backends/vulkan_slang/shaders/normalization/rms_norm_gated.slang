// RMSNormGated: out[n,d] = weight[d] * (input[n,d] / sqrt(mean(input[n,:]^2) + eps)) * silu(gate[n,d])
// Used by Qwen3.5-0.8B GatedDeltaNet layers (Qwen3_5RMSNormGated).
// One workgroup per row — each thread cooperates on the reduction.

struct Params {
    uint num_rows;    // N (batch * seq_len * num_heads)
    uint norm_size;   // D (head_v_dim)
    float eps;
};

[[vk::binding(0)]] StructuredBuffer<float> input;    // [num_rows, norm_size]
[[vk::binding(1)]] StructuredBuffer<float> gate;     // [num_rows, norm_size] — z in GatedDeltaNet
[[vk::binding(2)]] StructuredBuffer<float> weight;   // [norm_size]
[[vk::binding(3)]] RWStructuredBuffer<float> out;    // [num_rows, norm_size]
[[vk::binding(4)]] RWStructuredBuffer<float> rstd;   // [num_rows] — saved for backward
[[vk::push_constant]] Params params;

groupshared float smem[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID)
{
    uint row = gid.x;
    if (row >= params.num_rows) return;

    uint tid = lid.x;
    uint base = row * params.norm_size;

    // Step 1: Compute sum of squares (partial per thread)
    float partial_sq = 0.0f;
    for (uint d = tid; d < params.norm_size; d += 256) {
        float v = input[base + d];
        partial_sq += v * v;
    }
    smem[tid] = partial_sq;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    // Step 2: rstd = rsqrt(mean_sq + eps)
    float mean_sq = smem[0] / float(params.norm_size);
    float rstd_val = rsqrt(mean_sq + params.eps);
    if (tid == 0) rstd[row] = rstd_val;

    // Step 3: Normalize, scale by weight, apply silu(gate)
    for (uint d = tid; d < params.norm_size; d += 256) {
        float x_norm = input[base + d] * rstd_val * weight[d];
        // silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
        float g = gate[base + d];
        float silu_g = g / (1.0f + exp(-g));
        out[base + d] = x_norm * silu_g;
    }
}
