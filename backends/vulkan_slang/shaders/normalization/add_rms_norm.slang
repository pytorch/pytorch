// Fused add + RMSNorm: h_new = residual + shortcut; out = weight * (h_new / rms(h_new))
// One workgroup per row — fuses the add and RMSNorm into a single pass.
// Saves 1 dispatch vs separate add + rms_norm in every transformer layer.
//
// Returns:
//   out:   weight * h_new * rstd  — normalized+scaled output
//   h_new: residual + shortcut    — updated hidden state (for next layer residual)
//   rstd:  1/sqrt(mean(h_new^2) + eps)  — saved for backward

struct Params {
    uint num_rows;    // N (batch * seq_len)
    uint norm_size;   // D (hidden_size)
    float eps;
};

[[vk::binding(0)]] StructuredBuffer<float> residual;  // [num_rows, norm_size]
[[vk::binding(1)]] StructuredBuffer<float> shortcut;  // [num_rows, norm_size]
[[vk::binding(2)]] StructuredBuffer<float> weight;    // [norm_size]
[[vk::binding(3)]] RWStructuredBuffer<float> out;     // [num_rows, norm_size]
[[vk::binding(4)]] RWStructuredBuffer<float> h_new;   // [num_rows, norm_size] — for next layer
[[vk::binding(5)]] RWStructuredBuffer<float> rstd;    // [num_rows]
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

    // Step 1: Compute h_new = residual + shortcut, accumulate sum of squares
    float partial_sq = 0.0f;
    for (uint d = tid; d < params.norm_size; d += 256) {
        float v = residual[base + d] + shortcut[base + d];
        h_new[base + d] = v;
        partial_sq += v * v;
    }
    smem[tid] = partial_sq;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }

    // Step 2: Compute rstd
    float mean_sq = smem[0] / float(params.norm_size);
    float rstd_val = rsqrt(mean_sq + params.eps);

    if (tid == 0) rstd[row] = rstd_val;

    // Step 3: Normalize and scale
    for (uint d = tid; d < params.norm_size; d += 256) {
        out[base + d] = h_new[base + d] * rstd_val * weight[d];
    }
}
