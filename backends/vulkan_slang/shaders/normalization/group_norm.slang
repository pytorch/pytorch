// Fused group normalization: mean/var/normalize/scale in one workgroup per (batch, group).
// Input: [N*G, group_size] where group_size = (C/G) * spatial_size.
// Supports any group_size via strided thread access (each thread handles group_size/256 elements).
// Outputs mean/rstd per row for use in backward pass.

groupshared float sdata[256];

struct Params {
    uint num_groups;           // G
    uint group_size;           // (C / G) * spatial_size
    uint num_rows;             // N * G
    uint channels_per_group;   // C / G
    uint spatial_size;         // product of spatial dims
    float eps;
};

[[vk::binding(0)]] StructuredBuffer<float> input;   // [N*G, group_size]
[[vk::binding(1)]] StructuredBuffer<float> weight;  // gamma [C]
[[vk::binding(2)]] StructuredBuffer<float> bias;    // beta [C]
[[vk::binding(3)]] RWStructuredBuffer<float> out;   // [N*G, group_size]
[[vk::binding(4)]] RWStructuredBuffer<float> out_mean;  // [N*G] — per-row mean for backward
[[vk::binding(5)]] RWStructuredBuffer<float> out_rstd;  // [N*G] — per-row rstd for backward
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(
    uint3 lid : SV_GroupThreadID,
    uint3 gid : SV_GroupID)
{
    uint row = gid.x;  // batch * G + group_idx
    uint tid = lid.x;
    uint S = params.group_size;

    if (row >= params.num_rows) return;

    uint base = row * S;

    // Step 1: Compute mean via strided partial sum
    float partial_sum = 0.0f;
    for (uint d = tid; d < S; d += 256) {
        partial_sum += input[base + d];
    }
    sdata[tid] = partial_sum;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction for sum
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = sdata[0] / float(S);
    GroupMemoryBarrierWithGroupSync();

    // Step 2: Compute variance via strided partial sum of (x - mean)^2
    float partial_var = 0.0f;
    for (uint d = tid; d < S; d += 256) {
        float diff = input[base + d] - mean;
        partial_var += diff * diff;
    }
    sdata[tid] = partial_var;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction for variance
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float var = sdata[0] / float(S);
    float inv_std = rsqrt(var + params.eps);

    // Step 3: Normalize and apply per-channel affine (strided write)
    uint group_idx = row % params.num_groups;
    for (uint d = tid; d < S; d += 256) {
        uint channel_in_group = d / params.spatial_size;
        uint channel = group_idx * params.channels_per_group + channel_in_group;

        float normed = (input[base + d] - mean) * inv_std;
        out[base + d] = normed * weight[channel] + bias[channel];
    }

    // Write per-row statistics for backward pass (only thread 0 per workgroup)
    if (tid == 0) {
        out_mean[row] = mean;
        out_rstd[row] = inv_std;
    }
}
