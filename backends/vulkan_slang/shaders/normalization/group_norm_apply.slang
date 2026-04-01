// Group normalization: elementwise apply step.
// Input has been reshaped to [N*G, group_size].
// mean/rstd are [N*G] (one per group).
// weight/bias are [C] (per-channel), expanded to match group_size.
//
// out[row, col] = (input[row, col] - mean[row]) * rstd[row] * weight[ch] + bias[ch]
//
// This avoids shared-memory reduction (unreliable on SwiftShader).

struct Params {
    uint numel;          // N * G * group_size
    uint group_size;     // (C / G) * spatial
    uint channels_per_group;  // C / G
    uint spatial_size;   // H * W (or product of spatial dims)
    uint num_groups;     // G
};

[[vk::binding(0)]] StructuredBuffer<float> input;      // [N*G, group_size]
[[vk::binding(1)]] StructuredBuffer<float> mean;       // [N*G]
[[vk::binding(2)]] StructuredBuffer<float> rstd;       // [N*G]
[[vk::binding(3)]] StructuredBuffer<float> weight;     // [C]
[[vk::binding(4)]] StructuredBuffer<float> bias;       // [C]
[[vk::binding(5)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    // Decompose: tid.x = row * group_size + col
    uint row = tid.x / params.group_size;   // which (batch, group) pair
    uint col = tid.x % params.group_size;   // position within group

    // Compute channel index: group_idx * channels_per_group + col / spatial_size
    uint group_idx = row % params.num_groups;
    uint channel_in_group = col / params.spatial_size;
    uint channel = group_idx * params.channels_per_group + channel_in_group;

    float x = input[tid.x];
    float m = mean[row];
    float r = rstd[row];

    float normed = (x - m) * r;
    out[tid.x] = normed * weight[channel] + bias[channel];
}
