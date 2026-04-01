// Batch normalization (inference mode only: uses running_mean/running_var).
// Input: [N, C, ...] flattened as [N, C, spatial_size]
// Each thread handles one element.

struct Params {
    uint num_channels;
    uint spatial_size;   // product of spatial dims (H*W)
    uint total;          // N * C * spatial_size
    float eps;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> running_mean;   // [C]
[[vk::binding(2)]] StructuredBuffer<float> running_var;    // [C]
[[vk::binding(3)]] StructuredBuffer<float> weight;         // gamma [C]
[[vk::binding(4)]] StructuredBuffer<float> bias;           // beta [C]
[[vk::binding(5)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= params.total) return;

    // Determine which channel this element belongs to
    uint c = (idx / params.spatial_size) % params.num_channels;

    float x = input[idx];
    float mean = running_mean[c];
    float var = running_var[c];
    float gamma = weight[c];
    float beta = bias[c];

    out[idx] = gamma * (x - mean) * rsqrt(var + params.eps) + beta;
}
