// Masked scatter: writes src values into self at positions where mask is true.
// indices_buf[i] = source index for position i (packed as uint in float buffer).
// If indices_buf[i] != 0xFFFFFFFF, out[i] = src[indices_buf[i]].
// Otherwise out[i] = input[i].

struct Params {
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<uint> indices;
[[vk::binding(2)]] StructuredBuffer<float> src;
[[vk::binding(3)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= params.numel) return;

    uint src_idx = indices[idx];
    if (src_idx != 0xFFFFFFFF) {
        out[idx] = src[src_idx];
    } else {
        out[idx] = input[idx];
    }
}
