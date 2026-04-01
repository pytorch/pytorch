// index_put_: write values at indexed positions.
// self[indices[i]] = values[i]  (1D case)
// For accumulate mode: self[indices[i]] += values[i]
//
// Note: non-accumulate mode — last write wins (no atomics needed).

struct Params {
    uint num_indices;
    uint accumulate;  // 0 = replace, 1 = accumulate
};

[[vk::binding(0)]] StructuredBuffer<uint> indices;
[[vk::binding(1)]] StructuredBuffer<float> values;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.num_indices) return;

    uint dst_idx = indices[tid.x];
    if (params.accumulate != 0) {
        out[dst_idx] = out[dst_idx] + values[tid.x];
    } else {
        out[dst_idx] = values[tid.x];
    }
}
