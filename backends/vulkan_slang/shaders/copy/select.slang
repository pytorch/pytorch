// Select a slice along a dimension: out = input[dim][index]
// Copies a contiguous block from the selected position.

struct Params {
    uint numel;        // total output elements
    uint outer_size;   // product of dims before selected dim
    uint inner_size;   // product of dims after selected dim
    uint dim_size;     // size of the selected dimension
    uint index;        // which index along dim to select
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    uint outer = tid.x / params.inner_size;
    uint inner = tid.x % params.inner_size;

    uint src_idx = outer * (params.dim_size * params.inner_size) +
                   params.index * params.inner_size + inner;
    out[tid.x] = input[src_idx];
}
