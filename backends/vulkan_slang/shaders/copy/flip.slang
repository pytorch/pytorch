// Flip tensor along a single dimension.
// For multi-dim flip, dispatch multiple times.

struct Params {
    uint numel;
    uint outer_size;  // product of dims before flip dim
    uint dim_size;    // size of the flip dim
    uint inner_size;  // product of dims after flip dim
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    uint outer_inner = params.dim_size * params.inner_size;
    uint outer = tid.x / outer_inner;
    uint rem = tid.x % outer_inner;
    uint dim_idx = rem / params.inner_size;
    uint inner = rem % params.inner_size;

    // Reverse the dim index
    uint flipped_dim_idx = params.dim_size - 1 - dim_idx;
    uint src = outer * outer_inner + flipped_dim_idx * params.inner_size + inner;
    out[tid.x] = input[src];
}
