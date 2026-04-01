// Roll (circular shift) along a single dimension.

struct Params {
    uint numel;
    uint outer_size;
    uint dim_size;
    uint inner_size;
    uint shift;       // positive shift amount (pre-normalized to [0, dim_size))
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

    // Source index: shift backward (undo the roll)
    uint src_dim_idx = (dim_idx + params.dim_size - params.shift) % params.dim_size;
    uint src = outer * outer_inner + src_dim_idx * params.inner_size + inner;
    out[tid.x] = input[src];
}
