// Slice along a dimension: out = input[dim][start:end:step]

struct Params {
    uint numel;        // total output elements
    uint outer_size;   // product of dims before sliced dim
    uint inner_size;   // product of dims after sliced dim
    uint dim_size;     // original dim size
    uint out_dim_size; // output dim size
    uint start;        // start index
    uint step;         // step size
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    uint outer_inner = params.out_dim_size * params.inner_size;
    uint outer = tid.x / outer_inner;
    uint rem = tid.x % outer_inner;
    uint dim_idx = rem / params.inner_size;
    uint inner = rem % params.inner_size;

    uint src_dim_idx = params.start + dim_idx * params.step;
    uint src_idx = outer * (params.dim_size * params.inner_size) +
                   src_dim_idx * params.inner_size + inner;
    out[tid.x] = input[src_idx];
}
