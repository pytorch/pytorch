// Cat (concatenate) up to 8 tensors along a dimension in a single dispatch.
// Uses an offset table to route each output element to the correct input tensor.

static const uint MAX_INPUTS = 8;

struct Params {
    uint numel;              // total output elements
    uint outer_size;         // product of dims before cat dim
    uint inner_size;         // product of dims after cat dim
    uint n_inputs;           // number of input tensors (2..8)
    uint dim_offsets[MAX_INPUTS]; // cumulative offsets: dim_offsets[0]=0, dim_offsets[i] = sum of dim_sizes[0..i-1]
    uint dim_sizes[MAX_INPUTS];   // size of cat dim for each input
};

[[vk::binding(0)]] StructuredBuffer<float> in0;
[[vk::binding(1)]] StructuredBuffer<float> in1;
[[vk::binding(2)]] StructuredBuffer<float> in2;
[[vk::binding(3)]] StructuredBuffer<float> in3;
[[vk::binding(4)]] StructuredBuffer<float> in4;
[[vk::binding(5)]] StructuredBuffer<float> in5;
[[vk::binding(6)]] StructuredBuffer<float> in6;
[[vk::binding(7)]] StructuredBuffer<float> in7;
[[vk::binding(8)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

float read_input(uint which, uint idx) {
    switch (which) {
        case 0: return in0[idx];
        case 1: return in1[idx];
        case 2: return in2[idx];
        case 3: return in3[idx];
        case 4: return in4[idx];
        case 5: return in5[idx];
        case 6: return in6[idx];
        default: return in7[idx];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    uint total_dim = params.dim_offsets[params.n_inputs - 1] + params.dim_sizes[params.n_inputs - 1];
    uint outer_inner = total_dim * params.inner_size;
    uint outer = tid.x / outer_inner;
    uint rem = tid.x % outer_inner;
    uint dim_idx = rem / params.inner_size;
    uint inner = rem % params.inner_size;

    // Find which input tensor this dim_idx belongs to
    uint which = 0;
    for (uint i = 0; i < params.n_inputs - 1; i++) {
        if (dim_idx >= params.dim_offsets[i + 1]) {
            which = i + 1;
        }
    }

    uint local_dim_idx = dim_idx - params.dim_offsets[which];
    uint src = outer * (params.dim_sizes[which] * params.inner_size) +
               local_dim_idx * params.inner_size + inner;

    out[tid.x] = read_input(which, src);
}
