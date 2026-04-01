// Gather along a dimension.
// out[i][j][k] = input[i][index[i][j][k]][k]  (example for dim=1, 3D)
//
// Flattened: for each output element, decompose into (outer, dim_pos, inner),
// read index[flat_idx] to get the source dim position, then read input.

struct Params {
    uint numel;         // total output elements
    uint outer_size;    // product of dims before gather dim
    uint gather_dim;    // size of output along gather dim (= index dim size)
    uint input_dim;     // size of input along gather dim
    uint inner_size;    // product of dims after gather dim
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<uint> indices;  // int32 packed as uint
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    // Decompose flat output index: (outer, dim_pos, inner)
    uint inner = tid.x % params.inner_size;
    uint tmp = tid.x / params.inner_size;
    uint dim_pos = tmp % params.gather_dim;
    uint outer = tmp / params.gather_dim;

    // Read gather index for this output position
    uint src_dim = indices[tid.x];

    // Compute source flat index: same outer and inner, but use src_dim
    uint src_idx = outer * params.input_dim * params.inner_size
                 + src_dim * params.inner_size
                 + inner;

    out[tid.x] = input[src_idx];
}
