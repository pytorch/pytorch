// Scatter along a dimension (non-accumulate version).
// For each src element, write to output at the position given by index.
// self[i][index[i][j][k]][k] = src[i][j][k]  (example for dim=1, 3D)
//
// We iterate over src elements, writing into output.
// Note: for non-accumulate, last write wins (no atomics needed).

struct Params {
    uint numel;         // total src elements
    uint outer_size;    // product of dims before scatter dim
    uint scatter_dim;   // size of src/index along scatter dim
    uint self_dim;      // size of self (output) along scatter dim
    uint inner_size;    // product of dims after scatter dim
};

[[vk::binding(0)]] StructuredBuffer<float> src;
[[vk::binding(1)]] StructuredBuffer<uint> indices;  // int32 packed as uint
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    // Decompose flat src index: (outer, dim_pos, inner)
    uint inner = tid.x % params.inner_size;
    uint tmp = tid.x / params.inner_size;
    uint dim_pos = tmp % params.scatter_dim;
    uint outer = tmp / params.scatter_dim;

    // Read scatter index for this src position
    uint dst_dim = indices[tid.x];

    // Compute destination flat index
    uint dst_idx = outer * params.self_dim * params.inner_size
                 + dst_dim * params.inner_size
                 + inner;

    out[dst_idx] = src[tid.x];
}
