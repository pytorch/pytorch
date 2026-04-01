// Index select along dimension 0: out[i] = input[indices[i]]
// Generalized: each output element maps to a source element.

struct Params {
    uint num_indices;
    uint slice_size;    // product of dims after the indexed dim
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<uint> indices;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.num_indices * params.slice_size;
    if (tid.x >= total) return;

    uint row = tid.x / params.slice_size;
    uint col = tid.x % params.slice_size;

    uint src_row = indices[row];
    out[tid.x] = input[src_row * params.slice_size + col];
}
