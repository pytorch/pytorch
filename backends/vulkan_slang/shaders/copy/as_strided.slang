// as_strided: copy elements from source using arbitrary strides
// For each output element at multi-dim index, compute source offset
// using strides and storage_offset.

struct Params {
    uint numel;          // total output elements
    uint ndim;           // number of dimensions
    uint sizes[8];       // output sizes
    uint strides[8];     // input strides (in elements)
    uint storage_offset; // offset into source buffer
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> output;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    // Decompose flat output index into multi-dim coords
    uint remaining = tid.x;
    uint src_offset = params.storage_offset;

    for (int d = int(params.ndim) - 1; d >= 0; d--) {
        uint coord = remaining % params.sizes[d];
        remaining /= params.sizes[d];
        src_offset += coord * params.strides[d];
    }

    output[tid.x] = input[src_offset];
}
