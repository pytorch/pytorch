// N-dimensional permute: rearrange data according to stride mapping.
// Push constants encode input strides and output shape for index remapping.
// Supports up to 8 dimensions.

struct Params {
    uint numel;
    uint ndim;
    uint in_strides[8];   // strides of input tensor (in elements)
    uint out_shape[8];     // shape of output tensor
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    // Convert flat output index to multi-dimensional index
    uint remaining = tid.x;
    uint in_offset = 0;

    for (uint d = 0; d < params.ndim; d++) {
        // Compute stride for this output dimension
        uint out_stride = 1;
        for (uint k = d + 1; k < params.ndim; k++) {
            out_stride *= params.out_shape[k];
        }
        uint coord = remaining / out_stride;
        remaining = remaining % out_stride;
        in_offset += coord * params.in_strides[d];
    }

    out[tid.x] = input[in_offset];
}
