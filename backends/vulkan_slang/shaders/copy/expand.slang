// Expand (broadcast) tensor to larger shape.
// Maps each output index back to the input using modulo for broadcast dims.
// Supports up to 8 dimensions.
// in_strides[d] is the actual stride of the input tensor (handles non-contiguous inputs).

struct Params {
    uint numel;
    uint ndim;
    uint in_shape[8];
    uint out_shape[8];
    uint in_strides[8];
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    // Convert flat output index to multi-dim, map back to input using actual strides
    uint remaining = tid.x;
    uint in_offset = 0;

    for (uint d = 0; d < params.ndim; d++) {
        // Compute output stride for dim d (contiguous output layout)
        uint out_stride = 1;
        for (uint k = d + 1; k < params.ndim; k++) {
            out_stride *= params.out_shape[k];
        }
        uint coord = remaining / out_stride;
        remaining = remaining % out_stride;

        // Map to input coord (broadcast: if in_shape[d]==1, coord=0)
        uint in_coord = (params.in_shape[d] == 1) ? 0 : coord;

        // Use actual input stride (handles non-contiguous layouts)
        in_offset += in_coord * params.in_strides[d];
    }

    out[tid.x] = input[in_offset];
}
