// Repeat (tile) a tensor. Maps output indices back to input using modulo.
// Supports up to 8 dimensions.

struct Params {
    uint numel;
    uint ndim;
    uint in_shape[8];
    uint out_shape[8];
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    uint remaining = tid.x;
    uint in_offset = 0;

    for (uint d = 0; d < params.ndim; d++) {
        uint out_stride = 1;
        for (uint k = d + 1; k < params.ndim; k++) {
            out_stride *= params.out_shape[k];
        }
        uint coord = remaining / out_stride;
        remaining = remaining % out_stride;

        // Wrap coordinate to input shape via modulo
        uint in_coord = coord % params.in_shape[d];

        uint in_stride = 1;
        for (uint k = d + 1; k < params.ndim; k++) {
            in_stride *= params.in_shape[k];
        }
        in_offset += in_coord * in_stride;
    }

    out[tid.x] = input[in_offset];
}
