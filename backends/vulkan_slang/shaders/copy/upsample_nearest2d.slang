// Upsample nearest 2D: NCHW layout
// Each output pixel maps to the nearest input pixel.

struct Params {
    uint numel;
    uint batch_channels;  // N * C
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    uint out_spatial = params.out_h * params.out_w;
    uint bc = tid.x / out_spatial;
    uint spatial_idx = tid.x % out_spatial;
    uint oh = spatial_idx / params.out_w;
    uint ow = spatial_idx % params.out_w;

    // Nearest neighbor: map output coord to input coord
    uint ih = oh * params.in_h / params.out_h;
    uint iw = ow * params.in_w / params.out_w;

    uint in_spatial = params.in_h * params.in_w;
    uint src_idx = bc * in_spatial + ih * params.in_w + iw;

    out[tid.x] = input[src_idx];
}
