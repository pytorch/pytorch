// Upsample bilinear 2D: NCHW layout
// Each output pixel is bilinearly interpolated from 4 input neighbors.

struct Params {
    uint numel;
    uint batch_channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint align_corners;  // 1 = align_corners, 0 = half-pixel offset
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

    // Compute source coordinates (floating point)
    float src_h, src_w;
    if (params.align_corners != 0 && params.out_h > 1) {
        src_h = float(oh) * float(params.in_h - 1) / float(params.out_h - 1);
    } else {
        src_h = (float(oh) + 0.5f) * float(params.in_h) / float(params.out_h) - 0.5f;
    }
    if (params.align_corners != 0 && params.out_w > 1) {
        src_w = float(ow) * float(params.in_w - 1) / float(params.out_w - 1);
    } else {
        src_w = (float(ow) + 0.5f) * float(params.in_w) / float(params.out_w) - 0.5f;
    }

    // Clamp to valid range
    src_h = max(0.0f, min(src_h, float(params.in_h - 1)));
    src_w = max(0.0f, min(src_w, float(params.in_w - 1)));

    uint h0 = uint(src_h);
    uint w0 = uint(src_w);
    uint h1 = min(h0 + 1, params.in_h - 1);
    uint w1 = min(w0 + 1, params.in_w - 1);

    float fh = src_h - float(h0);
    float fw = src_w - float(w0);

    uint in_spatial = params.in_h * params.in_w;
    uint base = bc * in_spatial;

    float v00 = input[base + h0 * params.in_w + w0];
    float v01 = input[base + h0 * params.in_w + w1];
    float v10 = input[base + h1 * params.in_w + w0];
    float v11 = input[base + h1 * params.in_w + w1];

    float value = v00 * (1.0f - fh) * (1.0f - fw)
                + v01 * (1.0f - fh) * fw
                + v10 * fh * (1.0f - fw)
                + v11 * fh * fw;

    out[tid.x] = value;
}
