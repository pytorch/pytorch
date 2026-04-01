// Adaptive average pooling 2D: NCHW layout.
// For each output pixel, compute the input region and average it.
// start_index = floor(i * iSize / oSize), end_index = ceil((i+1) * iSize / oSize)

struct Params {
    uint batch_channels;
    uint iH;
    uint iW;
    uint oH;
    uint oW;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.batch_channels * params.oH * params.oW;
    if (tid.x >= total) return;

    uint out_spatial = params.oH * params.oW;
    uint bc = tid.x / out_spatial;
    uint spatial = tid.x % out_spatial;
    uint oh = spatial / params.oW;
    uint ow = spatial % params.oW;

    // Compute adaptive pooling window (matches PyTorch exactly)
    // start = floor(i * isize / osize), end = ceil((i+1) * isize / osize)
    uint h_start = oh * params.iH / params.oH;
    uint h_end = ((oh + 1) * params.iH + params.oH - 1) / params.oH;
    uint w_start = ow * params.iW / params.oW;
    uint w_end = ((ow + 1) * params.iW + params.oW - 1) / params.oW;

    // Clamp to valid range
    if (h_end > params.iH) h_end = params.iH;
    if (w_end > params.iW) w_end = params.iW;

    uint in_spatial = params.iH * params.iW;
    uint base = bc * in_spatial;

    float sum = 0.0f;
    uint count = 0;
    for (uint ih = h_start; ih < h_end; ih++) {
        for (uint iw = w_start; iw < w_end; iw++) {
            sum += input[base + ih * params.iW + iw];
            count++;
        }
    }

    out[tid.x] = sum / float(count);
}
