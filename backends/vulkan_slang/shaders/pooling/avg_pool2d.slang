// Average pooling 2D
// Input: [N, C, H, W] flattened; Output: [N, C, oH, oW]

struct Params {
    uint batch_channels;  // N * C
    uint iH; uint iW;
    uint oH; uint oW;
    uint kH; uint kW;
    uint sH; uint sW;
    uint pH; uint pW;
    uint count_include_pad;  // 1 = divide by kH*kW, 0 = divide by valid count
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

    uint idx = tid.x;
    uint ow = idx % params.oW; idx /= params.oW;
    uint oh = idx % params.oH; idx /= params.oH;
    uint bc = idx;

    float sum = 0.0f;
    uint count = 0;
    uint in_base = bc * params.iH * params.iW;

    for (uint kh = 0; kh < params.kH; kh++) {
        int ih = int(oh * params.sH + kh) - int(params.pH);
        if (ih < 0 || uint(ih) >= params.iH) continue;
        for (uint kw = 0; kw < params.kW; kw++) {
            int iw = int(ow * params.sW + kw) - int(params.pW);
            if (iw < 0 || uint(iw) >= params.iW) continue;
            sum += input[in_base + uint(ih) * params.iW + uint(iw)];
            count++;
        }
    }

    float divisor = (params.count_include_pad != 0) ? float(params.kH * params.kW) : float(count);
    out[tid.x] = (divisor > 0.0f) ? sum / divisor : 0.0f;
}
