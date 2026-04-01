// Max pooling 2D with indices output (for backward pass)
// Input: [N, C, H, W] flattened
// Output: [N, C, oH, oW] values + [N, C, oH, oW] indices

struct Params {
    uint batch_channels;  // N * C
    uint iH; uint iW;
    uint oH; uint oW;
    uint kH; uint kW;
    uint sH; uint sW;
    uint pH; uint pW;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::binding(2)]] RWStructuredBuffer<float> indices;  // stores linear index as float (reinterpret as uint)
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
    uint bc = idx;  // batch * channels

    float max_val = -3.402823466e+38f;
    uint max_idx = 0;
    uint in_base = bc * params.iH * params.iW;

    for (uint kh = 0; kh < params.kH; kh++) {
        int ih = int(oh * params.sH + kh) - int(params.pH);
        if (ih < 0 || uint(ih) >= params.iH) continue;
        for (uint kw = 0; kw < params.kW; kw++) {
            int iw = int(ow * params.sW + kw) - int(params.pW);
            if (iw < 0 || uint(iw) >= params.iW) continue;
            uint linear_idx = in_base + uint(ih) * params.iW + uint(iw);
            float v = input[linear_idx];
            if (v > max_val) {
                max_val = v;
                max_idx = linear_idx;
            }
        }
    }

    out[tid.x] = max_val;
    indices[tid.x] = asfloat(max_idx);  // store uint as float bits
}
