// Direct Conv2D forward
// Input: [N, C_in, iH, iW], Weight: [C_out, C_in/groups, kH, kW], Output: [N, C_out, oH, oW]
// Each thread computes one output element.

struct Params {
    uint N;
    uint C_in; uint C_out;
    uint iH; uint iW;
    uint oH; uint oW;
    uint kH; uint kW;
    uint sH; uint sW;
    uint pH; uint pW;
    uint groups;
    uint dH; uint dW;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> weight;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.N * params.C_out * params.oH * params.oW;
    if (tid.x >= total) return;

    uint idx = tid.x;
    uint ow = idx % params.oW; idx /= params.oW;
    uint oh = idx % params.oH; idx /= params.oH;
    uint co = idx % params.C_out; idx /= params.C_out;
    uint n  = idx;

    uint channels_per_group = params.C_in / params.groups;
    uint group = co / (params.C_out / params.groups);
    uint ci_start = group * channels_per_group;

    float sum = 0.0f;

    // Weight layout: [C_out, C_in/groups, kH, kW]
    uint w_base = co * channels_per_group * params.kH * params.kW;

    for (uint ci = 0; ci < channels_per_group; ci++) {
        uint in_c = ci_start + ci;
        uint in_base = (n * params.C_in + in_c) * params.iH * params.iW;
        uint w_ci = w_base + ci * params.kH * params.kW;

        for (uint kh = 0; kh < params.kH; kh++) {
            int ih = int(oh * params.sH + kh * params.dH) - int(params.pH);
            if (ih < 0 || uint(ih) >= params.iH) continue;

            for (uint kw = 0; kw < params.kW; kw++) {
                int iw = int(ow * params.sW + kw * params.dW) - int(params.pW);
                if (iw < 0 || uint(iw) >= params.iW) continue;

                float in_val = input[in_base + uint(ih) * params.iW + uint(iw)];
                float w_val = weight[w_ci + kh * params.kW + kw];
                sum += in_val * w_val;
            }
        }
    }

    out[tid.x] = sum;
}
