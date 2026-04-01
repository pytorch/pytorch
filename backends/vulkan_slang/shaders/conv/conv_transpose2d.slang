// Conv Transpose 2D forward
// Input: [N, C_in, iH, iW], Weight: [C_in, C_out/groups, kH, kW], Output: [N, C_out, oH, oW]
// Each thread computes one output element.
// Transpose conv: output[n,co,oh,ow] = sum over ci,kh,kw where
//   ih = (oh + pH - kh) / sH and iw = (ow + pW - kw) / sW (only when divisible)

struct Params {
    uint N;
    uint C_in; uint C_out;
    uint iH; uint iW;
    uint oH; uint oW;
    uint kH; uint kW;
    uint sH; uint sW;
    uint pH; uint pW;
    uint groups;
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

    uint co_per_group = params.C_out / params.groups;
    uint ci_per_group = params.C_in / params.groups;
    uint group = co / co_per_group;
    uint co_local = co % co_per_group;
    uint ci_start = group * ci_per_group;

    float sum = 0.0f;

    for (uint ci = 0; ci < ci_per_group; ci++) {
        uint in_c = ci_start + ci;
        uint in_base = (n * params.C_in + in_c) * params.iH * params.iW;
        // Weight layout: [C_in, C_out/groups, kH, kW]
        uint w_base = (in_c * co_per_group + co_local) * params.kH * params.kW;

        for (uint kh = 0; kh < params.kH; kh++) {
            int oh_minus = int(oh) + int(params.pH) - int(kh);
            if (oh_minus < 0 || oh_minus % int(params.sH) != 0) continue;
            int ih = oh_minus / int(params.sH);
            if (ih < 0 || uint(ih) >= params.iH) continue;

            for (uint kw = 0; kw < params.kW; kw++) {
                int ow_minus = int(ow) + int(params.pW) - int(kw);
                if (ow_minus < 0 || ow_minus % int(params.sW) != 0) continue;
                int iw = ow_minus / int(params.sW);
                if (iw < 0 || uint(iw) >= params.iW) continue;

                float in_val = input[in_base + uint(ih) * params.iW + uint(iw)];
                float w_val = weight[w_base + kh * params.kW + kw];
                sum += in_val * w_val;
            }
        }
    }

    out[tid.x] = sum;
}
