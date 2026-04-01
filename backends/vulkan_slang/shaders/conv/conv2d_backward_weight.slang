// Conv2D backward pass w.r.t. weight (grad_weight)
// grad_weight[c_out, c_in/groups, kh, kw] = sum over n, oh, ow of
//   grad_output[n, c_out, oh, ow] * input[n, c_in, oh*sH + kh - pH, ow*sW + kw - pW]

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

[[vk::binding(0)]] StructuredBuffer<float> grad_output;   // [N, C_out, oH, oW]
[[vk::binding(1)]] StructuredBuffer<float> input;         // [N, C_in, iH, iW]
[[vk::binding(2)]] RWStructuredBuffer<float> grad_weight; // [C_out, C_in/groups, kH, kW]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint c_in_per_group = params.C_in / params.groups;
    uint c_out_per_group = params.C_out / params.groups;
    uint weight_numel = params.C_out * c_in_per_group * params.kH * params.kW;
    if (tid.x >= weight_numel) return;

    uint idx = tid.x;
    uint kw = idx % params.kW; idx /= params.kW;
    uint kh = idx % params.kH; idx /= params.kH;
    uint ci_local = idx % c_in_per_group; idx /= c_in_per_group;
    uint c_out = idx;

    uint group = c_out / c_out_per_group;
    uint c_in = group * c_in_per_group + ci_local;

    float sum = 0.0f;

    for (uint n = 0; n < params.N; n++) {
        for (uint oh = 0; oh < params.oH; oh++) {
            int ih = int(oh * params.sH + kh) - int(params.pH);
            if (ih < 0 || uint(ih) >= params.iH) continue;
            for (uint ow = 0; ow < params.oW; ow++) {
                int iw = int(ow * params.sW + kw) - int(params.pW);
                if (iw < 0 || uint(iw) >= params.iW) continue;

                uint go_idx = ((n * params.C_out + c_out) * params.oH + oh) * params.oW + ow;
                uint in_idx = ((n * params.C_in + c_in) * params.iH + uint(ih)) * params.iW + uint(iw);

                sum += grad_output[go_idx] * input[in_idx];
            }
        }
    }

    grad_weight[tid.x] = sum;
}
