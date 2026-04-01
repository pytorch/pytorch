// Conv2D backward pass w.r.t. input (grad_input)
// grad_input[n, c_in, ih, iw] = sum over c_out, kh, kw of
//   weight[c_out, c_in/groups, kh, kw] * grad_output[n, c_out, oh, ow]
// where oh = (ih + pH - kh) / sH, ow = (iw + pW - kw) / sW (if divisible and in bounds)

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

[[vk::binding(0)]] StructuredBuffer<float> grad_output;  // [N, C_out, oH, oW]
[[vk::binding(1)]] StructuredBuffer<float> weight;       // [C_out, C_in/groups, kH, kW]
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input; // [N, C_in, iH, iW]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.N * params.C_in * params.iH * params.iW;
    if (tid.x >= total) return;

    uint idx = tid.x;
    uint iw = idx % params.iW; idx /= params.iW;
    uint ih = idx % params.iH; idx /= params.iH;
    uint c_in = idx % params.C_in; idx /= params.C_in;
    uint n = idx;

    uint c_in_per_group = params.C_in / params.groups;
    uint c_out_per_group = params.C_out / params.groups;
    uint group = c_in / c_in_per_group;
    uint c_in_local = c_in % c_in_per_group;

    float sum = 0.0f;

    for (uint co = 0; co < c_out_per_group; co++) {
        uint c_out = group * c_out_per_group + co;
        for (uint kh = 0; kh < params.kH; kh++) {
            int oh_num = int(ih + params.pH) - int(kh);
            if (oh_num < 0 || oh_num % int(params.sH) != 0) continue;
            uint oh = uint(oh_num) / params.sH;
            if (oh >= params.oH) continue;

            for (uint kw = 0; kw < params.kW; kw++) {
                int ow_num = int(iw + params.pW) - int(kw);
                if (ow_num < 0 || ow_num % int(params.sW) != 0) continue;
                uint ow = uint(ow_num) / params.sW;
                if (ow >= params.oW) continue;

                uint go_idx = ((n * params.C_out + c_out) * params.oH + oh) * params.oW + ow;
                uint w_idx = ((c_out * c_in_per_group + c_in_local) * params.kH + kh) * params.kW + kw;

                sum += grad_output[go_idx] * weight[w_idx];
            }
        }
    }

    grad_input[tid.x] = sum;
}
