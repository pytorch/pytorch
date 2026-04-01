// Max pooling 2D backward pass (gather-based, no race conditions)
// Each thread handles one element of grad_input.
// For each input position, iterate over output positions whose pooling
// window covers it, and accumulate gradients where the index matches.

struct Params {
    uint input_numel;     // N * C * iH * iW
    uint output_numel;    // N * C * oH * oW
    uint NC;              // N * C
    uint iH; uint iW;
    uint oH; uint oW;
    uint kH; uint kW;
    uint sH; uint sW;
    uint pH; uint pW;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;   // [N, C, oH, oW]
[[vk::binding(1)]] StructuredBuffer<float> indices;        // [N, C, oH, oW] max indices (as float bits)
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;   // [N, C, iH, iW]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.input_numel) return;

    // Decode input position [bc, ih, iw]
    uint idx = tid.x;
    uint iw = idx % params.iW; idx /= params.iW;
    uint ih = idx % params.iH; idx /= params.iH;
    uint bc = idx;

    float sum = 0.0f;
    uint out_base = bc * params.oH * params.oW;

    // Compute range of output positions whose window covers (ih, iw)
    // oh * sH - pH <= ih < oh * sH - pH + kH
    // oh >= ceil((ih + pH - kH + 1) / sH), oh <= floor((ih + pH) / sH)
    int oh_start_raw = (int(ih) + int(params.pH) - int(params.kH) + int(params.sH)) / int(params.sH);
    int oh_end_raw = (int(ih) + int(params.pH)) / int(params.sH);
    uint oh_start = uint(max(oh_start_raw, 0));
    uint oh_end = min(uint(oh_end_raw), params.oH - 1);

    int ow_start_raw = (int(iw) + int(params.pW) - int(params.kW) + int(params.sW)) / int(params.sW);
    int ow_end_raw = (int(iw) + int(params.pW)) / int(params.sW);
    uint ow_start = uint(max(ow_start_raw, 0));
    uint ow_end = min(uint(ow_end_raw), params.oW - 1);

    for (uint oh = oh_start; oh <= oh_end; oh++) {
        for (uint ow = ow_start; ow <= ow_end; ow++) {
            uint out_idx = out_base + oh * params.oW + ow;
            uint stored_idx = asuint(indices[out_idx]);
            if (stored_idx == tid.x) {
                sum += grad_output[out_idx];
            }
        }
    }

    grad_input[tid.x] = sum;
}
