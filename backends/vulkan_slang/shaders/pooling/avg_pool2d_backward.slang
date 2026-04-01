// avg_pool2d_backward: distribute grad_output back to grad_input
// For each input element, sum contributions from all output elements
// that pooled over it.
// Input: grad_output [N, C, oH, oW], Output: grad_input [N, C, iH, iW]

struct Params {
    uint total;     // N * C * iH * iW
    uint N;
    uint C;
    uint iH; uint iW;
    uint oH; uint oW;
    uint kH; uint kW;
    uint sH; uint sW;
    uint pH; uint pW;
    uint count_include_pad;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.total) return;

    // Decompose into (n, c, ih, iw)
    uint idx = tid.x;
    uint iw = idx % params.iW; idx /= params.iW;
    uint ih = idx % params.iH; idx /= params.iH;
    uint c  = idx % params.C;  idx /= params.C;
    uint n  = idx;

    float grad = 0.0f;

    // For each output position that uses this input position
    for (uint oh = 0; oh < params.oH; oh++) {
        for (uint ow = 0; ow < params.oW; ow++) {
            // Window covered by (oh, ow)
            int h_start = int(oh * params.sH) - int(params.pH);
            int w_start = int(ow * params.sW) - int(params.pW);
            int h_end = h_start + int(params.kH);
            int w_end = w_start + int(params.kW);

            // Check if (ih, iw) is in this window
            if (int(ih) >= h_start && int(ih) < h_end &&
                int(iw) >= w_start && int(iw) < w_end) {
                // Count pool size
                uint pool_size;
                if (params.count_include_pad != 0) {
                    pool_size = params.kH * params.kW;
                } else {
                    uint ch_start = max(h_start, 0);
                    uint ch_end = min(uint(h_end), params.iH);
                    uint cw_start = max(w_start, 0);
                    uint cw_end = min(uint(w_end), params.iW);
                    pool_size = (ch_end - ch_start) * (cw_end - cw_start);
                }
                if (pool_size > 0) {
                    uint go_idx = ((n * params.C + c) * params.oH + oh) * params.oW + ow;
                    grad += grad_output[go_idx] / float(pool_size);
                }
            }
        }
    }

    grad_input[tid.x] = grad;
}
