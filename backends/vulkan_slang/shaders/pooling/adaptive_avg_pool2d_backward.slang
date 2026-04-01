// Adaptive average pooling 2D backward: NCHW layout.
// Thread per INPUT pixel. Each input pixel gathers gradients from all
// output pixels whose adaptive window includes it.
// grad_input[ih,iw] = sum over (oh,ow) of grad_output[oh,ow] / window_size(oh,ow)
//   where ih is in [h_start(oh), h_end(oh)) and iw is in [w_start(ow), w_end(ow))

struct Params {
    uint numel;          // total input elements (N*C*iH*iW)
    uint batch_channels; // N * C
    uint iH;
    uint iW;
    uint oH;
    uint oW;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;   // [N, C, oH, oW]
[[vk::binding(1)]] RWStructuredBuffer<float> grad_input;   // [N, C, iH, iW]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    uint in_spatial = params.iH * params.iW;
    uint bc = tid.x / in_spatial;
    uint spatial_idx = tid.x % in_spatial;
    uint ih = spatial_idx / params.iW;
    uint iw = spatial_idx % params.iW;

    // Find the range of output pixels whose window includes this input pixel.
    // Forward: h_start(oh) = oh * iH / oH
    //          h_end(oh)   = ((oh+1) * iH + oH - 1) / oH
    // We need: h_start(oh) <= ih < h_end(oh)
    // oh * iH / oH <= ih  =>  oh <= ih * oH / iH  (with floor)
    // ((oh+1) * iH + oH - 1) / oH > ih  =>  oh+1 > (ih * oH + 1) / iH  (approx)
    // Safe bounds: iterate all oh where the window could contain ih.
    // oh_start: smallest oh where h_end(oh) > ih, i.e., oh where ((oh+1)*iH+oH-1)/oH > ih
    // oh_end: largest oh where h_start(oh) <= ih, i.e., oh*iH/oH <= ih => oh <= (ih+1)*oH/iH - 1

    // Conservative but correct bounds:
    // oh_end: h_start(oh) <= ih => oh * iH / oH <= ih => oh <= ih * oH / iH (integer)
    // But we need floor division, so oh_end = ih * oH / iH + 1 (exclusive upper bound, iterate up to)
    // Actually, let's compute it correctly:
    // h_start(oh) = floor(oh * iH / oH) <= ih means oh <= ceil((ih + 1) * oH / iH) - 1
    // oh_start: h_end(oh) > ih means ceil((oh+1) * iH / oH) > ih means oh + 1 > floor(ih * oH / iH)
    // Simplest correct approach: iterate conservatively

    uint oh_start = ih * params.oH / params.iH;
    // Go back to find the first oh whose h_end > ih
    while (oh_start > 0) {
        uint prev = oh_start - 1;
        uint h_end_prev = ((prev + 1) * params.iH + params.oH - 1) / params.oH;
        if (h_end_prev > params.iH) h_end_prev = params.iH;
        if (h_end_prev <= ih) break;
        oh_start = prev;
    }
    uint oh_end = min((ih + 1) * params.oH / params.iH + 1, params.oH);

    uint ow_start = iw * params.oW / params.iW;
    while (ow_start > 0) {
        uint prev = ow_start - 1;
        uint w_end_prev = ((prev + 1) * params.iW + params.oW - 1) / params.oW;
        if (w_end_prev > params.iW) w_end_prev = params.iW;
        if (w_end_prev <= iw) break;
        ow_start = prev;
    }
    uint ow_end = min((iw + 1) * params.oW / params.iW + 1, params.oW);

    uint out_spatial = params.oH * params.oW;
    uint out_base = bc * out_spatial;

    float sum = 0.0f;
    for (uint oh = oh_start; oh < oh_end; oh++) {
        uint h_start_oh = oh * params.iH / params.oH;
        uint h_end_oh = ((oh + 1) * params.iH + params.oH - 1) / params.oH;
        if (h_end_oh > params.iH) h_end_oh = params.iH;
        if (ih < h_start_oh || ih >= h_end_oh) continue;

        for (uint ow = ow_start; ow < ow_end; ow++) {
            uint w_start_ow = ow * params.iW / params.oW;
            uint w_end_ow = ((ow + 1) * params.iW + params.oW - 1) / params.oW;
            if (w_end_ow > params.iW) w_end_ow = params.iW;
            if (iw < w_start_ow || iw >= w_end_ow) continue;

            uint window_size = (h_end_oh - h_start_oh) * (w_end_ow - w_start_ow);
            sum += grad_output[out_base + oh * params.oW + ow] / float(window_size);
        }
    }

    grad_input[tid.x] = sum;
}
