// Upsample nearest 2D backward: accumulate grad_output into grad_input.
// Each input pixel gathers gradients from all output pixels that map to it.
// Thread per input pixel.

struct Params {
    uint numel;          // total input elements (N*C*in_h*in_w)
    uint batch_channels; // N * C
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
};

[[vk::binding(0)]] StructuredBuffer<float> grad_output;   // [N, C, out_h, out_w]
[[vk::binding(1)]] RWStructuredBuffer<float> grad_input;   // [N, C, in_h, in_w]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;

    uint in_spatial = params.in_h * params.in_w;
    uint bc = tid.x / in_spatial;
    uint spatial_idx = tid.x % in_spatial;
    uint ih = spatial_idx / params.in_w;
    uint iw = spatial_idx % params.in_w;

    // Find the range of output pixels that map to this input pixel
    // Forward mapping: oh * in_h / out_h = ih  =>  oh in [ih * out_h / in_h, (ih+1) * out_h / in_h)
    uint oh_start = ih * params.out_h / params.in_h;
    uint oh_end = (ih + 1) * params.out_h / params.in_h;
    uint ow_start = iw * params.out_w / params.in_w;
    uint ow_end = (iw + 1) * params.out_w / params.in_w;

    uint out_spatial = params.out_h * params.out_w;
    uint out_base = bc * out_spatial;

    float sum = 0.0f;
    for (uint oh = oh_start; oh < oh_end; oh++) {
        for (uint ow = ow_start; ow < ow_end; ow++) {
            sum += grad_output[out_base + oh * params.out_w + ow];
        }
    }

    grad_input[tid.x] = sum;
}
