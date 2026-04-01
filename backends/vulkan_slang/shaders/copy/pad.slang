// 2D constant padding: pad last 2 dims of input tensor
// pad = [left, right, top, bottom]

struct Params {
    uint numel;       // total output elements
    uint batch_size;  // product of all dims except last 2
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint pad_left;
    uint pad_top;
    float pad_value;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    uint hw = params.out_h * params.out_w;
    uint batch = tid.x / hw;
    uint rem = tid.x % hw;
    uint oh = rem / params.out_w;
    uint ow = rem % params.out_w;

    // Map output coords to input coords
    int ih = (int)oh - (int)params.pad_top;
    int iw = (int)ow - (int)params.pad_left;

    if (ih >= 0 && ih < (int)params.in_h && iw >= 0 && iw < (int)params.in_w) {
        uint src = batch * (params.in_h * params.in_w) + (uint)ih * params.in_w + (uint)iw;
        out[tid.x] = input[src];
    } else {
        out[tid.x] = params.pad_value;
    }
}
