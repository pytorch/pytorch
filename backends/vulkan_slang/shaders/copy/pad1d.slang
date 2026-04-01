// 1D constant padding: pad last dim of input tensor
// pad = [left, right]

struct Params {
    uint numel;       // total output elements
    uint batch_size;  // product of all dims except last
    uint in_w;        // input last dim size
    uint out_w;       // output last dim size
    uint pad_left;
    float pad_value;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    uint batch = tid.x / params.out_w;
    uint ow = tid.x % params.out_w;

    int iw = (int)ow - (int)params.pad_left;

    if (iw >= 0 && iw < (int)params.in_w) {
        uint src = batch * params.in_w + (uint)iw;
        out[tid.x] = input[src];
    } else {
        out[tid.x] = params.pad_value;
    }
}
