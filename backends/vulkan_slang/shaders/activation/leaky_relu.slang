// Leaky ReLU: out[i] = x >= 0 ? x : negative_slope * x

struct Params {
    float negative_slope;
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    out[tid.x] = x >= 0.0f ? x : params.negative_slope * x;
}
