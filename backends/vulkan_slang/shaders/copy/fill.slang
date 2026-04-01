// Fill buffer with a constant value

struct Params {
    float value;
    uint numel;
};

[[vk::binding(0)]] RWStructuredBuffer<float> output;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    output[tid.x] = params.value;
}
