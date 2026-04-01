// Tanh activation: out[i] = tanh(input[i])

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    // Clamp to avoid NaN from tanh overflow on some drivers (NaN at |x|>=89)
    x = clamp(x, -88.0f, 88.0f);
    out[tid.x] = tanh(x);
}
