// Check isinf: out[i] = isinf(input[i]) ? 1.0 : 0.0

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float v = input[tid.x];
    // inf check: abs(v) == infinity and v is not NaN
    float a = abs(v);
    out[tid.x] = (a == 1.0f / 0.0f) ? 1.0f : 0.0f;
}
