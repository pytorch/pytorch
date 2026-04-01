// Check isfinite: out[i] = isfinite(input[i]) ? 1.0 : 0.0

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float v = input[tid.x];
    // finite = not NaN and not inf
    float a = abs(v);
    bool is_nan = (v != v);
    bool is_inf = (a == 1.0f / 0.0f);
    out[tid.x] = (!is_nan && !is_inf) ? 1.0f : 0.0f;
}
