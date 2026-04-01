// Check isnan: out[i] = isnan(input[i]) ? 1.0 : 0.0
// Output is float but represents bool (will be cast to bool by caller)

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    // NaN != NaN is the standard NaN check
    float v = input[tid.x];
    out[tid.x] = (v != v) ? 1.0f : 0.0f;
}
