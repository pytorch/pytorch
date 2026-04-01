// Unary natural log: out[i] = ln(input[i])

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    // Handle edge cases: log(0) = -inf, log(negative) = NaN
    // Some drivers return finite values instead of -inf for log(0)
    if (x == 0.0f) {
        out[tid.x] = asfloat(0xFF800000u); // -inf
    } else if (x < 0.0f) {
        out[tid.x] = asfloat(0x7FC00000u); // NaN
    } else {
        out[tid.x] = log(x);
    }
}
