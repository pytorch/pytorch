// Unary erf: out[i] = erf(input[i])
// Abramowitz & Stegun approximation (max error ~1.5e-7)

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

float erf_approx(float x) {
    float sign_x = (x >= 0.0f) ? 1.0f : -1.0f;
    float ax = abs(x);
    // Abramowitz & Stegun 7.1.26
    float p = 0.3275911f;
    float a1 = 0.254829592f;
    float a2 = -0.284496736f;
    float a3 = 1.421413741f;
    float a4 = -1.453152027f;
    float a5 = 1.061405429f;
    float t = 1.0f / (1.0f + p * ax);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    float y = 1.0f - (a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5) * exp(-ax*ax);
    return sign_x * y;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    out[tid.x] = erf_approx(input[tid.x]);
}
