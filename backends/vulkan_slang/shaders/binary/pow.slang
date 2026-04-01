// Element-wise power: out[i] = a[i] ^ b[i]
// Handles negative bases correctly (matching PyTorch behavior):
// - negative base + integer exponent → correct signed result
// - negative base + non-integer exponent → NaN

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> a;
[[vk::binding(1)]] StructuredBuffer<float> b;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    float base = a[tid.x];
    float exponent = b[tid.x];

    if (base >= 0.0) {
        out[tid.x] = pow(base, exponent);
    } else {
        // Negative base: check if exponent is integer
        float rounded = round(exponent);
        if (abs(exponent - rounded) < 1e-6) {
            // Integer exponent: pow(abs(base), exp) with correct sign
            float mag = pow(abs(base), exponent);
            // Odd exponent → negative result
            int iexp = int(rounded);
            out[tid.x] = (iexp & 1) != 0 ? -mag : mag;
        } else {
            // Non-integer exponent with negative base → NaN (matches PyTorch)
            out[tid.x] = 0.0 / 0.0;
        }
    }
}
