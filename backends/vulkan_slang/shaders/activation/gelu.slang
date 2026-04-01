// GELU activation: out[i] = x * 0.5 * (1 + erf(x / sqrt(2)))
// Approximation: x * sigmoid(1.702 * x)

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    // Tanh approximation of GELU
    float k = 0.7978845608f; // sqrt(2/pi)
    float inner = k * (x + 0.044715f * x * x * x);
    // Clamp to avoid NaN from tanh overflow on SwiftShader (NaN at |x|>=89)
    inner = clamp(inner, -88.0f, 88.0f);
    out[tid.x] = 0.5f * x * (1.0f + tanh(inner));
}
