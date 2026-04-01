// Mish activation: out[i] = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    // softplus(x) = log(1 + exp(x)), use stable version for large x
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    sp = clamp(sp, -88.0f, 88.0f);
    out[tid.x] = x * tanh(sp);
}
