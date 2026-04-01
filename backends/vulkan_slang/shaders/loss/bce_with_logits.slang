// BCE with Logits Loss: out[i] = max(x,0) - x*t + log(1 + exp(-|x|))
// Numerically stable version of: -t*log(sigmoid(x)) - (1-t)*log(1-sigmoid(x))

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] StructuredBuffer<float> target;
[[vk::binding(2)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    float t = target[tid.x];
    float max_val = max(x, 0.0f);
    out[tid.x] = max_val - x * t + log(1.0f + exp(-abs(x)));
}
