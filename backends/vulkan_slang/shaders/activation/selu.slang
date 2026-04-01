// SELU activation: scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
// alpha = 1.6732632423543772, scale = 1.0507009873554805

struct Params {
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    float alpha = 1.6732632423543772f;
    float scale = 1.0507009873554805f;
    float result = (x > 0.0f) ? scale * x : scale * alpha * (exp(x) - 1.0f);
    out[tid.x] = result;
}
