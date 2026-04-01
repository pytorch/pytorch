// ELU: out[i] = x >= 0 ? x : alpha * (exp(x) - 1)

struct Params {
    float alpha;
    uint numel;
};

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float x = input[tid.x];
    out[tid.x] = x >= 0.0f ? x : params.alpha * (exp(x) - 1.0f);
}
