// Hardtanh: clamp(x, min_val, max_val)

struct Params { uint numel; float min_val; float max_val; };

[[vk::binding(0)]] StructuredBuffer<float> input;
[[vk::binding(1)]] RWStructuredBuffer<float> out;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    out[tid.x] = clamp(input[tid.x], params.min_val, params.max_val);
}
