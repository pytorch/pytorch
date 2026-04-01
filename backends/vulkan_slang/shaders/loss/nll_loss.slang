// NLL Loss forward: loss = -input[target] for each sample
// Input: [N, C] (log-probabilities), Target: [N] (as uint packed in float)
// Output: [N] per-sample losses (0.0 for ignored indices)

struct Params {
    uint N;             // batch size
    uint C;             // num classes
    int ignore_index;   // target value to ignore (-100 typically)
};

[[vk::binding(0)]] StructuredBuffer<float> input;   // [N, C]
[[vk::binding(1)]] StructuredBuffer<float> target;   // [N] packed as int
[[vk::binding(2)]] RWStructuredBuffer<float> out;     // [N]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.N) return;
    uint n = tid.x;
    int t = asint(target[n]);
    if (t == params.ignore_index) {
        out[n] = 0.0f;
        return;
    }
    out[n] = -input[n * params.C + uint(t)];
}
