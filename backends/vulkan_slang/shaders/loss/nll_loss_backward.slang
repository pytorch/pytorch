// NLL Loss backward: grad_input[n, c] = -grad_output_val / total_weight  if c == target[n]
//                                      = 0.0                              otherwise
// grad_output_val is passed as a push constant (avoids GPU allocation for scalar 1.0).
// total_weight is the number of non-ignored samples (for mean reduction), stored as a GPU buffer.

struct Params {
    uint N;              // batch size
    uint C;              // num classes
    int ignore_index;    // target value to ignore
    float grad_output_val; // upstream gradient (typically 1.0)
};

[[vk::binding(0)]] StructuredBuffer<float> target;        // [N] packed as int
[[vk::binding(1)]] StructuredBuffer<float> total_weight;  // [1] scalar on GPU
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;  // [N, C]
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint total = params.N * params.C;
    if (tid.x >= total) return;

    uint n = tid.x / params.C;
    uint c = tid.x % params.C;

    int t = asint(target[n]);
    if (t == params.ignore_index) {
        grad_input[tid.x] = 0.0f;
        return;
    }

    if (c == uint(t)) {
        float tw = total_weight[0];
        if (tw == 0.0f) tw = 1.0f;
        grad_input[tid.x] = -params.grad_output_val / tw;
    } else {
        grad_input[tid.x] = 0.0f;
    }
}
