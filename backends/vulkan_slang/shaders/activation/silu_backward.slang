// Fused SiLU backward: grad_input = grad_output * sigmoid(x) * (1 + x - x * sigmoid(x))
// Single-pass instead of 6 separate element-wise ops.
// SiLU(x) = x * sigmoid(x)
// SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//           = sigmoid(x) * (1 + x - x * sigmoid(x))

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> self;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= params.numel) return;
    float x = self[tid.x];
    float sig = 1.0f / (1.0f + exp(-x));
    float deriv = sig * (1.0f + x - x * sig);
    grad_input[tid.x] = grad_output[tid.x] * deriv;
}
