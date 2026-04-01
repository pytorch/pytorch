// Softplus backward: grad * sigmoid(beta * x) if beta*x <= threshold, else grad

struct Params { uint numel; float beta; float threshold; };

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
    float bx = params.beta * x;
    float g;
    if (bx > params.threshold) {
        g = 1.0f;
    } else {
        g = 1.0f / (1.0f + exp(-bx));
    }
    grad_input[tid.x] = grad_output[tid.x] * g;
}
