// Mish backward: grad_input = grad_output * d/dx[x * tanh(softplus(x))]
// d/dx[mish(x)] = tanh(sp) + x * sech^2(sp) * sigmoid(x)
// where sp = softplus(x) = log(1 + exp(x))

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> self;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;
    float x = self[tid.x];
    float sp = (x > 20.0f) ? x : log(1.0f + exp(x));
    sp = clamp(sp, -88.0f, 88.0f);
    float tanh_sp = tanh(sp);
    float sig = 1.0f / (1.0f + exp(-x));
    float sech2 = 1.0f - tanh_sp * tanh_sp;
    float dmish = tanh_sp + x * sech2 * sig;
    grad_input[tid.x] = grad_output[tid.x] * dmish;
}
