// GELU backward: grad_input = grad_output * d/dx[GELU(x)]
// d/dx[GELU(x)] = 0.5*(1+tanh(k*(x+0.044715*x^3))) + 0.5*x*sech^2(k*(x+0.044715*x^3))*k*(1+3*0.044715*x^2)

struct Params { uint numel; };

[[vk::binding(0)]] StructuredBuffer<float> grad_output;
[[vk::binding(1)]] StructuredBuffer<float> input;
[[vk::binding(2)]] RWStructuredBuffer<float> grad_input;
[[vk::push_constant]] Params params;

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= params.numel) return;

    float x = input[tid.x];
    float go = grad_output[tid.x];

    float k = 0.7978845608f; // sqrt(2/pi)
    float c = 0.044715f;
    float inner = k * (x + c * x * x * x);
    // Clamp to avoid NaN from tanh overflow (NaN at |x|>=89 on some drivers)
    inner = clamp(inner, -88.0f, 88.0f);
    float t = tanh(inner);
    float sech2 = 1.0f - t * t;
    float d_inner = k * (1.0f + 3.0f * c * x * x);

    float grad = 0.5f * (1.0f + t) + 0.5f * x * sech2 * d_inner;
    grad_input[tid.x] = go * grad;
}
