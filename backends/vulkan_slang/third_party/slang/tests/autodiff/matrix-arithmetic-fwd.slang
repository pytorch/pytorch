//TEST(compute):COMPARE_COMPUTE_EX:-wgpu -compute  -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[ForwardDifferentiable]
float3x3 g(float3x3 x, float3x3 y)
{
    float3x3 a = x + y;
    float3x3 b = x - y;
    return a * b + 2 * x * y;
}

[ForwardDifferentiable]
float h(float2x2 x, float2x2 y)
{
    let t = mul(x, y);
    return t[0][0] + t[0][1] + t[1][0] + t[1][1];
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float3x3 a = float3x3(2.0);
    float3x3 b = float3x3(1.5);
    float3x3 da = float3x3(1.0);

    outputBuffer[0] = __fwd_diff(g)(
                          diffPair(a, da),
                          diffPair(b, da)).d._11; // Expect: 8

    float2x2 l = float2x2(1.0, 2.0, 3.0, 4.0);
    float2x2 r = float2x2(10.0, 11.0, 12.0, 13.0);
    float2x2 d = float2x2(1.0, 0.0, 1.0, 1.0);
    
    //float2x2 epsilon = d * 0.001f;
    //outputBuffer[1] = (h(l + epsilon, r + epsilon) - h(l - epsilon, r - epsilon)) / (epsilon[0][0] * 2.0));

    outputBuffer[1] = __fwd_diff(h)(diffPair(l, d), diffPair(r, d)).d; // Expect 83.0
}
