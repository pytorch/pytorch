//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;
typedef DifferentialPair<float4> dpfloat4;

[ForwardDifferentiable]
float3 f(float3 x)
{
    return x;
}

[ForwardDifferentiable]
float3 g(float3 x, float3 y)
{
    float3 a = x + y;
    float3 b = x - y;
    return a * b + 2 * x * y;
}

[ForwardDifferentiable]
float2 h(float2 x, float2 y)
{
    float2 a = x + y;
    float2 b = x - y;
    return a * b + 2 * x * y;
}

[ForwardDifferentiable]
float4 j(float4 x, float4 y)
{
    float4 a = x + y;
    float4 b = x - y;
    return a * b + 2 * x * y;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        float3 a = float3(2.0, 2.0, 2.0);
        float3 b = float3(1.5, 1.5, 1.5);
        float3 da = float3(1.0, 1.0, 1.0);

        float2 a2 = float2(2.0, 1.0);
        float2 b2 = float2(1.5, -2.0);

        float4 a4 = float4(2.0, 1.0, 0.0, 2.0);
        float4 b4 = float4(1.5, -2.0, 1.0, 1.5);

        outputBuffer[0] = __fwd_diff(f)(dpfloat3(a, da)).d.z;    // Expect: 1

        outputBuffer[1] = __fwd_diff(g)(
            dpfloat3(a, da),
            dpfloat3(b, float3(2.0, 1.0, 0.0))).d.y;        // Expect: 8

        outputBuffer[2] = __fwd_diff(h)(
            dpfloat2(a2, float2(1.0, 0.0)),
            dpfloat2(b2, float2(1.0, 1.0))).d.x;            // Expect: 8

        outputBuffer[3] = __fwd_diff(j)(
            dpfloat4(a4, float4(1.0)),
            dpfloat4(b4, float4(2.0))).d.w;                 // Expect: 9
    }
}
