//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;
typedef DifferentialPair<float4> dpfloat4;

[ForwardDifferentiable]
float2 f(float3 x)
{
    return x.zy;
}

[ForwardDifferentiable]
float2 g(float3 x, float4 y)
{
    float3 a = x + y.zyx;
    float2 b = x.zx - y.yw;
    return a.yz * b + 2 * x.xy * y.yz;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        float3 a = float3(2.0, 2.0, 2.0);
        float3 da = float3(1.0, 0.5, 1.0);

        outputBuffer[0] = __fwd_diff(f)(dpfloat3(a, da)).d.x;  // Expect: 1
        outputBuffer[1] = __fwd_diff(f)(dpfloat3(a, da)).d.y;  // Expect: 0.5
        
        float3 x = float3(0.5, 2.0, 0.5);
        float4 y = float4(-1.5, 1.0, 4.0, 2.0);
        float3 dx = float3(1.0, 0.0, -1.0);
        float4 dy = float4(0.0, 0.5, -0.25, 1.0);

        outputBuffer[2] = __fwd_diff(g)(dpfloat3(x, dx), dpfloat4(y, dy)).d.x; // Expect: -2.25
        outputBuffer[3] = __fwd_diff(g)(dpfloat3(x, dx), dpfloat4(y, dy)).d.y; // Expect: 0.5
        
    }
}
