//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;
typedef DifferentialPair<float4> dpfloat4;

[Differentiable]
float4 f(float4 x)
{
    return float4(x.rgb, x.a * x.a);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        float4 a = float4(1.0, 2.0, 3.0, 4.0);
        float4 da = float4(1.0, 0.5, 1.5, 2.5);

        outputBuffer[0] = fwd_diff(f)(dpfloat4(a, da)).d.x;
    }

    {
        float4 a = float4(1.0, 2.0, 3.0, 4.0);
        
        var dpa = diffPair(a);

        bwd_diff(f)(dpa, float4(1.0, 0.5, 1.5, 2.5));

        outputBuffer[1] = dpa.d.x; // 1.0
        outputBuffer[2] = dpa.d.y; // 0.5
        outputBuffer[3] = dpa.d.z; // 0.0
        outputBuffer[4] = dpa.d.a; // 0.0
    }
}
