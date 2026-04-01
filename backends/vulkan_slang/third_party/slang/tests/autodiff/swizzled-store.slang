//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef DifferentialPair<float2> dpfloat2;
typedef DifferentialPair<float3> dpfloat3;
typedef DifferentialPair<float4> dpfloat4;

[Differentiable]
float2 f(float3 x)
{
    float3 u;
    u.zy = x.yx;
    return u.zy;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        float3 a = float3(2.0, 2.0, 2.0);
        float3 da = float3(1.0, 0.5, 1.0);

        outputBuffer[0] = fwd_diff(f)(dpfloat3(a, da)).d.x;
    }

    {
        float3 a = float3(2.0, 2.0, 2.0);
        var dpa = diffPair(a);

        bwd_diff(f)(dpa, float2(0.5, 1.0));

        outputBuffer[1] = dpa.d.x; // 1.0
        outputBuffer[2] = dpa.d.y; // 0.5
        outputBuffer[3] = dpa.d.z; // 0.0
    }
}
