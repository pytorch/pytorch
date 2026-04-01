//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[Differentiable] [PreferRecompute]
float3 fetch(float2 uv)
{
    if (uv.x > 0.5f)
    {
        if (uv.x > 0.7f)
            return float3(2.) * uv.y;
        else
            return float3(1.) * uv.y;
    }
    else
    {
        if (uv.x > 0.3f)
            return float3(4.) * uv.y;
        else
            return float3(3.) * uv.y;
    }
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float2 uv = (float2)dispatchThreadID.xy / float2(512, 512);
    float3 color = fetch(uv);
    outputBuffer[0] = color.x; // Expect: 0.0

    {
        DifferentialPair<float2> dpuv = diffPair(float2(0.6f));
        bwd_diff(fetch)(dpuv, float3(1.f));

        outputBuffer[1] = dpuv.d.y; // Expect: 1.0 * 3 = 3
    }

    {
        DifferentialPair<float2> dpuv = diffPair(float2(0.8f));
        bwd_diff(fetch)(dpuv, float3(1.f));

        outputBuffer[2] = dpuv.d.y; // Expect: 2.0 * 3 = 6
    }

    {
        DifferentialPair<float2> dpuv = diffPair(float2(0.1f));
        bwd_diff(fetch)(dpuv, float3(1.f));
 
        outputBuffer[3] = dpuv.d.y; // Expect: 3.0 * 3 = 9
    }

    {
        DifferentialPair<float2> dpuv = diffPair(float2(0.4f));
        bwd_diff(fetch)(dpuv, float3(1.f));

        outputBuffer[4] = dpuv.d.y; // Expect: 4.0 * 3 = 12
    }
}