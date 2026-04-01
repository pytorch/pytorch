//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

static const uint levels = 8;

[Differentiable]
void eval(float3 p, out float4 output[levels])
{
    [ForceUnroll] for (int level = 0; level < 3; ++level) 
    {
        float4 f = 0.f;

        // tri-linear time!
        [ForceUnroll] for (int z = 0; z < 2; ++z)
        {
            float wx = 0;
            if (z != 0)
                wx = p.x;
            else
                wx = p.y;

            f += wx;
        }

        output[level] = f;
    }
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float3 p = float3(2.0, 3.0, 0);

    float4 output[levels];
    eval(p, output);
    DifferentialPair<float3> dp = DifferentialPair<float3>(p, 0);
    __bwd_diff(eval)(dp, output);

    // Write output
    outputBuffer[0] = dp.d.x;
}
