//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
float diffDeterminant(float2x2 x)
{
    return determinant(x);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        var dpx = diffPair(float2x2(1.0, 2.0, 3.0, 4.0));
        __bwd_diff(diffDeterminant)(dpx, 1.0);
        outputBuffer[0] = dpx.d[0][0];
        outputBuffer[1] = dpx.d[0][1];
        outputBuffer[2] = dpx.d[1][0];
        outputBuffer[3] = dpx.d[1][1];
    }
}
