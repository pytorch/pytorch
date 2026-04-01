//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

float lengthDiffX(float3 x)
{
    return (length(float3(x.x + 0.001, x.yz)) - length(float3(x.x - 0.001, x.yz))) / 0.002;
}
float lengthDiffY(float3 x)
{
    return (length(float3(x.x, x.y + 0.001, x.z)) - length(float3(x.x, x.y - 0.001, x.z))) / 0.002;
}
float lengthDiffZ(float3 x)
{
    return (length(float3(x.xy, x.z + 0.001)) - length(float3(x.xy, x.z - 0.001))) / 0.002;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        let x = float3(12, 23, 31);
        var dpx = diffPair(x);
        __bwd_diff(length)(dpx, 1.0);
        outputBuffer[0] = dpx.d[0];
        outputBuffer[1] = dpx.d[1];
        outputBuffer[2] = dpx.d[2];

        // for reference:
        //outputBuffer[3] = lengthDiffX(x);
        //outputBuffer[4] = lengthDiffY(x);
        //outputBuffer[5] = lengthDiffZ(x);
    }
}
