//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    int sum = 0;
    [ForceUnroll]
    for (int i = 0; i < 2; i++)
    {
        [ForceUnroll(2)]
        for (int j = 1; j < 3; j++)
        {
            if (i == 1 && j == 1)
                continue;
            sum += (i+j);
        }
    }
    outputBuffer[0] = sum;
}
