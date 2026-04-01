//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    int i = 0;
    int sum = 0;
    [ForceUnroll]
    while (i < 2)
    {
        int j = 1;
        [ForceUnroll(2)]
        while (j < 3)
        {
            sum += (i+j);
            j++;
        }
        i++;
    }
    outputBuffer[0] = sum;
}
