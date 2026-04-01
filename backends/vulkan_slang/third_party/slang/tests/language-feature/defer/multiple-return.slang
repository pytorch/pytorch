//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
// ^ Due to DeviceMemoryBarrier() missing.

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

void testFunc(int index1, int index2, int condition)
{
    outputBuffer[index1] = 128;
    defer outputBuffer[index1] = 0;

    outputBuffer[index2] = 3;
    DeviceMemoryBarrier();

    if (condition == 0)
    {
        outputBuffer[index2] = 1;
        return;
    }

    if (condition == 1)
    {
        outputBuffer[index2] = 2;
        return;
    }
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    for (int i = 0; i < 3; ++i)
        testFunc(2*i, 2*i+1, i);
}
