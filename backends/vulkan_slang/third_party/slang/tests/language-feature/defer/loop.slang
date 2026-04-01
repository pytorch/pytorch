//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int i = 0;
    defer outputBuffer[i++] = 128;
    for (int j = 0; j < 4; j++)
    {
        defer outputBuffer[i++] = j*2+1;
        outputBuffer[i++] = j*2;
    }

    for (int j = 0; j < 5; j++)
    {
        if (j == 1)
            continue;
        defer outputBuffer[i++] = j*2+1;
        if (j == 2)
            continue;
        outputBuffer[i++] = j*2;
        if (j == 3)
            break;
    }
}
