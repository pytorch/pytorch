//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int i = 0;
    defer defer outputBuffer[i++] = 0xFF;

    outputBuffer[i++] = 0;

    for (int j = 0; j < 4; j++)
        defer outputBuffer[i++] = j;

    if (i == 5)
        defer outputBuffer[i++] = 0x80;

    while (i < 7)
        defer outputBuffer[i++] = 0x81;

    do defer outputBuffer[i++] = 0x82; while (false);

    outputBuffer[i++] = 5;
}
