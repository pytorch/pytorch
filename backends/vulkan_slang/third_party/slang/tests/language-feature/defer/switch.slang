//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

void testFunc(int i, int param)
{
    defer outputBuffer[i++] = param*3+2;
    switch(param)
    {
    case 0:
        defer outputBuffer[i++] = 1;
        outputBuffer[i++] = 0;
        break;
    case 1:
        defer outputBuffer[i++] = 4;
        outputBuffer[i++] = 3;
        break;
    default:
        defer outputBuffer[i++] = 7;
        outputBuffer[i++] = 6;
        break;
    };
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    testFunc(0, 0);
    testFunc(3, 1);
    testFunc(6, 2);
}
