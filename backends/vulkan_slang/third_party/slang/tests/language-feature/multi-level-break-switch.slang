// multi-level-break-switch.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

int test(int t, int r)
{
    int result = 0;
outter:
    switch (t)
    {
    case 0:
    case 1:
    jLoop:
        for (int j = 0; j < 3; j++)
        {
            switch (r)
            {
            case 0:
                break outter;
            case 1:
                break jLoop;
            default:
            {
                result++;
                break;
            }
            }
        }
        result++;
        break;
    case 2:
        return 100;
    }
    return result;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    outputBuffer[0] = test(dispatchThreadID.x, 0);
    outputBuffer[1] = test(dispatchThreadID.x, 1);
    outputBuffer[2] = test(dispatchThreadID.x, 2);
    outputBuffer[3] = 0;
}
