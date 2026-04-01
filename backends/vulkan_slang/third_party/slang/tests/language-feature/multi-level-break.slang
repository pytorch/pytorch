// multi-level-break.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

int test(int r)
{
    int result = 0;
iLoop:
    for (int i = 0; i < 2; i++)
    {
    jLoop:
        for (int j = 0; j < 3; j++)
        {
            for (;;)
            {
                result++;
                if (r == 0)
                {
                    // When r == 0, we break out the outter most loop,
                    // resulting the inner most statement being run only once.
                    break iLoop;
                }
                else if (r == 1)
                {
                    // When r == 1, we break out the `j` loop,
                    // resulting the inner most statement being run loop-i times.
                    break jLoop;
                }
                else
                {
                    // When r takes other values, we break out the inner most loop p,
                    // resulting the inner most statement being run i*j times.
                    break;
                }
            }
        }
    }
    return result;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    outputBuffer[0] = test(0);
    outputBuffer[1] = test(1);
    outputBuffer[2] = test(2);
    outputBuffer[3] = 0;
}
