// general-inline.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

[ForceInline]
int test(int r)
{
    int result = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            result++;
            if (r == 0)
            {
                return result;
            }
        }
        if (r == 1)
            return result;
    }
    return result;
}

[ForceInline]
void testVoid(int r, out int result)
{
    result = 0;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            result++;
            if (r == 0)
            {
                return;
            }
        }
        if (r == 1)
            return;
    }
    return;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    outputBuffer[0] = test(0);
    outputBuffer[1] = test(1);
    outputBuffer[2] = test(2);

    int rs;
    testVoid(2, rs);
    outputBuffer[3] = rs;
}
