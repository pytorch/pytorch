// simple-inline.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

[ForceInline]
int test(int r)
{
    return r+2;
}

[ForceInline]
void testVoid(int r, out int result)
{
    if (r == 0)
    {
        result = 2;
        return;
    }
    result = r + 3;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    outputBuffer[0] = test(dispatchThreadID.x);
    int rs;
    testVoid(dispatchThreadID.x + 2, rs);
    outputBuffer[1] = rs;
}
