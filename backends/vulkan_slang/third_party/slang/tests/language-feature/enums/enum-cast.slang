//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

enum MyFlags
{
    A,B,C
}

bool test(MyFlags f1)
{
    return f1 == MyFlags(uint(1));
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int outVal = test(MyFlags.B) ? 1 : 0;
    // CHECK: 1
    outputBuffer[0] = outVal;
}
