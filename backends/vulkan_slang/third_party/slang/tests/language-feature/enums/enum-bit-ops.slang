//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

[Flags]
enum MyFlags
{
    Zero = 0,
    BitOne, // = 1
    BitTwo, // = 2
    BitThree // = 4
}

bool test(MyFlags f1)
{
    return (f1 & MyFlags.BitTwo) != 0;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int outVal = test(MyFlags.BitOne | MyFlags.BitTwo) ? 1 : 0;
    // CHECK: 1
    outputBuffer[0] = outVal;
    // CHECK: 4
    outputBuffer[1] = (int)MyFlags.BitThree;
    // CHECK: 0
    outputBuffer[2] = test(MyFlags.BitOne | MyFlags.BitThree) ? 1 : 0;
    // CHECK: 8
    outputBuffer[3] = MyFlags(8);
}
