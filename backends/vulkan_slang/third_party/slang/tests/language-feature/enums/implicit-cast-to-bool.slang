// Test enums can be implicitly casted to bool. 

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

[Flags]
enum BitFlags
{
    One,
    Two,
    Three
}

int test<let n : bool>()
{
    return n;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    BitFlags x = BitFlags.One | BitFlags.Two;
    if (x & BitFlags.One)
        outputBuffer[0] = 1;
    else
        outputBuffer[0] = 0;
    if (x & BitFlags.Three)
        outputBuffer[1] = 1;
    else
        outputBuffer[1] = 0;
    if (x & BitFlags.Two)
        outputBuffer[2] = 1;
    else
        outputBuffer[2] = 0;

    outputBuffer[3] = test<BitFlags.One & BitFlags.One>();
    // CHECK: 1
    // CHECK: 0
    // CHECK: 1
    // CHECK: 1
}
