//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

// Test that a enum case can reference a previously defined case.

enum Check { V = 1, X = V+2 };

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK: 3
    outputBuffer[0] = Check.X;
}
