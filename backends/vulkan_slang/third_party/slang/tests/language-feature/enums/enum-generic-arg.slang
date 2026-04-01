//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

// Test that enum values can be used as compile time constants
// to specialize generics.

[Flags]
enum BitFlags
{
    One, Two, Three
}

int test<let F : BitFlags>()
{
    return F;
}

int testInt<let f : int>()
{
    return f;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK: 3
    outputBuffer[0] = test<BitFlags.One | BitFlags.Two>();

    // CHECK: 3
    outputBuffer[1] = testInt<BitFlags.One | BitFlags.Two>();
}
