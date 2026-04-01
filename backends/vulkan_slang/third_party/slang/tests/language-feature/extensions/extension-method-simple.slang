// interface-extension.slang

// Test that an `extension` applied to an interface type works as users expect

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

struct MyType
{
    int v;
}

extension MyType
{
    int foo()
    {
        return v;
    }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    MyType t;
    t.v = 1;
    // CHECK: 1
    outputBuffer[dispatchThreadID.x] = t.foo();
}
