// Test that the same generic type specialization does not get emitted as different types in target code.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

struct MyGeneric<let addOne: bool>
{
    int value;

    [mutating]
    void load(RWStructuredBuffer<MyGeneric<addOne>> buffer)
    {
        var m = buffer.Load(0);
        if (addOne)
            value = m.value + 1;
        else
            value = m.value;
    }
};

//TEST_INPUT:set myBuffer = ubuffer(data=[1],stride=4)
RWStructuredBuffer<MyGeneric<true>> myBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    MyGeneric<true> obj;
    obj.load(myBuffer);
    outputBuffer[dispatchThreadID.x] = obj.value;
}
