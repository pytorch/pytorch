// scope.slang
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

// Confirm that scoping on enums and types works 

struct Thing
{
    int a;
    static Thing make(int v)
    {
        Thing thing;
        thing.a = v;
        return thing;
    }
};

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;

    Thing thing = Thing::make(tid);

    outputBuffer[tid] = thing.a;
}
