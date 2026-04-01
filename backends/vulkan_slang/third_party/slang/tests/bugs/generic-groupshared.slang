//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[ForceInline]
Ref<uint> table<let n: int>(int index)
{
    static groupshared uint array[n];
    return array[index];
}

struct S<let n : int>
{
    static const int M = n * 2;
    int doSomething()
    {
        table<M>(0) = M;
        return int(table<M>(0));
    }
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    S<2> s;
    outputBuffer[0] = s.doSomething();
}
