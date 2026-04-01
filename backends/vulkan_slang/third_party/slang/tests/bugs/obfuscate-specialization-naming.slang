//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -compile-arg -obfuscate -shaderobj 
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -compile-arg -obfuscate -shaderobj

//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):name a
RWStructuredBuffer<int> a;
//TEST_INPUT:ubuffer(data=[0 2 4 6], stride=4):name b
RWStructuredBuffer<int> b;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name output
RWStructuredBuffer<int> output;

int doThing(RWStructuredBuffer<int> buf, int index)
{
    return buf[index];
}

[numthreads(16, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = int(dispatchThreadID.x); 

    output[tid] = doThing(a, tid) | doThing(b, tid);
}