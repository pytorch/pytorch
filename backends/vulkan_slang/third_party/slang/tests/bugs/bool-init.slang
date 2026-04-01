//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

struct Thing
{
    bool a;
    bool b;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Thing thing = {};
    int index = int(dispatchThreadID.x);
    
    if (index % 3 != 0)
    {
        thing.a = (index & 1) != 0;
    }
    else
    {
        thing.b = (index & 2) != 0;
    }
 
    int v = (thing.a ? 2 : 0) + (thing.b ? 1: 0);

    outputBuffer[index] = v;
}