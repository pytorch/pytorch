//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = int(dispatchThreadID.x);   
    
    int a = index;
    
    // This is kind of silly - but it is a valid construct.
    // We want to check condition expression is executed though
    switch (++a)
    {
    }

    switch (index)
    {
        // This should not be executed
        a += 10;
    }

    outputBuffer[index] = a;
}