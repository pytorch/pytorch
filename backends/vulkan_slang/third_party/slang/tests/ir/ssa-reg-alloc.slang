//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

int test1(uint p)
{
    int a, b;
    if (p > 1)
    {
        a = 1;
        b = 2;
    }
    else
    {
        a = 2;
        b = 3;
    }
    // b is not used and should not interfere the result of a.
    return a;
}

int test2(uint p)
{
    int a, b;
    if (p > 1)
    {
        a = 1;
        b = 2;
    }
    else
    {
        a = 2;
        b = 3;
    }
    // a is not used and should not interfere the result of b.
    return b;
}

int test3(uint p)
{
    int a = 1;
    int b = 5;
    
    if (p > 0) a = 2;
    if (p > 0) b = 3;

    // a and b are now register allocated.
    // The first block of the loop will have IRParams in the form of (a, b)
    for (int i = 0; i <= p + 2; i++)
    {
        let tmp = a;
        a = b;
        b = tmp;
        // The branch back to the loop header will have phi args: (b, a)
        // Phi-elmination must handle this case of concurrent assignment correctly.
    }
    return a - b; // should be 4 when p == 0.
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    let rs1 = test1(dispatchThreadID.x) + test2(dispatchThreadID.x);
    outputBuffer[0] = rs1;
    outputBuffer[1] = test3(0);
}
