//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHK):-dx12 -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHK):-vk -compute  -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHK):-mtl -compute  -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-cuda -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-cpu -compute -compile-arg -O3 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -shaderobj

// Test doing vector comparisons 

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

bool assignFunc(int index)
{
    outputBuffer[index] = 1;
    return true;
}

[numthreads(16, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    // Only the first 4 elements will be 1
    (index < 4) && assignFunc(index);

    // Only the last 4 elements will be 1.
    (index < 12) || assignFunc(index);

    //CHK-COUNT-4: 1
    //CHK-COUNT-8: 0
    //CHK-COUNT-4: 1
}
