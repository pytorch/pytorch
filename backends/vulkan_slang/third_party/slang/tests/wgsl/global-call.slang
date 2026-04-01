//TEST(compute):COMPARE_COMPUTE:-shaderobj

//TEST_INPUT:ubuffer(data=[3 7 8 10], stride=4):name=inputBuffer
RWStructuredBuffer<uint> inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

uint f(uint a)
{
    return a*inputBuffer[0];
}

static uint b = f(2);

[numthreads(1,1,1)]
void computeMain()
{
    outputBuffer[0] = b;
}