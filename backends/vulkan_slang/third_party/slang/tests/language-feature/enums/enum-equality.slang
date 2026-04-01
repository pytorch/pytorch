// enum-equality.slang

// Test that equality (and inequality) of `enum`
// types works as expected.

//TEST(compute):COMPARE_COMPUTE: -shaderobj

enum Channel
{
    Red,
    Green,
    Blue,
    Alpha,
}

int test(int val)
{
    Channel channel= Channel(val);

    int result = 0;
    if(channel == Channel.Red)      result += 1;
    if(channel != Channel.Green)    result += 16;
    if(channel == Channel.Blue)     result += 16*16;
    if(channel != Channel.Alpha)    result += 16*16*16;

    return result;
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
