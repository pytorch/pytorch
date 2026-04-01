// explicit-tag-type.slang

// Test that the underlying "tag type" of an `enum` can be set.

//TEST(compute):COMPARE_COMPUTE: -shaderobj

enum Channel : uint
{
    Red,
    Green,
    Blue,
    Alpha,
}

uint test(uint val)
{
    Channel channel = Channel(val);
    uint u = uint(channel) + 1;
    channel = Channel(u & 0x3);
    return uint(channel);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    uint inVal = tid;
    uint outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
