//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT: set buffer = [ubuffer(data=[0]), ubuffer(data=[1])]
RWByteAddressBuffer buffer[2];

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain( uint2 dispatchThreadID : SV_DispatchThreadID )
{
    buffer[0].InterlockedAddF32(0, 1.0f);
    outputBuffer[dispatchThreadID.x] = buffer[0].Load<float>(0);
}