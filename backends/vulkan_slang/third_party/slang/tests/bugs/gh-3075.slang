//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile sm_6_0 -use-dxil -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11 -profile sm_5_0 -output-using-type

//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute  -output-using-type
//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer

RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    uint3 vLookup = 3;
    vLookup.yx += int2(1, 2); // calling += (inout int2, int2).
    outputBuffer[dispatchThreadID.x] = vLookup.x; // expect 5
}
