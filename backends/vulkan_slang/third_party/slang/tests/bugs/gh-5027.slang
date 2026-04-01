//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-d3d11 -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

in uint3 gid : SV_GroupID;

[numthreads(1,1,1)]
void computeMain()
{
    uint expr = WorkgroupSize().x + gid.x;
    // CHECK: type: uint32_t
    // CHECK: 2
    outputBuffer[0] = expr + 1;
}