// array-resource.slang

// Test direct SPIR-V emit on arrays of buffers.

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry computeMain -stage compute -emit-spirv-directly
//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)

// Note: we can't run this test at the moment because gfx doesn't support allocating shader objects with unsized arrays.
//TEST_DISABLED(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-vk -compute -output-using-type

RWStructuredBuffer<uint> inputBuffers[];

RWStructuredBuffer<uint> resultBuffer;

//TEST_INPUT: set inputBuffers = {ubuffer(data=[1 0 0 0], stride=4), ubuffer(data=[2 0 0 0], stride=4)}

[numthreads(4,1,1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint threadId = dispatchThreadID.x;
    resultBuffer[threadId] = inputBuffers[0][threadId] + inputBuffers[1][threadId];
    // CHECK: OpCapability RuntimeDescriptorArray
    // BUFFER: 3
}
