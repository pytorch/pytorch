// atomic-image-access.slang
//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry computeMain -stage compute -emit-spirv-directly

// The executable test is disabled because it somehow doesn't work. Might be a gfx issue.
//TEST_DISABLED(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-vk -compute -output-using-type

// Test direct SPIR-V emit on image atomics.

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> resultBuffer;

// TEST_INPUT: set tex = RWTexture2D(format=R32Uint, size=4, content=zero, mipMaps = 1)
[vk::image_format("r32ui")]
RWTexture2D<uint> tex;

[numthreads(1,1,1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK: OpImageTexelPointer
    InterlockedAdd(tex[uint2(0, 0)], 1);
    uint oldVal;
    InterlockedAdd(tex[uint2(0, 0)], 1, oldVal);
    resultBuffer[0] = oldVal;
}
