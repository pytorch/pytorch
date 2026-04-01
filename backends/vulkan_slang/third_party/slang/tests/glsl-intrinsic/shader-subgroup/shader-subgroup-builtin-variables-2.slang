//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-wgpu -compute -entry computeMain -allow-glsl


// There are some issues with the Metal backend when using glsl-style syntax - this should be fixed.
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-metal -compute -entry computeMain -allow-glsl

#version 430

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
buffer MyBlockName2
{
    int data[];
} outputBuffer;

layout(local_size_x = 4) in;

[shader("compute")]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    // There may be some issues with structure padding for global context containing
    // global builtin variables.
    // int idx = gl_GlobalInvocationID.x;
    int idx = dispatchThreadID.x;

    uint laneId = gl_SubgroupInvocationID;
    // The laneCount will be dependent on target hardware. It seems a count of 1 is valid in spec. 
    // For now we'll just check it's not 0.
    uint laneCount = gl_SubgroupSize;
    outputBuffer.data[idx] = int(((laneCount > 0) ? 0x100 : 0) + laneId);

    // BUF: 100
    // BUF-NEXT: 101
    // BUF-NEXT: 102
    // BUF-NEXT: 103
}
