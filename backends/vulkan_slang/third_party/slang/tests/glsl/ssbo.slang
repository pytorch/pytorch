//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
#version 430
precision highp float;
precision highp int;

//TEST_INPUT:ubuffer(data=[2 3 5 7], stride=4):name=inputBuffer
buffer MyBlockName
{
    uint data[];
} inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
buffer MyBlockName2
{
    uint data[];
} outputBuffer;

layout(local_size_x = 4) in;
[shader("compute")]
void computeMain()
{
    outputBuffer.data[gl_GlobalInvocationID.x] = inputBuffer.data[gl_GlobalInvocationID.x];
    // BUF: 2
    // BUF-NEXT: 3
    // BUF-NEXT: 5
    // BUF-NEXT: 7
}
