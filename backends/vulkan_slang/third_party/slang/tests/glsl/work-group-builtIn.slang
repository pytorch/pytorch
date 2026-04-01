//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
#version 430
precision highp float;
precision highp int;

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name=outputBuffer
layout(binding = 0) buffer MyBlockName
{
    int data[6];
} outputBuffer;

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
void computeMain()
{
    outputBuffer.data[0] = gl_NumWorkGroups.x;
    outputBuffer.data[1] = gl_NumWorkGroups.y;
    outputBuffer.data[2] = gl_NumWorkGroups.z;
    // BUF: 1
    // BUF-NEXT: 1
    // BUF-NEXT: 1
}