//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl

#version 430
precision highp float;
precision highp int;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
layout(scalar) buffer MyBlockName2
{
    ivec3 data0;
    ivec3 data1;
} outputBuffer;

layout(local_size_x = 1) in;
void computeMain()
{
    outputBuffer.data0 = ivec3(1,2,3);
    outputBuffer.data1 = ivec3(4,5,6);

    // BUF: 1
    // BUF-NEXT: 2
    // BUF-NEXT: 3
    // BUF-NEXT: 4
    // BUF-NEXT: 5
    // BUF-NEXT: 6
}
