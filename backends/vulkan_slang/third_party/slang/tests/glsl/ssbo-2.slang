//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
#version 430
precision highp float;
precision highp int;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
layout(binding = 0) buffer MyBlockName
{
    int a;
    int b;
    int c;
    int d;
} outputBuffer;

layout(local_size_x = 1) in;
void computeMain()
{
    outputBuffer.a = 1;
    outputBuffer.b = 2;
    outputBuffer.c = 3;
    outputBuffer.d = 4;
    // BUF: 1
    // BUF-NEXT: 2
    // BUF-NEXT: 3
    // BUF-NEXT: 4
}
