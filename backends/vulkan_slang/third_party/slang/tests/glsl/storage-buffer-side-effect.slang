//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
#version 430
precision highp float;
precision highp int;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer

buffer MyBlockName2
{
    uint data[1];
} outputBuffer;

layout(local_size_x = 4) in;

void sideEffectInit(int val)
{
    outputBuffer.data[0] = val;
}
void computeMain()
{
    outputBuffer.data[0] = 1000;
    sideEffectInit(4);
    // BUF: 4
}
