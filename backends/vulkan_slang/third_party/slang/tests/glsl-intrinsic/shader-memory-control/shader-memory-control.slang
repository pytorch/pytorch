//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
#version 460

// CHECK_GLSL: void main(
// CHECK_SPV: OpEntryPoint

//TEST_INPUT:ubuffer(data=[2 2 2 2], stride=4):out,name=raceCondition
buffer MyBlockName1
{
    uint data[];
} raceCondition;

groupshared uint raceConditionShared;

//TEST_INPUT: set raceConditionImage = RWTexture1D(format=R32Sint, size=4, content=one, mipMaps = 1)
uniform layout(binding=0,r32i) iimage1D raceConditionImage;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName2
{
    uint data[];
} outputBuffer;

layout(local_size_x = 32) in;

bool testMemoryBarrier()
{
    for (int i = 0; i < 4; i++)
    {
        raceCondition.data[0] = gl_GlobalInvocationID.x;
        memoryBarrier();
        raceCondition.data[0] = 1;
        memoryBarrier();
        if (raceCondition.data[0] != 1)
            return false;
    }
    return true
        && raceCondition.data[0] == 1
        ;
}
bool testMemoryBarrierAtomicCounter()
{
    for (int i = 0; i < 4; i++)
    {
        atomicExchange(raceCondition.data[1], 2);
        memoryBarrierAtomicCounter();
        atomicExchange(raceCondition.data[1], 1);
        memoryBarrierAtomicCounter();
        if (raceCondition.data[1] != 1)
            return false;
    }
    return true
        && raceCondition.data[1] == 1
        ;
}
bool testMemoryBarrierBuffer()
{
    for (int i = 0; i < 4; i++)
    {
        raceCondition.data[2] = gl_GlobalInvocationID.x;
        memoryBarrierBuffer();
        raceCondition.data[2] = 1;
        memoryBarrierBuffer();
        if (raceCondition.data[2] != 1)
            return false;
    }
    return true
        && raceCondition.data[2] == 1
        ;
}
bool testMemoryBarrierShared()
{
    for (int i = 0; i < 4; i++)
    {
        raceConditionShared = 2;
        memoryBarrierShared();
        raceConditionShared = 1;
        memoryBarrierShared();
        if (raceConditionShared != 1)
            return false;
    }
    return true
        && raceConditionShared == 1
        ;
}
bool testMemoryBarrierImage()
{
    for (int i = 0; i < 4; i++)
    {
        imageStore(raceConditionImage, 0, 2);
        memoryBarrierShared();
        imageStore(raceConditionImage, 0, 1);
        memoryBarrierShared();
        if (imageLoad(raceConditionImage, 0).x != 1)
            return false;
    }
    return true
        && imageLoad(raceConditionImage, 0).x == 1
        ;
}
bool testGroupMemoryBarrier()
{
    for (int i = 0; i < 4; i++)
    {
        raceCondition.data[3] = gl_GlobalInvocationID.x;
        groupMemoryBarrier();
        raceCondition.data[3] = 1;
        groupMemoryBarrier();
        if (raceCondition.data[3] != 1)
            return false;
    }
    return true
        && raceCondition.data[3] == 1
        ;
}
bool testMemoryBarriers()
{
    return true
        && testMemoryBarrier()
        && testMemoryBarrierAtomicCounter()
        && testMemoryBarrierBuffer()
        && testMemoryBarrierShared()
        && testMemoryBarrierImage()
        && testGroupMemoryBarrier()
        ;
}

void computeMain()
{
    outputBuffer.data[0] = true
        && testMemoryBarriers()
        ;
    // BUF: 1
}
