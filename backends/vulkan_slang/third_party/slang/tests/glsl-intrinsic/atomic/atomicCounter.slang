//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl -DTARGET_GLSL
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly -DTARGET_SPIRV
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
#version 430

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    uint data[2];
} outputBuffer;

// CHECK_GLSL-DAG: void main(
// CHECK_SPV-DAG: OpEntryPoint

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):name=one
layout(binding = 1, offset = 12) uniform atomic_uint one;
bool testSetterAndGetter()
{
    return true

// CHECK_GLSL-DAG: atomicExchange
// CHECK_SPV-DAG: OpAtomicExchange
        && atomicCounterExchange(one, 1) == 0

// CHECK_GLSL-DAG: atomicExchange
// no idea how to check the spirv reliabley...
        && atomicCounter(one) == 1
        && atomicCounterExchange(one, 5) == 1
        && atomicCounter(one) == 5
        ;
}

bool counterAsParam(atomic_uint param)
{
    return true
        && atomicCounterExchange(param, 5) != 100

// CHECK_GLSL-DAG: atomicAdd(
// CHECK_SPV-DAG: OpAtomicIAdd
        && atomicCounterIncrement(param) == 5
        && atomicCounter(param) == 6
        ;
}

// GLSL_CHECK-LABEL: bool testAtomicUint
// SPV_CHECK-LABEL: testAtomicUint
bool testAtomicUint()
{
// ensure the code emits for `one` index into [3] for 12/4
// CHECK_GLSL-DAG: {{.*}}_data_{{.*}}[3]{{.*}}
    return true
    
        && atomicCounterExchange(one, 5) != 100
// CHECK_GLSL-DAG: atomicAdd(
// CHECK_SPV-DAG: OpAtomicIIncrement
        && atomicCounterIncrement(one) == 5
        && atomicCounter(one) == 6

        && atomicCounterExchange(one, 5) != 100
// CHECK_GLSL-DAG: atomicExchange(
// CHECK_SPV-DAG: OpAtomicIDecrement
        && atomicCounterDecrement(one) == 4

        && atomicCounterExchange(one, 5) != 100
// CHECK_GLSL-DAG: atomicAdd(
        && atomicCounterAdd(one, 1) == 5
        && atomicCounter(one) == 6

        && atomicCounterExchange(one, 5) != 100
// CHECK_GLSL-DAG: atomicExchange
// CHECK_SPV-DAG: OpAtomicISub
        && atomicCounterSubtract(one, 1) == 5
        && atomicCounter(one) == 4

        && atomicCounterExchange(one, 5) != 100
// CHECK_GLSL-DAG: atomicMin(
// CHECK_SPV-DAG: OpAtomicUMin
        && atomicCounterMin(one, 1) == 5
        && atomicCounter(one) == 1

        && atomicCounterExchange(one, 5) != 100
// CHECK_GLSL-DAG: atomicMax(
// CHECK_SPV-DAG: OpAtomicUMax
        && atomicCounterMax(one, 1) == 5 
        && atomicCounter(one) == 5

// CHECK_GLSL-DAG: atomicAnd(
// CHECK_SPV: OpAtomicAnd
        && atomicCounterExchange(one, 5) != 100
        && atomicCounterAnd(one, 2) == 5
        && atomicCounter(one) == 0

// CHECK_GLSL-DAG: atomicOr(
// CHECK_SPV-DAG: OpAtomicOr
        && atomicCounterExchange(one, 5) != 100
        && atomicCounterOr(one, 8) == 5
        && atomicCounter(one) == 13

// CHECK_GLSL-DAG: atomicXor(
// CHECK_SPV-DAG: OpAtomicXor
        && atomicCounterExchange(one, 5) != 100
        && atomicCounterXor(one, 4) == 5
        && atomicCounter(one) == 1

// CHECK_GLSL-DAG: atomicCompSwap(
// CHECK_SPV-DAG: OpAtomicCompareExchange
        && atomicCounterExchange(one, 5) != 100
        && atomicCounterCompSwap(one, 5, 3) == 5 
        && atomicCounter(one) == 3

// CHECK_GLSL-DAG: atomicCompSwap(
// CHECK_SPV-DAG: OpAtomicCompareExchange
        && atomicCounterExchange(one, 5) != 100
        && atomicCounterCompSwap(one, 5, 3) == 5 
        && atomicCounter(one) == 3

        && counterAsParam(one);
        ;
}

void computeMain()
{
    outputBuffer.data[0] = true
            && testSetterAndGetter()
            ;
    outputBuffer.data[1] = true
            && testAtomicUint()
            ;
    // BUF: 1
    // BUF-NEXT: 1
}
