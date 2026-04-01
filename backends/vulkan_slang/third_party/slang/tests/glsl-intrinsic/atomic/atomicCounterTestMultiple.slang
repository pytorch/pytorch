//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl -DTARGET_GLSL
#version 430

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName
{
    uint data[1];
} outputBuffer;

layout(binding = 1, offset = 12) uniform atomic_uint one; 
layout(binding = 1) uniform atomic_uint two; 

// Layout qualifiers can be specified in any order.
layout(offset = 4, binding = 1) uniform atomic_uint three; 

layout(binding = 1) uniform atomic_uint four; 
layout(binding = 2) uniform atomic_uint five; 

void computeMain()
{

    outputBuffer.data[0] = true
// CHECK_GLSL: one_0._data_{{.*}}[3]
        && atomicCounter(one) == 0
// CHECK_GLSL: one_0._data_{{.*}}[4]
        && atomicCounter(two) == 0
// CHECK_GLSL: one_0._data_{{.*}}[1]
        && atomicCounter(three) == 0
// CHECK_GLSL: one_0._data_{{.*}}[2]
        && atomicCounter(four) == 0
// CHECK_GLSL: five_0._data_{{.*}}[0]
        && atomicCounter(five) == 0
        
        ;

}
