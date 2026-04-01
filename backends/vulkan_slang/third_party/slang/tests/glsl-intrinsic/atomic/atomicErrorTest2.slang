//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target glsl -DTARGET_GLSL
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly -DTARGET_SPIRV
#version 430

// CHECK: error 20001

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):name=one
layout(binding 1) uniform atomic_uint one;

void computeMain()
{

}
