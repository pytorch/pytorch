//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target glsl -DTARGET_GLSL
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):  -allow-glsl -stage compute -entry computeMain -target spirv -skip-spirv-validation -emit-spirv-directly -DTARGET_SPIRV
#version 430

uint notShared;

void computeMain()
{
    // CHECK: ([[# @LINE+1]]): error 41403
    atomicAdd(notShared, 1u);
    // CHECK: ([[# @LINE+1]]): error 41403
    atomicAnd(notShared, 1u);
    // CHECK: ([[# @LINE+1]]): error 41403
    atomicCompSwap(notShared, 1u, 2u);
    // CHECK: ([[# @LINE+1]]): error 41403
    atomicExchange(notShared, 1u);
    // CHECK: ([[# @LINE+1]]): error 41403
    atomicMax(notShared, 1u);
    // CHECK: ([[# @LINE+1]]): error 41403
    atomicMin(notShared, 1u);
    // CHECK: ([[# @LINE+1]]): error 41403
    atomicOr(notShared, 1u);
    // CHECK: ([[# @LINE+1]]): error 41403
    atomicXor(notShared, 1u);
}
