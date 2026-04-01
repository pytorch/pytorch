//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_NOT_SUPP): -entry computeMain -stage compute -target spirv
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_NOT_SUPP): -entry computeMain -stage compute -target glsl
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_NOT_SUPP): -entry computeMain -stage compute -target wgsl

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_SUPP):
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_SUPP): -entry computeMain -stage compute -target hlsl
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_SUPP): -entry computeMain -stage compute -target metal
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_SUPP): -entry computeMain -stage compute -target cpp
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_SUPP): -entry computeMain -stage compute -target cuda

// Some compilation targets allow missing returns while some do not.
// This test ensures that either errors and warnings are emitted appropriately.

RWStructuredBuffer<uint> out;

// CHECK_NOT_SUPP: warning 41010: non-void function
// CHECK_NOT_SUPP: error 41009: non-void function

// CHECK_SUPP: warning 41010: non-void function

uint func()
{
}

[shader("compute")]
void computeMain()
{
    out[0] = func();
}
