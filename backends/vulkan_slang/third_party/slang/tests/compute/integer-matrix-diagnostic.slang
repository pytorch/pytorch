// Check that using matrices with integer floating point type yields the correct diagnostic

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target glsl -entry computeMain -stage compute
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target metal -entry computeMain -stage compute
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv -entry computeMain -stage compute
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target wgsl -entry computeMain -stage compute

cbuffer MatrixBuffer
{
    // CHECK: error 38202
    int4x4 iMatrix;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name=outputBuffer
RWStructuredBuffer<int4> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    outputBuffer[index] = iMatrix[0][0];
}