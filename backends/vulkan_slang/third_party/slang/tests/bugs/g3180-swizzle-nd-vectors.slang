//TEST:SIMPLE(filecheck=CHECK):-target cpp -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK):-target glsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK):-target hlsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-via-glsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-directly -entry computeMain -stage compute

// This test tests that assigning to a subscript of a unary swizzle works

RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    vector<vector<int, 2>, 2> v;
    v.x[0] = 1;
    // CHECK: ([[# @LINE+1]]): error 55200: 'vector<vector<{{.*}}>' is not a supported builtin type for the target.
    outputBuffer[dispatchThreadID.x] = v.x[0];
}
